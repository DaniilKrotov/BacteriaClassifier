import cv2
import numpy as np
import os
import glob
from pathlib import Path


def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    """
    Безопасное чтение изображений с кириллицей/Unicode в пути (Windows).
    """
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        img = cv2.imdecode(data, flags)
        return img
    except Exception:
        return None


def imwrite_unicode(path, img):
    """
    Безопасная запись изображений с Unicode в пути (Windows).
    """
    try:
        ext = Path(path).suffix
        if not ext:
            ext = ".png"
            path = str(path) + ext
        ok, buf = cv2.imencode(ext, img)
        if not ok:
            return False
        buf.tofile(path)
        return True
    except Exception:
        return False

def auto_canny(gray, sigma=0.33):
    v = np.median(gray)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(gray, lower, upper)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def resize_long_side(img, target=1400):
    h, w = img.shape[:2]
    long_side = max(h, w)
    if long_side <= target:
        return img, 1.0
    scale = target / long_side
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    out = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return out, scale

def build_edge_map(gray):
    # Усиление локального контраста + сглаживание (блики/шум)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    g = clahe.apply(gray)
    g = cv2.bilateralFilter(g, d=7, sigmaColor=40, sigmaSpace=40)
    g = cv2.GaussianBlur(g, (7, 7), 0)

    # Градиенты + canny (вместе работают лучше на “ободке” чашки)
    edges1 = auto_canny(g, sigma=0.33)
    sobx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    soby = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(sobx, soby)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, edges2 = cv2.threshold(mag, 35, 255, cv2.THRESH_BINARY)

    edges = cv2.bitwise_or(edges1, edges2)

    # Закрытие разрывов
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)

    return edges


def ellipse_edge_support(edges, ellipse, band=3):
    """
    Сколько пикселей ребра попадает в тонкую "полосу" вдоль эллипса.
    Возвращает support_ratio (0..1 примерно).
    """
    (cx, cy), (MA, ma), angle = ellipse  # MA/ma: длины осей
    a = MA / 2.0
    b = ma / 2.0
    if a < 10 or b < 10:
        return 0.0

    h, w = edges.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    # Полоса: эллипс толщиной 2*band
    cv2.ellipse(mask, (int(cx), int(cy)), (int(a), int(b)), angle, 0, 360, 255, thickness=2 * band)
    edge_hits = cv2.countNonZero(cv2.bitwise_and(edges, mask))
    band_area = cv2.countNonZero(mask)
    if band_area == 0:
        return 0.0
    return edge_hits / band_area


def perimeter_in_bounds_ratio(shape, ellipse, n=180, margin=2):
    """Доля периметра эллипса, попадающая в границы кадра."""
    h, w = shape[:2]
    (cx, cy), (MA, ma), angle = ellipse
    a = MA / 2.0
    b = ma / 2.0
    if a < 2 or b < 2:
        return 0.0
    ang = np.deg2rad(angle)
    cosA, sinA = np.cos(ang), np.sin(ang)
    inside = 0
    for t in np.linspace(0, 2 * np.pi, n, endpoint=False):
        x = a * np.cos(t)
        y = b * np.sin(t)
        xr = x * cosA - y * sinA + cx
        yr = x * sinA + y * cosA + cy
        if margin <= xr < (w - margin) and margin <= yr < (h - margin):
            inside += 1
    return inside / float(n)


def ellipse_edge_support_scaled(edges, ellipse, band=3, scale=1.0):
    """Поддержка ребрами вдоль эллипса, но с масштабированием полуосей."""
    (cx, cy), (MA, ma), angle = ellipse
    MA2 = MA * scale
    ma2 = ma * scale
    a = max(2, int(round(MA2 / 2.0)))
    b = max(2, int(round(ma2 / 2.0)))
    h, w = edges.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    cv2.ellipse(mask, (int(round(cx)), int(round(cy))), (a, b), angle, 0, 360, 255, thickness=2 * band)
    edge_hits = cv2.countNonZero(cv2.bitwise_and(edges, mask))
    band_area = cv2.countNonZero(mask)
    if band_area == 0:
        return 0.0
    return edge_hits / band_area


def refine_ellipse_axes_by_edge_support(edges, ellipse, band=3, scale_min=0.85, scale_max=1.25, step=0.01):
    """Подбирает масштаб полуосей, чтобы эллипс "сел" на внешний ободок."""
    (cx, cy), (MA, ma), angle = ellipse
    best = ellipse
    best_score = -1e18

    for s in np.arange(scale_min, scale_max + 1e-9, step):
        cand = ((cx, cy), (MA * s, ma * s), angle)

        # Не допускаем эллипс, который сильно вылезает за кадр
        if perimeter_in_bounds_ratio(edges.shape, cand, n=180, margin=2) < 0.85:
            continue

        sup = ellipse_edge_support_scaled(edges, ellipse, band=band, scale=s)

        # Лёгкая регуляризация: не "раздувать" без надобности
        score = sup - 0.15 * abs(s - 1.0)

        if score > best_score:
            best_score = score
            best = cand

    return best


def contour_circularity(cnt):
    area = cv2.contourArea(cnt)
    per = cv2.arcLength(cnt, True)
    if per <= 1e-6:
        return 0.0
    return float(4.0 * np.pi * area / (per * per))



def score_ellipse_candidate(edges, cnt, ellipse, img_center, img_area):
    (cx, cy), (MA, ma), angle = ellipse
    a = MA / 2.0
    b = ma / 2.0

    # Фильтры здравого смысла
    area = cv2.contourArea(cnt)
    if area < 0.05 * img_area:   # слишком маленькое
        return -1e9
    if area > 0.98 * img_area:   # слишком большое (скорее фон/рамка)
        return -1e9

    # Эллипс не должен быть слишком вытянутым
    aspect = max(a, b) / (min(a, b) + 1e-6)
    if aspect > 2.8:
        return -1e9

    # Эллипс должен почти полностью лежать в кадре (иначе часто ловим "огромные" ложные круги)
    inb = perimeter_in_bounds_ratio(edges.shape, ellipse, n=180, margin=2)
    if inb < 0.85:
        return -1e9

    # Поддержка ребрами вдоль эллипса:
    # берём максимум по нескольким масштабам, чтобы "поймать" внешний ободок
    supports = [
        ellipse_edge_support_scaled(edges, ellipse, band=3, scale=1.00),
        ellipse_edge_support_scaled(edges, ellipse, band=3, scale=1.06),
        ellipse_edge_support_scaled(edges, ellipse, band=3, scale=1.12),
    ]
    support = float(max(supports))

    # Круговость контура
    circ = contour_circularity(cnt)

    # Близость центра к центру кадра (мягкий бонус)
    icx, icy = img_center
    dist = np.hypot(cx - icx, cy - icy)
    diag = np.hypot(edges.shape[1], edges.shape[0])
    center_bonus = 1.0 - (dist / (diag + 1e-6))  # ~0..1

    # Нормализованная площадь (сильнее стимулируем "крупную" чашку,
    # чтобы не выбирать внутреннюю границу среды вместо внешнего ободка)
    area_norm = area / (img_area + 1e-6)

    # Итоговый скоринг
    score = (
        4.2 * support +
        1.2 * circ +
        1.1 * center_bonus +
        1.8 * area_norm +
        0.4 * (1.0 / aspect) +
        1.0 * inb
    )
    return score


def detect_petri_model(img_bgr, debug=False):
    """
    Возвращает модель: (cx, cy, a, b, angle_deg) в координатах img_bgr,
    где a,b — полуоси (pixels), angle — поворот эллипса (deg).
    """
    img_small, scale = resize_long_side(img_bgr, target=1400)
    gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
    edges = build_edge_map(gray)

    h, w = gray.shape[:2]
    img_area = float(h * w)
    img_center = (w / 2.0, h / 2.0)

    best = None
    best_score = -1e18

    # --- 0) Кандидат из цветовой/яркостной маски (обычно даёт "внешний" контур чашки) ---
    mask_color = build_color_dish_mask(img_small)

    # Доп. маска по яркости: помогает, когда среда почти однородная/тёмная, а ободок прозрачный
    mask_luma = build_luma_dish_mask(img_small)

    mask = cv2.bitwise_or(mask_color, mask_luma)
    cnts_m, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if cnts_m:
        cnt_m = max(cnts_m, key=cv2.contourArea)
        area_m = cv2.contourArea(cnt_m)
        if len(cnt_m) >= 80 and area_m > 0.08 * img_area:
            try:
                ell_m = cv2.fitEllipse(cnt_m)
                # небольшой бонус доверия к маске
                s_m = score_ellipse_candidate(edges, cnt_m, ell_m, img_center, img_area) + 0.6
                if s_m > best_score:
                    best, best_score = ell_m, s_m
            except cv2.error:
                pass

    # --- 1) Контуры по ребрам -> fitEllipse кандидаты ---
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        if len(cnt) < 80:
            continue
        area = cv2.contourArea(cnt)
        if area < 0.03 * img_area:
            continue

        try:
            ell = cv2.fitEllipse(cnt)
        except cv2.error:
            continue

        s = score_ellipse_candidate(edges, cnt, ell, img_center, img_area)
        if s > best_score:
            best_score = s
            best = ell

    
    # --- 1b) HoughCircles как дополнительные кандидаты (часто ловит внешний ободок лучше контуров) ---
    g = cv2.GaussianBlur(gray, (9, 9), 0)
    minR = int(min(h, w) * 0.25)
    maxR = int(min(h, w) * 0.65)
    circles = cv2.HoughCircles(
        g, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=min(h, w) * 0.35,
        param1=120, param2=22,
        minRadius=minR, maxRadius=maxR
    )
    if circles is not None and len(circles) > 0:
        circles = circles[0]
        # берём несколько лучших по поддержке ребрами
        for x, y, r in circles[:10]:
            # круг должен быть почти полностью в кадре
            # допускаем небольшое "вылезание" за кадр (обрежется при нормализации)
            if x - r < -0.05 * r or y - r < -0.05 * r or x + r > w + 0.05 * r or y + r > h + 0.05 * r:
                continue

            ell_c = ((float(x), float(y)), (float(2 * r), float(2 * r)), 0.0)
            # подгоним радиус под реальный ободок по ребрам
            ell_c = refine_ellipse_axes_by_edge_support(edges, ell_c, band=3, scale_min=0.85, scale_max=1.25, step=0.01)

            inb = perimeter_in_bounds_ratio(edges.shape, ell_c, n=180, margin=2)
            if inb < 0.85:
                continue

            support = max(
                ellipse_edge_support_scaled(edges, ell_c, band=3, scale=1.00),
                ellipse_edge_support_scaled(edges, ell_c, band=3, scale=1.08),
                ellipse_edge_support_scaled(edges, ell_c, band=3, scale=1.16),
            )

            (cx, cy), (MA, ma), _ = ell_c
            a = MA / 2.0
            b = ma / 2.0
            area_est = np.pi * a * b
            area_norm = area_est / (img_area + 1e-6)
            dist = np.hypot(cx - img_center[0], cy - img_center[1])
            diag = np.hypot(w, h)
            center_bonus = 1.0 - (dist / (diag + 1e-6))

            s_c = 4.6 * support + 1.8 * area_norm + 0.9 * center_bonus + 1.0 * inb
            if s_c > best_score:
                best_score = s_c
                best = ell_c


# --- 2) Fallback: эллипс по всем точкам ребер (если контур рваный) ---
    if best is None or best_score < 1.0:
        ell_e = fit_ellipse_from_edge_points(edges)
        if ell_e is not None and perimeter_in_bounds_ratio(edges.shape, ell_e) >= 0.90:
            support = max(
                ellipse_edge_support_scaled(edges, ell_e, band=3, scale=1.00),
                ellipse_edge_support_scaled(edges, ell_e, band=3, scale=1.08),
                ellipse_edge_support_scaled(edges, ell_e, band=3, scale=1.16),
            )

            (cx, cy), (MA, ma), angle = ell_e
            a = MA / 2.0
            b = ma / 2.0
            aspect = max(a, b) / (min(a, b) + 1e-6)

            dist = np.hypot(cx - img_center[0], cy - img_center[1])
            diag = np.hypot(w, h)
            center_bonus = 1.0 - (dist / (diag + 1e-6))

            area_est = np.pi * a * b
            area_norm = area_est / (img_area + 1e-6)

            s_e = 4.0 * support + 1.0 * center_bonus + 1.6 * area_norm + 0.3 * (1.0 / aspect)
            if s_e > best_score:
                best_score = s_e
                best = ell_e

    # --- 3) Fallback: HoughCircles (когда вид сверху почти круг) ---
    if best is None or best_score < 1.0:
        g = cv2.GaussianBlur(gray, (9, 9), 0)
        minR = int(min(h, w) * 0.25)
        maxR = int(min(h, w) * 0.60)
        circles = cv2.HoughCircles(
            g, cv2.HOUGH_GRADIENT,
            dp=1.2, minDist=min(h, w) * 0.35,
            param1=120, param2=32,
            minRadius=minR, maxRadius=maxR
        )
        if circles is not None and len(circles) > 0:
            circles = np.round(circles[0]).astype(int)

            best_c = None
            best_cs = -1e18
            for x, y, r in circles:
                # допускаем небольшое "вылезание" за кадр (обрежется при нормализации)
                if x - r < -0.05 * r or y - r < -0.05 * r or x + r > w + 0.05 * r or y + r > h + 0.05 * r:
                    continue
                dist = np.hypot(x - img_center[0], y - img_center[1])
                diag = np.hypot(w, h)
                center_bonus = 1.0 - dist / (diag + 1e-6)
                score = 0.9 * (r / (maxR + 1e-6)) + 0.4 * center_bonus
                if score > best_cs:
                    best_cs = score
                    best_c = (x, y, r)

            if best_c is not None:
                x, y, r = best_c
                best = ((float(x), float(y)), (float(2 * r), float(2 * r)), 0.0)

    if best is None:
        return None

    # --- 4) ВАЖНО: подгоняем полуоси под внешний ободок (часто самый тонкий контур) ---
    best = refine_ellipse_axes_by_edge_support(edges, best, band=3, scale_min=0.85, scale_max=1.25, step=0.01)

    (cx, cy), (MA, ma), angle = best
    a = MA / 2.0
    b = ma / 2.0

    # обратно в исходный масштаб
    inv = 1.0 / scale
    cx *= inv
    cy *= inv
    a *= inv
    b *= inv

    return (float(cx), float(cy), float(a), float(b), float(angle))

def warp_ellipse_to_circle(img_bgr, model, output_size=1024, padding=0.06, fill_black=True):
    """
    Поворот + масштабирование, чтобы эллипс стал кругом.
    padding — доля радиуса, добавляемая вокруг чашки.
    """
    cx, cy, a, b, angle = model
    h, w = img_bgr.shape[:2]

    # 1) Повернуть так, чтобы большая ось стала “горизонтальной”
    Mrot = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(img_bgr, Mrot, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # 2) После поворота — сделать b -> a (масштаб по Y)
    # (в координатах после поворота)
    scale_y = a / (b + 1e-6)

    new_h = int(round(h * scale_y))
    scaled = cv2.resize(rotated, (w, new_h), interpolation=cv2.INTER_LINEAR)

    # центр тоже масштабируется по Y
    cy2 = cy * scale_y
    cx2 = cx

    # 3) Кроп по кругу радиуса a (с небольшим паддингом)
    r = a * (1.0 + padding)
    x1 = int(round(cx2 - r))
    x2 = int(round(cx2 + r))
    y1 = int(round(cy2 - r))
    y2 = int(round(cy2 + r))

    # паддинг, если вышли за границы
    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - w)
    pad_bottom = max(0, y2 - new_h)

    if any([pad_left, pad_top, pad_right, pad_bottom]):
        scaled = cv2.copyMakeBorder(
            scaled, pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_REPLICATE
        )
        x1 += pad_left
        x2 += pad_left
        y1 += pad_top
        y2 += pad_top

    crop = scaled[y1:y2, x1:x2].copy()

    # 4) Маска круга (вырезаем чашку)
    if fill_black:
        ch, cw = crop.shape[:2]
        mask = np.zeros((ch, cw), np.uint8)
        ccx, ccy = cw // 2, ch // 2
        rr = int(round(min(ccx, ccy) * (1.0 - 0.02)))
        cv2.circle(mask, (ccx, ccy), rr, 255, -1)
        crop[mask == 0] = (0, 0, 0)

    # 5) Нормализация размера
    out = cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_AREA)
    return out


def draw_debug(img_bgr, model):
    cx, cy, a, b, angle = model
    vis = img_bgr.copy()
    cv2.ellipse(vis, (int(cx), int(cy)), (int(a), int(b)), angle, 0, 360, (0, 255, 0), 3)
    cv2.circle(vis, (int(cx), int(cy)), 4, (0, 255, 0), -1)
    return vis

def process_image(path, cfg):
    # ВАЖНО: Unicode-safe чтение (кириллица в путях)
    img = imread_unicode(path)
    if img is None:
        # возвращаем None как раньше, чтобы не ломать внешний код
        return None, None, None

    model = detect_petri_model(img, debug=cfg["debug"])
    if model is None:
        return None, None, None

    norm = warp_ellipse_to_circle(
        img, model,
        output_size=cfg["output_size"],
        padding=cfg["padding"],
        fill_black=cfg["fill_black"]
    )

    dbg = draw_debug(img, model) if cfg["debug"] else None
    return norm, dbg, model


def process_folder(input_dir, output_dir, cfg):
    input_dir = str(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    debug_dir = output_dir / "debug"
    if cfg["debug"]:
        debug_dir.mkdir(parents=True, exist_ok=True)

    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
    files = []
    for f in glob.glob(os.path.join(input_dir, "*.*")):
        if f.lower().endswith(exts):
            files.append(f)
    files.sort()

    if not files:
        print("Нет изображений в:", input_dir)
        return

    ok = 0
    for f in files:
        name = Path(f).stem
        print("Обработка:", Path(f).name)

        norm, dbg, model = process_image(f, cfg)

        if norm is None:
            print("Чашка не найдена")
            continue

        out_path = output_dir / f"processed_{name}.png"
        imwrite_unicode(str(out_path), norm)

        if cfg["debug"] and dbg is not None:
            imwrite_unicode(str(debug_dir / f"debug_{name}.png"), dbg)

        ok += 1
        cx, cy, a, b, ang = model
        print(f"  model: cx={cx:.1f}, cy={cy:.1f}, a={a:.1f}, b={b:.1f}, angle={ang:.1f}")

    print(f"\nГотово: {ok}/{len(files)} успешно.")


def build_color_dish_mask(img_bgr):
    """
    Делает маску области чашки/среды на фоне (особенно хорошо для белого стола).
    Возвращает бинарную mask (uint8 0/255).
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    # 1) Фон белый/серый обычно с НИЗКОЙ насыщенностью S
    #    Чашка/среда (розовая/красная) — S выше
    #    Порог лучше брать адаптивно от картинки:
    s_med = np.median(S)
    s_thr = int(clamp(max(25, 0.9 * s_med), 20, 80))  # обычно работает 25..80

    mask = (S > s_thr).astype(np.uint8) * 255

    # 2) Убираем шум и склеиваем области
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
                            iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)),
                            iterations=2)

    # 3) Берём самый большой компонент (обычно это сама чашка/среда)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return mask

    # stats: [label, x, y, w, h, area], 0 - фон
    biggest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    out = np.zeros_like(mask)
    out[labels == biggest] = 255

    # Ещё немного "подровнять" края
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
                           iterations=1)
    return out


def build_luma_dish_mask(img_bgr):
    """
    Маска чашки по яркости/контрасту (работает, когда насыщенность низкая или ободок прозрачный).
    Возвращает бинарную mask (uint8 0/255).
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]

    # Сглаживаем, чтобы убрать мелкий шум/колонии
    Lb = cv2.GaussianBlur(L, (9, 9), 0)

    # Otsu даёт разделение "объект/фон", но не знаем, что темнее.
    _, th = cv2.threshold(Lb, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask1 = th
    mask2 = cv2.bitwise_not(th)

    def post(mask):
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
                                iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)),
                                iterations=2)
        # самый большой компонент
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num <= 1:
            return mask
        biggest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        out = np.zeros_like(mask)
        out[labels == biggest] = 255
        return out

    m1 = post(mask1)
    m2 = post(mask2)

    # Выбираем, какая маска больше похожа на чашку: по площади и круговости
    def mask_score(mask):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            return -1e9
        cnt = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        if area <= 1:
            return -1e9
        circ = contour_circularity(cnt)
        h, w = mask.shape[:2]
        area_norm = area / float(h * w)
        # штраф за маску, которая "липнет" к краям (часто это фон)
        x, y, ww, hh = cv2.boundingRect(cnt)
        touches = (x <= 1) or (y <= 1) or (x + ww >= w - 2) or (y + hh >= h - 2)
        penalty = 0.25 if touches else 0.0
        return 1.3 * circ + 1.7 * area_norm - penalty

    return m1 if mask_score(m1) >= mask_score(m2) else m2



def fit_ellipse_from_edge_points(edges, min_points=250, max_points=6000):
    """
    Fallback: пытаемся аппроксимировать эллипс по всем точкам ребер.
    Помогает, когда контур чашки не замкнут/порвало бликами.
    Возвращает ellipse в формате OpenCV: ((cx,cy),(MA,ma),angle) или None.
    """
    ys, xs = np.where(edges > 0)
    n = len(xs)
    if n < min_points:
        return None

    # Если ребер слишком много — подвыборка, чтобы fitEllipse не был тяжелым
    if n > max_points:
        idx = np.random.choice(n, size=max_points, replace=False)
        xs = xs[idx]
        ys = ys[idx]

    pts = np.stack([xs, ys], axis=1).astype(np.int32).reshape(-1, 1, 2)

    try:
        ell = cv2.fitEllipse(pts)
    except cv2.error:
        return None

    # Быстрая валидация на адекватность
    (cx, cy), (MA, ma), angle = ell
    a = MA / 2.0
    b = ma / 2.0
    if a < 10 or b < 10:
        return None

    aspect = max(a, b) / (min(a, b) + 1e-6)
    if aspect > 3.0:  # слишком вытянуто — скорее мусор
        return None

    return ell

# ---------------------------
# Запуск с фиксированными настройками
# ---------------------------

def main():
    input_dir = "petri"
    output_dir = "petri_processed"

    cfg = {
        "output_size": 1024,
        "padding": 0.0,
        "fill_black": True,
        "debug": False,
    }
    process_folder(input_dir, output_dir, cfg)

if __name__ == "__main__":
    main()