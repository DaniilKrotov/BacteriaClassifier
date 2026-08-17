import os
import cv2
import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict
from collections import Counter

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Ваш файл
from petri_dish_detector import process_image


DEFAULT_CFG = {
    "output_size": 1024,
    "padding": 0.06,
    "fill_black": True,
    "debug": False,
}


def circular_mask(size: int, radius_ratio: float = 0.96) -> np.ndarray:
    m = np.zeros((size, size), dtype=np.uint8)
    c = size // 2
    r = int(c * radius_ratio)
    cv2.circle(m, (c, c), r, 255, -1)
    return m


def compute_lbp_u8(gray: np.ndarray) -> np.ndarray:
    if gray.ndim != 2:
        raise ValueError("LBP expects grayscale image")

    c = gray[1:-1, 1:-1]
    lbp = np.zeros_like(c, dtype=np.uint8)

    neighbors = [
        gray[0:-2, 0:-2],
        gray[0:-2, 1:-1],
        gray[0:-2, 2:  ],
        gray[1:-1, 2:  ],
        gray[2:  , 2:  ],
        gray[2:  , 1:-1],
        gray[2:  , 0:-2],
        gray[1:-1, 0:-2],
    ]

    for bit, n in enumerate(neighbors):
        lbp |= ((n >= c).astype(np.uint8) << bit)

    out = np.zeros_like(gray, dtype=np.uint8)
    out[1:-1, 1:-1] = lbp
    return out


def patch_features(patch_bgr: np.ndarray, patch_mask: np.ndarray = None) -> np.ndarray:
    """
    Фиксированный размер вектора признаков: 64
    """
    target_len = 64

    if patch_bgr.size == 0:
        return np.zeros(target_len, dtype=np.float32)

    h, w = patch_bgr.shape[:2]
    if h < 8 or w < 8:
        return np.zeros(target_len, dtype=np.float32)

    if patch_mask is None:
        patch_mask = np.ones((h, w), dtype=np.uint8) * 255

    valid = patch_mask > 0
    if valid.sum() < 16:
        return np.zeros(target_len, dtype=np.float32)

    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)

    feats = []

    # Цветовые статистики BGR/HSV/Lab
    for arr in [patch_bgr, hsv, lab]:
        for ch in range(3):
            vals = arr[..., ch][valid]
            feats.extend([
                float(np.mean(vals)),
                float(np.std(vals)),
                float(np.percentile(vals, 10)),
                float(np.percentile(vals, 50)),
                float(np.percentile(vals, 90)),
            ])

    # Края / детализация
    edges = cv2.Canny(gray, 60, 160)
    edge_density = float((edges[valid] > 0).mean())
    lap_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())

    # Грубая оценка "пятен"
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thr = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 4
    )
    thr = cv2.bitwise_and(thr, thr, mask=patch_mask)
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thr, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA] if num_labels > 1 else np.array([], dtype=np.int32)
    if len(areas) > 0:
        areas = areas[areas >= 3]

    blob_count = int(len(areas))
    blob_area_sum = float(areas.sum()) if len(areas) else 0.0
    blob_area_mean = float(areas.mean()) if len(areas) else 0.0
    blob_area_std = float(areas.std()) if len(areas) else 0.0
    blob_fill_ratio = float(blob_area_sum / (valid.sum() + 1e-6))

    # LBP histogram (16 bins)
    lbp = compute_lbp_u8(gray)
    lbp_vals = lbp[valid]
    lbp_hist, _ = np.histogram(lbp_vals, bins=16, range=(0, 256))
    lbp_hist = lbp_hist.astype(np.float32)
    lbp_hist /= (lbp_hist.sum() + 1e-6)

    feats.extend([
        edge_density,
        lap_var,
        blob_count,
        blob_area_sum,
        blob_area_mean,
        blob_area_std,
        blob_fill_ratio,
    ])
    feats.extend(lbp_hist.tolist())

    f = np.asarray(feats, dtype=np.float32)

    if f.shape[0] < target_len:
        f = np.pad(f, (0, target_len - f.shape[0]), mode="constant")
    elif f.shape[0] > target_len:
        f = f[:target_len]

    return f


def grid_extract_features(
    norm_bgr: np.ndarray,
    grid_rows: int = 6,
    grid_cols: int = 6,
    circle_radius_ratio: float = 0.96,
    min_cell_valid_ratio: float = 0.20
) -> np.ndarray:
    H, W = norm_bgr.shape[:2]
    if H != W:
        raise ValueError("Ожидается квадратное нормализованное изображение")

    dish_mask = circular_mask(H, radius_ratio=circle_radius_ratio)

    cell_h = H // grid_rows
    cell_w = W // grid_cols

    all_feats = []
    cell_presence = []

    for r in range(grid_rows):
        for c in range(grid_cols):
            y1 = r * cell_h
            y2 = H if r == grid_rows - 1 else (r + 1) * cell_h
            x1 = c * cell_w
            x2 = W if c == grid_cols - 1 else (c + 1) * cell_w

            patch = norm_bgr[y1:y2, x1:x2]
            pmask = dish_mask[y1:y2, x1:x2]

            valid_ratio = float((pmask > 0).mean())
            cell_presence.append(valid_ratio)

            if valid_ratio < min_cell_valid_ratio:
                all_feats.append(np.zeros(64, dtype=np.float32))
                continue

            all_feats.append(patch_features(patch, pmask))

    global_f = patch_features(norm_bgr, dish_mask)
    cell_presence = np.asarray(cell_presence, dtype=np.float32)

    feature_vector = np.concatenate([
        np.concatenate(all_feats, axis=0),
        global_f,
        cell_presence
    ], axis=0)

    return feature_vector.astype(np.float32)


def collect_dataset(
    dataset_dir: str,
    grid_rows: int,
    grid_cols: int,
    cfg: Dict,
    exts=(".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"),
    augment: bool = False,
    augment_per_image: int = 4,
    seed: int = 42
):
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"dataset dir not found: {dataset_dir}")

    X, y, paths = [], [], []
    class_names = sorted([p.name for p in dataset_dir.iterdir() if p.is_dir()])

    if not class_names:
        raise ValueError("В dataset_dir нет подпапок классов")

    print("Найдены классы:", class_names)

    total_files = 0
    skipped_no_petri = 0

    for class_name in class_names:
        class_dir = dataset_dir / class_name
        files = sorted([p for p in class_dir.iterdir() if p.suffix.lower() in exts])

        if not files:
            print(f"[WARN] Папка класса пуста: {class_dir}")
            continue

        for fp in files:
            total_files += 1
            try:
                norm, dbg, model = process_image(str(fp), cfg)

                # process_image возвращает None и когда файл не прочитан (Unicode/imread), и когда чашка не найдена
                if norm is None:
                    skipped_no_petri += 1
                    print(f"[SKIP] Не удалось обработать (чтение файла или чашка не найдена): {fp}")
                    continue

                feats = grid_extract_features(
                    norm,
                    grid_rows=grid_rows,
                    grid_cols=grid_cols,
                    circle_radius_ratio=0.96
                )

                X.append(feats)
                y.append(class_name)
                paths.append(str(fp))

            except Exception as e:
                print(f"[ERR] {fp}: {e}")

    if len(X) == 0:
        raise ValueError("Не удалось собрать ни одного образца. Проверьте данные/детекцию.")

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)

    print(f"\n[INFO] Всего файлов: {total_files}")
    print(f"[INFO] Успешно обработано: {len(X)}")
    print(f"[INFO] Пропущено: {skipped_no_petri}")

    return X, y, paths, class_names


def train_model(X, y, model_type: str = "rf", random_state: int = 42):
    if model_type == "rf":
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1
        )
        pipe = Pipeline([("clf", clf)])
    elif model_type == "svm":
        clf = SVC(
            C=5.0,
            kernel="rbf",
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=random_state
        )
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", clf)
        ])
    else:
        raise ValueError("model_type must be 'rf' or 'svm'")

    pipe.fit(X, y)
    return pipe


def save_artifacts(out_dir: str, model, meta: dict):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    model_path = out / "bacteria_grid_model.pkl"
    meta_path = out / "bacteria_grid_meta.json"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] model: {model_path}")
    print(f"[OK] meta : {meta_path}")

def rotate_bound(image: np.ndarray, angle: float) -> np.ndarray:
    """Поворот с сохранением размера холста (здесь потом вернём к исходному размеру)."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )
    return rotated


def adjust_brightness_contrast(img: np.ndarray, alpha: float = 1.0, beta: float = 0.0) -> np.ndarray:
    """
    alpha: контраст (0.9..1.1)
    beta : яркость (-20..20)
    """
    out = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return out


def add_gaussian_noise(img: np.ndarray, sigma: float = 4.0, rng=None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.normal(0, sigma, img.shape).astype(np.float32)
    out = img.astype(np.float32) + noise
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def random_small_affine(img: np.ndarray, rng=None) -> np.ndarray:
    """Небольшой сдвиг/масштаб без сильных искажений."""
    if rng is None:
        rng = np.random.default_rng()

    h, w = img.shape[:2]
    tx = int(rng.integers(-w // 50, w // 50 + 1))   # ~ ±2%
    ty = int(rng.integers(-h // 50, h // 50 + 1))
    scale = float(rng.uniform(0.96, 1.04))

    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, 0, scale)
    M[0, 2] += tx
    M[1, 2] += ty

    out = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )
    return out


def generate_augmentations(norm_bgr: np.ndarray, rng_seed: int = 42):
    """
    Возвращает список аугментированных версий нормализованной чашки.
    Первая картинка НЕ включается (оригинал добавляем отдельно).
    """
    rng = np.random.default_rng(rng_seed)

    augs = []

    # Безопасные повороты (для чашки Петри обычно допустимы)
    for angle in [90, 180, 270]:
        augs.append(rotate_bound(norm_bgr, angle))

    # Небольшие углы (чтобы не слишком искажать)
    for angle in [-10, -5, 5, 10]:
        augs.append(rotate_bound(norm_bgr, angle))

    # Варианты яркости/контраста
    augs.append(adjust_brightness_contrast(norm_bgr, alpha=0.95, beta=-10))
    augs.append(adjust_brightness_contrast(norm_bgr, alpha=1.05, beta=10))
    augs.append(adjust_brightness_contrast(norm_bgr, alpha=1.10, beta=0))

    # Лёгкий blur и шум
    augs.append(cv2.GaussianBlur(norm_bgr, (3, 3), 0))
    augs.append(add_gaussian_noise(norm_bgr, sigma=3.0, rng=rng))

    # Небольшая affine-трансформация
    augs.append(random_small_affine(norm_bgr, rng=rng))

    return augs


def main():
    dataset_dir = "petri_dataset1"
    out_dir = "trained_bacteria_model1"
    grid_rows = 6
    grid_cols = 6
    model_type = "rf"
    test_size = 0.2
    seed = 42

    cfg = {
        "output_size": 1024,
        "padding": 0.0,
        "fill_black": True,
        "debug": False,
    }

    X, y, paths, class_names = collect_dataset(
        dataset_dir=dataset_dir,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        cfg=cfg
    )

    print(f"\nСобрано образцов: {len(X)}")
    print(f"Размер вектора признаков: {X.shape[1]}")

    # ИСПРАВЛЕНО: защита от падения stratify при классах с 1 образцом
    class_counts = Counter(y.tolist())
    
    print("\nКоличество примеров по классам:")
    for cls, cnt in sorted(class_counts.items()):
        print(f"  {cls}: {cnt}")
    
    n_samples = len(y)
    n_classes = len(class_counts)

    if n_classes < 2:
        raise ValueError("Для обучения нужно минимум 2 класса с образцами.")
    
    min_count = min(class_counts.values())

    # stratify можно использовать только если в каждом классе >= 2 примеров
    stratify = y if min_count >= 2 else None

    if stratify is None:
        print("[WARN] Stratified split отключен: есть классы с < 2 примерами.")
    
    # ---- НОВОЕ: проверка test_size для stratify ----
    test_size = test_size  # фиксированное значение

    if stratify is not None:
    # Сколько объектов попадет в тест
        if isinstance(test_size, float):
            n_test = int(np.ceil(n_samples * test_size))
        else:
            n_test = int(test_size)

        # Для stratify нужно хотя бы по 1 образцу на класс в тесте
        if n_test < n_classes:
            # Подбираем минимально допустимый test_size
            min_test_float = n_classes / n_samples
            safe_test_size = max(test_size, min_test_float)

            # Небольшой запас, чтобы избежать пограничных округлений
            if isinstance(safe_test_size, float):
                safe_test_size = min(0.5, max(safe_test_size + 1e-6, test_size))

            print(
                f"[WARN] test_size слишком мал для stratify: "
                f"n_test={n_test}, classes={n_classes}. "
                f"Авто-увеличение test_size: {test_size} -> {safe_test_size:.3f}"
            )
            test_size = safe_test_size

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=seed,
        stratify=stratify
    )

    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    model = train_model(X_train, y_train, model_type=model_type, random_state=seed)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("\n=== RESULT ===")
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    meta = {
        "grid_rows": grid_rows,
        "grid_cols": grid_cols,
        "feature_version": "grid_patch_v1",
        "petri_cfg": cfg,
        "classes_seen": sorted(list(set(y.tolist()))),
        "model_type": model_type,
        "feature_dim": int(X.shape[1]),
    }

    save_artifacts(out_dir, model, meta)


if __name__ == "__main__":
    main()