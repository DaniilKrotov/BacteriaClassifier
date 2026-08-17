import cv2
import json
import pickle
import numpy as np
from pathlib import Path

# Ваш файл
from petri_dish_detector import process_image


def circular_mask(size: int, radius_ratio: float = 0.96) -> np.ndarray:
    m = np.zeros((size, size), dtype=np.uint8)
    c = size // 2
    r = int(c * radius_ratio)
    cv2.circle(m, (c, c), r, 255, -1)
    return m


def compute_lbp_u8(gray: np.ndarray) -> np.ndarray:
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

    edges = cv2.Canny(gray, 60, 160)
    edge_density = float((edges[valid] > 0).mean())
    lap_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())

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


def load_artifacts(model_dir: str):
    model_dir = Path(model_dir)
    with open(model_dir / "bacteria_grid_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(model_dir / "bacteria_grid_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    return model, meta


def draw_grid_overlay(norm_bgr: np.ndarray, grid_rows: int, grid_cols: int) -> np.ndarray:
    """Опционально: нарисовать сетку на нормализованной чашке"""
    vis = norm_bgr.copy()
    h, w = vis.shape[:2]

    for r in range(1, grid_rows):
        y = int(r * h / grid_rows)
        cv2.line(vis, (0, y), (w, y), (0, 255, 0), 1)

    for c in range(1, grid_cols):
        x = int(c * w / grid_cols)
        cv2.line(vis, (x, 0), (x, h), (0, 255, 0), 1)

    return vis


def main():
    image_path = r"Dataset2\2 (Staphylococcus aureus - золотистый стафилококк - рана)\IMG_20260122_120832_135.jpg"
    model_dir = "trained_bacteria_model1"
    save_norm_path = ""
    save_grid_path = ""

    model, meta = load_artifacts(model_dir)

    cfg = meta["petri_cfg"]
    grid_rows = meta["grid_rows"]
    grid_cols = meta["grid_cols"]

    norm, dbg, geom = process_image(image_path, cfg)
    if norm is None:
        raise RuntimeError("Чашка Петри не найдена на изображении")

    feats = grid_extract_features(norm, grid_rows=grid_rows, grid_cols=grid_cols).reshape(1, -1)

    pred = model.predict(feats)[0]
    result = {"class": str(pred)}

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(feats)[0]
        classes = model.classes_

        # вероятности по всем классам (0..1)
        proba_dict = {str(c): float(p) for c, p in zip(classes, proba)}
        result["proba"] = proba_dict

        # процент уверенности в выбранном классе
        result["confidence_percent"] = round(proba_dict[str(pred)] * 100.0, 2)

        # топ-3 наиболее вероятных класса
        top_n = min(3, len(classes))
        top_idx = np.argsort(proba)[::-1][:top_n]

        result["top_predictions"] = [
            {
                "class": str(classes[i]),
                "probability": float(proba[i]),
                "confidence_percent": round(float(proba[i]) * 100.0, 2)
            }
            for i in top_idx
        ]
    else:
        result["confidence_percent"] = None
        result["top_predictions"] = []

    print(json.dumps(result, ensure_ascii=False, indent=2))

    if save_norm_path:
        cv2.imwrite(save_norm_path, norm)
        print(f"[OK] Сохранено нормализованное изображение: {save_norm_path}")

    if save_grid_path:
        grid_vis = draw_grid_overlay(norm, grid_rows, grid_cols)
        cv2.imwrite(save_grid_path, grid_vis)
        print(f"[OK] Сохранено изображение с сеткой: {save_grid_path}")


if __name__ == "__main__":
    main()