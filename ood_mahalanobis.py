"""
OOD 偵測 - Mahalanobis Distance 方法

流程：
  1. 用訓練集圖片跑 YOLO，擷取每個偵測框的特徵向量
  2. 計算 black / white 各自的特徵分佈（mean + covariance）
  3. 對測試圖片的每個框計算 Mahalanobis Distance
  4. 距離超過門檻 → 標記為 ALIEN（紅框）

輸出：runs/ood_results/
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO
from scipy.spatial.distance import mahalanobis

# ── 設定 ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.resolve()
MODEL_PATH   = PROJECT_ROOT / "runs" / "go_black_white_100ep" / "weights" / "best.pt"
TRAIN_DIR    = PROJECT_ROOT / "datasets" / "go-games-1" / "train" / "images"
TEST_DIR     = PROJECT_ROOT / "runs" / "ood_test"
OUTPUT_DIR   = PROJECT_ROOT / "runs" / "ood_results"

CONF             = 0.20    # 推論信心門檻（稍低讓黃棋有機會被偵測到）
IOU              = 0.45
ALIEN_THRESHOLD  = 7.5     # Mahalanobis 距離超過此值 → ALIEN（HSV 特徵校準後）
FEATURE_SIZE     = 32      # 截圖後 resize 的大小
MAX_TRAIN_IMGS   = 100     # 取多少張訓練圖建立分佈（太多會慢）

CLASS_NAMES = ["black", "white"]
COLORS = {
    "black": (50, 50, 50),
    "white": (200, 200, 200),
    "ALIEN": (0, 0, 255),    # 紅色
}
# ─────────────────────────────────────────────────────


def extract_features(crop: np.ndarray) -> np.ndarray:
    """
    從截圖中提取 HSV 色彩直方圖特徵向量。
    HSV 能有效區分黃色（高飽和度 + 特定色相）vs 黑/白（低飽和度）。
    H: 16 bins, S: 8 bins, V: 8 bins → 共 32 維
    """
    resized = cv2.resize(crop, (FEATURE_SIZE, FEATURE_SIZE))
    hsv     = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    h_hist  = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
    s_hist  = cv2.calcHist([hsv], [1], None, [8],  [0, 256]).flatten()
    v_hist  = cv2.calcHist([hsv], [2], None, [8],  [0, 256]).flatten()
    feat    = np.concatenate([h_hist, s_hist, v_hist])
    feat    = feat / (feat.sum() + 1e-6)   # 正規化
    return feat.astype(np.float32)


def build_distribution(model: YOLO, image_dir: Path) -> dict:
    """
    對訓練集跑推論，收集每個類別的特徵向量，
    回傳各類別的 mean 和 inverse covariance。
    """
    print(f"Building feature distribution from {image_dir}...")
    img_files = sorted(image_dir.glob("*.jpg"))[:MAX_TRAIN_IMGS]

    features = {0: [], 1: []}   # 0=black, 1=white

    for img_path in img_files:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        results = model.predict(str(img_path), conf=CONF, iou=IOU,
                                verbose=False, save=False)
        if not results:
            continue

        h, w = img.shape[:2]
        boxes = results[0].boxes
        if boxes is None:
            continue

        for box in boxes:
            cls_id = int(box.cls.item())
            if cls_id not in features:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            feat = extract_features(crop)
            features[cls_id].append(feat)

    print(f"  black: {len(features[0])} samples")
    print(f"  white: {len(features[1])} samples")

    distribution = {}
    for cls_id, feats in features.items():
        if len(feats) < 2:
            continue
        arr  = np.array(feats)
        mean = arr.mean(axis=0)
        cov  = np.cov(arr.T)
        # 加入小量正則化避免奇異矩陣
        cov += np.eye(cov.shape[0]) * 1e-4
        inv_cov = np.linalg.inv(cov)
        distribution[cls_id] = {"mean": mean, "inv_cov": inv_cov}
        print(f"  class {CLASS_NAMES[cls_id]}: mean shape={mean.shape}")

    return distribution


def compute_ood_score(feat: np.ndarray, distribution: dict) -> float:
    """
    計算特徵向量對所有已知類別的最小 Mahalanobis Distance。
    距離越大 = 越像外星人。
    """
    min_dist = float("inf")
    for cls_id, stats in distribution.items():
        try:
            dist = mahalanobis(feat, stats["mean"], stats["inv_cov"])
            min_dist = min(min_dist, dist)
        except Exception:
            continue
    return min_dist


def draw_result(img: np.ndarray, box_info: list) -> np.ndarray:
    """畫出偵測結果，ALIEN 用紅框，正常棋子用對應顏色。"""
    result = img.copy()
    for (x1, y1, x2, y2, label, conf, dist) in box_info:
        color = COLORS.get(label, (128, 128, 128))
        thickness = 3 if label == "ALIEN" else 2

        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)

        if label == "ALIEN":
            text = f"ALIEN! dist={dist:.1f}"
            bg_color = (0, 0, 200)
        else:
            text = f"{label} {conf:.2f}"
            bg_color = (40, 40, 40)

        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(result, (x1, y1 - th - 6), (x1 + tw + 4, y1), bg_color, -1)
        cv2.putText(result, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return result


def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(MODEL_PATH))

    # Step 1: 建立訓練集特徵分佈
    print("=" * 55)
    print("Step 1: Building normal distribution from training set")
    print("=" * 55)
    distribution = build_distribution(model, TRAIN_DIR)

    # Step 2: 對測試圖片做 OOD 偵測
    print("\n" + "=" * 55)
    print("Step 2: OOD detection on test images")
    print(f"        ALIEN threshold = {ALIEN_THRESHOLD}")
    print("=" * 55)

    test_images = sorted(TEST_DIR.glob("*.jpg"))
    alien_found = normal_found = 0

    for img_path in test_images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        results = model.predict(str(img_path), conf=CONF, iou=IOU,
                                verbose=False, save=False)
        h, w = img.shape[:2]
        box_info = []

        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls.item())
                conf_val = float(box.conf.item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                feat = extract_features(crop)
                dist = compute_ood_score(feat, distribution)

                if dist > ALIEN_THRESHOLD:
                    label = "ALIEN"
                    alien_found += 1
                else:
                    label = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else "unknown"
                    normal_found += 1

                box_info.append((x1, y1, x2, y2, label, conf_val, dist))

        result_img = draw_result(img, box_info)
        out_path = OUTPUT_DIR / img_path.name
        cv2.imwrite(str(out_path), result_img)

        alien_boxes = [b for b in box_info if b[4] == "ALIEN"]
        tag = f"  ** {len(alien_boxes)} ALIEN(s) detected **" if alien_boxes else ""
        print(f"  {img_path.name[:45]} | {len(box_info)} boxes{tag}")

    print(f"\n{'=' * 55}")
    print(f"  Normal stones : {normal_found}")
    print(f"  ALIEN detected: {alien_found}")
    print(f"  Output        : {OUTPUT_DIR}")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    run()
