"""
OOD 方法評估：計算 stone-level Precision / Recall

流程：
  1. 用相同 seed 重建 Ground Truth（每顆黃棋的 bounding box）
  2. 重新跑 Mahalanobis 偵測，記錄每個 ALIEN 框的座標
  3. 重新跑 AutoEncoder 偵測，記錄每個 ALIEN 框的座標
  4. 對每個 GT 框，判斷是否有偵測框與之 IoU >= 0.1（算命中）
  5. 輸出 Precision / Recall / F1 報表

判斷邏輯：
  True Positive  (TP)：GT 黃棋 被偵測到（IoU >= threshold）
  False Negative (FN)：GT 黃棋 沒被偵測到（漏抓）
  False Positive (FP)：偵測到但不對應任何 GT（誤抓）
"""

import cv2
import numpy as np
import random
import torch
import torch.nn as nn
from pathlib import Path
from torchvision import transforms
from PIL import Image as PILImage

# ── 共用設定 ─────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).parent.resolve()
SOURCE_DIR    = PROJECT_ROOT / "datasets" / "go-games-1" / "test" / "images"
TEST_DIR      = PROJECT_ROOT / "runs" / "ood_test"
YOLO_WEIGHTS  = PROJECT_ROOT / "runs" / "go_black_white_100ep" / "weights" / "best.pt"
AE_WEIGHTS    = PROJECT_ROOT / "runs" / "ood_ae_model.pth"
TRAIN_DIR     = PROJECT_ROOT / "datasets" / "go-games-1" / "train" / "images"
VIS_DIR       = PROJECT_ROOT / "runs" / "eval_vis"   # 視覺化輸出目錄

# synthesize_alien.py 的相同參數
NUM_IMAGES     = 10
ALIENS_PER_IMG = 2
STONE_RADIUS   = 28
SEED           = 42

# Mahalanobis 設定（與 ood_mahalanobis.py 一致）
MAHA_CONF       = 0.10    # 與 AE 統一，公平比較
MAHA_IOU        = 0.45
ALIEN_THRESHOLD = 7.5
FEATURE_SIZE    = 32
MAX_TRAIN_IMGS  = 100

# AutoEncoder 設定（與 ood_autoencoder.py 一致）
AE_CONF      = 0.1
AE_CROP_SIZE = 64
AE_DEVICE    = "cpu"

# IoU 門檻：GT 框 和 偵測框 IoU 超過此值才算命中
IOU_MATCH_THRESHOLD = 0.1
# ─────────────────────────────────────────────────────


# ════════════════════════════════════════════════════
# Step 1: 重建 Ground Truth（與 synthesize_alien.py 邏輯完全相同）
# ════════════════════════════════════════════════════

def find_empty_positions(img_shape, labels_path: Path, radius: int, n: int,
                          rng_random, rng_np) -> list:
    h, w = img_shape[:2]
    occupied = []
    if labels_path.exists():
        for line in labels_path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) == 5:
                cx = float(parts[1]) * w
                cy = float(parts[2]) * h
                rw = float(parts[3]) * w / 2
                rh = float(parts[4]) * h / 2
                occupied.append((cx, cy, max(rw, rh)))

    positions = []
    attempts  = 0
    margin    = radius + 10
    while len(positions) < n and attempts < 1000:
        attempts += 1
        x = rng_random.randint(margin, w - margin)
        y = rng_random.randint(margin, h - margin)
        too_close = False
        for ox, oy, or_ in occupied:
            if np.hypot(x - ox, y - oy) < (radius + or_ + 5):
                too_close = True; break
        for px, py in positions:
            if np.hypot(x - px, y - py) < radius * 2 + 5:
                too_close = True; break
        if not too_close:
            positions.append((x, y))
    return positions


def build_ground_truth() -> dict:
    """
    重跑 synthesize_alien.py 的邏輯，記錄每張 alien 圖的黃棋 bounding box。
    回傳 { img_name: [(x1,y1,x2,y2), ...] }
    """
    rng_random = random.Random(SEED)
    rng_np     = np.random.RandomState(SEED)

    label_dir   = SOURCE_DIR.parent.parent / "test" / "labels"
    image_files = sorted(SOURCE_DIR.glob("*.jpg"))[:NUM_IMAGES]

    gt = {}
    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        labels_path = label_dir / img_path.with_suffix(".txt").name
        positions   = find_empty_positions(img.shape, labels_path,
                                           STONE_RADIUS, ALIENS_PER_IMG,
                                           rng_random, rng_np)
        alien_name = f"alien_{img_path.name}"
        boxes = []
        for (cx, cy) in positions:
            x1 = cx - STONE_RADIUS
            y1 = cy - STONE_RADIUS
            x2 = cx + STONE_RADIUS
            y2 = cy + STONE_RADIUS
            boxes.append((int(x1), int(y1), int(x2), int(y2)))
        gt[alien_name] = boxes

    return gt


# ════════════════════════════════════════════════════
# 工具：視覺化輸出
# ════════════════════════════════════════════════════

def save_vis(method_name: str, img_name: str, gt_boxes: list, pred_boxes: list,
             iou_thr: float = IOU_MATCH_THRESHOLD):
    """
    儲存對照圖：
      綠框  = GT 黃棋（GROUND TRUTH）
      紅框  = 偵測到的 ALIEN（TP 或 FP）
      橘框  = 漏掉的 GT（FN）
    """
    img_path = TEST_DIR / img_name
    img = cv2.imread(str(img_path))
    if img is None:
        img = np.array(PILImage.open(str(img_path)).convert("RGB"))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    vis = img.copy()

    # 找哪些 GT 有被命中
    matched_gt   = set()
    matched_pred = set()
    for pi, pb in enumerate(pred_boxes):
        for gi, gb in enumerate(gt_boxes):
            if gi in matched_gt:
                continue
            if iou(pb, gb) >= iou_thr:
                matched_gt.add(gi)
                matched_pred.add(pi)
                break

    # 畫 GT 框
    for gi, (x1, y1, x2, y2) in enumerate(gt_boxes):
        color = (0, 200, 0) if gi in matched_gt else (0, 165, 255)   # 綠=命中 / 橘=漏掉
        label = "GT-HIT" if gi in matched_gt else "GT-MISS(FN)"
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)
        cv2.putText(vis, label, (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 畫 Pred 框
    for pi, (x1, y1, x2, y2) in enumerate(pred_boxes):
        color = (0, 0, 255) if pi not in matched_pred else (180, 0, 180)  # 紅=FP / 紫=TP
        label = "ALIEN-FP" if pi not in matched_pred else "ALIEN-TP"
        cv2.rectangle(vis, (x1+3, y1+3), (x2-3, y2-3), color, 2)
        cv2.putText(vis, label, (x1, y2 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

    method_slug = method_name.replace(" ", "_").lower()
    out_dir = VIS_DIR / method_slug
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / img_name), vis)


# ════════════════════════════════════════════════════
# 工具：IoU 計算 & NMS & 匹配
# ════════════════════════════════════════════════════

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    aA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    aB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return inter / (aA + aB - inter)


def nms_boxes(boxes_with_score: list, iou_thr: float = 0.5) -> list:
    """
    對預測的 ALIEN 框做 NMS，去除重複偵測。
    boxes_with_score: [(x1,y1,x2,y2, score), ...]  score 越大越優先保留
    回傳: [(x1,y1,x2,y2), ...]  去重後的框
    """
    if not boxes_with_score:
        return []
    # 依 score 由高到低排序
    sorted_boxes = sorted(boxes_with_score, key=lambda b: b[4], reverse=True)
    kept = []
    suppressed = set()
    for i, b in enumerate(sorted_boxes):
        if i in suppressed:
            continue
        kept.append(b[:4])
        for j in range(i + 1, len(sorted_boxes)):
            if j in suppressed:
                continue
            if iou(b[:4], sorted_boxes[j][:4]) > iou_thr:
                suppressed.add(j)
    return kept


def compute_metrics(gt_boxes: list, pred_boxes: list, iou_thr: float = IOU_MATCH_THRESHOLD):
    """
    gt_boxes   : [(x1,y1,x2,y2), ...]  Ground truth ALIEN boxes
    pred_boxes : [(x1,y1,x2,y2), ...]  Predicted ALIEN boxes
    回傳 (TP, FP, FN)
    """
    matched_gt   = set()
    matched_pred = set()

    for pi, pb in enumerate(pred_boxes):
        for gi, gb in enumerate(gt_boxes):
            if gi in matched_gt:
                continue
            if iou(pb, gb) >= iou_thr:
                matched_gt.add(gi)
                matched_pred.add(pi)
                break

    TP = len(matched_gt)
    FP = len(pred_boxes) - len(matched_pred)
    FN = len(gt_boxes)   - len(matched_gt)
    return TP, FP, FN


# ════════════════════════════════════════════════════
# Step 2: Mahalanobis 偵測（重跑，記錄座標）
# ════════════════════════════════════════════════════

def extract_features(crop):
    resized = cv2.resize(crop, (FEATURE_SIZE, FEATURE_SIZE))
    hsv     = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    h_hist  = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
    s_hist  = cv2.calcHist([hsv], [1], None, [8],  [0, 256]).flatten()
    v_hist  = cv2.calcHist([hsv], [2], None, [8],  [0, 256]).flatten()
    feat    = np.concatenate([h_hist, s_hist, v_hist])
    return (feat / (feat.sum() + 1e-6)).astype(np.float32)


def run_mahalanobis(gt: dict) -> dict:
    from ultralytics import YOLO
    from scipy.spatial.distance import mahalanobis as scipy_maha

    yolo = YOLO(str(YOLO_WEIGHTS))

    # 建立特徵分佈
    print("  [Mahal] Building feature distributions...")
    features = {0: [], 1: []}
    for img_path in sorted(TRAIN_DIR.glob("*.jpg"))[:MAX_TRAIN_IMGS]:
        img = cv2.imread(str(img_path))
        if img is None: continue
        results = yolo.predict(str(img_path), conf=MAHA_CONF, iou=MAHA_IOU,
                               verbose=False, save=False)
        if not results: continue
        h, w = img.shape[:2]
        for box in results[0].boxes:
            cls  = int(box.cls[0].cpu())
            x1,y1,x2,y2 = map(int, box.xyxy[0].cpu().numpy())
            x1,y1 = max(0,x1), max(0,y1)
            x2,y2 = min(w,x2), min(h,y2)
            crop = img[y1:y2, x1:x2]
            if crop.size == 0: continue
            features[cls].append(extract_features(crop))

    distrib = {}
    for cls_id, feats in features.items():
        if len(feats) < 2: continue
        X   = np.stack(feats)
        mu  = X.mean(axis=0)
        cov = np.cov(X.T) + np.eye(X.shape[1]) * 1e-4
        distrib[cls_id] = {"mean": mu, "inv_cov": np.linalg.inv(cov)}
    print(f"  [Mahal] black={len(features[0])} white={len(features[1])} crops")

    # 推論
    preds = {}
    for img_name in gt.keys():
        img_path = TEST_DIR / img_name
        img = cv2.imread(str(img_path))
        if img is None: continue
        results = yolo.predict(str(img_path), conf=MAHA_CONF, iou=MAHA_IOU,
                               verbose=False, save=False)
        h, w = img.shape[:2]
        alien_boxes_with_score = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                x1,y1,x2,y2 = map(int, box.xyxy[0].cpu().numpy())
                x1,y1 = max(0,x1), max(0,y1)
                x2,y2 = min(w,x2), min(h,y2)
                crop = img[y1:y2, x1:x2]
                if crop.size == 0: continue
                feat = extract_features(crop)
                min_dist = float("inf")
                for d in distrib.values():
                    try:
                        dist = scipy_maha(feat, d["mean"], d["inv_cov"])
                        min_dist = min(min_dist, dist)
                    except Exception:
                        pass
                if min_dist > ALIEN_THRESHOLD:
                    alien_boxes_with_score.append((x1, y1, x2, y2, min_dist))
        # NMS：去除同位置的重複偵測（YOLO 可能對同顆棋子輸出 black + white 兩個框）
        preds[img_name] = nms_boxes(alien_boxes_with_score, iou_thr=0.5)
    return preds


# ════════════════════════════════════════════════════
# Step 3: AutoEncoder 偵測（重跑，記錄座標）
# ════════════════════════════════════════════════════

class CropAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid(),
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))


def run_autoencoder(gt: dict) -> dict:
    from ultralytics import YOLO

    ae_model = CropAutoEncoder().to(AE_DEVICE)
    ae_model.load_state_dict(torch.load(str(AE_WEIGHTS), map_location=AE_DEVICE))
    ae_model.eval()
    yolo = YOLO(str(YOLO_WEIGHTS))

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((AE_CROP_SIZE, AE_CROP_SIZE)),
        transforms.ToTensor(),
    ])

    def crop_error(crop_bgr):
        rgb  = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        x    = transform(rgb).unsqueeze(0).to(AE_DEVICE)
        with torch.no_grad():
            recon = ae_model(x)
        return float((x - recon).pow(2).mean().cpu())

    # 校準門檻（用 normal 圖）
    print("  [AE] Calibrating threshold...")
    normal_errors = []
    for img_path in sorted(TEST_DIR.glob("normal_*.jpg")):
        results = yolo(str(img_path), conf=AE_CONF, verbose=False)
        img_bgr = np.array(PILImage.open(str(img_path)).convert("RGB"))
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)
        if not results or len(results[0].boxes) == 0: continue
        h, w = img_bgr.shape[:2]
        for box in results[0].boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0].cpu().numpy())
            x1,y1 = max(0,x1), max(0,y1)
            x2,y2 = min(w,x2), min(h,y2)
            if x2-x1 < 5 or y2-y1 < 5: continue
            crop = img_bgr[y1:y2, x1:x2]
            normal_errors.append(crop_error(crop))

    threshold = float(np.percentile(normal_errors, 95)) * 1.8 if normal_errors else 0.005
    print(f"  [AE] threshold={threshold:.6f}  (from {len(normal_errors)} normal crops)")

    # 推論
    preds = {}
    for img_name in gt.keys():
        img_path = TEST_DIR / img_name
        img_bgr  = np.array(PILImage.open(str(img_path)).convert("RGB"))
        img_bgr  = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)
        results  = yolo(str(img_path), conf=AE_CONF, verbose=False)
        h, w     = img_bgr.shape[:2]
        alien_boxes_with_score = []
        if results and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                x1,y1,x2,y2 = map(int, box.xyxy[0].cpu().numpy())
                x1,y1 = max(0,x1), max(0,y1)
                x2,y2 = min(w,x2), min(h,y2)
                if x2-x1 < 5 or y2-y1 < 5: continue
                crop = img_bgr[y1:y2, x1:x2]
                err  = crop_error(crop)
                if err > threshold:
                    alien_boxes_with_score.append((x1, y1, x2, y2, err))
        # NMS：去除重複偵測
        preds[img_name] = nms_boxes(alien_boxes_with_score, iou_thr=0.5)
    return preds


# ════════════════════════════════════════════════════
# Step 4: 計算並輸出報表
# ════════════════════════════════════════════════════

def print_report(method_name: str, gt: dict, preds: dict):
    total_TP = total_FP = total_FN = 0
    rows = []

    for img_name, gt_boxes in gt.items():
        pred_boxes = preds.get(img_name, [])
        TP, FP, FN = compute_metrics(gt_boxes, pred_boxes)
        total_TP += TP
        total_FP += FP
        total_FN += FN

        # 同步輸出視覺化圖片
        save_vis(method_name, img_name, gt_boxes, pred_boxes)

        status = []
        if FN > 0: status.append(f"MISS x{FN}")
        if FP > 0: status.append(f"FP x{FP}")
        if not status: status.append("perfect")
        rows.append((img_name[:45], len(gt_boxes), len(pred_boxes), TP, FP, FN, " / ".join(status)))

    precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
    recall    = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n{'='*70}")
    print(f"  {method_name}")
    print(f"{'='*70}")
    header = f"{'Image':<46} {'GT':>3} {'Pred':>5} {'TP':>4} {'FP':>4} {'FN':>4}  Status"
    print(header)
    print("-" * 70)
    for r in rows:
        print(f"  {r[0]:<44} {r[1]:>3} {r[2]:>5} {r[3]:>4} {r[4]:>4} {r[5]:>4}  {r[6]}")
    print("-" * 70)
    print(f"  TOTAL                                            "
          f"GT={total_TP+total_FN}  TP={total_TP}  FP={total_FP}  FN={total_FN}")
    print(f"\n  Precision : {precision:.3f}  ({total_TP}/{total_TP+total_FP})")
    print(f"  Recall    : {recall:.3f}  ({total_TP}/{total_TP+total_FN})")
    print(f"  F1 Score  : {f1:.3f}")


# ════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("  OOD Evaluation: Stone-Level Precision / Recall")
    print("=" * 70)

    print("\n[Step 1] Rebuilding Ground Truth...")
    gt = build_ground_truth()
    total_gt = sum(len(v) for v in gt.values())
    print(f"  {len(gt)} alien images  |  {total_gt} GT alien stones total")
    for name, boxes in gt.items():
        print(f"    {name[:50]}  GT boxes: {[b for b in boxes]}")

    print("\n[Step 2] Running Mahalanobis...")
    maha_preds = run_mahalanobis(gt)

    print("\n[Step 3] Running AutoEncoder...")
    ae_preds = run_autoencoder(gt)

    print("\n[Step 4] Computing Metrics...")
    print_report("Mahalanobis Distance", gt, maha_preds)
    print_report("AutoEncoder",          gt, ae_preds)

    # 最終對比
    print(f"\n{'='*70}")
    print("  FINAL COMPARISON")
    print(f"{'='*70}")
    print(f"  {'Method':<25} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print(f"  {'-'*55}")
    for name, preds_dict in [("Mahalanobis", maha_preds), ("AutoEncoder", ae_preds)]:
        TP = FP = FN = 0
        for img_name, gbs in gt.items():
            tp, fp, fn = compute_metrics(gbs, preds_dict.get(img_name, []))
            TP += tp; FP += fp; FN += fn
        prec = TP/(TP+FP) if TP+FP > 0 else 0
        rec  = TP/(TP+FN) if TP+FN > 0 else 0
        f1   = 2*prec*rec/(prec+rec) if prec+rec > 0 else 0
        print(f"  {name:<25} {prec:>10.3f} {rec:>8.3f} {f1:>8.3f}")
    print()
