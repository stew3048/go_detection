"""
診斷 alien_frame_00999 的 Mahalanobis 偵測結果：
列出每個被標為 ALIEN 的 box，計算它和兩個 GT box 的 IoU，
並檢查 box 中心的像素顏色。
"""
import cv2, numpy as np
from pathlib import Path
from ultralytics import YOLO
from scipy.spatial.distance import mahalanobis

PROJECT_ROOT = Path(__file__).parent.resolve()
YOLO_WEIGHTS = PROJECT_ROOT / "runs" / "go_black_white_100ep" / "weights" / "best.pt"
TRAIN_DIR    = PROJECT_ROOT / "datasets" / "go-games-1" / "train" / "images"
TEST_DIR     = PROJECT_ROOT / "runs" / "ood_test"

CONF            = 0.20
IOU_NMS         = 0.45
ALIEN_THRESHOLD = 7.5
FEATURE_SIZE    = 32
MAX_TRAIN_IMGS  = 100

GT_BOXES = [(248, 527, 304, 583), (568, 439, 624, 495)]   # alien_00999 的 GT
IMG_NAME = "alien_frame_00999_png.rf.6bfd635f12778b091ed35405acf0bec6.jpg"


def extract_features(crop):
    resized = cv2.resize(crop, (FEATURE_SIZE, FEATURE_SIZE))
    hsv     = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    h_hist  = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
    s_hist  = cv2.calcHist([hsv], [1], None, [8],  [0, 256]).flatten()
    v_hist  = cv2.calcHist([hsv], [2], None, [8],  [0, 256]).flatten()
    feat    = np.concatenate([h_hist, s_hist, v_hist])
    return (feat / (feat.sum() + 1e-6)).astype(np.float32)


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    return inter / ((boxA[2]-boxA[0])*(boxA[3]-boxA[1]) +
                    (boxB[2]-boxB[0])*(boxB[3]-boxB[1]) - inter)


# ── 建立特徵分佈 ─────────────────────────────
yolo = YOLO(str(YOLO_WEIGHTS))
features = {0: [], 1: []}
for img_path in sorted(TRAIN_DIR.glob("*.jpg"))[:MAX_TRAIN_IMGS]:
    img = cv2.imread(str(img_path))
    if img is None: continue
    results = yolo.predict(str(img_path), conf=CONF, iou=IOU_NMS, verbose=False, save=False)
    if not results: continue
    h, w = img.shape[:2]
    for box in results[0].boxes:
        cls = int(box.cls[0].cpu())
        x1,y1,x2,y2 = map(int, box.xyxy[0].cpu().numpy())
        x1,y1 = max(0,x1), max(0,y1)
        x2,y2 = min(w,x2), min(h,y2)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0: continue
        features[cls].append(extract_features(crop))

distrib = {}
for cls_id, feats in features.items():
    if len(feats) < 2: continue
    X = np.stack(feats)
    mu = X.mean(axis=0)
    cov = np.cov(X.T) + np.eye(X.shape[1]) * 1e-4
    distrib[cls_id] = {"mean": mu, "inv_cov": np.linalg.inv(cov)}

# ── 對 alien_00999 推論 ───────────────────────
img_path = TEST_DIR / IMG_NAME
img = cv2.imread(str(img_path))
h, w = img.shape[:2]

results = yolo.predict(str(img_path), conf=CONF, iou=IOU_NMS, verbose=False, save=False)
print(f"Image: {IMG_NAME}")
print(f"GT boxes: {GT_BOXES}")
print()
print(f"All YOLO detections on this image:")
print(f"  {'Box':30s}  {'cls':6s}  {'conf':6s}  {'dist':7s}  {'ALIEN?':8s}  {'IoU_GT1':8s}  {'IoU_GT2':8s}  CenterPixelHSV  YellowCenter?")
print("-" * 120)

alien_count = 0
if results and results[0].boxes is not None:
    for box in results[0].boxes:
        cls_id   = int(box.cls[0].cpu())
        conf_val = float(box.conf[0].cpu())
        x1,y1,x2,y2 = map(int, box.xyxy[0].cpu().numpy())
        x1,y1 = max(0,x1), max(0,y1)
        x2,y2 = min(w,x2), min(h,y2)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0: continue

        feat = extract_features(crop)
        min_dist = float("inf")
        for d in distrib.values():
            try:
                dist = mahalanobis(feat, d["mean"], d["inv_cov"])
                min_dist = min(min_dist, dist)
            except: pass

        is_alien = min_dist > ALIEN_THRESHOLD
        iou1 = iou((x1,y1,x2,y2), GT_BOXES[0])
        iou2 = iou((x1,y1,x2,y2), GT_BOXES[1])

        # 檢查 box 中心顏色
        cx, cy = (x1+x2)//2, (y1+y2)//2
        pixel_bgr = img[cy, cx]
        px = np.uint8([[pixel_bgr]])
        hsv = cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0][0]
        is_yellow_center = (15 <= int(hsv[0]) <= 35) and int(hsv[1]) > 100

        cls_name = ["black","white"][cls_id] if cls_id < 2 else "?"
        box_str = f"({x1},{y1},{x2},{y2})"
        print(f"  {box_str:30s}  {cls_name:6s}  {conf_val:.3f}  {min_dist:7.2f}  {'ALIEN':8s}  {iou1:.3f}    {iou2:.3f}    H{hsv[0]:3d}S{hsv[1]:3d}V{hsv[2]:3d}  {'YELLOW' if is_yellow_center else 'not yellow'}")
        if is_alien:
            alien_count += 1
            matched = iou1 >= 0.1 or iou2 >= 0.1
            tp_fp = "TP" if matched else "FP ← THIS IS THE PROBLEM"
            print(f"       ^ ALIEN box -> {'IoU matches GT' if matched else 'NO GT MATCH! IoU1={:.3f} IoU2={:.3f}'.format(iou1,iou2)}  -> {tp_fp}")
