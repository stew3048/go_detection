import cv2, numpy as np
from pathlib import Path
from ultralytics import YOLO
from scipy.spatial.distance import mahalanobis

MODEL_PATH   = Path("runs/go_black_white_100ep/weights/best.pt")
TRAIN_DIR    = Path("datasets/go-games-1/train/images")
TEST_DIR     = Path("runs/ood_test")
FEATURE_SIZE = 32
THRESHOLD    = 7.5

def extract_features(crop):
    r = cv2.resize(crop, (FEATURE_SIZE, FEATURE_SIZE))
    hsv = cv2.cvtColor(r, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [8],  [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], None, [8],  [0, 256]).flatten()
    feat = np.concatenate([h_hist, s_hist, v_hist])
    return (feat / (feat.sum() + 1e-6)).astype(np.float32)

model = YOLO(str(MODEL_PATH))

# 建立分佈
features = {0: [], 1: []}
for img_path in sorted(TRAIN_DIR.glob("*.jpg"))[:100]:
    img = cv2.imread(str(img_path))
    results = model.predict(str(img_path), conf=0.2, iou=0.45, verbose=False, save=False)
    if not results or results[0].boxes is None:
        continue
    h, w = img.shape[:2]
    for box in results[0].boxes:
        cls_id = int(box.cls.item())
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        crop = img[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
        if crop.size == 0:
            continue
        features[cls_id].append(extract_features(crop))

dist_info = {}
for cls_id, feats in features.items():
    arr = np.array(feats)
    mean = arr.mean(axis=0)
    cov = np.cov(arr.T) + np.eye(len(mean)) * 1e-4
    dist_info[cls_id] = {"mean": mean, "inv_cov": np.linalg.inv(cov)}

targets = [
    "alien_frame_00530_png.rf.d584836cfbc516a6c67f422c7d2d35d3.jpg",
    "alien_frame_00545_png.rf.b1330f4bdd39ee2f4e894f39c023fec2.jpg",
]

for fname in targets:
    img_path = TEST_DIR / fname
    img = cv2.imread(str(img_path))
    results = model.predict(str(img_path), conf=0.2, iou=0.45, verbose=False, save=False)
    h, w = img.shape[:2]
    print(f"\n{'='*65}")
    print(f"{fname[:55]}")
    print(f"{'cls':6} {'conf':6} {'dist_black':12} {'dist_white':12} {'min':8} {'判定':8}")
    print("-"*65)
    for box in results[0].boxes:
        cls_id = int(box.cls.item())
        conf_v = float(box.conf.item())
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        crop = img[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
        if crop.size == 0:
            continue
        feat = extract_features(crop)
        d_b = mahalanobis(feat, dist_info[0]["mean"], dist_info[0]["inv_cov"])
        d_w = mahalanobis(feat, dist_info[1]["mean"], dist_info[1]["inv_cov"])
        min_d = min(d_b, d_w)
        cls_name = ["black", "white"][cls_id]
        label = "*** ALIEN" if min_d > THRESHOLD else cls_name
        print(f"{cls_name:6} {conf_v:.3f}  {d_b:12.2f} {d_w:12.2f} {min_d:8.2f} {label}")
