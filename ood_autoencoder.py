"""
OOD 偵測 - AutoEncoder 方法（Crop-Level）

架構：
  訓練：
    1. YOLO 偵測訓練圖片中的棋子 → 截取每個棋子的 crop
    2. AutoEncoder 學習重建「正常棋子（黑/白）的 crop」
  推論：
    1. YOLO 偵測測試圖片中的棋子
    2. 每個 crop 送進 AE → 計算重建誤差
    3. 誤差大的 crop = 外星人（AE 沒見過 → 還原效果差）

與 Mahalanobis 的差異：
  Mahalanobis : 用 HSV 直方圖 + 統計距離
  AutoEncoder : 學習重建外觀，依賴深度學習特徵

用法：
  python ood_autoencoder.py --mode train    # 訓練
  python ood_autoencoder.py --mode predict  # 推論
  python ood_autoencoder.py                 # 訓練 + 推論
"""

import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from PIL import Image as PILImage

# ── 設定 ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.resolve()
TRAIN_IMG_DIR = PROJECT_ROOT / "datasets" / "go-games-1" / "train" / "images"
TEST_DIR      = PROJECT_ROOT / "runs" / "ood_test"
OUTPUT_DIR    = PROJECT_ROOT / "runs" / "ood_ae_results"
MODEL_SAVE    = PROJECT_ROOT / "runs" / "ood_ae_model.pth"
YOLO_WEIGHTS  = PROJECT_ROOT / "runs" / "go_black_white_100ep" / "weights" / "best.pt"

CROP_SIZE     = 64         # 每個棋子 crop 縮放大小
BATCH_SIZE    = 16
EPOCHS        = 80
LR            = 5e-4
DEVICE        = "cpu"      # 避免 GPU 記憶體問題

# 門檻：推論時自動從 normal 圖片的誤差校準
# ─────────────────────────────────────────────────────


# ════════════════════════════════════════════════════
# 1. AutoEncoder 架構（針對 64x64 棋子 crop）
# ════════════════════════════════════════════════════

class CropAutoEncoder(nn.Module):
    """
    輸入：64x64x3 棋子截圖
    Encoder：64→32→16→8
    Decoder：8→16→32→64
    """
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            # 64x64 → 32x32 × 32ch
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),
            # 32x32 → 16x16 × 64ch
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            # 16x16 → 8x8 × 64ch
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            # 8x8 → 16x16 × 64ch
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            # 16x16 → 32x32 × 32ch
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            # 32x32 → 64x64 × 3ch
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ════════════════════════════════════════════════════
# 2. 從圖片中提取棋子 crops（使用 YOLO）
# ════════════════════════════════════════════════════

def extract_crops_from_dir(img_dir: Path, conf: float = 0.25) -> list:
    """用 YOLO 偵測圖片，回傳所有棋子 crop（numpy array）"""
    from ultralytics import YOLO
    model = YOLO(str(YOLO_WEIGHTS))

    crops = []
    img_paths = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    print(f"  Extracting crops from {len(img_paths)} images...")

    for img_path in img_paths:
        results = model(str(img_path), conf=conf, verbose=False)
        if not results:
            continue
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            img_pil = PILImage.open(str(img_path)).convert("RGB")
            img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            h, w = img_bgr.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 - x1 < 5 or y2 - y1 < 5:
                continue
            crop = img_bgr[y1:y2, x1:x2]
            crops.append(crop)

    print(f"  Found {len(crops)} crops total")
    return crops


# ════════════════════════════════════════════════════
# 3. Dataset
# ════════════════════════════════════════════════════

class CropDataset(Dataset):
    def __init__(self, crops: list, crop_size: int):
        self.crops = crops
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((crop_size, crop_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, idx):
        crop = self.crops[idx]
        rgb  = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        return self.transform(rgb)


# ════════════════════════════════════════════════════
# 4. 訓練
# ════════════════════════════════════════════════════

def train():
    print(f"Device: {DEVICE}")
    print(f"YOLO weights: {YOLO_WEIGHTS}")
    print(f"Training images: {TRAIN_IMG_DIR}\n")

    crops = extract_crops_from_dir(TRAIN_IMG_DIR, conf=0.25)
    if len(crops) == 0:
        raise RuntimeError("No crops found. Check YOLO_WEIGHTS and TRAIN_IMG_DIR.")

    dataset    = CropDataset(crops, CROP_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model     = CropAutoEncoder().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    print(f"Crops: {len(crops)}  |  Batch: {BATCH_SIZE}  |  Epochs: {EPOCHS}\n")

    best_loss = float("inf")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for batch in dataloader:
            batch = batch.to(DEVICE)
            recon = model(batch)
            loss  = criterion(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS}  loss={avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), MODEL_SAVE)

    print(f"\nBest loss: {best_loss:.6f}")
    print(f"Model saved: {MODEL_SAVE}")


# ════════════════════════════════════════════════════
# 5. 推論：對測試圖片逐一偵測 + AE 誤差
# ════════════════════════════════════════════════════

def predict():
    from ultralytics import YOLO

    assert MODEL_SAVE.exists(), f"Model not found: {MODEL_SAVE}\nRun with --mode train first."
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ae_model = CropAutoEncoder().to(DEVICE)
    ae_model.load_state_dict(torch.load(MODEL_SAVE, map_location=DEVICE))
    ae_model.eval()

    yolo = YOLO(str(YOLO_WEIGHTS))

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((CROP_SIZE, CROP_SIZE)),
        transforms.ToTensor(),
    ])

    def crop_error(crop_bgr: np.ndarray) -> float:
        """計算單一 crop 的 AE 重建誤差"""
        rgb  = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        x    = transform(rgb).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            recon = ae_model(x)
        return float((x - recon).pow(2).mean().cpu())

    # ── 從 normal 圖片校準門檻 ────────────────────────
    print("Calibrating threshold from normal images...")
    normal_errors = []
    for img_path in sorted(TEST_DIR.glob("normal_*.jpg")):
        results = yolo(str(img_path), conf=0.1, verbose=False)
        img_bgr = np.array(PILImage.open(str(img_path)).convert("RGB"))
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)
        if not results or len(results[0].boxes) == 0:
            continue
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            h, w = img_bgr.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 - x1 < 5 or y2 - y1 < 5:
                continue
            crop = img_bgr[y1:y2, x1:x2]
            normal_errors.append(crop_error(crop))

    if len(normal_errors) == 0:
        threshold = 0.005
        print("  No normal crops found, using default threshold=0.005")
    else:
        p95 = float(np.percentile(normal_errors, 95))
        threshold = p95 * 1.8
        print(f"  Normal crops: {len(normal_errors)}")
        print(f"  Error mean={np.mean(normal_errors):.5f}  p95={p95:.5f}")
        print(f"  Auto threshold = {threshold:.5f}\n")

    # ── 對所有測試圖片推論 ─────────────────────────────
    test_images = sorted(TEST_DIR.glob("*.jpg"))
    for img_path in test_images:
        img_bgr  = np.array(PILImage.open(str(img_path)).convert("RGB"))
        img_bgr  = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)
        vis      = img_bgr.copy()

        results  = yolo(str(img_path), conf=0.1, verbose=False)
        if not results or len(results[0].boxes) == 0:
            print(f"  {img_path.name[:55]}  (no detection)")
            cv2.imwrite(str(OUTPUT_DIR / img_path.name), vis)
            continue

        alien_count  = 0
        total_stones = len(results[0].boxes)
        h, w = img_bgr.shape[:2]

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 - x1 < 5 or y2 - y1 < 5:
                continue
            crop   = img_bgr[y1:y2, x1:x2]
            err    = crop_error(crop)
            is_alien = err > threshold

            if is_alien:
                color = (0, 0, 255)   # 紅色 = ALIEN
                label = f"ALIEN e={err:.4f}"
                alien_count += 1
            else:
                color = (0, 200, 0)   # 綠色 = 正常
                label = f"OK e={err:.4f}"

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis, label, (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        cv2.imwrite(str(OUTPUT_DIR / img_path.name), vis)

        tag = f"  *** {alien_count} ALIEN" if alien_count > 0 else ""
        print(f"  {img_path.name[:55]}  stones={total_stones}{tag}")

    print(f"\nOutput: {OUTPUT_DIR}")


# ════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "predict", "both"], default="both")
    args = parser.parse_args()

    if args.mode in ("train", "both"):
        print("=" * 55)
        print("TRAIN AutoEncoder (crop-level)")
        print("=" * 55)
        train()

    if args.mode in ("predict", "both"):
        print("\n" + "=" * 55)
        print("PREDICT (AE anomaly score per stone)")
        print("=" * 55)
        predict()
