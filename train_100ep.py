"""
YOLOv8 訓練腳本 - 圍棋黑白棋子偵測 (CPU) - 100 epochs
資料集: go-games-ztg5w v1  (nc=2, black / white)
輸出資料夾: runs/go_black_white_100ep  (與 30ep 結果完全分開)
"""

from pathlib import Path
from ultralytics import YOLO

# ── 路徑設定 ────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_YAML    = PROJECT_ROOT / "datasets" / "go-games-1" / "data.yaml"
RUNS_DIR     = PROJECT_ROOT / "runs"

# ── 訓練參數 ────────────────────────────────────────────
PARAMS = dict(
    data      = str(DATA_YAML),
    epochs    = 100,
    imgsz     = 640,
    batch     = 8,
    device    = "cpu",
    workers   = 0,          # Windows 多 worker 容易 crash，固定 0
    amp       = False,      # CPU 不支援 AMP，必須關閉否則 Access Violation
    patience  = 20,         # 20 個 epoch 無進步自動提早停止
    optimizer = "AdamW",
    lr0       = 0.001,
    lrf       = 0.01,
    momentum  = 0.937,
    weight_decay = 0.0005,
    warmup_epochs = 3,
    cos_lr    = True,
    augment   = True,
    project   = str(RUNS_DIR),
    name      = "go_black_white_100ep",   # 獨立資料夾，不覆蓋 30ep 結果
    exist_ok  = False,                    # 若已存在就報錯，避免誤覆蓋
    verbose   = True,
)

if __name__ == "__main__":
    assert DATA_YAML.exists(), f"data.yaml not found: {DATA_YAML}"

    model = YOLO("yolov8n.pt")

    print("Training config:")
    print(f"  model   : YOLOv8n (pretrained)")
    print(f"  data    : {DATA_YAML}")
    print(f"  epochs  : {PARAMS['epochs']} (early stop if no improvement for 20 epochs)")
    print(f"  imgsz   : {PARAMS['imgsz']}")
    print(f"  batch   : {PARAMS['batch']}")
    print(f"  device  : {PARAMS['device']}")
    print(f"  output  : {RUNS_DIR / PARAMS['name']}\n")

    results = model.train(**PARAMS)

    print("\nTraining complete.")
    print(f"Best weights: {RUNS_DIR / PARAMS['name'] / 'weights' / 'best.pt'}")
