"""
YOLOv8 訓練腳本 - 圍棋黑白棋子偵測 (CPU)
資料集: go-games-ztg5w v1  (nc=2, black / white)
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
    epochs    = 30,         # 先跑 30 epoch 驗證模型效果
    imgsz     = 640,        # 標準輸入尺寸
    batch     = 8,          # CPU 記憶體有限，用 8
    device    = "cpu",
    workers   = 0,          # Windows 多 worker 容易 crash，固定 0
    amp       = False,      # CPU 不支援 AMP，必須關閉否則 Access Violation
    patience  = 20,         # 20 個 epoch 無進步自動提早停止
    optimizer = "AdamW",    # 小資料集用 AdamW 較穩定
    lr0       = 0.001,      # 初始學習率
    lrf       = 0.01,       # 最終學習率 = lr0 * lrf
    momentum  = 0.937,
    weight_decay = 0.0005,
    warmup_epochs = 3,      # 前 3 epoch 學習率暖身
    cos_lr    = True,       # cosine 學習率衰減
    augment   = True,       # 開啟資料增強（小資料集必備）
    project   = str(RUNS_DIR),
    name      = "go_black_white",
    exist_ok  = True,       # 若資料夾已存在則覆蓋
    verbose   = True,
)

if __name__ == "__main__":
    assert DATA_YAML.exists(), f"data.yaml not found: {DATA_YAML}"

    # 使用 yolov8n（nano）: 最小最快，適合 CPU + 小資料集
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
