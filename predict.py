"""
圍棋黑白棋子偵測 - 推論腳本
用法：
  python predict.py --source image.jpg          # 單張圖片
  python predict.py --source images/            # 整個資料夾
  python predict.py --source 0                  # 即時攝影機
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

# ── 設定區 ────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.resolve()
MODEL_PATH   = PROJECT_ROOT / "runs" / "go_black_white_100ep" / "weights" / "best.pt"
OUTPUT_DIR   = PROJECT_ROOT / "runs" / "predict_100ep"

CONF_THRESHOLD = 0.25   # 信心分數低於此值的偵測結果會被丟棄
IOU_THRESHOLD  = 0.45   # NMS 的 IoU 閾值，避免同一棋子被框多次
# ─────────────────────────────────────────────────────


def run_predict(source: str, save_img: bool = True, show: bool = False):
    assert MODEL_PATH.exists(), f"找不到模型權重：{MODEL_PATH}\n請先執行 train.py 完成訓練。"

    model = YOLO(str(MODEL_PATH))

    results = model.predict(
        source   = source,
        conf     = CONF_THRESHOLD,
        iou      = IOU_THRESHOLD,
        save     = save_img,        # 將結果圖片儲存到 runs/predict/
        show     = show,            # 是否即時彈出視窗顯示
        project  = str(OUTPUT_DIR.parent),
        name     = "predict_100ep",
        exist_ok = True,
        verbose  = True,
    )

    # ── 統計每張圖的黑白棋數量 ──────────────────────
    print("\n" + "=" * 55)
    print(f"  {'檔案':<30} {'黑棋':>5} {'白棋':>5} {'合計':>5}")
    print("=" * 55)

    total_black = total_white = 0
    for r in results:
        img_name = Path(r.path).name if r.path else "camera"
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            print(f"  {img_name:<30} {'0':>5} {'0':>5} {'0':>5}")
            continue

        black_count = int((boxes.cls == 0).sum())
        white_count = int((boxes.cls == 1).sum())
        total       = black_count + white_count
        total_black += black_count
        total_white += white_count

        print(f"  {img_name:<30} {black_count:>5} {white_count:>5} {total:>5}")

    print("-" * 55)
    print(f"  {'合計':<30} {total_black:>5} {total_white:>5} {total_black + total_white:>5}")
    print("=" * 55)

    if save_img:
        print(f"\n結果圖片已儲存至：{OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="圍棋黑白棋子偵測推論")
    parser.add_argument(
        "--source", type=str,
        default=str(PROJECT_ROOT / "datasets" / "go-games-1" / "test" / "images"),
        help="輸入來源：圖片路徑 / 資料夾路徑 / 0（攝影機）",
    )
    parser.add_argument("--show",     action="store_true", help="即時彈出視窗顯示結果")
    parser.add_argument("--no-save",  action="store_true", help="不儲存結果圖片")
    args = parser.parse_args()

    run_predict(
        source   = args.source,
        save_img = not args.no_save,
        show     = args.show,
    )
