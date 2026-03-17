"""
用 best.pt 對圖片預標註，輸出 YOLO 格式的 .txt 標註檔
方便上傳到 Roboflow 後只需人工確認/修正，不用從零標記。

輸出結構：
  output/
    images/   ← 原始圖片
    labels/   ← YOLO 格式標註 (.txt)
"""

import shutil
from pathlib import Path
from ultralytics import YOLO

# ── 設定 ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.resolve()
MODEL_PATH   = PROJECT_ROOT / "runs" / "go_black_white_100ep" / "weights" / "best.pt"
SOURCE_DIR   = Path(r"C:\Users\yiching\ai_projects\go_dection\datasets\go-board-1\valid\images")
OUTPUT_DIR   = PROJECT_ROOT / "prelabel_output"
CONF         = 0.25
IOU          = 0.45
# ─────────────────────────────────────────────────────

def run():
    assert MODEL_PATH.exists(), f"Model not found: {MODEL_PATH}"
    assert SOURCE_DIR.exists(), f"Image folder not found: {SOURCE_DIR}"

    out_images = OUTPUT_DIR / "images"
    out_labels = OUTPUT_DIR / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(MODEL_PATH))
    image_files = list(SOURCE_DIR.glob("*.jpg")) + list(SOURCE_DIR.glob("*.png"))

    print(f"Found {len(image_files)} images in {SOURCE_DIR}")
    print(f"Running inference (conf={CONF}, iou={IOU})...\n")

    results = model.predict(
        source  = str(SOURCE_DIR),
        conf    = CONF,
        iou     = IOU,
        save    = False,
        verbose = False,
    )

    saved = skipped = 0
    for r in results:
        img_path = Path(r.path)
        stem     = img_path.stem

        # 圖片複製過去
        shutil.copy(img_path, out_images / img_path.name)

        # 沒偵測到任何東西 → 空的 txt（Roboflow 仍需要這個檔案）
        label_path = out_labels / f"{stem}.txt"
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            label_path.write_text("")
            skipped += 1
            continue

        # 寫成 YOLO 格式：class_id cx cy w h（全部正規化到 0~1）
        lines = []
        for cls, xywhn in zip(boxes.cls.tolist(), boxes.xywhn.tolist()):
            cls_id = int(cls)
            cx, cy, w, h = xywhn
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        label_path.write_text("\n".join(lines))
        saved += 1

    # 產生 classes.txt（Roboflow 上傳需要）
    (OUTPUT_DIR / "classes.txt").write_text("black\nwhite\n")

    print(f"Done!")
    print(f"  Labelled  : {saved} images")
    print(f"  No detect : {skipped} images (empty label files)")
    print(f"\nOutput folder: {OUTPUT_DIR}")
    print("  images/   <- upload these to Roboflow")
    print("  labels/   <- upload these to Roboflow")
    print("  classes.txt")

if __name__ == "__main__":
    run()
