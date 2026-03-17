"""
將 labelme 的 .json 標註轉換成 YOLO 格式的 .txt 標註
支援 rectangle（bounding box）格式

輸入：labels/*.json
輸出：yolo_labels/*.txt
"""

import json
from pathlib import Path

# ── 設定 ─────────────────────────────────────────────
LABELS_DIR = Path(r"C:\Users\yiching\ai_projects\go_dection\prelabel_output\labels")
OUTPUT_DIR = Path(r"C:\Users\yiching\ai_projects\go_dection\prelabel_output\yolo_labels")

# 類別順序（必須與 data.yaml 的 names 一致）
CLASSES = ["black", "white"]
# ─────────────────────────────────────────────────────


def convert_one(json_path: Path) -> int:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_w = data["imageWidth"]
    img_h = data["imageHeight"]
    shapes = data.get("shapes", [])

    lines = []
    for shape in shapes:
        label      = shape["label"]
        shape_type = shape["shape_type"]

        if shape_type != "rectangle":
            print(f"  [skip] {json_path.name}: shape_type={shape_type} (only rectangle supported)")
            continue

        if label not in CLASSES:
            print(f"  [skip] {json_path.name}: unknown label '{label}'")
            continue

        cls_id = CLASSES.index(label)
        (x1, y1), (x2, y2) = shape["points"]

        # 確保 x1 < x2, y1 < y2
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        # 轉成 YOLO 格式（正規化的 cx, cy, w, h）
        cx = (x1 + x2) / 2 / img_w
        cy = (y1 + y2) / 2 / img_h
        w  = (x2 - x1) / img_w
        h  = (y2 - y1) / img_h

        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    return lines


def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_files = list(LABELS_DIR.glob("*.json"))

    if not json_files:
        print(f"No .json files found in {LABELS_DIR}")
        return

    print(f"Found {len(json_files)} json files, converting...\n")
    total_boxes = 0

    for json_path in sorted(json_files):
        lines = convert_one(json_path)
        out_path = OUTPUT_DIR / json_path.with_suffix(".txt").name
        out_path.write_text("\n".join(lines), encoding="utf-8")
        total_boxes += len(lines)
        print(f"  {json_path.name} -> {out_path.name}  ({len(lines)} boxes)")

    print(f"\nDone! {len(json_files)} files, {total_boxes} total boxes")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    run()
