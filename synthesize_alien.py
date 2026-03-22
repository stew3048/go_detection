"""
在現有棋盤圖片上合成「外星黃棋」，製造 OOD 測試圖。

輸出：
  runs/ood_test/
    alien_*.jpg   ← 混入黃棋的圖片（測試用）
    normal_*.jpg  ← 未修改的原圖（對照用）
"""

import cv2
import numpy as np
import random
import shutil
from pathlib import Path

# ── 設定 ─────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).parent.resolve()
SOURCE_DIR    = PROJECT_ROOT / "datasets" / "go-games-1" / "test" / "images"
OUTPUT_DIR    = PROJECT_ROOT / "runs" / "ood_test"
NUM_IMAGES    = 10       # 要合成幾張
ALIENS_PER_IMG = 2       # 每張混入幾顆黃棋
STONE_RADIUS  = 28       # 黃棋半徑（px），與真實棋子接近
ALIEN_COLOR   = (0, 215, 255)   # BGR：黃色
SEED          = 42
# ─────────────────────────────────────────────────────

random.seed(SEED)
np.random.seed(SEED)


def draw_alien_stone(img: np.ndarray, cx: int, cy: int, radius: int) -> np.ndarray:
    """在圖片上畫一顆仿真的黃棋（有光澤感）"""
    img = img.copy()
    # 主體：黃色填充圓
    cv2.circle(img, (cx, cy), radius, ALIEN_COLOR, -1)
    # 外圈：深黃色邊框（像真棋子的立體感）
    cv2.circle(img, (cx, cy), radius, (0, 160, 200), 2)
    # 高光：左上角白色小圓（反光效果）
    highlight_x = cx - radius // 3
    highlight_y = cy - radius // 3
    cv2.circle(img, (highlight_x, highlight_y), radius // 5, (255, 255, 220), -1)
    return img


def find_empty_positions(img_shape, labels_path: Path, radius: int, n: int) -> list:
    """
    從標註檔讀取已有棋子位置，在空白區域隨機找放置位置，
    避免黃棋疊在已有棋子上。
    """
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
    attempts = 0
    margin = radius + 10

    while len(positions) < n and attempts < 1000:
        attempts += 1
        x = random.randint(margin, w - margin)
        y = random.randint(margin, h - margin)

        # 確認不與已有棋子或已選位置重疊
        too_close = False
        for ox, oy, or_ in occupied:
            if np.hypot(x - ox, y - oy) < (radius + or_ + 5):
                too_close = True
                break
        for px, py in positions:
            if np.hypot(x - px, y - py) < radius * 2 + 5:
                too_close = True
                break

        if not too_close:
            positions.append((x, y))

    return positions


def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    label_dir = SOURCE_DIR.parent.parent / "test" / "labels"

    image_files = sorted(SOURCE_DIR.glob("*.jpg"))[:NUM_IMAGES]
    if not image_files:
        print(f"No images found in {SOURCE_DIR}")
        return

    alien_count = 0
    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # 對照組：原圖直接存
        normal_out = OUTPUT_DIR / f"normal_{img_path.name}"
        shutil.copy(img_path, normal_out)

        # 找空位並畫黃棋
        labels_path = label_dir / img_path.with_suffix(".txt").name
        positions = find_empty_positions(img.shape, labels_path, STONE_RADIUS, ALIENS_PER_IMG)

        alien_img = img.copy()
        for (x, y) in positions:
            alien_img = draw_alien_stone(alien_img, x, y, STONE_RADIUS)
            alien_count += 1

        alien_out = OUTPUT_DIR / f"alien_{img_path.name}"
        cv2.imwrite(str(alien_out), alien_img)
        print(f"  {img_path.name[:45]} → {len(positions)} aliens placed")

    print(f"\nDone! {len(image_files)} images, {alien_count} aliens total")
    print(f"Output: {OUTPUT_DIR}")
    print(f"  normal_*.jpg  ← clean images (no alien)")
    print(f"  alien_*.jpg   ← images with yellow stone mixed in")


if __name__ == "__main__":
    run()
