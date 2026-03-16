"""
從 Roboflow 下載圍棋棋盤偵測資料集，並整理成符合 YOLO 規範的資料夾結構。

下載後結構：
  datasets/go-4/
    train/images/  train/labels/
    valid/images/  valid/labels/
    test/images/   test/labels/
    data.yaml  ← 路徑自動修正為絕對路徑
"""

import os
import shutil
import yaml
from pathlib import Path

# ── 設定區 ──────────────────────────────────────────────
API_KEY      = "WMFmCYH28XKimZKxa0sV"
WORKSPACE    = "drug-pmaup"
PROJECT_NAME = "go-games-ztg5w"
VERSION      = 1
FORMAT       = "yolov8"

# 本腳本所在目錄即為專案根目錄
PROJECT_ROOT = Path(__file__).parent.resolve()
DATASETS_DIR = PROJECT_ROOT / "datasets"
# ─────────────────────────────────────────────────────────


def download_from_roboflow() -> Path:
    """下載資料集到 datasets/ 下，回傳資料集根目錄。"""
    from roboflow import Roboflow

    rf = Roboflow(api_key=API_KEY)
    project = rf.workspace(WORKSPACE).project(PROJECT_NAME)
    version = project.version(VERSION)

    # Roboflow 會在目前工作目錄下建立資料夾，先切到 datasets/
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    os.chdir(DATASETS_DIR)

    dataset = version.download(FORMAT)

    # 取得 Roboflow 實際建立的資料夾路徑
    dataset_path = Path(dataset.location).resolve()
    return dataset_path


def ensure_yolo_structure(dataset_path: Path) -> None:
    """
    確保資料集符合 YOLO 標準結構：
      split/images/  split/labels/
    Roboflow 下載的格式通常已符合，此函式做防禦性檢查。
    """
    for split in ("train", "valid", "test"):
        split_dir = dataset_path / split
        if not split_dir.exists():
            continue
        for sub in ("images", "labels"):
            (split_dir / sub).mkdir(parents=True, exist_ok=True)

        # 若圖片直接放在 split/ 下（非 images/ 子目錄），移進去
        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        label_exts = {".txt"}
        for f in list(split_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in image_exts:
                shutil.move(str(f), str(split_dir / "images" / f.name))
            elif f.is_file() and f.suffix.lower() in label_exts:
                shutil.move(str(f), str(split_dir / "labels" / f.name))


def fix_yaml_paths(dataset_path: Path) -> None:
    """
    將 data.yaml 中的路徑改為絕對路徑，避免因工作目錄不同而找不到資料。
    同時印出 nc（類別數）與 names（類別名稱）供確認。
    """
    yaml_path = dataset_path / "data.yaml"
    if not yaml_path.exists():
        print(f"[WARNING] data.yaml not found at {yaml_path}")
        return

    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 修正各 split 路徑為絕對路徑
    for split in ("train", "val", "test"):
        if split in cfg:
            # 相容舊版 Roboflow 使用 valid/ 或 val/
            folder = "valid" if split == "val" else split
            abs_path = (dataset_path / folder / "images").as_posix()
            cfg[split] = abs_path

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, sort_keys=False)

    print("\n---- data.yaml updated --------------------------------")
    for split in ("train", "val", "test"):
        if split in cfg:
            print(f"  {split}: {cfg[split]}")
    print(f"  nc   : {cfg.get('nc', '?')}")
    print(f"  names: {cfg.get('names', '?')}")
    print("-------------------------------------------------------\n")


def print_structure(dataset_path: Path) -> None:
    """印出最終資料夾結構摘要。"""
    print("---- Dataset structure --------------------------------")
    for split in ("train", "valid", "test"):
        split_dir = dataset_path / split
        if not split_dir.exists():
            continue
        imgs   = list((split_dir / "images").glob("*")) if (split_dir / "images").exists() else []
        labels = list((split_dir / "labels").glob("*")) if (split_dir / "labels").exists() else []
        print(f"  {split}/images/  -> {len(imgs):>4} images")
        print(f"  {split}/labels/  -> {len(labels):>4} labels")
    print(f"\n  data.yaml: {dataset_path / 'data.yaml'}")
    print("-------------------------------------------------------")


if __name__ == "__main__":
    print("[1/4] Downloading dataset from Roboflow...")
    dataset_path = download_from_roboflow()
    print(f"      Done: {dataset_path}\n")

    print("[2/4] Verifying YOLO folder structure...")
    ensure_yolo_structure(dataset_path)
    print("      OK\n")

    print("[3/4] Fixing data.yaml paths...")
    fix_yaml_paths(dataset_path)

    print("[4/4] Summary:")
    print_structure(dataset_path)
    print("\nAll done! Use the data.yaml path above for training.")
