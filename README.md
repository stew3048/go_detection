# Go Detection

圍棋黑白棋子偵測專案，使用 YOLOv8 模型訓練。

## 專案結構

```
go_dection/
├── datasets/               # 資料集（git 忽略，需重新下載）
├── runs/                   # 訓練輸出與推論結果（git 忽略）
├── download_dataset.py     # 從 Roboflow 下載資料集
├── train.py                # 訓練腳本（30 epochs）
├── train_100ep.py          # 訓練腳本（100 epochs）
├── predict.py              # 推論腳本（使用 100ep 模型）
└── export_to_excel.py      # 匯出兩模型比較結果至 Excel
```

## 資料集

- 來源：[Roboflow Universe - go-games-ztg5w](https://universe.roboflow.com/drug-pmaup/go-games-ztg5w)
- 類別：`black`（黑棋）、`white`（白棋）
- 訓練集：366 張 / 驗證集：100 張 / 測試集：25 張

## 模型成績

| 模型 | mAP50 | mAP50-95 | Precision | Recall |
|------|-------|----------|-----------|--------|
| 30 epochs  | 0.719 | 0.599 | 0.939 | 0.707 |
| 100 epochs | 0.940 | 0.816 | 0.963 | 0.925 |

## 使用方式

### 安裝套件

```bash
pip install roboflow ultralytics openpyxl pyyaml
```

### 下載資料集

```bash
python download_dataset.py
```

### 訓練

```bash
python train.py          # 30 epochs
python train_100ep.py    # 100 epochs
```

### 推論

```bash
# 對測試集
python predict.py

# 對指定圖片或資料夾
python predict.py --source "path/to/image.jpg"
python predict.py --source "path/to/folder/"
```

### 匯出 Excel 比較報告

```bash
python export_to_excel.py
```
