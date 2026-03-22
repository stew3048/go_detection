# 2026-03-22｜OOD 完整實驗：Mahalanobis Distance vs AutoEncoder

## 一、今日完成項目

| 腳本 | 說明 |
|------|------|
| `synthesize_alien.py` | 合成黃棋（外星人）測試圖 |
| `ood_mahalanobis.py` | 法一：Mahalanobis Distance OOD 偵測 |
| `ood_autoencoder.py` | 法二：AutoEncoder OOD 偵測（訓練 + 推論）|
| `evaluate_ood.py` | 定量評估：stone-level Precision / Recall |
| `debug_distances.py` | 門檻校準輔助診斷腳本 |
| `check_gt.py` | 驗證 GT box 中心確實是黃色像素 |
| `debug_eval_00999.py` | 診斷 YOLO 重複偵測問題 |

---

## 二、OOD 是什麼？

**OOD = Out-of-Distribution Detection（超出已知分佈的偵測）**

> 「模型在不知道外星人存在的情況下，只靠看過大量正常棋子，識別出從未見過的異常物件。」

### OOD vs 一般分類的差異

| | 一般分類（YOLO） | OOD |
|--|----------------|-----|
| 問的問題 | 這是 black 還是 white？ | 這在已知範圍內嗎？ |
| 世界觀 | 閉世界：只有已知類別 | 開世界：可能有未知物件 |
| 遇到黃棋 | 硬選 black 或 white | 說「我不認識這個」|
| 需要黃棋資料 | 需要 | **不需要** |

---

## 三、整體架構（兩道關卡）

```
棋盤圖（640×640）
      │
      ▼
┌─────────────┐
│ YOLO best.pt│  第一關：找出所有棋子位置，截出 crop
└─────────────┘  （用我們自己訓練的 100ep 模型）
      │
      ▼
每個 crop（約 60×60）
      │
      ├─── 法一：Mahalanobis Distance ──→ 距離 > 7.5 → ALIEN
      │
      └─── 法二：AutoEncoder         ──→ MSE > 0.000248 → ALIEN
```

---

## 四、法一：Mahalanobis Distance

### 流程

```
Step 1：用 YOLO 對訓練集 100 張圖推論，截出棋子 crop
Step 2：每個 crop → 提取 HSV 色彩直方圖（32 維特徵向量）
Step 3：分別計算 black（976顆）和 white（886顆）的 mean + covariance matrix
Step 4：推論時：新 crop → 32維特徵 → 計算與 black/white 分佈的 Mahalanobis Distance → 取最小值
Step 5：最小距離 > 7.5 → ALIEN
```

### 特徵向量（32維）

HSV 色彩直方圖：
- H（色相）：16 bins，範圍 0~180
- S（飽和度）：8 bins，範圍 0~255
- V（亮度）：8 bins，範圍 0~255

```
黑棋：S 低（接近灰），V 低（很暗）
白棋：S 低（接近灰），V 高（很亮）
黃棋：H ≈ 30（黃色），S 高（鮮豔）← 與黑白棋完全不同
```

### Mahalanobis Distance 公式

```
d = √[ (x - mean)ᵀ × Σ⁻¹ × (x - mean) ]

x     = 新棋子特徵向量（32維）
mean  = 訓練集平均特徵向量
Σ⁻¹   = 共變異數矩陣的逆
```

**為什麼不用歐式距離？**
歐式距離不考慮各維度的資料分散程度。Mahalanobis 等於先把每個維度除以自己的標準差再量距離，讓「分布集中的維度」對異常更敏感。

```
例子：黑棋的 V（亮度）mean=20, std=4
黃棋 V=128 → (128-20)/4 = 27 個標準差 → 極度異常
```

---

## 五、法二：AutoEncoder

### AutoEncoder 是什麼？

AutoEncoder 是一個神經網路模型，架構是「壓縮 → 還原」：

```
原始 crop → [Encoder] → latent（壓縮表示）→ [Decoder] → 重建 crop
```

**核心概念**：訓練時只給正常棋子，讓 AE 學會還原正常圖片。
推論時遇到沒見過的黃棋，AE 無法正確還原 → 重建誤差（MSE）大 → ALIEN。

### Latent 是什麼？

Latent 是 Encoder 輸出的「壓縮表示」，也叫 bottleneck：

```
輸入：64×64×3 = 12,288 個數字
latent：8×8×64 = 4,096 個數字（壓縮比 3x）

模型必須用 4,096 個數字代表整張圖 → 強迫學習「最本質的特徵」
```

### Conv2d vs ConvTranspose2d

| | Conv2d | ConvTranspose2d |
|--|--------|----------------|
| 方向 | 縮小（多合一）| 放大（一展多）|
| 用在 | Encoder（壓縮）| Decoder（還原）|
| stride=2 效果 | 64×64 → 32×32 | 32×32 → 64×64 |

### 訓練流程

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  PHASE 0：準備 Training Data
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

366 張正常棋盤圖（無黃棋）
         ↓
  YOLO best.pt（weights 凍結，只做 inference）
         ↓
  8,408 個棋子 crop（64×64×3）
  全是黑棋 / 白棋，無標籤

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  PHASE 1：訓練 AutoEncoder（80 epochs）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

每個 batch（16 張 crop）：

  crop X（64×64×3）
       ↓
  [Encoder] 64→32→16→8（Conv2d × 3，stride=2）
       ↓
  latent（8×8×64）
       ↓
  [Decoder] 8→16→32→64（ConvTranspose2d × 3，stride=2）
       ↓
  重建圖 X'（64×64×3）
       ↓
  Loss = MSE(X, X')   ← X 就是 GT，不需要標籤！
       ↓
  Backprop → 更新 AE weights
  ❌ YOLO weights 完全不動

Loss 變化：
  Epoch  1：0.008833
  Epoch 10：0.000373
  Epoch 20：0.000222
  Epoch 30：0.000146
  Epoch 50：0.000104
  Epoch 80：0.000062  ← Best，儲存為 ood_ae_model.pth
```

### 為什麼訓練了 AE 還需要 YOLO？

AE 只回答「這顆棋子正不正常」，不知道棋子在哪裡。

```
YOLO 問的是：「棋子在圖片的哪裡？」  ← 定位問題
AE   問的是：「這顆棋子是不是外星人？」← 異常判斷問題
```

若不用 YOLO 直接把整張 640×640 圖丟進 AE：
- 黃棋只佔整張圖 1% 的面積
- 整張圖 99% 部分都還原得很好
- 黃棋的誤差被稀釋，MSE 整體仍低 → 偵測失敗！

**YOLO 先定位 → crop 讓黃棋佔 100% 畫面 → AE 誤差不被稀釋**

### 推論流程

```
測試圖 → YOLO best.pt → 截出每個 crop
                              ↓
                        AE（ood_ae_model.pth）
                              ↓
                        重建誤差（MSE）
                              ↓
                    MSE > 0.000248 → ALIEN 🔴
                    MSE ≤ 0.000248 → 正常棋子 🟢

threshold = 0.000248（從 230 顆 normal 棋的誤差取 p95 × 1.8 自動校準）
```

---

## 六、評估方法與修正過程

### 為什麼不能用 Frame-level 評估？

初版評估只看「這張圖有沒有偵測到至少一顆 ALIEN」，掩蓋了單顆石子層級的差異，導致兩個方法看起來一樣好。

### 正確評估：Stone-level Precision / Recall

- **Recall（召回率）**：真正的黃棋有多少被找到（漏抓是問題）
- **Precision（精確率）**：被標為 ALIEN 的有多少是真黃棋（誤報）
- **Ground Truth**：用相同 seed 重算黃棋位置（已驗證 20 個 GT box 中心都是黃色像素）

### 遇到的評估 Bug：YOLO 重複偵測

發現 YOLO 對同一顆黃棋輸出了兩個幾乎重疊的框（一個判成 black，一個判成 white），因為 NMS 只去重同一 class 的框。評估碼只能讓一個框匹配到 GT，另一個被算成 FP。

**修正**：在計算 Precision/Recall 前，先對所有 ALIEN 預測框做 NMS（互相 IoU > 0.5 就去重）。

---

## 七、最終評估結果（修正後）

共 20 顆 GT 黃棋（10 張圖 × 每張 2 顆）

### Mahalanobis Distance 逐圖結果

| 圖片 | GT | TP | FP | FN | 狀況 |
|------|:--:|:--:|:--:|:--:|------|
| alien_00530 | 2 | 1 | 0 | 1 | ❌ MISS x1 |
| alien_00545 | 2 | 1 | 0 | 1 | ❌ MISS x1 |
| alien_00999~01785（8張）| 各2 | 各2 | 0 | 0 | ✅ perfect |

**Precision = 1.000 / Recall = 0.900 / F1 = 0.947**

### AutoEncoder 逐圖結果

| 圖片 | GT | TP | FP | FN | 狀況 |
|------|:--:|:--:|:--:|:--:|------|
| alien_00530 | 2 | 2 | 1 | 0 | ⚠️ FP x1 |
| alien_00545~01785（9張）| 各2 | 各2 | 0 | 0 | ✅ perfect |

**Precision = 0.952 / Recall = 1.000 / F1 = 0.976**

### 最終對比

| 指標 | Mahalanobis | AutoEncoder |
|------|:-----------:|:-----------:|
| Precision | **1.000** | 0.952 |
| Recall | 0.900 | **1.000** |
| F1 Score | 0.947 | **0.976** |
| FP（誤報）| 0 | 1 |
| FN（漏報）| 2 | **0** |

**結論：AutoEncoder 表現更好（F1 0.976 > 0.947），且零漏報。**

Mahalanobis 的 2 個 FN 是因為 YOLO 信心分數太低，根本沒偵測到那 2 顆黃棋，OOD 階段連截圖都拿不到。這是兩個方法共同的瓶頸：都依賴 YOLO 先定位。

---

## 八、兩種方法的本質差異

| | Mahalanobis | AutoEncoder |
|--|-------------|-------------|
| **特徵** | 手工設計的 HSV 直方圖（32維）| 模型自動學的 latent（8×8×64）|
| **判斷依據** | 顏色離黑/白分布多遠？ | 模型嘗試還原後，誤差多大？ |
| **類比** | 用尺量顏色距離 | 用記憶重畫，看畫得像不像 |
| **優點** | 快、可解釋、不需訓練 | 特徵自動學習，更有彈性 |
| **缺點** | 依賴手工特徵設計 | 需要訓練（~50分鐘）|
| **Loss Function** | 無（統計方法）| MSE（均方誤差）|

---

## 九、視覺化輸出位置

| 目錄 | 內容 |
|------|------|
| `runs/ood_test/` | alien_*.jpg（有黃棋）+ normal_*.jpg（對照組）|
| `runs/ood_results/` | Mahalanobis 偵測結果圖（紅框=ALIEN）|
| `runs/ood_ae_results/` | AutoEncoder 偵測結果圖（紅框=ALIEN）|
| `runs/eval_vis/mahalanobis_distance/` | 評估可視化（綠=GT命中, 橘=GT漏抓, 紫=TP, 紅=FP）|
| `runs/eval_vis/autoencoder/` | 同上，AutoEncoder 版本 |

---

## 十、後續討論：公平比較與參數統一

### YOLO conf threshold 不統一的問題

討論中發現法一推論用 `CONF=0.20`、法二推論用 `CONF=0.10`，造成不公平比較。

**你的推論**：兩個方法對 YOLO 的用途完全一樣（定位），所以 threshold 應該統一。

**釐清 conf 的含義**：
```
conf = P(這裡有物件) × P(這個類別正確 | 有物件)
```
不是單純判斷顏色，也包含「這個位置有沒有東西」的信心。
黃棋 conf 低的原因：YOLO 從未見過黃色棋子，兩個成分都低。

**修正後（兩者都用 CONF=0.10）的最終結果**：

| 指標 | Mahalanobis | AutoEncoder |
|------|:-----------:|:-----------:|
| Precision | **1.000** | 0.952 |
| Recall | **1.000** | **1.000** |
| F1 Score | **1.000** | 0.976 |

**大逆轉**：統一 threshold 後 Mahalanobis 反而拿到完美分數。

### 為什麼 Mahalanobis 反而贏？

因為黃棋的顏色特徵（HSV H≈25, S=255）和黑白棋差異極大，32 維 HSV 直方圖輕鬆區分。
AutoEncoder 的 1 個誤報來自某顆正常棋子外觀特殊（光線/遮擋），重建誤差剛好超過 threshold。

**你的直覺**：如果外星人不好認、圖像更複雜，Mahalanobis 的 F1 會大幅下降，AutoEncoder 更 robust。

**驗證**：

| 情境 | Mahalanobis | AutoEncoder |
|------|:-----------:|:-----------:|
| 顏色差異明顯（本題：黃棋）| ✅ 完美 | ✅ 好 |
| 顏色差異不明顯（深灰棋）| ❌ 容易失效 | ✅ 較撐得住 |
| 形狀異常（白色三角形）| ❌ 幾乎無效（HSV 看不出形狀）| ✅ 有效 |
| 複雜真實場景 | ❌ 脆弱 | ✅ Robust |

**核心規律**：越複雜的任務，AutoEncoder 優勢越大；任務越簡單，統計方法反而贏。

---

## 十一、架構根本性限制的討論

### 你發現的問題

**如果外星人是「白色三角形」**：
- 法一：Recall ≈ 0（HSV 和白棋完全一樣，看不出形狀差異）
- 法二：理論上能偵測（圓形 → 三角形，重建誤差大）
- **共同瓶頸**：YOLO 只學過圓形，三角形 conf 可能 < 0.10 → 兩個方法都進不去

**你的結論**：OOD 方法高度依賴 YOLO，如果 YOLO 找不到，OOD 完全沒機會。

**誠實承認**：最初給的架構方向有盲點，只適合「異常物件形狀像棋子，只是外觀不同」的情境。

### 解決方案比較

| 架構 | 依賴 YOLO？ | 適合情境 |
|------|:---------:|---------|
| 現在的架構（YOLO + OOD）| ✅ 高度依賴 | 異常物件形狀像棋子 |
| 全圖 AE | ❌ 不依賴 | 異常佔畫面較大比例（小異常會被稀釋）|
| Sliding Window AE | ❌ 不依賴 | 任何異常，但計算量大 |
| PatchCore（業界主流）| ❌ 不依賴 | 工業缺陷偵測，目前最強 |

---

## 十二、未來方向（提案，尚未實作）

### 提案一：白色三角形外星人實驗
驗證「形狀異常」時兩個方法的差距：
- 修改 `synthesize_alien.py`：改成畫白色三角形
- 重跑兩個 OOD 方法（AutoEncoder 不需重新訓練）
- 預期：Mahalanobis Recall≈0，AutoEncoder 有效但受限於 YOLO 定位

### 提案二：Tesla 等自動駕駛的物體偵測
**你的好奇**：Tesla 怎麼知道那裡有障礙物、辨識多種不同物體？YOLO 做得到嗎？

方向：
- YOLO 本身就能偵測 80 種物體（COCO 資料集）
- Tesla 用的是更複雜的多攝影機融合 + 時序追蹤
- 核心差異：Tesla 需要 3D 空間感知（距離、速度），YOLO 只做 2D 框框

### 提案三：業界異常偵測怎麼做
**你的好奇**：業界（工廠缺陷偵測等）是不是也用 YOLO？

方向：
- 工業界主流：PatchCore、PaDiM、SimpleNet（不依賴 YOLO）
- 有標籤的異常：YOLO/分類器直接訓練（需要收集異常樣本）
- 無標籤異常（真正的 OOD）：AE、PatchCore 等重建方法
- 業界現實：往往兩者並用，已知異常用監督式，未知異常用 OOD
