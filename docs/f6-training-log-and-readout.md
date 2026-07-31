# F6 訓練紀錄與判讀指引

> 建立：2026-08-01｜狀態：GAT 消融進行中（`no_gat` 完成、`old_kg` 訓練中、`v2_kg` 待跑）
>
> **這份文件是為了讓一個沒有前文脈絡的人（或新 session）能直接接手。**
> 每一輪訓練的設定、數字、要看什麼、以及不能怎麼比，全部寫在這裡。
>
> 相關：`docs/feature-protocol-v2.md` §9（F5 定案規格與 baseline 數字）、
> `V6/experimental/kg_ablation.py`（消融腳本）、
> `V6/experimental/result/f5_r_series_result.json`（F5 完整數字）

---

## 0. F6 是什麼、要回答什麼

F5（baseline 層，已完成）用 Ridge/GBDT 量出特徵工程各項改動的 IC delta，
定案了兩項規格。F6 是**用真正的架構（Mamba + GATv2）驗證**，有兩個問題：

1. **GATv2 到底有沒有貢獻？** —— 三組消融 `no_gat` / `old_kg` / `v2_kg`
2. **Group D（總經 12 維）有沒有貢獻？** —— 尚未開始，腳本待寫

**F6 定案規格（協定 v2.0 §9.4，凍結）**

| 項目 | 值 |
|---|---|
| `INPUT_DIM` | **59**（可得性旗標不採用） |
| `fundamentals_v2` | **True** |
| `availability_flags` | **False** |
| `macro_norm` | `"ts"` |
| `neutralize` | `"none"` |
| 切分 | `splitters.train_val_split_dates`，purge + embargo 20 |
| train 起點 | **2013-01-01** |
| train cutoff | 2023-12-31 |
| val 窗 | **2024-01-02 → 2026-06-02（582 天）** |
| 宇宙 | `filter_tradable_universe`（排除 ETF + 興櫃）→ 2,245 支 |

**特徵矩陣**：7,990,686 列 × 65 欄（59 特徵 + 6 meta），已備份到
Drive `V6_Feature_Matrix.parquet`（2.51 GB）。三層驗證全過（見 §5）。

**已知死維**：`FED_Rate` 整維恆為 0——`fed_rate.parquet` 只有 **8 列 / 1 個相異日期
（2004-01-01）**，等於沒有時間序列。所以 59 維中實質 **58 維有訊息**，
Group D 是 **11 活 + 1 死**。做 Group D 消融時結論必須寫成「11 維有效特徵的貢獻」。

---

## 1. 判讀的共同基準：同一批 582 天

F6 的 val 窗**刻意**設成與 F5 的 test 窗同一批交易日（2024-01-02 → 2026-06-02），
所以下面所有數字可以直接並列。IC 一律是**每日橫斷面 Spearman、再對交易日平均**
（`trainer.compute_ic` 與 F5 的 `daily_spearman_ic` 同一個統計量）。

| 模型 | 5d IC | ICIR | 年化 | Sharpe | MDD | 換手 |
|---|---|---|---|---|---|---|
| GBDT G-R2（旗標開） | **+0.1040** | 0.878 | +8.2% | 0.497 | −32.3% | 80% |
| GBDT G-R1（旗標關） | +0.1036 | 0.842 | +7.7% | 0.469 | −32.8% | 79% |
| GBDT G-R4a（中性化） | +0.1059 | 0.950 | **+14.7%** | 0.779 | −32.9% | 82% |
| Ridge R1 | +0.0899 | 0.843 | +11.3% | 0.699 | −26.2% | 76% |
| Ridge R2 | +0.0899 | 0.849 | +10.7% | 0.668 | −25.4% | 76% |
| Ridge R4a（中性化） | +0.0924 | 0.996 | +14.0% | 0.848 | −25.6% | 76% |
| **Mamba 單尺度 `no_gat`** | **+0.0884** | **0.733** | +10.7% | 0.572 | −33.4% | 77% |
| Mamba 多尺度 + 舊圖 | +0.0526 ⚠️ | — | — | — | — | — |

⚠️ 多尺度那筆**排程受損、不可直接比**，原因見 §3。

**目前的圖像**：11.5M→1.4M 參數的 Mamba 打不贏 300 維的 Ridge，更打不贏 GBDT。
這與方向二 Step 4 的收斂結論一致——**模型形式（線性/樹/RNN/SSM）只在 ±0.01 內移動，
訊號來自特徵/label/廣度**。但 GAT 兩組還沒出來，先不下定論。

---

## 2. 執行紀錄：GAT 三組消融

**腳本**：`V6/experimental/kg_ablation.py`
**結果**：Drive `MyDrive/MarketMamba_V6/kg_ablation_result.json`（每組跑完落盤、可續跑）
**設定**：`epochs=10, early_stop=5, dropout=0.2, seed=20260730`，
train 2,661 天（horizon=10 → purge 砍 30 天）、val 582 天

### 三組的定義

| arm | `use_gat` | 圖 | 節點 / 邊 | 參數量 |
|---|---|---|---|---|
| A `no_gat` | False | 不使用 | — | **1,394,301** |
| B `old_kg` | True | `knowledge_graph_cache.npz` | 42,864（真股票 2,510）/ 642,451 | **1,659,005** |
| C `v2_kg` | True | `knowledge_graph_v2.npz` | 2,245（全為真股票）/ 32,083 | 1,659,005 |

舊圖為什麼可疑：642,451 條邊中滾動相關性邊 **0 條**（動態層從未生效）、
供應鏈邊來自「regex 抓 HTML 裡所有 4 位數字」、2330 的鄰居是電器電纜/化學工業、
產業邊用**未設 seed 的 `random.sample`**。線上 IC ~0.08 是帶著這張雜訊圖達到的。

### A `no_gat` — 已完成

```
峰值 5d IC = +0.0884 @ep5 ｜ 峰後每 epoch 平均下滑 +0.0027 ｜ 參數 1,394,301
最佳 checkpoint 重評：582 天 ｜ mean IC +0.0884 ｜ ICIR 0.733 ｜ 正比例 80.1%
組合層：年化 +10.7% ｜ Sharpe 0.572 ｜ MDD −33.4% ｜ 換手 77%
每 epoch 1,293–1,296 秒（≈21.6 分）｜ ep10 early stop（5 ep 無改善）
```

逐 epoch val IC（5d）：
`0.0460 / 0.0768 / 0.0807 / 0.0787 / 0.0884 / 0.0790 / 0.0789 / 0.0744 / 0.0745 / 0.0750`

val_loss：`0.13409 / 0.12799 / 0.12807 / 0.12843 / 0.12829 / 0.12825 / 0.12920 / 0.13021 / 0.13022 / 0.13030`

**一個內部一致性檢查通過了**：重評的 mean IC **+0.0884 與訓練迴圈 ep5 的 val IC
完全相同** → checkpoint 重載 + 逐日 IC 那段程式是對的。

### B `old_kg` — 訓練中

參數 1,659,005（比 A 多 **264,704**）。開跑設定與 A 逐項相同（2,661/582 天、同 seed）。

### C `v2_kg` — 待跑

---

## 3. 執行紀錄：多尺度主模型（Cell 4）—— 已中止

**這一輪不可用來判斷多尺度架構的優劣**，因為 LR 排程被 `epochs=100` 拉壞了。

設定：`multi_scale_layers=[2,3,3]`、`loss_weights={mse_5d:1.0, mse_20d:0.3, mse_60d:0.3,
listnet_20d:0.0}`、train 2,611 天（horizon=60 → purge 砍 80 天）、val 582 天、**舊圖**。

```
best val IC = +0.0526 @ep7（ep14 已掉到 +0.0259，剩一半）
val_loss    ep4 谷底 1.5965 → ep14 1.9068（連續 10 個 epoch 上升，+19.4%）
train_loss  3.385 → 1.399（持續下降）
每 epoch ≈ 47.7 分
```

### 為什麼排程壞了（重要，會再犯）

`WARMUP_PCT=0.15` × `epochs=100` → **暖身長達 15 個 epoch**。

| epoch | LR | 佔 max_lr (7e-5) |
|---|---|---|
| ep7（IC 峰值） | 3.289e-5 | **47%，還在往上爬** |
| ep14 | 6.927e-5 | 99% |

**模型在暖身期就收斂並開始過擬合，從頭到尾沒進入退火階段。**
對照單尺度那輪 `epochs=10`：LR ep3 到頂 6.99e-5、ep10 退到 2.8e-10，完整走完一圈。

→ 根因是 **`epochs` 一個參數身兼兩職**：既是「最多跑幾輪」也是「OneCycle 排程長度」。
有 early stopping 時這兩者不該綁在一起。要重跑多尺度的話 `epochs` 應設 **15–20**。

### 白賺的發現：scale gate 再次塌到 Mid

ep14 收斂值 **Short 0.047 / Mid 0.890 / Long 0.063**。
Phase 0（5d 目標、舊資料基礎）是 Short 0.004 / Mid 0.80 / Long 0.20。
**同樣的形狀在全新的資料基礎上重現。**

加上 Phase 1 deep supervision 早已證實三尺度對 5d 冗餘，現在有**三條獨立證據**
指向同一件事：5d 目標下多尺度實質退化成單尺度（Mid，60 天回看）。

---

## 4. 消融跑完後要看什麼（判讀清單）

### 4.1 先檢查實驗本身有沒有壞

- [ ] **峰值 epoch 落在哪**。若 `old_kg` 或 `v2_kg` 的峰在 **ep9–ep10**，代表排程對它們
      太短、峰值被截斷 → **三組要一起用更長的 `epochs` 重跑**，不能只補那一組
      （`epochs` 決定 LR 曲線，改了就不可比）。落在 ep7 以前就沒問題。
- [ ] **三組的 train/val 天數是否都是 2,661 / 582**。任一組不同就代表切分沒對齊。
- [ ] **`v2_kg` 的圖摘要**是否印出 2,245 節點 / 32,083 邊。若印出 42,864 就是載錯圖。

### 4.2 主判讀

腳本尾端會自動印對照表 + 配對 Δ + Newey-West t（三組共用同一批 582 天）。

| 結果 | 結論 |
|---|---|
| `B > A` 明顯 | 圖有貢獻，即使是雜訊圖（可能只是多一層平滑/正則） |
| `C > B` 明顯 | 圖的**品質**有貢獻 → 值得再投資（動態相關性邊、產業鏈邊，即 Phase 4-A） |
| `A ≈ B ≈ C` | GATv2 對 5d 沒加值 → **砍掉**，省參數與算力，也不必再做產業鏈邊 |
| `C ≈ A < B` | 反直覺，**先懷疑實驗本身**（例如 v2 圖的 stock_id 對不上） |

門檻：峰值 Δ ≥ **+0.009**（Phase 3-A 的 dropout 效應，對訓練雜訊校準過）
**且**配對 NW t 的 |t| ≥ 2。

### 4.3 三個不能忽略的干擾項

1. **參數量**：B/C 比 A 多 264,704 個參數（`graph_layer`/`gate`/`norm_fuse`）。
   若 `B > A`，**分不出是圖的資訊還是多出來的參數容量**。要分開需補第四組
   `random_kg`（保留 GAT 層但邊全部隨機重連、度分布與 v2 圖相同）。
   **只在 `B > A` 且顯著時才值得跑那一組。**
2. **A 與 B/C 的隨機性不同**（2026-08-01 實測發現，比原本記載的更廣）：
   固定 seed 只保證 **B vs C 逐參數同初始化、同資料順序**（架構相同 → RNG 消耗相同）。
   **A 因少建 GAT 層而少消耗 RNG**，導致 (i) head 初始化不同 (ii) DataLoader 的
   `RandomSampler` 取到不同種子 → **資料打亂順序也不同**。
   證據：A 的第一個 batch `stocks=1558`、B 是 `stocks=1668`（同一份 dataset、同一個 seed）。
   → **B vs C 是乾淨的配對比較；A vs B 有三個干擾項，要保守讀。**
3. **三組落在 0.01 以內時不能直接宣告「A≈B≈C」**。要拿其中一組多跑 **2 個額外 seed**
   量 run-to-run σ（資料順序造成的變異正好會被這個 σ 涵蓋），否則是拿雜訊當結論。

### 4.4 引用紀律

- **不要跟 Phase 3-A 的 0.0959 比**。那是 2005 起、4,651 天、無 purge 的數字；
  F6 是 2013 起、2,661 天、有 purge。只能組內比較。
- **與 F5 的 Ridge/GBDT 並列時**，有一個二階差異要一起講：F6 矩陣自 2005 建、
  F5 的 v2 矩陣自 2011 建，而 `macro_norm="ts"` 是 **expanding** z-score
  → 兩者 macro 12 維數值不完全相同。對 F5 的線性模型完全免疫（macro 是每日橫斷面
  常數），但對 Mamba 不是嚴格免疫（常數會經非線性層與個股特徵交互）。
  不值得為它重建矩陣，但不能宣稱是純粹的架構比較。
- **IC 與組合層可能不同調**（Step 3 教訓）。已經出現兩次：GBDT IC 最高（+0.1036）
  年化最差（+7.7%）；`no_gat` IC 最低（+0.0884）年化中段（+10.7%）。

---

## 5. Colab 執行的三個坑（都踩過，會再踩）

1. **模組快取**：`git pull` 之後 Python 仍用 `sys.modules` 裡的舊版，**完全不會報錯**，
   只會看到一個「看起來正常」的訓練跑起來。實際踩到的後果：`train_start` 沒生效、
   訓練集變成 2005 起的 4,651 天。
   → 每次 pull 後必須清快取並**印出模組常數確認**：
   ```python
   import sys
   for m in [k for k in sys.modules if k.startswith("experimental")]:
       del sys.modules[m]
   import experimental.kg_ablation as ka
   assert ka.TRAIN_START_DEFAULT == "2013-01-01"
   assert ka.VAL_END_DEFAULT == "2026-06-02"
   ```
2. **特徵矩陣快取不帶規格資訊**：Drive 上留著舊的 `V6_Feature_Matrix.parquet` 時，
   Cell 3 只印一行 "Loading cached feature matrix..." 就直接用它。
   → Cell 3 尾端的**三層驗證**專門擋這個，其中只有第三層擋得住：
   **內容指紋** `Book_Value` 相異值數（舊財報 = **1**，修好後 = 5,033,154）、
   `Gross_Margin`（舊 = 524，修好後 = 7,002,546）。
3. **`train_short_model` 的全域洩漏**（已於 `d71b8ac` 修好）：它為了吃 5d/10d 標籤而改
   `T.TARGET_COLS` / `PRED_HORIZONS`，原本跑完沒還原 → 同一 session 接著跑 Cell 4 的
   多 horizon 訓練會 `IndexError: index 2 is out of bounds`，而錯誤訊息指向
   `trainer.py`，與真正的原因差很遠。現已用 try/finally 保證還原並印出還原值。

---

## 6. Colab 訓練期間可並行的工作（本機，不碰 GPU）

### 6.1 拆解 F5 那個 −0.0079（本機 CPU，約 30 分鐘）

`R0b − R0 = −0.0079` 是整個 F5 最大的單項效應，**至今無法歸因**：
`fundamentals_v2` 只佔 −0.0014、起點 +0.0001，剩約 −0.0065 混在
「宇宙過濾（ETF＋興櫃）」與「外資持股 9xxx 回補」兩者裡。
協定 §9.5 目前寫的是「要拆開需再建一份矩陣，未做」。

做法：在 `baseline_common.py` 的 `_VARIANT_SPECS` 加一個「v2 規格 + v1 宇宙規則」的變體
（`_filter_universe` 目前是依 `PROTOCOL_VERSION` 分支，需要改成也吃變體旗標），
建一份矩陣（~27 分鐘）+ 跑一級 Ridge（1.5 分鐘），對照 R0b。

### 6.2 寫 Group D 消融腳本（純寫程式）

照 `kg_ablation.py` 同一套模式：獨立 checkpoint、固定 seed、逐日 IC、組合層、可續跑。
兩組：`with_macro`（現況 59 維）vs `no_macro`（砍掉 Group D 12 維 → 47 維）。

⚠️ **結論必須寫成「11 維有效特徵的貢獻」**——`FED_Rate` 是死維。
⚠️ 砍維度會改變 `INPUT_DIM` 與 `GROUP_DIMS`，而 `FactorGroupedEmbedding` 的
`group_dims` 預設參數在 **def 執行時**求值 → patch 必須在 `import marketmamba.models.*`
**之前**（同 `feature_spec.patch_config_67d` 那個陷阱，見協定 §9 與核對結果 F2）。

### 6.3 macro 停更 → regime 閘門即將失效（線上系統）

實測 `macro_raw` 最新是 **2026-07-23**（5,620 列、無斷層），
但 `run_daily_update` **完全不碰 macro**（已逐項列出它呼叫的 17 個 fetcher，沒有 macro）
→ 會再度停更。scanner 的 regime 判斷有 **10 天新鮮度檢查**，超過就退回 NORMAL、
保守模式 BUY 門檻從 ≥90 變回 ≥70。

⚠️ **CLAUDE.md 原記載「macro_raw 停在 2026-04-24」已過時**，實際是 07-23。

修法：把 `fetch_macro` 接進 `run_daily_update`（模式同其他源）。
**但若 V6.1 要退役（使用者 2026-08-01 表示可以停），這一項就不必做。**

### 6.4 不建議現在做：修 `fed_rate`

依既有決策紀錄：「先證實 Group D 有貢獻，再補來源；否則是為未經證實的特徵接資料源。」
`fed_rate` 屬同一類 → **先做 6.2 的消融**。

（`fear_greed` 3,885 列到 2026-04-24、`business_indicator` 266 列到 2026-02-01，
兩者在訓練窗 2013–2023 內都是活的，所以層 2b 只報 `FED_Rate` 死是正確的。）
