# 特徵工程層收尾檢查表 — 核對結果

> 對照：`MarketMamba_特徵工程檢查表.md`
> 核對日：2026-07-30
> 核對方式：對真實資料與程式碼實測，不從記憶或文件推論。
> 標記：**✅ 通過** / **⚠️ 有條件通過（附說明）** / **❌ 未完成**

**第一輪核對（2026-07-30 上午）**：A–F 21 項 → 通過 17、有條件 2、未完成 2；
G 類 4 項 → 通過 1、未完成 3。

**第二輪（同日，補做完 A2 / E1 / F3 / G1 之後）**：
A–F **21 項全數通過**；G 類通過 2、未完成 2（皆待 F5 全量結果，非程式問題）。

核對過程共**發現並修掉 3 個實質問題**：

| # | 問題 | 嚴重度 |
|---|---|---|
| 1 | **`^\d{4}$` 擋不住 ETF**（`0050`/`0056` 是 4 位數）→ 35 支混進矩陣 | 中 |
| 2 | **import 連鎖讓 67 維 patch 失效** → v2 矩陣會靜默變成 59 維、8 個旗標全部丟失 | **高** |
| 3 | **purged CV 沒有任何腳本在用** | 中 |

第 2 項是本次核對最大的收穫，詳見 F1。

---

## A. 特徵計算正確性與順序

| # | 項目 | 結果 |
|---|---|---|
| A1 | 時序特徵先在各股序列算完才橫斷面標準化，無路徑順序顛倒 | **✅** |
| A2 | 每個新增/修改特徵都跑過子集不變性測試 | **❌ 未完成** |
| A3 | `clean_and_scale` 順序（winsorize → 中性化 → z-score）為唯一生效路徑 | **✅** |

**A1**：子集不變性測試 [1] 實測 59 欄 × 16,346 列**最大絕對差 `0.000e+00`**。
`baseline_common` 的 rolling 順序已於 2026-07-27 修正（G4），
`Mom_*` 一直是正確順序。生產路徑與研究路徑皆已確認。

**A3**：`feature_engineer.py:1673-1692` 實測只有一條路徑，無殘留舊分支。
且該重構經 git HEAD 逐位元比對（`cross` / `ts` 兩種 macro_norm 皆 **`0.000e+00`**）。

**A2 — 已補做（第二輪）**：新增 `--protocol v2` 選項。
實測 `--protocol v2 --full 40 --sub 8`：比對欄位由 59 → **70 欄**（67 特徵 + 3 標籤），
**最大絕對差 `0.000e+00`**。8 個 `Avail_*` 旗標的子集不變性從「應該是」變成「驗過了」。

⚠️ 而這一補做**直接挖出了本次核對最嚴重的問題**——見 F1。
原本的判斷「旗標從實作看應該是子集不變的」是對的，
但如果沒有真的跑一次，就不會發現 patch 根本沒生效。

---

## B. 中性化與標準化

| # | 項目 | 結果 |
|---|---|---|
| B1 | 中性化統計量只用當天實際可得的股票池 | **✅** |
| B2 | 產業分類已套 `resolve_sector()` 與 `canonical_sector()` | **✅** |
| B3 | 宇宙過濾與 `filter_tradable_universe` 一致 | **✅（本次核對發現問題並已修）** |

**B1**：`_neutralize_cross_section` 逐 `Date` 分組，只用當日實際存在的列。
殘差驗收：`industry` 產業內均值 |max| 3.33e-16、與 log 市值相關 −0.0001；
`industry_mktcap` 5.55e-14 / +0.0005。

**B2**：`_neutralize_cross_section` 呼叫 `resolve_sector(load_stock_info(latest_only=False))`。
`resolve_sector` 內部已套 `canonical_sector`，同時解決兩個問題：
跨市場命名（上櫃「運動休閒類」vs 上市「運動休閒」）與大類/細類並存
（605 組 (股票,日期) 同時有「電子工業」與「半導體業」）。

**B3 — 本次核對抓到的實質問題**

檢查表這一項的措辭把「不參與中性化的**欄位**」與「排除的**標的**」混在一起
（`NEUTRALIZE_EXCLUDE` 實際上是欄位清單），但**它指向的疑慮是真的**：

> `baseline_common` 的協定 §2 只做 `^\d{4}$` 過濾，
> 而 **ETF 的 `0050`／`0056` 正好是 4 位數字**。

實測：**35 支混入**（23 支 ETF + 12 支興櫃），**52,998 列 = 0.64%**。
ETF 在橫斷面裡會污染 winsorize 與 z-score，中性化時全部落進「Unknown」產業組，
而它們的 Alpha 對選股毫無意義。

**已修**：新增 `_filter_universe()`，v2 改用 `hygiene.filter_tradable_universe()`
（與 `run_daily_inference._sanitize` 同一套規則）；
**v1 刻意維持原樣**——已發表的 Ridge/GBDT/GRU 結果是在那個宇宙下跑的，改了無法重現。
實測 v1 `2,396 → 2,202` 支（不變）、v2 `2,396 → 2,169` 支。

---

## C. 缺失處理與可得性旗標

| # | 項目 | 結果 |
|---|---|---|
| C1 | 旗標完全不經標準化直接進入模型輸入 | **✅** |
| C2 | 旗標語意（值是否為真 vs 今天有無新觀測）程式與文件一致 | **✅** |
| C3 | 死旗標處理依**全量**結果定案並同步 `INPUT_DIM` | **❌ 待 F5** |
| C4 | 「整欄全缺」與「部分缺」維持區分，無新特徵用回統一 `fillna(0)` | **✅** |

**C1**：`clean_and_scale` 對 `AVAIL_COLS` 完全跳過；測試 [3] 實測標準化後仍只有 {0.0, 1.0}。
對照組 `Return_5d` mean +0.0000 / std 0.9874，證明標準化本身確實有跑。

**C2**：語意 = 「該來源對這筆有真實觀測（可能經 ffill 帶下來，但源頭存在過）」，
實作為「ffill 之後、`fillna(0)` 之前的 `notna`」，
`feature_spec.py` 的說明與 `_mark_avail()` 一致。
**已揭露的限制**：旗標不描述新鮮度（2013 有融資、2020 被取消資格者旗標仍為 1）。

**C3 — 待 F5**：40 支樣本下 `Avail_Margin` / `Avail_Valuation` / `Avail_Financials`
在 2013+ 訓練窗內是常數（std=0.0000）。樣本偏老牌大型股，不足以定案。
若全量仍為常數 → 旗標砍到 5 個、`INPUT_DIM` 降為 **64**。腳本已內建自動偵測。

---

## D. 圖結構（GATv2）正確性

| # | 項目 | 結果 |
|---|---|---|
| D1 | 節點全為可交易真股票，無 ETF/權證 | **✅** |
| D2 | 產業邊決定性，無未設種子的隨機抽樣 | **✅** |
| D3 | 排序鍵（權重 + `stock_id` 決勝）一致套用 | **✅** |
| D4 | 集團邊與至少一個獨立公開來源交叉比對 | **✅** |
| D5 | 相關性邊與供應鏈邊確認未啟用 | **✅** |
| D6 | 人工可讀的抽驗 | **✅** |

**D1**：2,245 節點全數通過 `filter_tradable_universe`（舊圖 42,864 節點僅 2,510 是真股票）。
**D2**：`build_sector_edges_v2` 依 (市值 desc, stock_id asc) 排序，取代舊版未設 seed 的 `random.sample`。
**D3**：`merge_edges` 排序鍵 `(-weight, dst_index)`——第二個鍵是為了避免同權重時
取決於 dict 迭代順序。
**D4**：與 Yahoo 股市「集團股」分類頁比對，**0 誤收 / 37 漏收**（13 vs 50 檔）。
順帶驗出舊表另一錯誤：3532 台勝科屬台塑集團而非友達集團。
完整名單存於 `CONGLOMERATE_TABLE_FULL`，待 GAT 消融後再決定是否採用。
**D5**：v2 不含這兩類邊。相關性邊的啟用前置條件（GAT 消融證明圖有貢獻）寫在
`kg_builder_v2.py` 與協定 §6.3，不會被意外引入。
**D6**：2330 → 聯電/聯發科/日月光/環球晶/創意（舊圖為電器電纜/化學工業/綠能環保）；
1101 台泥 → 亞泥/嘉泥/環泥/幸福/信大/東泥；2412 中華電 → 台灣大/遠傳。
**這一關正是發現舊圖是壞的方式**——舊圖的節點數與邊數看起來完全正常。

---

## E. 特徵—Label 時間對齊與切分

| # | 項目 | 結果 |
|---|---|---|
| E1 | G1 對照表涵蓋所有納入模型的特徵組，與 `feature_spec.py` 同步 | **⚠️ 有條件通過** |
| E2 | `assert_no_leakage()` 本身用正確/錯誤案例各測過一次 | **✅** |
| E3 | purge/embargo 參數依據寫入協定，標註「這是選擇不是定理」 | **✅** |
| E4 | 訓練起點與 CV 切分範圍同步一致 | **✅** |

**E1 — 有條件**：`AVAILABILITY_TABLE` 涵蓋 13 個特徵群，
但 **RS_5d/20d/60d 被歸在「OHLCV / 技術指標」列裡、未單獨列出**其相依性
（它們需要 `_merge_macro` 先提供 TWII_Return_*，是唯一跨 Group 的相依）。
不影響正確性，但對照表的用途是「一次看出有沒有特徵漏掉檢查」，建議補一列。

**E2**：`splitters.py:__main__` 有明確的正/反兩個案例——
39 個 fold 全部通過斷言（正例），以及「不做 purge/embargo」的對照組
**正確抓出現行切分洩漏到 2024-04-08**（反例）。
斷言本身也曾寫錯過一次（拿 `train ∪ test` 當日曆，把正確切分誤報成洩漏），
已修正並在程式碼中記錄該教訓。

**E3**：協定 §5 明載「embargo 取 20 是嚴謹度與樣本量的折衷，**這是個選擇不是定理**」。

**E4**：v2 的 `PROTOCOL` 統一定義 `TRAIN_START=2013-01-01` 與
`PURGE_HORIZON=60` / `EMBARGO_DAYS=20`，`splitters` 以參數接收、無硬編碼日期。

---

## F. 版本管理與回歸驗證

| # | 項目 | 結果 |
|---|---|---|
| F1 | patch 順序保護遵循且該保護本身被驗證會觸發 | **✅** |
| F2 | v1 快取與程式碼路徑未被本輪異動影響 | **⚠️ 有條件通過** |
| F3 | 新增腳本/規格文件已標註凍結或版本狀態 | **⚠️ 部分** |

**F1 — 本次核對抓到的最嚴重問題（已修）**

原本的保護只擋 `marketmamba.models`。但實測發現：

```
marketmamba/data/__init__.py:
    from marketmamba.data.feature_engineer import build_features, clean_and_scale
```

**光是 `from marketmamba.data.feature_spec import patch_config_67d` 這一行，
就會先觸發 `__init__`、把 `feature_engineer` 連帶載入並綁定 56 維的 `FEATURE_COLS`**，
patch 才執行。

後果（三層都是靜默的）：
1. `build_features` 尾端 `df[meta_cols + FEATURE_COLS]` 用舊清單重排 → **8 個旗標被丟掉**
2. `_cfg_has_flags` 也讀舊清單 → 防呆的 `ValueError` **不會觸發**
3. 只剩一行 warning，而 `MM_PROTOCOL=v2 --build` 的 log 有數百行

→ **`MM_PROTOCOL=v2` 建出來的矩陣會是 59 維、沒有任何旗標，而且沒有人會發現。**
整個決策1 的工作等於白做。

**修法**：`patch_config_67d()` 在 patch 完 config 後，**顯式同步已載入模組的 module 級綁定**，
並當場 assert 同步成功（這個 bug 的可怕之處就是沒有徵兆，所以不能只是「寫了同步的程式碼」）。

`architecture.py` 為何仍需 strict 擋下而不能事後同步：
`FactorGroupedEmbedding.__init__(group_dims=GROUP_DIMS)` 的**預設參數在 def 執行時就求值**，
改 module 全域救不回來。`feature_engineer` 則是呼叫時才讀，所以事後覆寫有效。
**同一個「import 期綁值」的陷阱，兩個模組要用不同的解法。**

實測驗證：
```
patch 前 feature_engineer.FEATURE_COLS = 56 維
INPUT_DIM = 67
patch 後 feature_engineer.FEATURE_COLS = 67 維  ✓
baseline_common v1: 59 / 59 一致 ✓   v2: 67 / 67 一致 ✓
```

**F2 — 有條件，這一點很重要**：
- **程式碼層面 ✅**：`build_features` 與 `clean_and_scale` 對 git HEAD 逐位元 `0.000e+00`；
  v1 的 `_filter_universe`、`lag_names`、flat 維度（300）皆維持不變。
- **資料層面 ⚠️**：`baseline_cache/`（v1 的 300 維矩陣）是在
  **外資持股 9xxx 缺口修復之前**建的。
  → v1 的既有 Ridge/GBDT/GRU 數字是**舊資料基礎**下的結果。
  這不影響 v2 的 F5，但若之後把 v1 當對照組並列，**必須註明兩者資料基礎不同**，
  或重建 v1 快取。建議寫進協定的引用紀律。

**F3 — 部分**：`docs/feature-protocol-v2.md` 開頭已標「規格已凍結」；
`feature_spec.py` 有完整的設計說明但**無顯式的版本/凍結標記**。建議補一行。

---

## G. 進入訓練/實驗層前的交接確認

| # | 項目 | 結果 |
|---|---|---|
| G1 | 訓練腳本已改用 `splitters.py`，而非有洩漏的 `walk_forward.py` | **❌ 未完成** |
| G2 | 死旗標最終取捨已依全量定案並反映到 `INPUT_DIM` | **❌ 待 F5** |
| G3 | 集團表縮減版本已定案並記錄理由 | **✅** |
| G4 | R0–R5 六組對照已執行，只有正貢獻才帶進重訓 | **❌ 待 F5** |

**G1 — 已補做（第二輪）**

第一輪發現：全專案 grep **沒有任何檔案 import `splitters`**。
purged CV 寫好、驗證過、39 fold 全通過，但沒有任何實驗在用它。

已修：新增 `splitters.train_val_split_dates()` 便利函式（換一行即可套用），
並接進 **`kg_ablation.py`**（預設 `purge=True`）與
**`v6_colab_training.py` 的兩個切分點**（Cell 4 與 Cell 4b）。

實測代價很小：

| 用途 | horizon | 砍掉的訓練日 | 佔 train |
|---|---|---|---|
| 短線（5d/10d 標籤） | 10 | 30 天（4,681 → 4,651）| 0.6% |
| 多 horizon（5/20/60）| 60 | 80 天（4,681 → 4,601）| **1.7%** |

1.7% 的樣本換掉每個 fold 邊界的 label 洩漏，是很划算的交易。

切分設定（`purge` / `purge_horizon` / `embargo_days` / 訓練日數）已寫進
`kg_ablation` 的結果 JSON——**沒記錄的話，半年後看到那份 JSON 會不知道
它能不能跟別的數字並列**。

`phase3_a~d` 刻意**未改**：A 已跑完，改了會讓 B/C/D 與 A 不可比。
若之後恢復 Phase 3，需整組重跑並統一切分。

---

## 結論：特徵工程層還剩什麼

### ✅ 第二輪已補做完成
1. **A2** 不變性測試 `--protocol v2`：70 欄 `0.000e+00`
2. **E1** `AVAILABILITY_TABLE` 補 RS_* 的跨 Group 相依說明
3. **F3** `feature_spec.py` 補版本/凍結標記（含「哪兩項仍允許改」）
4. **G1** `splitters` 接進 `kg_ablation.py` 與 `v6_colab_training.py`
5. **（額外）** 修掉 ETF 混入與 import 連鎖兩個 bug

**程式層面已無未完成項目。**

### 仍待 F5 全量結果才能定案（**這兩項屬特徵規格本身，不是實驗**）
6. **C3 / G2**：死旗標取捨 → `INPUT_DIM` 是 **67 還是 64**
7. **G4 / R4**：中性化取捨 → `NEUTRALIZE` 是 `none` / `industry` / `industry_mktcap`

### 誠實的判斷

**程式已備妥，但特徵規格尚未凍結**——`INPUT_DIM` 與 `NEUTRALIZE` 都還是開放的。
若現在開新階段並宣告「資料與特徵沒問題」，等於把兩個未決的規格選擇帶進去。

**建議順序**：跑 F5 全量 + R0–R5 → 兩項規格定案 →
**那時特徵工程才真正凍結**，再開新階段。

F5 是無人值守的 CPU 工作，可以在做前端重構時並行跑：

```powershell
$env:MM_PROTOCOL="v2"; python V6/experimental/baseline_common.py --build
```

⚠️ 引用紀律：`baseline_cache/`（v1）是在**外資持股 9xxx 修復前**建的。
v1 的既有 Ridge/GBDT/GRU 數字是舊資料基礎下的結果，
與 v2 並列時必須註明，或重建 v1 快取。
