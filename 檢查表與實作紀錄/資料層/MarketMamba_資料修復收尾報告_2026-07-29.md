# MarketMamba 資料修復收尾報告

> 期間：2026-07-27 ~ 2026-07-29
> 起點：使用者的目標是「先把資料完全修好，修好之後再談模型」
> 相關文件：`MarketMamba_資料問題處理記錄_2026-07-27.md`（問題逐項）、
> `docs/data-source-implementation-traps.md`（31 項前瞻式檢查清單）

---

## 一、成果總覽

| 指標 | 修復前 | 修復後 |
|---|---|---|
| 健檢警告 | **14 項** | **2 項** |
| `prices_raw` 列數 | 8,762,966 | 8,282,049（剔除興櫃/極低量雜訊） |
| `\|單日報酬\|>40%` | 850 筆（2026 年 93） | **471 筆（2026 年 4）** |
| `\|單日報酬\|>100%` | 184 筆 | **144 筆** |
| 日報酬 std | 0.0817 | **0.0349** |
| p99.9 日報酬 | **+12.52%** | **+10.00%** |
| 最近交易日檔數 | 1,948 | **2,100** |
| `(Date, stock_id)` 重複 | 每日約 1,550 列 | **0** |
| `Close <= 0` | 329 列 | **0** |

`p99.9 = +10.00%` 正好是台股單日漲跌幅上限。**修復前的 +12.52% 在制度上不可能存在**
——那個數字本身就是資料污染的證據。

---

## 二、修了什麼

### 2.1 資料源直連化（擺脫 FinMind）

| 資料源 | 新來源 | 動機 |
|---|---|---|
| `prices` | TWSE `MI_INDEX` + TPEX `afterTrading/otc` | 舊 TPEX 端點忽略 date，把當天資料蓋上歷史日期數月 |
| `institutional` | TWSE `T86` + TPEX `insti/dailyTrade` | 缺 `selectType` 只回水泥類 |
| `margin` | TWSE `MI_MARGN` + TPEX `margin/balance` | 停更 3 個月 |
| `daytrade` | TWSE `TWTB4U` + TPEX | 全歷史 4,263,330 列皆為 0 |
| `per`/`securities`/`market_value` | TWSE 直連 + MOPS 股本 | 完全不在每日更新裡 |
| `foreign_shareholding` | TWSE `MI_QFIIS` + TPEX `insti/qfii` | 停更 84 天 |
| `futures`/`options_inst` | TAIFEX 下載端點 | FinMind 免費層已擋（需付費） |
| `dividend` | MOPS `t187ap45_L`/`_O` | FinMind 逐股查詢會撞爆額度 |
| `holdings` | TDCC 開放資料 | FinMind 免費層已擋 |
| `revenue`/`financials` | FinMind（滾動逐股） | 月/季頻，免費層仍可用 |

### 2.2 抓出的第三方資料錯誤

- **FinMind 對上櫃股把「券賣/券買」標反**（全歷史 2010–2026）。
  `corr(ΔShort_Balance, Sale−Cover)` 在 OTC 為 −0.85~−0.93、符號一致率僅 1%。
  交換 2,638,958 列後，符號一致率 **65.69% → 97.18%**。
- **`Day_Trade_Volume` 自 2014 年起整欄皆為 0**（4,263,330 列）。重建後年度中位數
  0.026(2014) → 0.220(2026) 單調上升，對得上當沖制度放寬的時序。
- **財報三維（`Gross_Margin`/`ROE`/`Book_Value`）自 2005 年起是死常數**。
  根因是 FinMind 在損益表把「淨利歸屬母公司業主」的英文 type 標成
  `EquityAttributableToOwnersOfParent`，且 `df_balance_sheet` 一直是**從未被讀取的參數**。
- **`Whale_Hold_Ratio` 在全部 848,269 列都是 100.0**。舊聚合把「合計」列重複計入。

### 2.3 全歷史官方還原價重建

重抓 22 年未還原原始價（3.84 小時），套用交易所官方因子：

```
adjusted(t) = raw(t) × Π{ adj_factor(e) : e 為除權息日, e > t }
```

因子表 `ex_rights_raw`（**26,385 筆 / 2,143 支**）＝
TWSE `TWT49U` + TPEX `exDailyQ` + **減資恢復買賣表 667 筆**。

- 除權息日報酬 median **−4.01% → +0.48%**；`<−2%` 比例 73.08% → 13.69%
  （一般交易日基準 12.87%）
- 2412 於 2026-07-09 由 −4.30% → **−0.60%**，與官方參考價隱含的 −3.73% 完全吻合
- 殘留的 +0.48% 經三條證據確認是**真實填息**而非公式錯誤

### 2.4 結構性修復（防止問題重演）

- **`hygiene.py`**（新）：宇宙過濾 + 每日健檢（停更、股票池變動、損壞列、
  法人覆蓋率、交易日曆缺口）。全部 non-fatal
- **`run_daily_update` 從 5 個源擴到 13 個**——「有 fetcher 但沒接進每日流程」
  是本次三個停更案例的共同根因
- **衛生處理前移到 `build_features` 之前**（原本在 `clean_and_scale` 之後，
  時序特徵與當日 z-score 早已被污染）

---

## 三、補不齊的部分（重要）

### 3.1 起始日缺口（**永久性，不會因為等待而改善**）

| 資料源 | 起始 | 訓練期缺頭 | 影響維度 |
|---|---|---|---|
| `daytrade` | 2014-01 | 9.0 年 | `Day_Trade_Volume`（1 維） |
| `holdings` | 2018-01 | 13.0 年 | `Holdings_Large_Pct`/`Change`（2 維） |
| `foreign_shareholding` | 2018-01 | 13.0 年 | `Foreign_Holding_Pct`（1 維） |
| `futures`/`options_inst` | 2018-06 | 13.4 年 | Group D（2 維） |
| `balance_sheet` | 2011-12 | 7.0 年 | `Book_Value`/`ROE`（2 維） |
| `fear_greed` | 2011-01 | 6.0 年 | Group D（1 維） |

**這些不是「還沒補」，是資料本身不存在**——交易所/集保並未公布那麼早的歷史。
`daytrade` 更是根本性的：**當沖制度 2014 年才開放**，2014 年以前沒有當沖這件事。

⚠️ **這個狀態一直都在，不是本次造成的**。V6.1 就是在這個資料基礎上訓練出來的。

### 3.2 近期缺口（可隨時間自然癒合）

| 資料源 | 現況 | 說明 |
|---|---|---|
| `holdings` | 2026-05-08 ~ 07-24 有 11 週的洞 | TDCC 開放資料只給最新一週，查詢頁需逐股（22,000 次請求）。**洞永久留著**，但之後逐週累積 |
| `financials` | 現役 1,942 支中 1,925 支停在 2025-12 | FinMind 額度 600 次/日 + IP 曾被封；滾動補齊約需 3 週 |
| `revenue` | 現役 1,924 支中 1,373 支停在 2026-04 | 同上 |
| `fear_greed` / `business_indicator` | 停更 96 / 178 天 | Group D，**V6.1 下恆為 0**，刻意暫緩 |

---

## 四、Group D 的重大發現

實測 `clean_and_scale` 後，**Group D 全部 12 維的 std 皆為 0.000000**：

```
TWII_Return  SPX_Return  VIX  TNX  Gold_Return  Oil_Return  USD_TWD
Futures_OI_Foreign  Options_PC_Ratio  Fear_Greed  Business_Signal  FED_Rate
```

原始值有訊息（`Fear_Greed` 5~78 共 67 個相異值），是被 `macro_norm="cross"` 消滅的
——macro 對同一天所有股票同值，橫斷面 z-score 就是 std=0。

**推論：V6.1 是在整組 Group D 都死掉的狀態下達到 IC ~0.08 的。**
目前沒有任何證據顯示這 12 維有用。故已暫停追 `fear_greed`/`business_indicator` 來源，
等 V6.2 改 `macro_norm="ts"` 後先做消融實驗再決定。

---

## 五、方法論教訓（完整清單見雷區文件）

本次最反覆出現的三個模式：

1. **「有程式在抓」≠「會更新」**。三個停更案例的共同形狀是
   *程式碼看起來都有在抓，但沒有任何一條路徑會在日常執行時真的寫入*，而且都不報錯。
   判準應該是「**哪一行程式碼會在今天執行、並且真的寫入**？」

2. **聚合數字會蓋掉問題**。除權息公式驗證聚合 90.28% 看似及格，
   按類別拆解才發現「權」只有 45.95%；回補「最大日期 2026-03-31」正確，
   但那只代表 16 支股票。**收尾一定要印分布，不能只印總數與最大值。**

3. **拒絕服務不可以偽裝成「沒有資料」**。FinMind 的 402/403 落進 `logger.debug` 回 `None`，
   與「這支沒新資料」無法區分，導致回補空轉六輪而日誌一切正常。

**我自己在過程中被量測推翻了三次**（PER 偏差的解釋、乘數單調性方向、
「補最舊 N 筆」的假設），每次都是先有一個說得通的故事、再被數據否定。

---

## 六、Commit 清單

| Commit | 內容 |
|---|---|
| `0cb262f` | 資料源直連化 + 全歷史官方還原價重建 |
| `79a0715` | `prices_raw` 切換為官方還原價 + 月/季源接進每日更新 |
| `50f5a52` | 極端報酬歸因、健檢日曆缺口、`stock_info` 快照去重、FinMind 額度處理 |
| `21fcce1` | 期貨/選擇權三大法人改 TAIFEX 直連 |
| `ae2f793` | 股利分派改 MOPS 直連 |
| `cbdd8b6` | 集保股權分散改 TDCC 直連 + 修 `Whale_Hold_Ratio` 死常數 |
