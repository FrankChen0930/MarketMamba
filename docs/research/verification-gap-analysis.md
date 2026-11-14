# Verification Gap Analysis — 已知但尚未自動化的失敗模式

> 日期：2026-08-11 ｜ 掃描對象：`V6/`、`app/backend/` 全樹
> 性質：**盤點與最小 ratchet 提案。本輪未修改任何 production code。**
> 判準：只列「已經真實發生過」或「機制已被證實存在」的失敗模式。純推測不列。

---

## 0. 一句話結論

> **MarketMamba 的驗收紀律寫得比多數專案都好，但沒有任何一條會自己失敗。**

repo 內與驗證有關的資產全部是「人要記得去跑」的形式：

| 資產 | 形式 | 會回非零 exit code 擋下改動嗎 |
|---|---|---|
| `scripts/test_feature_invariance.py`（17 KB） | 一次性腳本 | ❌ |
| `scripts/test_availability_flags.py`（14 KB） | 一次性腳本 | ❌ |
| `scripts/verify_quality_checklist.py`（12 KB） | 一次性腳本 | ❌ |
| `scripts/data_quality_check.py` / `verify_fixes.py` | 一次性腳本 | ❌ |
| `data/hygiene.py: check_data_health`（18 源各設容許值） | **每日執行** | ❌ 設計上 non-fatal |
| `run_v62_daily.check_freshness()` | 每日執行 | ⚠️ 只發 Telegram |
| CLAUDE.md「與 git HEAD 逐位元相同」 | **文字紀律** | ❌ |
| `quant_rigor_checklist.md` | **要人把整份丟給 Claude 才會跑** | ❌ |

- 無 `.github/`（**零 CI**）
- 無 pytest / conftest（`scripts/test_*.py` 是可執行腳本，不是測試套件）

---

## 1. Gap 1 — Parquet index 序列化（★ 最高優先，已肇事一次）

### 1.1 完整掃描結果

掃描 `V6/**/*.py` 與 `app/backend/**/*.py` 的全部 `to_parquet(` 呼叫。

**原始命中：86 處。逐一人工核對後分類如下**（**核對本身很重要——原始 regex 有 4 個偽陽性**）：

| 分類 | 數量 | 說明 |
|---|---|---|
| ✅ 已帶 `index=False`（同一行） | 56 | 安全 |
| ✅ `index=False` 在**下一行**（多行呼叫） | 1 | `backfill_daytrade_direct.py:96` — **regex 偽陽性** |
| ✅ 呼叫 `_append_to_parquet()`（受守門保護） | 2 | `fetcher.py:3918`（prices）、`3938`（institutional） |
| ➖ 是 `def _append_to_parquet(` 定義行 | 1 | `fetcher.py:4202` |
| ➖ 是註解文字（`fix_prices_index_column.py:22,28`） | 2 | 說明用 |
| 🔴 **真實缺少 `index=False`** | **22** | 見下表 |

> ⚠️ **這段核對本身就是 CLAUDE.md「檢查腳本本身也要驗」那條紀律的實例。**
> 若直接把 regex 命中數當成缺陷數，會多報 4 個。未來的 ratchet 腳本必須附偽陽性測試樣本。

### 1.2 22 個真實缺口（依風險排序）

風險 = **①缺 `index=False`** × **②寫入前索引可能不連續** × **③該檔是否被讀回後重寫**。
**三者同時成立才會複製 2026-08-10 的致命形態（同名欄重複 → `ArrowInvalid`）。**

#### Tier A — 索引必然不連續，且在**每日/每次同步的活躍路徑**上

| 檔案:行 | 寫什麼 | 為什麼索引必然不連續（已逐行確認） |
|---|---|---|
| `fetcher.py:90` | `ticker_universe.parquet` | `_fetch_universe_from_finmind()` 第 100 行 `df = df[df["stock_id"].str.match(r"^\d{4}$")].copy()` — **filter + `.copy()` 不重設索引** |
| `fetcher.py:4151` | `macro_raw.parquet` | `df_macro.sort_values("Date", inplace=True)` — **排序後索引亂序** |
| `fetcher.py:4347` | 期貨法人快取 | `df = df[df["futures_id"].str.contains("TX", ...)].copy()` — filter |
| `fetcher.py:4385` | 選擇權法人快取 | 同型 |
| `fetcher.py:4423` | 總報酬指數快取 | 需逐行確認（同一區塊模式） |
| `fetcher.py:4464` | 股利快取 | 同上 |
| `fetcher.py:4503` | 外資持股快取 | 同上 |
| `fetcher.py:2269 / 2296 / 2315` | `run_full_data_sync` 的 prices / institutional / margin 快取 | `pd.concat([...], ignore_index=True)` 之後仍可能經過 filter |
| `fetcher.py:2171` | FinMind 價格年度合併快取 | concat 後 |
| `fetcher.py:4170 / 4186` | revenue / financials 快取 | 同步路徑 |

#### Tier B — 活躍但單次寫入（產生 stray 欄，暫不致命）

| 檔案:行 | 寫什麼 |
|---|---|
| `scripts/build_feature_matrix.py:160` | **特徵矩陣** |
| `notebooks/v6_colab_training.py:329` | **Colab 端特徵矩陣**（訓練時活躍） |
| `knowledge/graph_builder.py:198` | TPEX 供應鏈快取 — 但 `pd.DataFrame(pairs, columns=...)` 是全新 RangeIndex → **實際觸發機率低** |
| `evaluation/walk_forward.py:390` | WF 結果 — 手動執行 |

#### Tier C — 一次性/歷史腳本（不在每日路徑）

`scripts/backfill_institutional.py:99`、`scripts/fetch_v61_data.py:121/140/182`、`scripts/refetch_shareholding.py:129`

### 1.3 這件事的真正機制（比「還有 18 個炸彈」更精確）

2026-08-10 的致命錯誤是 **`Multiple matches for FieldRef.Name(__index_level_0__)`——同名欄出現兩次**。要走到那一步需要兩個動作：

```
動作 1：某次寫入把索引實體化成 __index_level_0__ 欄
        （08-08 的 prices 回補做的）
             ↓
動作 2：檔案被讀回（stray 欄變成一般資料欄），
        另一次寫入又實體化一次索引 → 同名欄 × 2
        （08-10 的每日 append 做的）
```

**因此上面 22 個站點目前多數只會完成「動作 1」——產生一個無害的 stray 欄。**
**但只要有任何一支 backfill / fix / refetch 腳本去讀寫同一個檔，就完成「動作 2」，重演 08-10。**
而 `V6/scripts/` 底下**正好有 20+ 支這種讀寫腳本**。

> **結論：這不是 22 個獨立的炸彈，是一個仍然開著的、會在下一次資料修復時引爆的機制。**
> 而資料修復在這個專案是**常態**（近一個月就做了 prices 回補、margin 券賣券買交換、除權息還原、MOPS 覆寫、欄名撞名修復）。

### 1.4 額外發現：`ticker_universe` 的特殊性

CLAUDE.md「資料管線注意事項」已經記載過這個檔損壞的後果（`prices_raw` 長出數萬支非股票工具、`[Dataset init] 46488 stocks`）。

它現在的寫入路徑是 `df.to_parquet(cache_path)`（第 90 行），而 df 剛經過正則過濾。
**目前的實際影響**：只多一個被忽略的欄（下游只取 `market` / `stock_id`），**不致命**。
**但它是快取，`run_full_data_sync(force_rebuild=True)` 不會重建它**——一旦被污染，污染會長期存在。

---

## 2. Gap 2 — 線上與回測的口徑一致性（已肇事，已修，但未鎖）

**事故**：2026-08-08 發現 `v62_portfolio.step()` 與 `portfolio_lab.replay()` 的並列打破方式不同
（`sort_values` 預設 quicksort 不穩定 vs `rank(method="first")`），582 天年化差 **0.18pp**。
一天有 **183 組完全相等的 float32 分數**。

**已修**：`sort_values(["score","stock_id"], ascending=[False,True], kind="mergesort")`。
**未鎖**：沒有任何自動檢查會在它再度分歧時失敗。

**為什麼特別危險**：0.18pp 看起來像捨入誤差。CLAUDE.md 自己寫過「太容易被當成捨入誤差放過」。
而 08-17 之後前瞻紀錄開始累積，**這類漂移會被誤讀成模型表現變化**，且在 n<60 時年化本身就有 ±39pp 誤差，**分不出來**。

---

## 3. Gap 3 — 「算不出來」的預設值方向（已肇事，已修，但未鎖）

**事故**：2026-08-09，`ann_stderr_pp` 在 n=1 時是 `None`，下游寫 `se = m.ann_stderr_pp ?? 0` 與
`"*" if se and abs(diff) < se` → **null 變成「零誤差」，19 個 arm 全部通過門檻被標成「明確差於回測 45~78pp」**，效果與設計意圖完全相反。

**已修**：終端印 `±?` 並一律標 `*`；前端 `solid` 要求 `!= null`。
**未鎖**：`?? 0` / `or 0` / `if x` 套在誤差欄位上，是一個**可被 grep 的模式**，但沒有人在 grep。

---

## 4. Gap 4 — 寫死的比較基準（已肇事 6 次，未鎖）

2026-08-09 一天內在**六個地方**踩到：前端寫死「主線是 38.0%」、router 測試寫死 `0.380`、
`score_window` 寫死 `對照 +0.1145`、前端文案寫死「從 +38.0% 掉到 −19.9%」（**兩個數字都過時，且 −19.9% 連正負號都錯**，實際是 37.3% → 25.0%）、router fallback 的 `backtest_ann: 0.380`、`--list` footer 引用過期 docs。

**已修**：改成從 manifest / API 取值。
**未鎖**：重跑一次就會再過時，而且**看起來還是很像正確的對照**。

---

## 5. Gap 5 — 資料寫入的「縮小」語意（已肇事，只修了一個呼叫點）

**事故**：2026-08-08 發現 `prices_raw` 缺 **194,909 列 = 15.15%**、859 支受影響（489 支在可交易宇宙內）、缺漏散布在 **98% 的交易日**。根因是 `_append_to_parquet` 的「整天替換」語意。

**已修**：`_append_to_parquet` 改成 merge 語意 + `allow_shrink` 明確授權。
**覆蓋範圍**：`_append_to_parquet` **只有 2 個呼叫端**（prices、institutional）。
**其餘 12+ 個資料源走各自的 `out.to_parquet(path, index=False)`**——它們沒有 merge 語意保護，但也沒有「整天替換」的邏輯（多數是整檔重寫）。

> ⚠️ **這一項需要逐源確認才能下結論，本次未做完。** 誠實標記為 **NEEDS EVIDENCE**，
> 不因為「另一個地方修好了」就假設整類問題消失。

---

## 6. Gap 6 — 非交易日寫入 gate（部分解決，未全面）

obsidian `02 問題追蹤/已知問題清單` R-7 記載：06-07（週日）、06-19（端午）被 API 異常寫入整日假資料。
`is_trading_day()` 閘門已存在於 `fetcher.py:3844` 的每日更新路徑（2026-08-08 實測五個日期全對，含端午節），
**但寫入端未全面補上**。

**未鎖**：沒有檢查會發現「某個 parquet 出現了非交易日的列」。

---

## 7. 最小 ratchet 計畫

**設計原則**（借自 FinceptTerminal 的 `arch-ratchet.yml`，見 `fincept-terminal-analysis.md` §1.4）：

> **對存量債務用棘輪，不用硬門檻。硬門檻會在第一天讓每個改動失敗，然後被刪掉。
> 「人們會保留的棘輪」勝過「人們會刪掉的門檻」。**

### 7.1 四條 ratchet（只能降，不能升）

| ID | 量什麼 | 對應 Gap | baseline | 備註 |
|---|---|---|---|---|
| **R1** | `to_parquet(` 呼叫點中真實缺少 `index=False` 的數量 | Gap 1 | **22**（本文件已量出，含偽陽性排除規則） | 必須處理多行呼叫與 `_append_to_parquet` 兩種偽陽性 |
| **R2** | `?? 0` / `or 0` / `if x` 套在誤差/不確定性欄位上的數量 | Gap 3 | **應為 0，直接鎖** | 欄位名 pattern：`*stderr*` / `*_se` / `*uncertainty*` / `*_err*` |
| **R3** | 非測試檔中的硬編比較基準字面值 | Gap 4 | 先量後鎖 | pattern：`0\.38\b` / `38\.0%` / `0\.1145` / `0\.373` / `37\.3%` |
| **R4** | 直接寫 `Data/processed_v6/*.parquet` 而未經守門函式的呼叫點數 | Gap 5 | 先量後鎖 | 遷移期用來量進度 |

### 7.2 兩條 golden value（精確比對）

| ID | 比對 | 對應 Gap |
|---|---|---|
| **G1** | `portfolio_lab.replay()` 對已知窗 → 年化 **37.28%** / 換手 **82.7%** | Gap 2 |
| **G2** | `v62_portfolio.step()` replay vs G1 → **0.000pp** | Gap 2 |

> ⚠️ **G1/G2 的基準值必須在建立時實際跑一次取得，不可從文件抄。**
> CLAUDE.md 裡同時出現過 38.0% / 38.02% / 38.15% / 37.28% / 37.3%，
> 分別對應不同的資料版本與換手定義——**抄錯一個就把 ratchet 變成假的安全感**。

### 7.3 三條伴隨紀律（人的部分）

1. **baseline 只能往下調。** 任何上調必須是獨立 commit + 寫明為什麼那個數字該變
2. **golden 值變動 = 正確性修正的一部分**，必須與該次修正同 commit，說明差多少、為什麼
3. **每支 ratchet 腳本必須附偽陽性樣本測試**——本文件 §1.1 就抓到 4 個偽陽性；
   CLAUDE.md 也記載過「第一版檢查腳本 regex 沒匹配大寫 `_BACKUP`，變成拿檔案跟自己比」

### 7.4 明確不做

- ❌ 覆蓋率門檻
- ❌ 對 `fetcher.py` 的網路抓取做 mock 測試（維護成本 > 價值）
- ❌ CI（先讓本機 `check.sh` 證明有人會跑）
- ❌ 前端測試
- ❌ 特徵矩陣**全量**黃金檔（27 分鐘重建，跑不動就沒人跑）→ 改成固定 20 天 × 50 檔小樣本雜湊

### 7.5 建議順序

```
Step 0  量測（不改任何東西）——跑一次 R1~R4 的掃描，記錄真實 baseline
Step 1  R1 + R2（R2 應為 0，可直接鎖）+ check.sh
Step 2  R3（先量後鎖）
Step 3  G1 / G2（需實跑一次取得基準）
──── 以上全部零 production 改動，可在 08-17 起跑前完成 ────
Step 4  R4 + 逐呼叫點修 index=False（Tier A 優先：ticker_universe → macro_raw → 期貨/選擇權）
```

**每修一個 `to_parquet` 呼叫點的驗收**（沿用 08-08 回補用過的方法，非新流程）：

```
舊檔鍵集合 ⊆ 新檔鍵集合        （不可縮小）
既有列 max|Δ| = 0.000e+00      （值不可變）
schema 型別逐欄相同             （守門要在寫入之前，不是之後）
```

---

## 8. 未解決的問題（交給下一輪決定，不在本輪自行處理）

依 CLAUDE.md 互動規則，以下記錄但不動手：

| Problem | Evidence | Impact | Affected paths | Recommended minimal fix | Priority |
|---|---|---|---|---|---|
| 22 個 `to_parquet` 缺 `index=False`，其中 Tier A 索引必然不連續 | 本文件 §1.2，逐行確認 | 下一次資料修復時可能重演 08-10 全系統停擺 | `fetcher.py` ×13、`build_feature_matrix.py`、`v6_colab_training.py` 等 | 逐點加 `index=False`，Tier A 優先 | **P0** |
| `step()` / `replay()` 一致性無自動檢查 | §2 | 前瞻紀錄被口徑漂移污染且不可事後區分 | `v62_portfolio.py` / `portfolio_lab.py` | G1/G2 golden | **P0** |
| 誤差欄位的 null 處理無檢查 | §3 | 呈現與設計意圖相反 | `v62_performance.py`、前端 | R2 ratchet | P1 |
| 非交易日寫入 gate 未全面 | §6、obsidian R-7 | 假資料進入面板 | `fetcher.py` 各寫入端 | 先加一條「面板不得含非交易日」的檢查 | P1 |
| 除 `_append_to_parquet` 外各源的縮小語意未稽核 | §5 | 未知 | 12+ 個資料源 | **先做證據蒐集，不要先改** | NEEDS EVIDENCE |
