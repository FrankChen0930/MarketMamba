# 不需要 Colab 的待辦（本機工程／資料／前端／分析／小模型）

> 掃描全專案（CLAUDE.md、docs/、obsidian_note/、本次 Cowork 討論）整理，2026-07-11。
> 涵蓋資料管線修復、前端頁面、訊號/回測工程、文件與分析、以及規模遠小於主模型、本機可訓練的小模型。標註「⚠️ 待重新盤點」的項目來源文件較舊，可能已被後續工作部分解決，動工前建議先確認現況。

---

## A. 資料管線 / 基礎設施

| # | 項目 | 狀態 | 優先級 | 來源 |
|---|---|---|---|---|
| 1 | macro_raw 停在 2026-04-24，需把 macro 加進每日更新，讓保守模式閘門（TWII vs MA60）啟用 | 未做 | 中 | CLAUDE.md 下一步 |
| 2 | prices_raw 每日 ~800 筆重複寫入，來源待查（存量已清） | 待查 | 低 | CLAUDE.md 下一步 |
| 3 | 兩個 `institutional_raw_backup_*.parquet` 備份檔清理 | 待做（scanner 穩定跑幾天後即可） | 低 | CLAUDE.md 下一步 |
| 4 | 本機 git 善後（`trainer.py` CRLF 衝突清理） | 待做 | 低 | CLAUDE.md 下一步 |
| 5 | SYS-12：WSL2 60 分鐘超時無輸出，根本原因仍待觀察（暖機+重試已部分緩解） | 部分緩解，持續觀察 | 中 | 已知問題清單.md |
| 6 | Render 免費方案是否夠用／是否換 Railway | 待決策 | 低 | 發想空間.md 雜想 |
| 7 | 知識圖譜相關性邊每月/季重建（目前用訓練時固定的 edge_index） | 未做 | 低 | 已知問題清單.md 優化方向 + 發想空間.md |
| 7a–7c | ~~Close=0 損壞列 / 還原-未還原價接縫 / 超限報酬複驗~~ | 落地方式已定案 | — | **改列進 `資料基礎升級計畫_baseline_common扶正.md` 階段二（2a/2b/2c）**，不在此重複追蹤，避免兩邊各自維護同一件事 |

---

## B. 訊號 / 回測工程

| # | 項目 | 狀態 | 優先級 | 來源 |
|---|---|---|---|---|
| 8 | 條件貢獻分析累積 20+ 天後回頭校準四條件權重（30/25/25/20）與 70/90 門檻；同時評估「掃描池擴到 Top200」要不要做 | 資料累積中 | 中 | CLAUDE.md 下一步 |
| 9 | 雙模型篩選條件（SQ 門檻/低不確定性/short∩trend 交集/跨日排名穩定） | 未做（原路線圖②） | 中 | CLAUDE.md 下一步 |
| 10 | 真正的 dual 模擬機器人（比照 sim_engine_v3 跑紙上交易含進退場，非目前僅算 IC/Top50 超額） | 未做（原路線圖③） | 中 | CLAUDE.md 下一步，注意勿與現有「雙模型驗證」混淆 |
| 11 | P0 後累積 20+ 天 archive，重跑 Uncertainty 校準分析確認結論維持 | 待資料累積 | 中 | `docs/uncertainty-calibration-2026-06-13.md` |
| 12 | U4 組合風控層：以 Net_Alpha_20d 為期望報酬、Uncertainty 為風險，做 mean-variance 或風險平價的 Top-N 權重，取代目前比例分配的 Suggested_Weight | 未做 | 中 | `docs/architecture-analysis-2026-06-12.md` |
| 13 | 組合層 sector 集中度限制（同產業最多 X 支，避免訊號全集中在半導體） | 未做 | 中 | 發想空間.md 策略面想法，與 12 可合併規劃 |
| 14 | O3 Signal_Quality 顯示邏輯優化（raw 欄位已存在，下游排序已改用 raw，前端顯示是否跟進待定） | 部分完成 | 低 | `docs/architecture-analysis-2026-06-12.md` |
| 15 | D4：刪除/封存重複的 `models/inference.py` 舊推論實作（與 `run_daily_inference.py` 邏輯分歧） | 未做 | 低 | `docs/architecture-analysis-2026-06-12.md` |
| 16 | conformal prediction 校準 Uncertainty（不需重訓，純推論後處理） | 未做，優先度已降低（現有排序力已夠用） | 低 | `docs/uncertainty-calibration-2026-06-13.md` |
| 17 | MC-Dropout 採樣次數能否從 30 次壓縮到 20 次（推論時間優化，本機測試即可） | 未做 | 低 | 發想空間.md 雜想 |

---

## C. Phase 4-B：產業變化 Agent（獨立層，LLM 為主）

| # | 項目 | 狀態 | 優先級 | 來源 |
|---|---|---|---|---|
| 18 | 守備清單（ic.tpex 鏈分組）+ 訊號源（FinMind 月營收/存貨/capex/法人 + MOPS 法說會PDF/重大訊息）串接 | 計畫定稿，尚未動工 | 中（待 Phase 3 完成後接續） | `docs/phase4-industry-chain-fusion-plan-2026-06-27.md` |
| 19 | LLM 做「本季 vs 上季」語氣/關鍵字 diff，輸出每週整條鏈同向變化摘要 | 同上 | 中 | 同上，注意此項有 LLM API 呼叫成本（非 GPU 訓練成本） |
| 20 | 付費報價催化劑資料源（集邦 TrendForce 等）：已記錄、因成本暫緩 | 明確暫緩 | 低 | 同上，待系統獲利後再評估 |

---

## D. 前端 / 產品

| # | 項目 | 狀態 | 優先級 | 來源 |
|---|---|---|---|---|
| 21 | PersonalOS 同步 K 線圖（Vercel 版已驗收，PersonalOS 版待同步） | 待做 | 中 | CLAUDE.md 下一步（含明確步驟：複製元件、npm install klinecharts、重新 build） |
| 22 | 即時看盤嵌入（參考 shioaji-pro-app，PersonalOS 桌面版專屬，view-only 不做下單） | 已分三層規劃，尚未動工 | 中（建議先做第 1 層：K線+即時報價內嵌） | 發想空間.md，範圍已明確（不做交易/下單） |
| 23 | ⚠️ 待重新盤點：個股分析中心（8-tab：總覽/技術/籌碼/基本面/估值/消息事件/同業比較/策略回測） | 大部分模組仍缺，但來源文件為 05-22（近 7 週前），期間已有 K 線圖等前端更新，需先確認現況 | 高（若確認仍缺，是最大結構缺口） | `docs/gap-analysis.md` |
| 24 | ⚠️ 待重新盤點：籌碼面模組（融資融券、當沖比、券資比、主力分點、期權情緒） | 同上，需先確認現況 | 高 | `docs/gap-analysis.md` |
| 25 | ⚠️ 待重新盤點：基本面模組（月營收、EPS、ROE、估值排序、財報摘要） | 同上 | 中 | `docs/gap-analysis.md` |
| 26 | ⚠️ 待重新盤點：消息/事件中心（法說會、除權息、處置股、重大訊息、產業消息） | 同上，且與 Phase 4-B 的產業變化 Agent 高度重疊，建議合併規劃避免重工 | 中 | `docs/gap-analysis.md` |
| 27 | 消息面個股新聞情緒（等模型主體穩定後再考慮，做個股層級、不做宏觀報告） | 明確延後 | 低 | 發想空間.md 雜想 |

---

## E. 本次 Cowork 討論新增的規劃文件（頁面架構 + 研究計畫）

以下皆為前端/文件/分析工程，**不需要 Colab**（其中方向二含 Ridge/Lasso/GBDT/簡單 LSTM 訓練，但規模遠小於主模型，預期本機可完成）：

| # | 項目 | 詳細規格 |
|---|---|---|
| 28 | 雙模型頁面架構重整（首頁定位敘事、`/conviction`、`/breadth`、`/compare`） | `planing/雙模型架構重整計畫.md` |
| 29 | 方向一：Pipeline 逐階段說明頁 | `planing/研究計畫_方向一_Pipeline說明頁.md` |
| 30 | 方向二：業界標準 Baseline 對照 | `planing/研究計畫_方向二_Baseline對照.md` |
| 31 | 方向三：從 breadth 萃取 conviction | `planing/研究計畫_方向三_Conviction萃取實驗.md` |

---

## F. 已完成、但舊筆記仍列為待辦的項目（僅供核對，不需再排程）

- sim_engine_v3 四層退場邏輯回放驗證 → 已於 2026-06-13 完成（32 天全量回放，`obsidian_note/04 發想/發想空間.md` 的「回測驗證 sim_engine_v3」條目已過時，可勾掉）
- Portfolio 損益曲線假資料 → 已於 PR 3 修復
- QuantAnalysis 假資料問題 → 已於 UX-01 修復

---

> 本檔案為 2026-07-11 全專案掃描整理產出，來源涵蓋 `CLAUDE.md`、`docs/`、`obsidian_note/` 與本次 Cowork 討論。
