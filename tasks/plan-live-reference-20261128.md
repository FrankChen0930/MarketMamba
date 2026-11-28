# 每日參考模型、既有網頁與模擬投組盤點／實作規劃

> 給協調者與 Human；2026-11-28；builder 離線盤點。狀態：**規劃候選／UNMERGED／NOT_DEPLOYED／NOT_ACCEPTED**。本文件不是啟用排程、部署或前瞻准入指令。完成不等於 Human acceptance。

目標是先讓第三步 B（LightGBM，190 欄）每天產生可追溯的參考排序，在現有 React／FastAPI 網頁顯示並維護模擬投組；後續再加入其他已訓練模型的獨立排序與整合排名。**目前沒有任何一個模型可僅憑本次盤點宣告完整每日鏈 READY**：現存權重與歷史評估不等於今天的特徵、來源時間、可交易性及服務接線已驗收。

Human 本次已決定：參考模型採第三步 B；使用現有網頁；系統僅輔助選股及模擬交易，不下單；目標包含多個已訓練模型的每日獨立排序及整合排名；任何參考模型觀察均不得用於修改前瞻家族、提早開盲或中止。下文將已決定事項與待選的工程細節分開；前瞻規則由協調者另立決定檔，本文件不冒充該凍結紀錄。

## 盤點邊界、身分與證據

- Source：本指派 worktree，detached HEAD `61f7505adec073e7c76f5cdcc95f928639040743`；起始 `git status --short --branch` 只有 `## HEAD (no branch)`，無既有 dirty／staged／untracked。沿用既有 `tasks/` 承載指定規劃，不新增實作、目錄、worktree 或報告樹。
- 已讀 [Current State](../knowledge/00_Project_Map/Current_State.md)、[Authority Map](../knowledge/00_Project_Map/Authority_Map.md)、[CLAUDE](../CLAUDE.md)、相關 machine contracts、原始碼、報告及 Data／Features／Labels／Training／Research domain。hierarchy 仍以 machine contracts／manifest／hash／tests 優先。
- Runtime：WSL Linux；`python3` 為 `/usr/bin/python3`，3.12.3；`python` 不在 PATH；Node v24.19.0。僅讀取 metadata／陣列日期索引時使用既有 `/home/frank/.venvs/marketmamba-gpu/bin/python` 的 NumPy 2.5.4，設 `CUDA_VISIBLE_DEVICES=''`，未載模型或啟動 GPU。
- `MARKETMAMBA_DATA_ROOT`／`VIRTUAL_ENV`／`CONDA_PREFIX` 未設定。本 worktree `Data/` 是普通目錄，realpath 就是本 worktree 的 `Data`；主 checkout `/home/frank/projects/MarketMamba/Data` 是另一個目錄，不推定兩者等價，也未讀正式 Data 的逐列內容。
- 唯讀抽查主 checkout 已有 `.artifacts/step3`、`step4`、`step5`、`p1`、`p4`、`p4b`、`p4d` 的檔名／manifest，以及 prepared 日期與 sample 索引；沒有讀取封盲區、訓練或重算 IC。檔案數量是 inventory，不是本輪逐檔 hash／load→predict 驗收。
- PUBLIC repo 是本次 Human 明示的限制；只離線核對 origin 指向 `FrankChen0930/MarketMamba`，未向 GitHub 查 visibility 或服務狀態。所有部署／scheduler 現況均不能由檔案推定。

## A. 既有網頁與資料契約

### A1. 架構、頁面與路由

[App.jsx](../app/frontend/src/App.jsx) 使用 React 19、React Router 7、Vite 8，圖表使用 Recharts／KLineCharts，Axios 包裝在 [client.js](../app/frontend/src/api/client.js)。[main.py](../app/backend/main.py) 是 FastAPI／Uvicorn，所有業務 router 掛在 `/api`。前端開發預設 `/api` 經 [vite.config.js](../app/frontend/vite.config.js) proxy 到 `localhost:8000`；部署可用 `VITE_API_URL`，這是公開於前端 bundle 的設定，不能放秘密。

| 頁面路徑 | 現有用途與限制 |
|---|---|
| `/` | Home 總覽，AppLayout 共用導覽 |
| `/conviction`、`/conviction/predictions`、`/conviction/backtest`、`/conviction/versions`、`/conviction/portfolio` | 高信念模型總覽、預測、回測、版本、既有持倉；最後一頁讀既有 portfolio，不能當作本案 paper 帳本 |
| `/breadth`、`/breadth/predictions`、`/breadth/portfolio`、`/breadth/versions` | V6.2 廣度模型及 arm 選擇；可借用列表／切換互動，但 V7 不沿用舊模型名稱 |
| `/legacy`、`/legacy/signals`、`/legacy/sim`、`/legacy/dual` | V6.1／掃描、歷史模擬、short/trend 雙模型對照 |
| `/compare` | CompareBoard 有路由，但檔案仍以尚未串接的比較看板定位，不能當作通用多模型 registry 已完成 |
| `/quant`、`/market` | 量化市場／形態、行情與 K 線 |
| `/research`、`/research/pipeline` | 研究作品集及 pipeline；不是交易或 production authority |
| `/v7/status` | V7 data-health 摘要 |
| `/v7/portfolio-history` | 僅 `VITE_V7_PORTFOLIO_HISTORY=1` 時掛載；後端亦需相應 flag；現有 history 契約需保持獨立 |

舊路徑 redirect：`/dashboard→/breadth/predictions`、`/scanner→/legacy/signals`、`/dual→/legacy/dual`、`/sim→/legacy/sim`、`/portfolio→/conviction/portfolio`、`/model→/breadth/versions`、`/pipeline→/research/pipeline`；另有 `conviction/signals`、`breadth/backtest`、`breadth/compare` 相容轉址。這些是 code presence，不證明部署版本相同。

### A2. 後端讀取來源與欄位

以下 URL 多由 `GITHUB_RESULTS_URL` 的 `df_kelly.csv` 替換檔名而得；除另註外是 HTTP 讀取，**不是後端直接載入 repo 內 CSV**。細節入口：[signals](../app/backend/routers/signals.py)、[schemas](../app/backend/schemas.py)、[v62](../app/backend/routers/v62.py)、[dual](../app/backend/routers/dual.py)。

| API router／主要 endpoint（均加 `/api`） | 資料契約與讀取方式 |
|---|---|
| `/signals`、`/{date}`、`/scanner`、`/history`、`/portfolio/exit-check`；POST `/run-inference`、`/cache/refresh` | `df_kelly.csv` 必需 `Ticker`、排序優先 `Signal_Quality_Raw` 否則 `Signal_Quality`；映射 `Exp_Alpha_5d/20d/60d`、`Uncertainty`、`vol_ratio`、`Suggested_Weight`、`Confidence`、`Date`。缺部分欄位會預設 0／1，缺日期會用今日，這對新契約不可接受。回傳 date/model_version/total_stocks/signals/freshness_warning；排行含 rank/stock_id/name/sector/alpha/weight。history 另讀 `history_index.json`。抓取失敗退 MOCK_SIGNALS，cache 1h，日期超過 3 個曆日才警告；新參考線須明確 unavailable／stale，不可假裝今日成功。POST 名稱不能當作已執行今日推論的證據。 |
| `/performance`；POST `/cache/refresh` | `training_status.json`（epoch/config/學習曲線/scale gates）及 `ic_analysis.json`（online points、horizon summary）；訓練 val IC 與成熟後的 paper IC 必須分開 |
| `/v62/arms`、`/portfolio?arm=...`、`/performance`；POST `/cache/refresh` | `v62_arms.json` 控制 arms/default/state_file/score_file，失敗 fallback 只剩主線；`v62_state_{arm}.json` 讀 spec、holdings、weights、last_date、last_rebalance、n_rebalances、days_since/to_next、data_complete 等；`df_v62*.csv` 讀 stock_id/score，依檔案列序給 rank，最多顯示 100 筆；state.spec 優先於 manifest 的展示參數；績效讀 `v62_performance.json` |
| `/dual/signals`、`/dual/ic`；POST `/cache/refresh` | `df_short.csv`／`df_trend.csv` 讀 stock_id/Date；short為Score_5d/10d、Unc_5d/10d、SQ_5d，trend為Score_20d/60d、Unc_20d/60d、SQ_20d；`dual_ic_analysis.json` 存 horizon IC；不是第三步 B 的 190 欄推論介面 |
| `/sim/backtest`、`/ic`、`/scanner-backtest`；POST `/cache/refresh` | 優先 raw URL 的 `sim_backtest.json`、`ic_analysis.json`、`scanner_backtest.json`，另有 repo `V6/results` 本機 fallback；屬既有歷史模擬 |
| `/quant`、`/patterns`；POST `/cache/refresh` | raw URL 的 `quant_market.json`、`pattern_signals.json`；市場統計及型態結果 |
| `/reports/latest`；POST `/cache/refresh` | `market_summary.json`：date/summary/model/top10/market_data/available；無需為參考排序啟用任何付費摘要 |
| `/market`、`/ticker`、`/kline/{ticker}` | 讀 TWSE MIS、Yahoo chart、匯率服務，另讀 df_kelly 判斷 run_date；K 線為 date/open/high/low/close/volume。這些 endpoint 會連網，這次沒有啟動／呼叫 |
| `/portfolio` | Supabase `portfolio_positions`，quantity/cost_price/current_price/pnl_pct/market_value 等；無設定回空持倉。沿用 Shioaji 資料標籤，不代表本案下單授權 |
| `/fin-news/analyze`、`/health` | 外部 Tavily／Anthropic 分析能力；與每日排序無必要依賴，本案不呼叫 |
| `/v7/status` | 從 `V7_RESULTS_DIR/health-summary.json` 本機讀，預設 `results/v7`；只投影 schema/state/publish_allowed/data_id/generated_at/counts，缺檔 not_ready、壞 schema 503；這是可參考的 fail-closed／公開 allowlist 模式 |
| `/v7/portfolio-history`（條件掛載） | `V7_PORTFOLIO_HISTORY_ENABLED=1` 才加入，僅loopback client可讀；`V7_PORTFOLIO_HISTORY_PATH`與`V7_PORTFOLIO_HISTORY_SHA256`控制fixture檔案及身分，no-store；不是正式paper history，不覆寫或挪用它保存本案每日投組 |

`/`、`/health`、`/api/health` 只回服務健康，不能證明資料 freshness／模型 identity／paper 帳本正確。現有排序缺 deterministic tie-break 的地方，後續新 endpoint 應明定 score 降冪、stock_id 字串升冪；不能順手改舊 success decision。

### A3. 既有多模型能力、部署檔與啟動

[run_v62_inference.py](../V6/run_v62_inference.py) 有 v2_kg_nomacro、同 checkpoint 的 h10 head、head10d/head20d、v2/v3/old KG、no_gat 等 arm；[run_v62_baselines.py](../V6/run_v62_baselines.py) 加入 ridge／gbdt／gru。對應 CSV 包含 `df_v62.csv`、`df_v62_gbdt.csv`、`df_v62_gru.csv`、`df_v62_ridge.csv`、各 h10/h20／KG 變體；[v62_portfolio.py](../V6/v62_portfolio.py) 產生 arm manifest 與各換股頻率 state。不同頻率的 state、同顆模型的不同 head 均不應重複算成獨立模型票數。dual 是另外的 short/trend 路線。

[render.yaml](../render.yaml) 指定 Render Python web、rootDir `app/backend`、pip requirements、`uvicorn main:app --host 0.0.0.0 --port $PORT`，health `/api/health`，設定公開 GitHub raw df_kelly URL 及允許的前端 origin。[vercel.json](../app/frontend/vercel.json) 位於 **app/frontend**，只有 SPA rewrite 到 index.html；root 沒有 vercel.json。前端 build／publish 專案設定與 Render 真實環境不在此驗證；檔案支持「設計上 Vercel 前端＋Render API」，不支持「目前服務正常」或「每天已推論」。現存 CSV 只能證明過去產物，不能證明今日仍執行。

本機啟動指引（供後續獲准工程使用，**本輪未執行**）：在已備妥相依套件的 Python 環境，工作目錄 `app/backend` 執行 `uvicorn main:app --reload --host 127.0.0.1 --port 8000`；`app/frontend` 執行 `npm ci`（需要套件來源時另核准）後 `npm run dev -- --host 127.0.0.1`。通常是 `localhost:5173`，使用 Vite proxy；既有前端可 `npm run build`。純離線啟動還需阻止自動呼叫外部行情／raw URL，配置本案本機唯讀 adapter，不能只刪 GITHUB_RESULTS_URL 後接受 mock 當真資料。本案不重跑舊 daily 腳本，因完整路徑有 fetch、通知及 push 副作用。

### A4. PUBLIC 發布與 TEJ 邊界

**將每日排名 CSV／JSON push 到此 PUBLIC repo，就會公開給所有可讀 raw／歷史 commit 的人；前端藏按鈕或 CORS 都不能保護它。** 名次、代碼、分數及模擬持倉本身也是對外披露內容。私有 bucket 也不會自動保護一個無驗證、可轉發內容的公開 API。

| 選項 | 取捨與建議 |
|---|---|
| 本機／區網跑既有 UI＋API，後端直接唯讀私有 artifact | **第一階段建議**；沒有 GitHub raw／對外 API 公開，容易保留完整 lineage；需本機開機及網路邊界，遠端便利性較低 |
| 私有 bucket／私有 artifact store＋有驗證的後端，前端只取得授權摘要 | 可延用雲端 UI；必須處理登入、存取控制、cache、secret、備份與成本；bucket token 只在後端，不能放 VITE_* |
| 後端讀同機私有檔案／volume | 簡化讀取；Render 不是 Windows／WSL 本機，需另設安全交付、持久化與更新機制；不能假設本機 Data 自動掛在雲端 |
| 私有 Git repo 儲存產物＋後端驗證讀取 | 仍需後端密鑰與可見範圍控制，更新／歷史管理較繁瑣；不把 source repo 改 private 當作本案既定決定 |
| PUBLIC 僅發布經允許的彙總或延遲排名 | 須 Human 明確同意披露範圍並確認資料授權；即使不含原始欄，也不能推定衍生排名可再散布 |

TEJ：這次沒有合約原文與公開再散布許可證據，**授權 UNKNOWN**。原始逐列營收／價格／特徵及可回推原值的明細不得因本案進公開 repo。衍生排名可能較不易還原，但「衍生」不等於已獲授權；由 Human／資料授權窗口核對自用、第三方雲端處理、對外展示／再散布範圍後決定。這是本地證據缺口記錄，非法律結論，本輪無網路查證。最小公開投影亦不得含私有 path、raw row、完整特徵、token、模型權重或封盲資訊。

## B. 已訓練模型／可重現配方清單

狀態定義：`READY`＝模型完整性、當日特徵、決策時鐘、load→predict、輸出契約皆驗收；`NEEDS_FEATURE_PIPELINE`＝有權重／組合配方但今天的同義輸入鏈未驗收；`NEEDS_RETRAIN`＝缺可用權重或僅有工程／訓練配方；`OBSOLETE`＝只保留舊版對照，不建議納入 V7 首發。表列為主要阻礙，其他限制仍適用；所有研究 IC 都不是 verified trading evidence。

私有 `.artifacts/...` 路徑以下相對 **主 checkout `/home/frank/projects/MarketMamba`**，不是本 worktree。沒有掃其他工作樹、Windows、Drive 或封盲區；因此「未找到」限本次已知位置。權重檔名／數量不等同 SHA 或載入驗證。

### B1. 第三／四／五步（均 CPU LightGBM、h20）

共同 target 是 `tools/step3.h20_labels` 的 `close[t+20]/open[t+1]−1`，再依凍結訓練口徑轉同日目標秩；不是預測原始報酬率的校準模型。最新是 **fold2026 的歷史年度模型**，不是用 2026-11-28 前全部資料新訓練。每折從 2013 起、成熟訓練列且 session `< origin−21`；seed17 抽樣日期實查：fold2015 最晚 2014-12-03、fold2021 最晚 2020-12-02、fold2026 最晚 2025-12-01。這是 sample 日期，不是供應商資料最後更新日；seed29/43 精確上限後續由各 manifest/sample 驗證，不能由 seed17 類推。第四／五步沿同凍結切分／抽樣，未另讀其各 seed 日期。

| 模型 | 位置／保有證據 | 特徵依賴／每日輸入 | 既有平均每日 IC（2015–2020；2021–2026） | 每日狀態 |
|---|---|---|---|---|
| 第三步 A | `.artifacts/step3/bundle-72/fits/fold{year}-A-seed{17,29,43}/model.txt`；A/B正式 fit 共72檔 | 185＝48基礎＋128 lag/rolling＋9 market；完整股鍵、session panel、W60 eligibility | 0.15714661；0.14702797 | NEEDS_FEATURE_PIPELINE |
| **第三步 B（Human 選定參考）** | 同上 `...-B-.../model.txt`；`bundle.json` 與各 job manifest；最新可候選 fold2026 三seed | A＋5 月營收＝190；另需營收事件、available_date、單位／口徑／版本、完整歷史比較值 | **0.16291603；0.15198103** | **NEEDS_FEATURE_PIPELINE**，不必為了每天推論每天重訓；是否另作 as-of refit 是獨立工作 |
| 第四步 A | `.artifacts/step4/fits/fold{year}-A-seed*/model.txt`；正式108 fits涵蓋 A/B1a/B1b | 同第三步 B190，重新擬合的工程對照 | 0.16291603；0.15198103 | NEEDS_FEATURE_PIPELINE；不多投一張重複基線票 |
| 第四步 B1a | 同目錄 `...-B1a-...` | 193＝B190＋peer_return/return_peer_count/market_return；需分類、研究在市集合及當日同業報酬 | 0.16077148；0.15246667 | NEEDS_FEATURE_PIPELINE；分類 known_at UNKNOWN、6161排除等仍在 |
| 第四步 B1b | 同目錄 `...-B1b-...` | 192＝B190＋revenue_spillover/revenue_peer_count；另需同業營收發布時序 | 0.16370460；0.15222099 | NEEDS_FEATURE_PIPELINE |
| 第五步 A | `.artifacts/step5/fits/fold{year}-A-seed*/model.txt`；正式72 fits涵蓋 A/B | 同第三步 B190，仍是本步獨立 refit | 0.16291603；0.15198103 | NEEDS_FEATURE_PIPELINE；只作重複基線識別 |
| 第五步 B | 同目錄 `...-B-...` | 194＝B190＋revenue_age_sessions／revenue_yoy_acceleration／revenue_yoy_signed_streak／revenue_seasonal_mom；依 `step5_features.py` frozen事件狀態 | 0.16368108；0.15239273 | NEEDS_FEATURE_PIPELINE；來源 strict PIT STOP |

數字直接引用 [第三步結果](../reports/step3_results_20261126.md)、[第四步結果](../reports/step4_results_20261127.md)、[第五步結果](../reports/step5_results_20261127.md)，未重新計算。它們都是已曝光歷史，兩時段分開，後期有效 1367／排定1382日；第四／五步新增臂未證實改善。原報告部分「原價未含股利」文字已由 [Labels](../knowledge/01_Domains/Labels.md)／[可交易性報告](../reports/tradability_diagnostic_20261128.md) 更正：來源為還原價，不把舊限制敘述當價格 authority，也不因此推定可執行成交。

檔案盤點：第三步 `.artifacts/step3` 共找到74個 `model.txt`，其中 bundle-72/fits 為72個正式 fit，其餘是量測，不能當新獨立模型；第四步 fits 108；第五步 fits 72、measurement 另1。沒有為挑新「最佳」而載入或評分。第三步 B 候選 latest job 的 manifest（fold2026/B/seed17）記載500,000訓練列、302,228歷史eval列、190欄；其eval推論約1.053秒、fit約43.463秒只是既有測量，不是本輪今日端到端效能承諾。

### B2. P1／P2、P4a/P4b 及 P4d

| 模型／配方 | 位置與 target／訓練截止 | 特徵與每日輸入／GPU | 歷史 IC 引用與每日判定 |
|---|---|---|---|
| P1 LightGBM baseline | `.artifacts/p1/full-scores-ace2eef/` 保存六折2021–2026、17/29/43分數；本目錄未找到權重；`matrix_baselines.py`／`rolling_scores.py` 配方可追 | 48原欄，y5模擬代理；逐折 train split 與成熟／purge，最新fold2026訓練在其eval前；精確日期需輸入manifest再核。CPU；當日48欄與同一 universe／preprocess | [P1報告](../reports/v7_p1_portfolio_evaluation.md)未列可直接引用的單一平均IC，只列組合結果，不能把年化當IC；**NEEDS_RETRAIN**＋每日特徵鏈 |
| P1 C0+ | 同上；`V6/research_lab/c0plus.py`；185欄，y5；無保存原模型的限制亦見WP2報告 | CPU；48 panel＋16 CORE lag/rolling＋market；逐fold as-of訓練截止，三seed平均秩 | [P3報告§排序品質](../reports/v7_p3_portfolio_evaluation.md)記 C0+ 約0.12–0.15（各折範圍，非全期間平均）；**NEEDS_RETRAIN**。不能用第三步 A(h20)冒稱同一P1(y5)模型 |
| P2 baseline | [P2報告](../reports/v7_p2_portfolio_evaluation.md) 明示沿用P1分數，沒有新增預測模型 | y5；只是top50／20-session／buffer等投組規則；CPU，不額外訓練 | 沿P1，沒有獨立P2 IC；**NEEDS_RETRAIN**（底層P1權重缺失），不可在ensemble當獨立模型票 |
| P4a/P4b c0plus | `.artifacts/p4b/eval-full-20261125/c0plus.parquet`，來自同P1固定分數 | y5；185欄；CPU；同P1缺原權重 | 同P1參照；**NEEDS_RETRAIN** |
| P4a/P4b crossstock | `.artifacts/p4/p4a-results-3/run-6fbaf815b0716a086a794dfbc12a21f67688263d116469e272e57783448c3730/Q{1,2}-{year}-crossstock-{17,29}-w60-lr0.0003/` 的complete／BEST checkpoint；P4b對應parquet | 神經雙頭y5/y10（P4b用y5）；48欄rank-gauss、W60完整窗、完整當日截面；每fold外層train，內層126-session validation並purge20，BEST IC5；最新2026 fold，精確fit／val最後日期本輪未抽取。GPU訓練；推論CPU可行性需量測，不能承諾不用GPU即達SLA | [P4b報告](../reports/v7_p4b_results.md)未列單臂平均IC，本次不從逐日檔重算；不套用P3舊run數字。**NEEDS_FEATURE_PIPELINE**＋鎖定runtime/native mixer驗證 |
| P4a/P4b gru | 同run `...-gru-...`／`gru.parquet`，BEST而非任一epoch | y5/y10、同48欄rank-gauss/W60；同outer／inner截止；CPU推論優先，訓練GPU較合適但非每日必需 | P4b單臂平均IC未列；**NEEDS_FEATURE_PIPELINE** |
| P4a/P4b mamba2 | 同run `...-mamba2-...`／`mamba2.parquet`，官方Mamba2 | y5/y10、同48/W60／截止；官方CUDA mixer依賴，CPU回退是否相同語義未驗，實務需相容GPU runtime或經認證CPU路徑 | P4b單臂平均IC未列；**NEEDS_FEATURE_PIPELINE**＋runtime gate；不因已有GPU配方就在本案啟動GPU |
| P4b R_all | `eval-full-20261125/R_all.parquet`；不是額外權重，是三神經臂rank後等權再rank | y5；依賴全部神經臂同日同股共同集合；訓練截止繼承各component；合成CPU | P4b未列單一平均IC；報告約0.79是與C0+排序相關，**不是預測IC**。**NEEDS_FEATURE_PIPELINE** |
| P4b B1 | `B1.parquet`；0.5 C0+ rank＋0.5 Mamba2 rank | y5；繼承C0+與Mamba2截止／輸入；合成CPU | IC未列；已列2021–2023日均淨超額差−0.0079%（不是IC）、未證實；**NEEDS_RETRAIN**（C0+缺權重）＋pipeline |
| P4b B2 | `B2.parquet`；0.5 C0+ rank＋0.5 R_all | y5；繼承所有component | IC未列；日均淨超額差−0.0171%、未證實；**NEEDS_RETRAIN**＋pipeline |
| P4b B3 | `B3.parquet`；0.5 C0+ rank＋0.5 GRU rank | y5；繼承component | IC未列；日均淨超額差−0.0165%、未證實；**NEEDS_RETRAIN**＋pipeline |
| P4d stage1 GRU六容量＋C0+ | `tools/p4d*.py`、[stage1 contract](p4d-stage1-contract-20261125.json)、[工程報告](../reports/v7_p4d_engineering.md)；本機p4d目錄未找到正式權重，只有measure；不可宣稱已完成140-fit | GRU y5/y10、48/W60及明示macro mask；C0+185/y5；2021前anchor選epoch／iteration、outer各fold固定更新；GPU完整訓練未驗；每日仍需features與CPU／GPU推論驗收 | 正式歷史IC **NOT_AVAILABLE**；短CPU synthetic fits不是研究結果；**NEEDS_RETRAIN**。Mamba2／D／N12仍DRAFT，不能列成已訓練權重 |

P4 run實際存在，盤點有552個 `.pt`（含多epoch與checkpoint，不是552個獨立模型）；P4b只准入凍結Q1/Q2的36 fits，不含Q3長窗／Q4調率／Q5 permutation。P4b 的B1/B2/B3沒有任何前瞻候選，不能因想顯示多模型而改此研究判定。[P4b spec](spec-p4b-blend-20261124.md) 是混合公式authority。歷史結果JSON的 `signals.summary.mean≈0.76–0.94` 是模型與C0+的秩相關，特別禁止誤貼成預測IC。

### B3. V6／V6.2 舊模型與其他歷史候選

| 模型 | 位置／截止／特徵／每日輸入 | 歷史 IC 與 GPU／狀態 |
|---|---|---|
| V6.1 v6_best | 程式預期 `V6/models/v6_best.pt`；本worktree及主checkout該目錄只見架構md，未見此權重。`run_daily_inference.py` 用56欄、252窗＋KG，5/20/60d輸出；模型實際訓練截止UNKNOWN，不能用CSV日期代替 | 無本輪可核且同版本的整體IC；有可追訓練程式；GPU訓練／MC推論成本待量測；**OBSOLETE**，缺權重，且protected models不得因本案補寫 |
| dual short／trend | `run_dual_inference.py` 讀 `V6/models/v6_short.pt`／`v6_trend.pt`；本機主checkout該處無權重。59欄、short單尺度／trend多尺度序列及KG，short為h5/h10、trend為h20/h60；實際checkpoint形狀仍需驗 | [舊對照表](../docs/baseline-comparison-table-2026-07-15.md)：short h5 0.0870（同harness），trend h20 0.0961是歷史峰值且非同harness；訓練截止UNKNOWN。GPU依架構／實測；**OBSOLETE** |
| V62 v2_kg_nomacro／其h10 head | `run_v62_inference.ARMS` 預期 `v6_short_GD_no_macro_gatv2.pt`＋knowledge_graph_v2；59欄、Group D輸入歸零、252窗；兩head同checkpoint；本機缺權重，截止UNKNOWN | 程式註解引用對Alpha20的0.1311，不能誤標為h10 IC，非本輪完整報告驗收；GPU依native path；**OBSOLETE** |
| V62 head10d／head20d | `v6_short_H_h10.pt`／`v6_short_H_h20.pt`，同59欄／KG；head20d程式仍用head欄位字串`10d`讀第二欄，實際target為h20；不能由輸出index認定horizon | head20d註解Alpha20 IC0.1388，僅舊版註解參考；截止UNKNOWN、本機缺權重；**OBSOLETE** |
| V62 v2_kg／v3_kg／old_kg／no_gat | `v6_short_KG_{v2_kg,v3_kg,old_kg,no_gat}.pt`；59欄，macro不歸零，各圖或無圖；h5為顯示主head，截止UNKNOWN、缺checkpoint | 本輪未定位到可直接引用的同版分臂IC，記UNKNOWN；GPU／CPU待量測；**OBSOLETE** |
| V62 ridge | `V6/experimental/result/ridge_5d__p30_rebuild20260808.npz`（本機未見）；配方`baseline_ridge_lasso.py`，59當日＋241 lag/rolling＝300欄；需cleaned/raw雙面板與完整rolling暖機 | 舊主切分train≤2023-12-31且purge；h5全市場0.1015／高流動0.0705，h20 0.1081（高流動未列）；CPU；**OBSOLETE**（要復用需NEEDS_RETRAIN） |
| V62 gbdt | `gbdt_5d__p30fix_20260808.txt`（本機未見）；只見另一個`gbdt_5d.txt`，不能當同版；300欄、h5、CPU | 舊300欄報告h5全市場0.1098／高流動0.0802，h20 0.1004／0.0685；這些不能轉貼到p30fix：daily程式明列new_model、原表分數ρ0.9203未重現。train≤2023-12-31為舊協定，p30fix精確截止待其artifact；**OBSOLETE** |
| V62 gru | `gru_5d__p30_s20260713.pt`（本機未見）；59欄W60，h5；`baseline_rnn.py`可重現配方；舊協定train≤2023-12-31且purge | h5全市場0.1113／高流動0.0867，h20 0.1081／0.0881為舊報告各自模型，不代表一顆h5權重能吐h20；CPU推論可優先測，完整訓練視成本；**OBSOLETE** |

上述舊IC均取自[七月對照表](../docs/baseline-comparison-table-2026-07-15.md)及其連結報告，與第三步48/190欄資料、標籤、支援／期間不同，**不可橫向排名挑冠軍**。W24/W30/W31等更早N0/Ridge/LightGBM、E1/E5、P3歷史候選仍由project map導航；不是本次多模型首發集合。P3報告有Mamba2/crossstock相對C0+ IC差約−0.012至−0.026，但不套到P4重新訓練的權重；E1/E5有selection/evaluation重用限制，若Human要恢復顯示須另逐run盤點，不能直接升格。沒有藉此新增訓練或復活V6/V7雙系統維護。

## C. 今天的190欄：已有函式與真正缺口

### C1. 來源到t日列的組裝

1. **既有ingestion只供來源**：[run_v62_daily.py](../V6/run_v62_daily.py) 的 `--fetch-only` 在抓取／每日源完整性檢查後退出，不建矩陣、不載模型、不推論、不發布。不可直接取消fetch-only來接第三步B，因完整路徑會走舊推論、通知及push。沿用既有供應，不改DataFetch。
2. **48欄與現行V6 59欄不等價**：[feature_engineer.py](../V6/marketmamba/data/feature_engineer.py) 可建立歷史／尾端量價籌碼／基本面／macro，但當前config依入口可能是56/59/67欄，且有補零／ffill／scaling。第三步使用 [step3_prepare.py](../tools/step3_prepare.py) 綁定 `/home/frank/projects/p4-bundles/prepared/shared` 原48欄float32與欄序；[T1稽核](../reports/leak_audit_20261128.md)顯示X保留未縮放還原價格。不能直接呼叫 `clean_and_scale` 或裁59取48便宣告相同；prepared原helper／source identity已有漂移，14/20來源整檔及helper hash與舊prepared不同，完整原shared端到端重播未證實。
3. **185欄可重用純函式，但缺today adapter**：[c0plus.build_features](../V6/research_lab/c0plus.py) 精確為48＋16 CORE×(lag1/2/5/10/20＋mean5/20/60)＋3 MARKET×mean/std/pos＝185。使用完整session×stock稠密panel，缺session維持NaN，rolling最低支援為ceil(window×0.5)，含當日；market是同日完整來源截面，不是TopN或通過交易篩選後的子集。今天需既定股鍵／欄序、歷史warmup、session索引、t日前panel與market截面；W60 eligibility及來源內部MA／RS／財務比較的lookback一併保留，不能只讀60列便推定足夠。
4. **5月營收可共用事件計算，不可直接由舊fundamentals替代**：[step3.monthly_events](../tools/step3.py) 要 stock_id/period/available_date/revenue/scope/currency/unit/comparability_id/version/source_row；五欄是monthly_yoy/mom/cumulative_yoy/high12/surprise。用已知逐月值重建，不讀匯出的去年／累計捷徑；2013前排除；單位／口徑不合格為NaN；多版本缺first-announcement evidence或碰撞STOP；`effective_session`嚴格在發布日期之後首session收盤生效。今天要有同義的新事件來源、版本／到達紀錄及所需歷史月序列，**現有DataFetch＋V6功能不證明已有TEJ 190欄日更**。
5. **獨立輸出而非覆寫研究矩陣**：後續在既有私有artifacts容器內設 `live-reference/<contract-id>/<signal-session>/<attempt-id>`（建目錄前再盤點），只載已凍結模型，產生snapshot、190欄row identity及predictions。來源缺失不偷偷ffill成當日，歷史按契約容許NaN與整個每日源失敗需分開；無合格來源、日曆或股池時當日發布blocked。

### C2. 訓練／服務一致性驗收（先驗證、再接今日）

- 以原bundle manifest及三seed模型hash為基準，固定重疊日期／股票集合，至少涵蓋月營收發布日前後、跨年、休市、資料缺口、除權息及最新歷史尾端；新today builder在每個as-of截斷原始輸入，不可讀未來列。
- 比較ordered `(session, stock_id)`、48/185/190欄名稱及dtype、NaN位置、finite值、每欄mismatch數／max_abs_delta／首個差異鍵，以及market截面分母、rolling count、營收effective session與lineage。先要求同環境逐bytes一致；若不能達到，隔離source drift／算法差異並STOP，不在本案自訂epsilon掩蓋。
- 將新列交給相同LightGBM版本／同權重／同threads，驗load→predict及各seed scores／ranks與既有golden rows；只可比已曝光歷史。T2子建構15/15無差異不代表全部190欄PASS；原shared完整重播仍是blocker。
- 加入未來值／missingness擾動不改過去輸出、盤後資料晚到、營收發布同日不提前、版本衝突、缺股／重複鍵、未知universe／停牌、NaN不得補0、模型或matrix mismatch、同分排序與重跑冪等反例。未先建立失敗驗證前不修實作。
- 如果新正確時間／還原價vintage配方無法重現原190欄，應登記**不同的參考模型輸入版本**並由Human決定是否重訓；不得私改第三步B frozen bytes或假稱已解決strict PIT。

### C3. 時鐘與建議推論時間

舊scheduler observation [2026-09-22](../knowledge/runtime-observations/scheduler-20260922T132135+0800.json) 記DataFetch `--fetch-only`、最近一次22:30、V62全流程disabled；距今超過7日，依authority只能標 **STALE_OBSERVATION**。[PIT ledger文件](../docs/operations/pit-ledger-task.md) 有11/25註冊、11/26 DataFetch完成事件及後續snapshot敘述，但不是本輪直接OS觀測，不能宣稱今天scheduler正常。現行Windows／WSL／Render是三個不同runtime。

[v62_daily.bat](../V6/scripts/v62_daily.bat) 記2026-08-05實測19:35融資／當沖未公布、21:11才有，建議21:30以後；T1指出法人／融資融券／期貨及其他來源仍缺實際receipt，不能把「通常盤後」當可知證據。美國SPX/VIX/TNX同曆日t收盤晚於台股t收盤；22:30台北仍未取得美股t日完整收盤，故不適合用同曆日美國欄直接推論。

**參考線建議：t為上一個台股完成session，於下一個台股session開盤前08:00 Asia/Taipei嘗試產生t訊號；08:30前驗收完成，才容許當日t+1開盤的虛擬成交。** 美國夏令／冬令收盤約台北04:00／05:00，08:00留接收緩衝；這是工程候選時窗，非已量測SLA或來源發表保證。跨美國假日／週末／DST按各市場calendar與timestamp映射，不能用曆日字串硬join。若當天是休市不產虛擬交易；11/28本身不能被硬當成t交易日。

08:00前需確認每個必要source的實際publication_at／observed_at、snapshot_at與model cutoff，且 `available_at≤decision_at`；22:30舊抓取若不包含隔晨美股t收盤，08:00空等不會變出新資料。需另案核准既有來源的隔晨refresh／交付或採另行版本化lag-us方案；**本輪不改排程、不新增資料取得**。缺資料就blocked／無新訊號，不用前一日結果假填今日，不延後完成後回填開盤成交。

此08:00參考線時鐘與前瞻WP1的18:00 cutoff不同，不能把隔晨補齊資料回注18:00前瞻快照。集保Week／Business fallback／還原價future action等問題也不因等到早上而自動消失。只有來源已實際記錄才更新observed_at；未知維持UNKNOWN／STOP。

## D. 多模型獨立排序與整合排名

建議先分 `h5`、`h10`、`h20` 看板：每張卡同時顯示target定義、as-of／決策時間、模型版本、訓練截止、當日股數、缺漏與freshness。h20不等於20-session再平衡頻率；不同head或同一基線在第四／五步的重fit，不是獨立研究信號。

第二階段預設只合成**同target定義、同horizon、同決策時鐘**的模型，先固定eligible component集合，seed層先rank平均，模型層再等權平均同日百分位。統計rank用average ties／N，最終展示與TopN同分以stock_id字串升冪；raw LightGBM值／神經logit不直接平均。所有component在同一共同股票截面先對齊再rank，列出各模型原股數、交集股數／覆蓋率與排除原因；缺分數不能補0。必要component缺失時整合排名unavailable，個別已驗收模型仍可單獨顯示；不能每天依輸贏或缺檔靜默改權重。

若Human希望跨horizon單一「綜合」：可用每模型同日百分位等權作**偏好指標**，明示「跨horizon／未經驗證」，不解讀成預期20日報酬，也不預設用於paper投組。建議保留分horizon整合為預設，跨horizon只作另外的探索視圖。不能把R_all連同其三component都等票納入而重複計票，亦不能把第三／四／五步同190欄基線都算一票。

可另案對**已曝光**OOF共同鍵做描述性IC／rank相關／TopN重疊／有成本投組評估，固定支援、NaN label成熟規則、版本與所有試法列表，不新增未曝光label存取，不以觀察調權再稱驗證成功。P4b既有三種混合未證實改善，是「等權也未必改善」的現成證據，不是新h20整合已驗證。網頁固定標記 **整合方法未經驗證／EXPOSED_DESCRIPTIVE_ONLY**，不把它變更前瞻primary或次要家族。

## E. 模擬投組規格候選

這是paper紀錄，所有成交均虛擬，無broker／Shioaji下單接線。建議先一個第三步B投組；第二階段每個模型及同horizon整合各自獨立本金／交易／NAV，不能共用持倉或看表現後切換主投組。

| 項目 | 候選固定規則 |
|---|---|
| 起始與換股 | 驗收後首個訊號session t為anchor，每20個**合格台股session**換股；t訊號在t+1開盤虛擬成交，下一次同anchor＋20，不用20曆日或每月日號 |
| 選股與權重 | 建議Top50等權、long-only、初始100萬元、無槓桿／借券；N、資金與零股政策待Human，未凍結前不產正式paper績效；首版不加績效導向buffer／Kelly |
| 可交易篩選 | t收盤原始成交價≥10元；建議**截至t的前20 sessions平均實際成交金額≥1,000萬元**，需20筆有效支援，不能以還原close×volume猜Trading_money。Human若指「t日金額」須明選，兩者不是同義。當日停牌／無交易／lifecycle未知fail closed，剔除與原因記錄 |
| 模擬成交 | t+1未調整open＋可核交易狀態；停牌、零量、開盤缺失、漲跌停無法證明成交等記blocked，不虛構fill。新買入失敗留現金；賣出失敗保留既有部位與unknown風險，不因未來缺價倒刪整個週期 |
| 成本 | 買佣金0.1425%＋單邊滑價0.1%；賣佣金0.1425%＋證交稅0.3%＋單邊滑價0.1%。相同名目金額來回約**0.785%**（買0.2425%、賣0.5425%）；採成交金額計，不對未成交金額扣費；留倉不重收全倉成本。首版比例費，不暗加折扣或最低20元；零股／整張／最低費用另由Human凍結 |
| 滑價實作 | 建議buy open×1.001、sell open×0.999後另記commission/tax；不得又扣一次10bp現金成本；持股不動只mark |
| 價格與公司行動 | 執行open與估值使用可核原始實際價；還原價只供模型配方。股利、分割、減資依生效事件記shares/cash，缺證據記UNKNOWN，不把人工還原價當虛擬成交價；不能混用股利已調整標籤與再加現金股利 |
| 持股與現金 | 建議整股floor、餘額留現金；現金不得負，weights＋cash對帳。合格股不足50時保留空缺現金，不在低支援日自動加槓桿；不足支援是否整日blocked在paper契約凍結 |
| 監控 | 每日來源receipt／特徵identity／模型hash／行數／缺值／signal freshness／被擋成交／現金部位對帳；label成熟後IC另外存，不影響封盲前瞻或自行選模型 |

建議在同一私有run namespace分表、append-only／atomic manifest（格式與欄位在實作前凍結）：

- `run_manifest.json`：schema/evidence_class/PAPER_ONLY、source_commit、contract/config hash、environment、calendar/universe/source snapshot身份、model／feature order／input hashes、signal_session、decision_at、generated_at、status、attempt與previous manifest身份。
- `predictions.parquet`：session/stock_id/model_id/seed/horizon/target_id/raw_score/seed_rank/model_percentile/rank/eligibility/reason、訓練截止與資料版本；未成熟label不寫假0。對外API只投影獲准欄位。
- `orders`／`fills`：paper_order_id、model／signal identity、planned_session、rebalance_id、side、target_qty、requested_qty、fill_qty、open_reference、slippage、execution_price、gross、commission、tax、cash_delta、source_receipt、status/reason；rerun同id不得重複fill。
- `positions`／`nav_daily`：session、stock_id、qty、cost_basis、mark、mark_quality、cash、equity、NAV、gross/net return、cost、turnover、unknown/stale counts、corporate_action lineage；缺估值就標NAV品質不足，不把未知當0。
- `events.jsonl`：人工決定／修訂／失敗／恢復的時間、身份及理由；保留原輸出，不覆寫失敗日。這是參考paper帳本，禁止與前瞻sealed journal共用檔案或authority。

驗收例：20-session跨長假正確；08:30後完成不能回填當日open；同分Top50穩定；買／賣費率與round-trip手算相符；同股留倉無重複費；停牌／缺open／UNKNOWN無fill；連續重跑無重複訂單；公司行動及缺mark不產假NAV；模型／輸入變更建立新版本，不重寫過往paper績效。

## F. 分階段實作工作包與驗收

以下是未執行的工作量估計，1人日約6–8小時，無付費GPU、資料採購或部署工時承諾；等待來源／授權確認的日曆時間另計。只修有界單一變因，先建立會失敗的驗證，再做最小實作；每包完成由協調者安排獨立審查，不由本builder新增agent。

### 第一階段：第三步B＋既有網頁＋paper＋監控（約8–13人日）

| 包／依賴 | 後續檔案責任與介面（候選，非本輪新增） | 驗收／估時 |
|---|---|---|
| F1 決定與凍結；無依賴 | 沿用tasks決定檔格式，凍結本機／雲端披露、三seed與fold、target/時鐘、paper config；協調者另立不得因參考觀察改前瞻決定。model registry含model_id/component/hash/feature_order/target/train_cutoff/runtime | 不以本文取代frozen authority；保存模型／manifest完整hash、load→predict probe與副本策略；不選績效最佳seed。0.5–1日 |
| F2 t日190欄；依F1 | 優先新增 `tools/live_reference_features.py` 薄adapter，重用c0plus、step3純函式；不修改原研究bundle／Data。介面輸入snapshot＋as_of＋calendar＋universe，輸出ordered rows＋features＋lineage/status；共用48來源重播先補證 | C2逐欄、NaN、發布日與未來擾動驗證；舊48完整重播有差異即STOP、提供差異報告；不能用測試通過掩蓋source STOP。3–5日，若原source不可恢復另估 |
| F3 推論與原子產物；依F2 | 候選 `tools/live_reference.py`，純本機`infer(snapshot, model_registry, out)`，不內建fetch/push/notify；每seed固定載入、rank合成、date與input/model身份必填、私有manifest最後發布 | mock／synthetic與已曝光golden rows：model mismatch、NaN score、半寫、重跑／kill-restart、stale／wrong session皆fail closed；量測CPU/RAM與08:30前SLA，不能只驗檔案存在。1–1.5日 |
| F4 既有UI/API；依F3，可先用synthetic fixture | 候選 `app/backend/routers/reference.py` 唯讀artifact、schemas/main附加掛載；frontend新增V7 reference page/API client與App路由，重用列表／狀態元件；顯示model/train_cutoff/horizon/as-of/freshness/PAPER_ONLY。私有storage reader與公開投影分界清楚 | 缺／壞／stale輸出明確blocked或unavailable、絕不mock假成功；不連盲區、不讀Data、不啟動推論；本機API schema＋UI build／必要互動測試＋敏感欄位拒絕；維持既有API回歸。1–2日 |
| F5 paper帳本；依F1/F3 | 候選 `tools/live_reference_paper.py`，只接受已驗收signal artifact＋後來可核open/status，產E表格與NAV；不呼叫broker。現有portfolio engine可借計算原語，但不能沿用其不同成本／門檻而不明說 | E列反例、成本與現金對帳、跨20session、冪等／恢復、缺價與公司行動；所有回傳PAPER_ONLY。1.5–2日 |
| F6 手動dry-run／監控；依F3–F5 | 沿用V7 health投影方式，日報只含來源品質／freshness／artifact完成、paper狀態；操作文件記錄snapshot／infer／publish三步與復原；最初手動執行，排程另案 | 至少5個連續合格session工程觀測，且用fixture覆蓋20-session換股；沒有5日就報PARTIAL，不能用fixture取代真來源接收。CPU smoke/故障演練、permission/secret allowlist、治理與diff檢查；不自動部署／啟用排程。0.5–1.5日＋觀測時間 |

### 第二階段：其他模型獨立排序＋同horizon整合（約5–10人日，不含重訓等待）

| 包 | 工作與驗收 | 估時 |
|---|---|---|
| F7 h20擴充 | 先第三步A、第四B1a/B1b、第五B；每模型獨立feature contract／hash／每日source receipts，先重播歷史相同欄再接t日。第四／五步重複A不重複投票。分類／同業來源缺時仍顯示unavailable，不以0家替代UNKNOWN | 2–4日，依同業事件供應缺口重估 |
| F8 y5/y10與神經runtime | P1缺模型需另核准有界refit，禁止假冒舊reference；P4 BEST／environment／native hashes、W60截面、CPU推論可行性先量測；Mamba2需要GPU則另案決定，本任務未授權。P4d未正式完成不先上 | 1–3日盤點接線；重訓與硬體費用另估、另准 |
| F9 ensemble與多paper | D的同horizon百分位等權、共同支援、缺component fail closed、同分tie-break；各模型與整合獨立帳本，UI並排；已曝光OOF描述性分析若獲准另輸出報告，固定試法不回頭挑權重 | 1–2日；若新增歷史描述分析再0.5–1日 |
| F10 驗收與交付 | 全鏈source/model/signal/paper身份對帳、權限檢查、瀏覽器可用性／資料stale示例、治理檢查；保留第一階段單B可獨立使用，不讓其他缺件拖成假成功 | 1日 |

### 需要Human／協調者明確選擇的事項

| 決定 | 選項 | 建議 |
|---|---|---|
| 首版存取／披露 | 本機或區網；私有雲端＋驗證；PUBLIC排名 | 本機／區網，TEJ衍生再散布確認前不public raw |
| 參考模型實體 | 最新fold2026三seed；單seed；另行as-of refit | 先凍結現有fold2026三seed，rank平均，標示最後訓練年；不因這樣選擇而聲稱最新模型或改前瞻 |
| 時鐘及US來源 | 隔晨08:00／08:30驗收；另版lag-US；不顯示缺源日 | 隔晨候選＋receipt gate；缺隔晨來源先blocked；變更fetch／排程須另行有界授權 |
| Paper N／本金／金額門檻 | Top20/50/100；100萬或自訂；t日金額或前20session均額 | Top50、100萬、前20有效session均額≥1000萬；價格≥10；整股留現金；雙邊費稅依E |
| 失敗／缺源與交易 | 缺component仍顯示其他單模型；混合停更；重分配權重 | 單模型可各自顯示；整合停更、無新虛擬買單，持倉估值品質另報；不靜默改權重 |
| 第二階段模型集合 | h20樹先行；同時加y5神經；復活V62 | h20樹先行，P4按runtime逐一驗；不復活V6平行維護 |
| 跨horizon整合 | 不做；偏好指標；直接paper決策 | 同horizon為預設；跨horizon若做僅未驗證探索，另議paper用途 |
| 日更自動化／雲端發布 | 手動；另核准排程／部署 | 先手動dry-run，本文件不啟用任何job、wakeup或通知 |

以上可在下一個明確工程核准中一併凍結；本次Human已批准的第三步B／現有網頁／只模擬／不得影響前瞻不需重問。

## G. 與WP1–WP6共用的界線及規則草案

前瞻authority仍為 [P2 spec](spec-forward-p2-primary-20261124.md)、[WP2–WP6決定](decision-forward-wp2-wp6-20261126.md)、[跨步家族把關決定](decision-forward-family-gatekeeping-20261127.md)與相應machine contracts；參考模型選擇不更改它們。WP工程通過不表示真來源PIT／calendar／lifecycle／execution已准入；目前整包NO_GO／各STOP不得由本文件解除。

| 模組 | 可共用能力 | 必須隔離／尚缺 |
|---|---|---|
| [WP1](../tools/forward_wp1.py) | Calendar identity、as-of／publication/observed時鐘、source/lifecycle fail closed | 前瞻18:00／A+2規則照原契約，參考08:00另contract；共享不可變原始來源版本與receipt，不共享較晚as-of快照。calendar或publication缺證據維持STOP |
| [WP2](../tools/forward_wp2.py) | keys／樣本／特徵及預測重播驗證原語、未來擾動oracle | P1歷史微差容差只限Human指定原reference，不能泛化190欄或新模型；參考B不是P1/y5 reference |
| [WP3](../tools/forward_wp3.py) | 完整session延伸、mask／matrix身份與immutable快照觀念 | 不能直接把190欄替換原48／185欄矩陣；新參考lineage獨立，嚴禁讀封盲predictions／labels填參考頁 |
| [WP4](../tools/forward_wp4.py)／summary | 模型serialize/fsync/hash／atomic pointer、probe roundtrip及environment封存模式；公開allowlist | model roots／pointer／seeds／context與前瞻隔離；同環境load→predict仍bitwise，不以歷史容差放寬；summary不能帶私有績效或native logs |
| [WP5](../tools/forward_wp5.py) | OPEN/CLOSE receipt、獨立attempt／checksum／來源品質監控觀念 | 正式retrain R/R+1/R+2、incident verdict、acceptance／unblind registry不可由參考run修改或重設；參考頁故障不成為前瞻停更、重啟或提前開盲理由 |
| [WP6](../tools/forward_wp6.py) | 可參考paper狀態／成本／對帳的純函式設計 | 現在`run_endpoint`明確拒絕非synthetic calendar；不能直接以真資料呼叫並宣稱驗收。20-session參考paper與252-session封存／開盲endpoint是兩本帳，不共享NAV、trials、重跑或selection |

建議資料流：同一唯讀來源快照／receipt store，各自按contract的as-of投影成「參考線輸入」與「前瞻封存輸入」；參考程序只持前者讀權及自己的寫root，API只持公開／授權投影讀權。前瞻程序保留原有封存／權限／attempt鏈，網頁與參考程序無法讀其權重、預測、績效或journal。這是後續權限／程序隔離設計，並非本輪已驗證OS隔離。

**供協調者另立決定檔的草案：**

> 第三步B及其後納入的每日參考模型、獨立排序、整合排名與模擬投組，均屬可觀察的選股輔助／PAPER_ONLY，不屬封盲前瞻檢定結果。其任何觀察，包括IC、報酬、損失、模型間勝負、排名、缺漏或人為解讀，均不得用於修改前瞻家族、primary/secondary、模型／特徵／horizon／seed、合成權重、成本／投組規則、樣本、期間、門檻、多重比較或開盲規則；不得據此提早開盲、中止、延長、重啟、挑選子期間或替換已封存版本。技術事故只依既有前瞻事故契約的非績效證據及既定期限處理；參考觀察不是事故豁免。若未來確有獨立工程／來源安全問題，保留原封存與事故證據，交由既有authority流程處理，不能以「參考模型不好／很好」作變更理由。所有可見參考資料與新探索決定均記錄曝光及時間，新研究另立版本與未曝光驗證，不追溯改寫原前瞻。

## 驗證、知識影響與固定交接

本案只交付規劃，沒有新功能、authority supersession、runtime變更或研究結論；目前blocker只是整理既有證據，不由本文解除。Knowledge-impact八項逐項判斷：semantics NO、authority NO、subsystem boundary NO、active runtime NO、correctness gate NO、canonical artifact新增／supersede NO（本檔為待審規劃）、deprecate NO（OBSOLETE是規劃分類，不刪除／停用既有能力）、major blocker／next decision的canonical狀態變更 NO。已檢查Current State與Authority Map，本任務不把未審規劃加入current authority。

`knowledge_update: not_required`；`NO_KNOWLEDGE_CHANGE_REQUIRED`理由：指定範圍只有離線盤點及待審計畫，未改接受的目標、實作語義、runtime／authority／gate，Human決定待協調者另立正式決定檔，故本輪不修改canonical maps。文件列出的未確認來源／缺模型／service狀態不等於新增production觀測。

| 固定欄位 | 本次交接 |
|---|---|
| Completed | A–G離線盤點與分期規劃；不是Human驗收或可運行交付 |
| Changed | 僅 `tasks/plan-live-reference-20261128.md` |
| Decisions | 重述Human已選第三步B／現有網頁／只模擬／觀察不得影響前瞻；其餘標候選建議 |
| Evidence | 版本化程式／契約／既有報告、私有manifest與檔案inventory；未訓練、未重算IC、未載模型；48引用有效、knowledge-impact PASS、knowledge-health WARNING且exit0、whitespace／範圍檢查通過，詳下方 |
| Remaining | 完整48→190重播、今天來源receipt／營收版本、各seed精確截止、模型逐檔hash／load→predict、可核universe／execution、TEJ許可、真實服務狀態；實作與部署均未執行 |
| Blockers | strict PIT／calendar／lifecycle／還原價vintage及來源漂移STOP；缺隔晨US來源；P1原權重缺失；P4d無正式模型；公開再散布授權UNKNOWN |
| Next recommended action | 協調者獨立審本計畫、另立不得因參考觀察改前瞻的Human決定，凍結F1選項，先核准F2來源重播／同義特徵工程 |
| Project memory updated | 否；knowledge_update: not_required，理由如上 |
| Authority map updated | 否；已檢查，候選規劃不升格authority |
| Git status | 起始clean；結束保留指定文件untracked；不為「clean」擅自stage／commit／刪除交付 |
| Commit | 無，依本次禁止commit |
| Push | 無；亦無merge／部署／排程／通知／wakeup |
| Worktree | 指派builder_live_inventory、detached `61f7505`；無新增或移除worktree |
| Runtime/environment | WSL／Python3.12.3／NumPy2.5.4（只讀metadata）／Node24.19.0；data root未設定，本worktree Data非主checkout Data；離線CPU，無GPU |

### 本輪實際檢查結果

1. 文件結構檢查：A–G七節皆存在，48個本機Markdown引用全部存在（缺失0），無trailing whitespace、無Unicode replacement character；自查修正dual欄位／模型目錄及P1配方名稱，未更動程式。
2. `tools/project_knowledge/check_knowledge_impact.py --changed-file tasks/plan-live-reference-20261128.md --ack-no-change <上述非空白理由>`：exit0、PASS；命中history-reports-tasks／NO_KNOWLEDGE_IMPACT，無自動更新；紀錄只在`/tmp/live-reference-knowledge-impact.json`及同名log。
3. `tools/project_knowledge/knowledge_health.py --ci`：exit0、Overall WARNING；Project Map／Agent Guides／Domains／Runtime Freshness／Generated Index PASS；Governance／Authority Drift／Legacy Docs WARNING，無structural FAIL。未改治理檔或刷新runtime觀測；log只在`/tmp/live-reference-knowledge-health.log`。
4. `git diff --check`及新增文件獨立whitespace／引用檢查通過；tracked差異為0，最終唯一untracked為本交付文件。受保護程式／Data／V6/models／排程未改；依禁止commit保留dirty交付，不聲稱clean-worktree完成。
5. 未跑應用／研究測試：本案僅Markdown規劃，沒有可執行語義改動，且應用與部分研究測試可能觸發網路或讀正式資料。本次驗證只支持文件完整性與範圍，不支持推論／部署／交易可用性。
