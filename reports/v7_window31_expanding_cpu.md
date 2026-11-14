# Window31 First Expanding CPU Development Pilot

Status: DEVELOPMENT_ONLY / UNMERGED / NOT_DEPLOYED。2023 全部 EXPOSED；本報告不建立模型選擇、Strict PIT、可交易收益或正式 Research Admission。

W30 integration：main `e83dfe68e4bc73b59cb6390ec130ba967138e7f1` → `b64ac353833a9248121f12ebe61e1333decb2dbf`，本機 ff-only。W30 970 daily＝940 defined＋30 None；2910 paired＝2820 defined＋90 None，未定義尾端／成熟度項未當成0納入Mean。原模型5d/10d分別351/379 split nodes，非退化預測；0重新Fit。

W31 branch `research/v7-window31-expanding`，base同整合後main；首次Fit前commit `ddcf5d835b43578ed9fc5567ee677d88e928e203` 固定契約、程序、支持集合與身份。

## 時界及支持集合

| Origin | Validation sessions | Forward sessions | H | Training N | Validation N | Fresh refit N | Forward label N |
|---|---|---|---:|---:|---:|---:|---:|
| O1 2023-01-03 | 2022-01-03–2022-12-30 | 2023-01-03–2023-03-31 | 5 | 14113 | 14816 | 29229 | 3353 |
| O1 2023-01-03 | 2022-01-03–2022-12-30 | 2023-01-03–2023-03-31 | 10 | 13711 | 14435 | 28743 | 3337 |
| O2 2023-04-06 | 2022-04-01–2023-03-31 | 2023-04-06–2023-06-30 | 5 | 17559 | 14710 | 32574 | 3698 |
| O2 2023-04-06 | 2022-04-01–2023-03-31 | 2023-04-06–2023-06-30 | 10 | 17143 | 14293 | 32046 | 3660 |
| O3 2023-07-03 | 2022-07-01–2023-06-30 | 2023-07-03–2023-09-28 | 5 | 21321 | 14645 | 36276 | 3944 |
| O3 2023-07-03 | 2022-07-01–2023-06-30 | 2023-07-03–2023-09-28 | 10 | 20890 | 14205 | 35715 | 3897 |

TRAIN exit 必須早於第一個 Validation Session；Validation/Refit exit 必須早於 Origin。日期以共同市場 session 索引處理；5d/10d依實際持有窗排除跨界，沒有套用E5固定30-session purge。額外embargo=0為本輪預先固定工程參數，非正式政策。Refit包含新成熟的validation及先前因TRAIN邊界被排除的樣本，故不只是兩個舊mask拼接。

所有模型只接收當次有資格的監督樣本。Ridge scaler各自在TRAIN或REFIT重新估計；LightGBM沿用W30唯一固定Recipe，無eval_set/early stopping/search。Validation模型及Refit模型IDs為 `W31-{O1/O2/O3}-H{5/10}-{Ridge/LightGBM}-{VALIDATION_FIT/REFIT}`，共24新模型；同名phase間不同身份，resume_checkpoint=None。L0方向固定。

每Origin先保存Validation Predictions及診斷，才fresh refit；所有Origin訓練完畢後才第一次計算Forward metrics。較早Forward時段已成熟labels可按固定規則進入後續Origin的Validation/Refit，但不使用Forward metrics調參、篩股票或選Recipe；重疊來源及曝露角色明確保留。

Feature-only Predictions由既有feature frame產生，沒有Label參數；保存後才Join評分mask。所有模型逐日股票集合/support hash相同；各模型完整預測均保留，不依結果正負刪除。

## Forward-style 診斷

| Origin | H | 有效日 | Daily N | L0+ Mean | L0− Mean | Ridge Mean | LGBM Mean | LGBM−Ridge |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| O1 | 5 | 54 | 60–64 | -0.029905 | 0.029905 | 0.028606 | 0.023592 | -0.005014 |
| O1 | 10 | 54 | 60–63 | -0.017516 | 0.017516 | -0.012394 | -0.021556 | -0.009162 |
| O2 | 5 | 59 | 60–64 | -0.024188 | 0.024188 | 0.035161 | 0.058135 | 0.022974 |
| O2 | 10 | 59 | 60–64 | 0.000856 | -0.000856 | 0.008584 | 0.026531 | 0.017947 |
| O3 | 5 | 63 | 60–64 | -0.020362 | 0.020362 | 0.024427 | 0.065496 | 0.041069 |
| O3 | 10 | 63 | 59–64 | -0.009608 | 0.009608 | -0.002463 | 0.000766 | 0.003229 |

平均差取同一每日支持集合，非不同Coverage的Mean相減。逐日N、ties、Mean/Median、六組pair及拒絕帳本保存在Private Root；完整48組Validation/Forward摘要見 `research/v7/window31-results.json`。三季176個Forward sessions×2h共352組均可計算，Daily N59–64，固定門檻40未降低。

Ridge5d三季正向；10d正負不一致。LGBM各季表現亦有差異。這僅支持固定Pipeline可按時界運作，不能宣稱模型勝者、統計顯著或獨立泛化。

## 實作與驗證

重用W07 fold/check_fold與run lineage、W10 Ridge、W29 pinned data/features、W30 LightGBM fit_arrays、W08 daily_metric與既有atomic/hash/lock工具。新增薄型origin-neutral numeric label及mature views是必要的：舊mask綁定2021/2022年度邊界，不能直接用於新Origin；沒有建立新Trainer/Evaluator/Registry。

10 focused tests（純數值與mock，不額外Fit）PASS。初始RED為缺元件；一個fixture缺少前一session造成預期錯誤已修fixture，未改Label規則。導航缺authority_class的structural FAIL已修復；原始失敗及修後health需在收尾證據如實保留。

實際Ridge 12 fits、LGBM 12 fits；453192 complete predictions，7264 daily，10896 paired（含明確None的Validation成熟度尾端，不納入Mean）。0 API/GPU/W19或V6操作。每階段GNUtimeout300s與OS RLIMIT_CPU300s，累積CPUledger1800s、disk512MiB含restore；詳執行／資源證據。

## 限制與下一次Human決策

Historical universe及普通股身份、hindsight/current-survivor選樣、UNKNOWN公司行動/停牌、Raw return非total/executable return、歷史availability/publication/vintage不足仍保留。幾何fold資格不是Strict PIT；實際2026執行時間與2023邏輯model-ready/close時鐘分開。資料不可直接升格正式准入。

正式Expanding研究最少須批准：合格Dataset/Universe/事件與UNKNOWN政策；精確Decision-time/Label/Calendar/maturity/purge；selection/refit與evaluation/statistical規則；過往曝露處理或真正prospective評估；CPU/GPU資源及獨立備份位置。不得由本輪分數選最佳模型或直接啟動Mamba。

Private Root `/home/frank/projects/MarketMamba/.artifacts/window31-expanding/`；同機Restore `.artifacts/window31-restore/`。BACKUP_PENDING，兩份同機副本不是獨立備份。W29/W30原證據與W11環境依賴均需保留。

恢復入口：`knowledge/00_Project_Map/Current_State.md` → `research/v7/research-program.md` → `window31-contract.json` / `window31-evidence.json`。本輪Fit額度已用完，禁止重跑O1/O2/O3；僅在後續明確授權下延續。完成後STOP。

獨立程序Replay：453192 predictions、7264daily、10896paired完全吻合，0額外Fit。72個immutable檔案91000410bytes/份同機restore，W29/W30既有310檔hash不變；總182002029bytes，研究CPU87.69063秒＋60秒驗證reserve。BACKUP_PENDING。

治理誤差：prefit shell未set-e，authority_class缺漏的導航FAIL未阻擋O1；補齊後health僅既有WARNING。未修改已固定數值程式、契約、模型或結果。後續收尾採set-e及獨立審查確認。

獨立主要Review GO：Critical0／Required0，0輪review rework。24角色view重建、24模型array hashes、12 Ridge own-scalers均吻合；Ridge normal-equation residual上限3.55e-15（無重新solve/Fit）。Warnings：歷史導航fail-fast疏漏、外層timeout未具獨立OS簽章、同機restore非備份及未解研究限制；均不提升正式准入。
