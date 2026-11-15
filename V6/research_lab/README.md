# Research Lab：可重現資料集 v1

`UNMERGED / NOT_DEPLOYED / RESEARCH_ONLY`，僅供 pilot09-w1 離線研究。
新 package 隔離 Yahoo 價量研究契約；不變更 production、V6 frozen feature spec 或 V7 admission。
已檢視 `marketmamba/data/{feature_spec,feature_engineer,hygiene,merger}.py`：既有實作依賴
production config、多來源合併／補值與 non-fatal hygiene，不適合此處 pure、嚴格缺值契約。
已比對 `experimental/v7_executable_labels.py`：重用其時間語義，另以 synthetic parity test
驗證公式；該模組的 executable admission 不適用沒有可成交證據的 Yahoo 輸入，因此不直接呼叫。
未新增外部依賴。

## 契約與 API

- `data.load_prices(path)`／`validate_prices(frame)`：欄位 `code,date,open,high,low,close,adj_close,volume`。
  代碼保留字串前導零；按 code/date 排序。完全相同重複列去重；衝突鍵、空代碼、無效日期拒絕。
  非有限／非正價格或 OHLC 幾何錯誤將整筆 OHLC 設 NaN；負值／非有限 volume 設 NaN。
  零成交量保留原值但 observation_valid=False；不補值、不插入輸出列。
- 原始 close 相對前次觀測變動絕對值 >35% 標記 `split_or_outlier_flag`，原因 UNKNOWN，
  不是已證實分割；不自動修價。coverage report 不參與當日特徵或 universe 篩選。
- `features.compute_features(prices)`：保留輸入欄並附加 `FEATURE_COLUMNS`（固定順序 33 欄）。
  1/5/10/20/60 日原始 close 報酬、20 日日報酬母體標準差（不年化）、20 日 volume z/均量比、
  close×volume 金額代理、60 日回撤、14 日簡單均值 RSI-like；每個特徵另有當日橫斷面 rank/z。
  只有截至 t 的值，沒有全期間擬合、中心視窗或未來標籤依賴。
  rank 平手取 average percentile；單一有效樣本未知；z 使用 ddof=0，零變異保留 NaN。
  RSI 全平盤=50、全上漲=100。均量比不是股本週轉率，無流通股本資料。
- `labels.compute_labels(prices)`：**收盤 t 訊號，t+1 開盤進場，t+h 收盤退出**，
  `fwd_return_{h}d = close[t+h]/open[t+1]-1`，h=5/10/20；禁止同 bar 進場。
  輸出 code/date、label_entry、label_end_{h}d、fwd_return_{h}d、rank_label_{h}d。
  t..t+h 任一缺漏／無效／零量／outlier 都令標籤 NaN；不延後進出場、尾端未知不補零。
- Calendar 使用全體輸入的 observed-date union **代理**；內部 reindex 防止缺漏壓縮為下一筆。
  不是官方交易曆，不能辨識全市場整日缺失，coverage 缺漏也不代表停牌或上市前。
- `splits.walk_forward_splits(labeled_frame, min_train_sessions=252, test_sessions=63,
  horizon=20, purge=20, embargo=0)`：expanding、完整 test blocks、跨股票共用日期邊界。
  回傳 `[{fold, train, test}]`，train/test 是含 `row_index,code,date,label_end_{h}d` 的 index frames；
  row_index 對應呼叫端唯一 index。purge 至少 horizon，embargo 為額外的 **test 前安全間隔**，
  合計排除 purge+embargo 個 observed sessions；同 fold 不訓練任何 test 後樣本。
  train 僅取已成熟有限標籤，拒絕錯誤 end-date 對齊。test 保留未知標籤尾端，
  只在事後 evaluation 排除未知標籤；禁止以未來 label coverage 篩選 predictions。
  所有模型/scaler 僅能 fit train，模型 fitting 與 splits 產物不在本 pipeline 範圍。

## 重建與驗證

於 repository root 執行（Python=`/home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python`）：

```bash
PYTHONPATH=V6 /home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python -m research_lab.build_dataset \
  --prices /home/frank/projects/ruflo-evaluation/pilot09/marketmamba/.local-data/public_tw/prices_daily.parquet \
  --out .local-data/research/dataset.parquet
/home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python -m pytest V6/research_lab/tests V6/experimental/v7_executable_labels_test.py V6/experimental/v7_temporal_contract_test.py -q
```

驗證：31 passed（含真實 parquet 檢查；外部資料不存在時該測試會 skip）。
先確認新套件不存在時測試 collection 失敗，再實作；含未來修改／截斷不改過去特徵、
壞資料、gap、平手／常數、label alignment、既有 holding-window parity、purge/embargo、
canonical hash 與 parquet round-trip/no-clobber。
預設 input `.local-data/public_tw/prices_daily.parquet`；預設 output 如上，已 gitignore。
輸出若存在即拒絕覆寫，重新執行必須提供新的獨立 `--out` 路徑。
完成標記為同名 `.manifest.json`，失敗留下只有 parquet 的產物不能視為 complete。
manifest 記錄輸入／輸出 canonical CSV SHA-256、原始檔 bytes hash、sidecar hashes、
source HEAD、實作逐檔 hashes、Python/pandas/numpy/pyarrow identity、品質及逐股 coverage。
Canonical form：按 code/date 與欄名排序、UTF-8 CSV、ISO 日期、float %.17g、null 缺值。
Path 是 metadata，不參與 content hash；不加入 volatile timestamp。`adj_close` 僅保留 provenance，
特徵與標籤使用原始 OHLC，不讓事後調整因子直接進模型。
下游應只使用 `FEATURE_COLUMNS` 訓練，禁止將原始 label／label_end 等欄位當特徵。

## 此次真實資料產物

來源 HEAD `4be3084948a453224ab1dbbd30342c04902ea7d4`；branch `w1-data`；指定 Python 3.12.3。
外部 input realpath 為上述 `--prices`；`MARKETMAMBA_DATA_ROOT` 未設定；未讀寫 `Data/` 或 weights。

| 項目 | 實測值 |
|---|---:|
| 列數／代碼數 | 378,571／150 |
| 日期範圍 | 2016-01-04～2026-11-13 |
| 完整特徵列數／特徵數 | 304,034／33 |
| 5／10／20 日有效標籤 | 366,365／358,070／343,938 |
| 完全重複／無效 OHLC／無效 volume | 0／0／0 |
| 零成交量／outlier | 3,346／4 |

Observed-union 共 2,645 sessions；每代碼 238～2,645 列，最低覆蓋率 8.9981%，
相對整段 union 共缺 18,179 個 code/session（包含尚未有觀測的時期，不能推定停牌）。
完整逐股覆蓋率在 `.local-data/research/dataset.manifest.json` 的 `coverage`。
Dataset canonical SHA-256：`325093035cf9eac936d4e151da26583aa90cb02c83af3be35a019e0c45c962b1`。

## 限制

存活者／當前成交額選樣偏差；包含 `0050` 等 ETF，非經資格驗證的純普通股 universe。
Yahoo adjusted prices／歷史修訂沒有當時可見 vintage，僅轉換的計算因果性經測試，
不代表歷史 strict PIT。未取得財報或 publication timestamps、歷史 universe、官方 calendar、
corporate-action 完整證據、可成交性、停牌／漲跌停／成本／滑價。
Raw returns 不是還原股利或分割的總報酬，小於 flag threshold 的事件仍可能污染；
flags 不是完整公司行動偵測。不得升格為 corrected V7、paper/forward 或 verified trading evidence。

治理：source-only impact PASS；完整 diff impact WARNING → ACKNOWLEDGED
（已檢視 GOVERNANCE.md，治理政策未改；Current State／Authority Map 已補此研究邊界）；`knowledge_health.py --ci` FAIL，
既有 authority 引用 `e3bce91` 等 commit 不存在於本 checkout 的 Git objects；
已確認 HEAD 原本的 authority-map.json／Features note 即含該引用，未改寫／放寬。
完整失敗輸出在 `.local-data/research/knowledge-health.json`。
工作樹保留原有 `.claude-flow/`、`.swarm/` dirty/untracked 狀態；Ruflo memory CLI 亦產生本機 runtime 狀態。
未 commit／push；後續先恢復既有 authority objects，再於整合前重跑治理。

## pilot09-w2 排序實驗

`models/`、`metrics.py`、`run_experiment.py` 為新增的 CPU-only 研究流程，
`UNMERGED / NOT_DEPLOYED / RESEARCH_ONLY`。重用本 package 的固定 feature 與 expanding split；
既有 `experimental/baseline_rnn.py` 綁定不同矩陣／舊 feature contract，
`marketmamba/models/architecture.py` 需要 CUDA mamba_ssm／torch_geometric；
既有 evaluation 將未知 IC 歸零，不能滿足此處未知值保留契約，因此另建小型模型與 metrics，
不改既有 authority 或任何 `V6/models/` 權重。

```bash
PYTHONPATH=V6 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python -m research_lab.run_experiment
PYTHONPATH=V6 /home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python -m pytest V6/research_lab/tests -q
```

預設資料與 prediction 路徑指向本次指定 main checkout `.local-data/research/`；
資料唯讀、驗 manifest bytes SHA-256；prediction 目錄已有 parquet 時拒絕覆寫。
全部模型共同採 60-session 完整歷史支援，不以未來 label 篩預測；因此與只需 20 日的
baseline 原生可預測範圍不同。日橫斷面 z 使用當日已知股票，不跨日 fit scaler。
訓練標籤為 5/20d percentile rank 減 0.5，GRU／selective 為雙頭 MSE；LightGBM 為 rank-label regression。
外層 min_train=504、test=252、purge=20，同一 coarse schedule 用於所有模型，未完成尾段略去。
神經網路每 fold 從頭初始化、seeds 17/29/43、width12、selective state4，最多2 epochs，
train 內末最多64 sessions 作 validation，fit 再隔20 sessions，每 seed 固定抽最多4096 fit／1024 validation rows；
使用 validation MSE 最低 checkpoint，不使用 OOS 選模。
此刻意低成本設定不構成充分收斂或模型家族優劣的證據。
各 seed 與逐日 percentile 等權集成各存獨立 parquet；模型 score 越高表示預期排名越佳。
輸出欄位 `code,date,fold,model,horizon,score`，code 保留字串前導零。
結果含整體／年度日期等權 Spearman IC、非年化 ICIR、Newey-West t／95% normal CI
（Bartlett lag=h-1，只處理指定長度的自相關）、前後各 floor(N/10) 股票的平均 h-session
報酬差與 top decile 正報酬比例。同分按 code 字串遞增排序，top 取末端；未知 IC 保留未知。
此為重疊 forward-return 診斷，非可執行多空策略、非年化 P&L，無成本／成交約束。
完整逐 seed、逐 fold、年度數字、manifest hash、source hash、runtime/config 見
[實驗報告](experiments/REPORT.md) 與 [校正後機器結果](experiments/results_evaluation.json)。

初始 `results.json` 保留原 run；`evaluate_predictions` 只讀既有 predictions 並重新計算完整session-grid NW，輸出 no-clobber 的 `results_evaluation.json`／`REPORT.md`，不重訓。未知target日留在session grid，未知IC不會將時間間隔壓縮。


## Pilot09 W3 投組研究 runner

[投組設定、CLI、測試及真實動能樣本](experiments/PORTFOLIO_README.md)。
`portfolio.run_backtest(prices, predictions, Config(...))` 與 `python -m research_lab.backtest`，
重用既有V7 metrics/tier，全部RESEARCH_ONLY。close訊號→next-open entry→下一再平衡open退出，
不等同上述t+h close-exit label；不更動dataset或production契約。w3-portfolio UNMERGED / NOT_DEPLOYED。
