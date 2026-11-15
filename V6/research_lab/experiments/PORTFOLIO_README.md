# Pilot09 W3：排名投組模擬

`UNMERGED / NOT_DEPLOYED / RESEARCH_ONLY`。Yahoo 價格、150 個當前大型名稱（含 ETF）、
存活者與選樣偏差、沒有 fundamentals；任何收益都不是投資、strict PIT、paper/forward 或可成交證據。
此為新增 research-only runner，沒有改 production endpoint、scheduler、模型權重或既有 success gate。

## 範圍與重用決策

已搜尋 `experimental`、`marketmamba` 與 project map；沿用
`experimental/v7_portfolio_metrics.py` 的 `SessionRecord`、`compute_performance`、
`canonical_sha256`，不重寫其回撤／turnover／cost drag／benchmark 算式；沿用
`v7_evidence_tier.classify_evidence_tier`，強制維持 RESEARCH_ONLY。
既有 V7 engine 綁定 Decimal execution/journal/qualified quote contract；Yahoo panel
沒有這些 admission evidence，因此另以 `research_lab/portfolio.py` 做純離線 fractional-share
模擬，轉接 SessionRecord。`backtest.py` 僅負責 parquet、CLI、provenance 與基準分數。
既有 research_lab 沒有 experiment 目錄；新增此指定 README 與 gitignored
`.local-data/research/portfolio/` 承載研究設定和報告，不建立另一套 production authority。

實作順序：手算與邊界測試先失敗 → 純引擎 → CLI/合成 sanity → 真實基準 →
獨立唯讀 review → regression／hash／knowledge gate。未 commit，依使用者明確指示交付 source diff。

## 輸入、時間與估值

- 單次 run 只接一種 model/horizon；可合併多個不重疊 fold shards。欄位為
  `code(str),date,fold(int>=0),model(nonempty str),horizon(int>0),score(finite numeric)`。
  重疊 code/date、混合 model/horizon、錯誤型別均拒絕。分數只代表排名，不當成預期報酬。
- Calendar 為 dataset observed-date union 代理，不是官方交易曆。從最早 prediction date
  起每 N sessions 取 signal，於下一 session open 執行；最後一個 schedule entry 只做退出，
  不建立新組合。只評估完整開盤到開盤區間，最後不足 N 的尾段不計入。
- close t 訊號 → open t+1 入場；之後持到下一 entry open。dataset label 則是
  `close[t+h]/open[t+1]-1`，本投組 open-to-open 報酬與該 close-exit label **不相同**。
  horizon 是預測身份；`rebalance_sessions` 是獨立持有設定，不偷偷改 labels。
- 非 terminal 日按 close 估值；terminal 日按 open 退出，不讀 terminal close。
  訊號日已知的正價、正成交量與 outlier flags 決定 eligibility；最低流動性是
  **signal 當日 close × volume**（TWD 代理，非 rolling ADV）。不使用 entry 當日總量決定開盤交易。
- entry open 缺失／非正／非有限，或前一訊號日已無效，該名稱不能交易；原股數保留，
  不以低排名補位。停牌原因 UNKNOWN，不能由缺價證明。缺 close 時沿用最後有效 mark，
  `stale_position_sessions` 揭露股數×session 次數，stale NAV 不是已驗證市值。
- 保留不能賣的 frozen holdings，從可投資資產扣除其估值，再對餘額套權重；可動部分
  因此可能少於 K。最後不能退出者記錄 `unresolved_positions`，不得稱已實現報酬。
  缺失當日 prediction 的排程不沿用舊訊號，可交易持倉轉現金。
- 日誌含 daily NAV/cash、signal/entry schedule、每筆 signed notional/price；不強制末日平倉成功。
  Fractional shares、lot size 忽略；沒有漲跌停、開盤競價、impact、成交量 participation、
  最低手續費、個別 ETF 稅率、配息／分割／稅務逐檔差異或 broker fill 模型。

## 權重、成本與基準

- Top-K：score 由高到低，同分 code 字串升序。等權與 score-weight 都可設 cap。
  Score-weight 用 `(score-min(score))+epsilon` 轉非負權重，平分數等權；逐步 water-fill
  至 cap。`K*cap<1` 留現金。這是明確研究權重規則，並非將模型輸出校準成報酬。
  Frozen 與缺開盤價會使實際權重偏離目標 cap；cap 是新可投資部分的目標，非強制賣出保證。
- 假設值（不是現行法規或券商報價）：commission=.001425，discount=1，sell tax=.003，
  slippage=0 bps。買費=`buy*(commission*discount+slip/10000)`；
  賣費=`sell*(commission*discount+tax+slip/10000)`，無負成本，沒有最低費。
- 以扣成本後可投資 NAV 解自籌資金 target；sell proceeds 支應 buy，成本不重複扣。
  求解保留 32 ULP 的微小現金安全餘額，避免巨額 oracle NAV 的浮點尾差變成借款；
  並未放寬 insolvency gate。每次交易後重算現金與股數價值，無槓桿、無借券。
- Benchmark 是相同 signal eligibility/liquidity、相同 schedule/cost/missing policy 的
  當時可觀測 universe 等權，不限制至 strategy predictions 的 coverage。
  無歷史 constituent evidence，不能聲稱 survivorship-free。
  `--benchmark-code 0050` 可改單一名稱對照，仍受同一 schedule 與可交易政策，
  並非另做不含費／無條件成交的 benchmark。

## 指標與不確定性

NAV 從 1 開始；共用 metrics 的 total return 含初始買與 terminal 賣成本；
turnover=`(buy+sell)/2/previous recorded NAV`，初始全買與終止全賣各為約 .5。
`cost_drag` 是累計實付費用／初始 NAV，長期獲利時可大於 1，不是年化百分點。
`gross_minus_net_return` 用同設定免成本重播與淨結果相減，含複利、股數與路徑差異。

額外 statistics：252 sessions 年化 CAGR、sample-ddof=1 日波動率、mean daily return／vol
的 Sharpe-like（risk-free=0）、日正報酬 hit rate、tracking error、information ratio，
零波動分母回傳 null。daily tracking difference 與 cumulative excess return 由共用 metrics 定義。
年度分解包含前一年度最後 NAV 作基點；首尾可能部分年度。任意 subperiod 用 CLI
`--start/--end` 另跑，會重新從現金 1 開始；不是沿用原持倉的 attribution。

Bootstrap：seed17，1000 次、20-session circular moving blocks，同時抽 strategy/benchmark
配對日報酬，估計整段 compounded strategy return 減 benchmark return 的 percentile 95% CI。
不是扣除選模偏差、Universe bias、共同市場因子或多重比較後的顯著性；單一路徑不能證明 alpha。
最後一個報酬為 previous close 到 terminal open，其時間長度與其他日 close-to-close 不同。

## 重現與 artifact identity

在此 worktree root，以指定環境執行，無下載、無訓練、CPU only：

```bash
export PYTHONPATH=V6
PY=/home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python
ROOT=/home/frank/projects/ruflo-evaluation/pilot09/marketmamba/.local-data/research
$PY -m research_lab.backtest --prices "$ROOT/dataset.parquet" \
  --predictions "$ROOT/predictions/w3_momentum20_h5.parquet" \
  --out .local-data/research/portfolio/replay_momentum20_h5.json \
  --top-k 20 --rebalance-sessions 5 --min-dollar-volume 10000000 --weight-cap .1
$PY -m pytest V6/research_lab/tests \
  V6/experimental/v7_portfolio_metrics_test.py V6/experimental/v7_evidence_tier_test.py \
  V6/experimental/v7_executable_labels_test.py V6/experimental/v7_temporal_contract_test.py -q
```

不覆寫既有 result/prediction；重跑使用新的明確 artifact identity。使用新分數可傳多個
`--predictions a.parquet b.parquet`，但同次 run 的 model/horizon 必須一致。
Baseline 亦可由 CLI `--baseline momentum20 --baseline-model w3_momentum20 --prediction-out ...`
產生（選 `--baseline` 時不傳 `--predictions`）；外部寫入仍須符合 sandbox 授權。

生成的 w3 預測位於主 checkout 上述 predictions 目錄：`w3_momentum20_h5.parquet`、
`w3_random_keyed_s17_h5.parquet`、`w3_oracle_h5.parquet`；期間 2020-01-02～2026-10-30。
Momentum 是 observed-union close/close.shift(20)-1，只用截至 t 的價格，不接未來 labels。
Random 使用 SHA256(seed,date,code) 的固定均勻分數，只對當日 observed keys 發出分數；
未來新增上市名稱或延長 panel 不會改過去分數。Oracle 故意使用
`open[t+N+1]/open[t+1]-1`，只供證明時間對齊／排名方向的污染診斷，不能當研究候選。
三者 fold=0，無 fitting；非 walk-forward OOS model。其他模型必須自行符合 upstream
purge>=horizon 與 train-only fitting 契約；本工具不把 predictions 的 fold 欄當作已通過審計。

CLI 對讀入的同一份 bytes 計算 SHA256，不在跑完後才對可變共享路徑補 hash。
每份 JSON 記錄 dataset/manifest/prediction bytes identity、source HEAD、四個實作檔 SHA、
Python/numpy/pandas、CPU/seed/fold/support 與實際運算量；實作若在執行間變更則拒絕發布。
`config_sha256` 固定設定；`result_sha256` 是不含 volatile timing/path provenance 的
確定性 engine payload；外層 `canonical_sha256` 驗整份報告含 provenance，故執行時間不同會改變。
Dataset 為唯讀，未設定 MARKETMAMBA_DATA_ROOT，未搬動 Data/、venv 或 runtime。

第一次 generic momentum 路徑被其他 worker 於執行間替換；該次 JSON 的 input hash 與實際
讀入 bytes 不一致，**INVALID_PROVENANCE / NOT_CURRENT_EVIDENCE**。
`.local-data/research/portfolio/` 根目錄及 `cash_conservation/` 是保留的工程診斷產物，
不作本交付 current evidence；後者 momentum 與其他兩組期間不同，不能配對比較。
正式本地研究報告只採 `w3_pinned/`：獨立 w3 model identity、固定輸入 bytes 與修後 engine。

## 真實資料樣本（採用的 w3_pinned artifacts）

共同設定：Top20、equal、cap .1、每 5 sessions、signal dollar volume >=10,000,000，
commission .001425、discount1、sell tax .003、slippage0；seed17；1656 日報酬觀測，
2020-01-02 signal 起、2026-10-30 open 終止。未依結果調整 K、成本、門檻或區間。
Dataset canonical SHA（上游 manifest）
`325093035cf9eac936d4e151da26583aa90cb02c83af3be35a019e0c45c962b1`。
Source base `814002275d67e7d755b530fceb9285a227e3ba10`，w3-portfolio；實作是未提交檔案，
不能用該 commit 單獨重建，請保留本 diff 及報告記錄的逐檔 source hashes。

| 指標 | Momentum20 | Seed17 keyed random | 故意前視 oracle |
|---|---:|---:|---:|
| prediction rows | 241872 | 242177 | 242040 |
| 累積淨報酬 | 1409.8169% | 4.0251% | 1.0682e15% |
| 等權 universe 淨報酬 | 535.8942% | 535.8942% | 535.8942% |
| 年化報酬 | 51.1477% | 0.6023% | 9507.7388% |
| 年化波動 | 31.5129% | 24.5744% | 33.3708% |
| Sharpe-like | 1.4701 | 0.1483 | 13.9665 |
| 日 hit rate | 58.0918% | 54.8309% | 85.0242% |
| terminal unresolved positions | 0 | 0 | 0 |
| blocked held orders / stale stock-sessions | 15 / 132 | 6 / 97 | 3 / 74 |
| CPU 秒 | 20.5295 | 21.1391 | 20.8842 |

Momentum 最大回撤 46.6637%（2021-12-09 peak → 2022-10-26 trough），
日 mean turnover .0782666（含全部 non-rebalance sessions），cost_drag 2.7284302（初始NAV單位），
免成本累積報酬 3122.1296%，相對淨值成本影響包含長期複利差異。
Cumulative excess=873.9227 百分點；bootstrap 95% CI **[-33.3649%, 6927.3669%]**，
跨零且極寬，不能證明穩定超額。Tracking error16.9822%、IR .9110。
Random 免成本累積報酬 452.8042%，與 universe 535.8942% 同量級，頻繁替換 Top20 的費用
將淨結果壓至 4.0251%；單一 seed 不保證恰等於 benchmark，不能以這個檢查推導隨機策略有效。
Oracle 刻意觀察未來開盤，遠勝 benchmark 是污染 sanity，不是候選模型。

| 年度（含部分首尾年） | Momentum 淨報酬 | Benchmark 淨報酬 |
|---|---:|---:|
| 2020 | 29.0347% | 23.7872% |
| 2021 | 69.5978% | 42.5821% |
| 2022 | -39.5360% | -20.3952% |
| 2023 | 74.7445% | 46.4667% |
| 2024 | 25.8248% | 15.5811% |
| 2025 | 106.4778% | 34.8920% |
| 2026 至 terminal | 151.3376% | 98.1950% |

結果看起來很高，已檢查：momentum 公式不引用 labels；未來價格／未來新上市名稱擾動測試；
entry 不使用當日全日 volume；無同日 close fill；兩個資金守恆測試含巨額 NAV；
與 universe 的同 schedule/cost 配對；輸入與 source byte hashes。這只排除測到的程式前視，
不能排除 Yahoo revised vintage、選樣／存活者偏差或公司行動污染。2026 約九成 universe 報酬本身
也不能當作市場代表指數。Raw OHLC 未調整配息／拆股，不是 total return。

三個 JSON 在本 worktree gitignored `.local-data/research/portfolio/w3_pinned/`，分別同名
`w3_momentum20_h5.json`、`w3_random_keyed_s17_h5.json`、`w3_oracle_h5.json`；
有完整 daily/trades/year/CI/provenance。Config SHA256
`8c6d8a12810a7b451635a7e59cf127e2d8768ea75db45e16c373377ecc730fc6`。
Momentum result SHA256
`17e829f81bd6c0053133720ae365251dc480c642eba3491f1c39f4aa4064f277`。
三組採用 run 合計62.5528 CPU秒，另有一次獨立動能 replay。
全輪含合成測試、初期失敗與診斷重跑，CPU估計低於10分鐘，無逐程序完整CPU ledger；
未啟動GPU，未下載或安裝依賴，未消耗付費外部服務。

## 驗證、審查與剩餘限制

上述 pytest：**61 passed, 8 subtests passed**。包含手算雙邊費用／sell tax／slip、
初末 turnover、drawdown、schedule、benchmark、missing/suspended proxy、terminal blocked、
cash/cap、future-price invariance、baseline future-listing invariance、型別 fail closed、
CLI 多 shard/no-clobber/共享輸入替換 provenance、config/payload/metrics hash 與既有資料契約。
獨立唯讀 code review 發現 score 字串排序與 random 全期 shape 影響過去分數兩項，
皆新增失敗重現後修正；oracle 巨額 NAV 浮點資金守恆另有 RED→GREEN 測試。

此交付不訓練、不選模型、不接 API/UI、不提交、不 push/merge/deploy、不建立排程。
Ruflo memory 指令曾回報自動啟動 daemon，已立即 stop（回覆未在執行），不授權背景服務。
Runtime metadata 的既有 .claude-flow/.swarm dirty/untracked 保留；不以清工作樹為由刪除。

完成前驗證：三份採用報告的外層／engine／nested metrics hashes、輸入 bytes 及四檔
實作 hashes 全部核對；動能另一次獨立 real-data replay 逐結果 hash 相同，1657 NAV rows。
Syntax compileall 與 git diff --check 通過；knowledge impact 對 source PASS，對 canonical
navigation 的 governance review WARNING 已 ACKNOWLEDGED（治理規則未改，導航已更新）。
`knowledge_health.py --ci --json --as-of 2026-11-15` **FAIL**：基線既有 authority commit
`e3bce91` 等不在 Git objects；已確認 HEAD 的 authority-map.json 與 Features.md 原本就引用。
Project Map／Domains／Governance／Authority Drift FAIL，Agent Guides／Runtime Freshness／
Generated Index PASS，Legacy Docs WARNING。保留 gate，不重寫舊 authority。
完整機器輸出在 `w3_pinned/knowledge-health.json`、`knowledge-impact.json`。
整合前需先由協作主代理恢復缺少的原 authority objects，再重跑治理；本任務不下載或repin。

Knowledge impact：新增研究能力與本地結論 YES；production authority/runtime/correctness gate
變更 NO。Current_State、Authority_Map、Portfolio domain、package README 已同輪更新。
既有 .claude-flow policy／.swarm DB dirty，以及 memory CLI 產生的 runtime metadata 均保留。
Git 維持未提交／未stage，符合使用者 no-commit 指令；本worktree不宣稱clean。

共享 memory 已以 `w3-portfolio-result` 寫入 `pilot09-w3-portfolio`，另一次獨立 retrieve
可讀到完整摘要。CLI fallback sql.js 回報 Invalid PRAGMA，警告 durable persistence 不保證；
不安裝 native binding，請以本 README/source/JSON 為可攜交付。後續 memory 操作均設
`RUFLO_DAEMON_AUTOSTART=0`，最終 daemon status=STOPPED。
最終 governance 重跑保存在 `w3_pinned/knowledge-health-final.json`，同樣保留既有 FAIL。
