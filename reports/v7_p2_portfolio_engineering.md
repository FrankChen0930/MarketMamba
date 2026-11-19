# P2 低換手引擎：修訂 1 bounded 修正

2026-11-19；`job-run-mwnpp2vy-e42792`，builder，detached base `1920fde0abfead53e84a2a9354a695e59a03764c`。UNMERGED / NOT_DEPLOYED；E2 RESEARCH_ONLY；完成不等於 Human acceptance。依 [P2 登記修訂 1](../tasks/spec-lowturnover-p2-20261119.md)；本節 supersede 下方「修訂前歷史工程紀錄」的現行網格、基準成本與耗時估計，不修改原始歷史 artifacts。

## Changed / Decisions

重用既有 [engine](../V6/research_lab/portfolio.py)、[evaluator](../V6/research_lab/portfolio_eval.py)、[tests](../V6/research_lab/tests/test_p2.py) 與 `.artifacts/p2/`。獨立 namespace `.artifacts/p2/job-run-mwnpp2vy-e42792/` 是 GENERATED 工程產物，避免覆寫舊 evidence；未另建實作或 worktree。

- P2 的 `benchmark_proportional_only=True`：策略照收最低費用，主要基準只收佣金×折扣、賣出稅與滑價。`benchmark_min_fee_sensitivity` 另存基準也收最低費用的 statistics、diagnostics 與 daily_returns，標示 descriptive／confirmation_eligible=false；H-P2a／b 只讀主要報酬，不讀敏感度。P1 預設 False，保留原成本算式。
- OFAT 將 buffer 視為 top_k 的倍數；單獨改 top_k 時維持主要 2× 倍數，即 (30,60)、(100,200)。13 個 OFAT unique cells（含主要格）加三指定組合：(a) 資金 1e7、(b) 折扣 .28＋資金 1e7、(c) rebalance40。其中 (a)/(c) 與 OFAT 重複，去重後 **14 格／fold×system，六折兩系統 168 格，另 12 個 P1 reference，共 180 cells**。三組合都有 executable membership tests；最低費用基準敏感度內嵌於各 P2 格，不另計網格或假設。
- P2 所有格明確啟用 `preserve_frozen_overflow=True`，包括 buffer_rank=top_k；P1 預設 False。啟用緩衝帶時，若選股與凍結持股聯集超過 top_k，該次再平衡保留全部既有股數，不新增，也不強制賣出可成交持股；不收無委託費用。測試涵蓋 frozen A 與可成交 B、其他較高排名股；中間委託為空。terminal liquidation 保留原語義。預設 None／top_k 路徑保留 P1 行為。
- Holm 重用既有 canonical `significance.holm_adjust`。測試涵蓋不等 p 值、反序、一個過關一個不過及 cumulative max：(.03,.04)→(.06,.06)。整合案例 p=(1/1001,522/1001)→(2/1001,522/1001)，只有 H-P2a 過關。這是合成測試，不是本次研究結果。
- `--fold`／`--system` 任一子集必使 `grid_complete=false`；CLI 與 evaluator 均有測試。分割輸出具有各自 run_hash，不能直接拼成完整確認性報告或跨 namespace 續跑。
- **相對 P1，結果 schema 與 run_hash 已改變，舊 P1 parts 無法續跑；位元一致僅指預設路徑的逐日策略與基準報酬。** JSON `caveats` 也逐項明載，不聲稱新舊整份 JSON/config/hash 相同。

## Evidence：RED → GREEN

原始輸出保存在上述 namespace 的 `red.log`、`green-final.log`、`mutations-final.log`、`full-suite-final.log`。初始 **8 failed, 16 passed / 3.76s**；補強相等緩衝案例 RED **1 failed, 27 deselected / 1.47s**；最終 P2 focused **28 passed / 6.97s**。

| 項目 | RED 實際輸出／原因 | GREEN |
|---|---|---|
| 1 基準成本 | benchmark 設定缺失；另 mutation 恢復基準最低費用，逐日 NAV 比較 AssertionError | 主要基準逐日等於 min_fee0 對照，策略等於原最低費用策略；敏感度完整比對 |
| 2 網格 | frozen_grid：972 != 14 | 14 unique、主要格、指定三組合與 top_k 倍數測試通過 |
| 3 凍結持股 | 中間委託非空 | 中間委託為 []；只持有 A/B；terminal unresolved=1；相等 buffer 也通過，停用保護 mutation 被拒絕 |
| 4 Holm | 新測試原缺 helper；實際 mutation 缺 max 得 (.06,.04)、未校正得 1/1001，均 AssertionError | 3 組手算＋不等 p 的確認判定整合通過 |
| 5 子集 | evaluate 不接受 selection_partial | 實際 full-cell 合成子集及兩 CLI flags 通過，grid_complete=false |
| 6 caveats | P2_CAVEATS 缺失 AttributeError | schema/run_hash/P1 parts/位元一致範圍測試及兩 CLI JSON 通過 |
| 7 舊引擎對照 | 改比例成本算式的 mutation 使固定期望 hash AssertionError | 還原後 golden、既有預設一致測試及實際 P1 part 比對通過 |

固定合成面板來自 `fixture(31)`，29 日×策略與基準，費用／價格漂移／換股非零；期望由 **pre-P2 `ace2eef:V6/research_lab/portfolio.py` 原始引擎**產生，而非兩個新分支互比。舊程式 SHA-256 `dd53da1f4ced12cd7031afa9fd0112e742277eeba85d78168156d0401d93ea19`，little-endian interleaved IEEE754 returns SHA-256 `67d4f183fcc8124611be78fc508ed5a4db2d7b515468e7e574223260e479ee98`；重現來源 `legacy_hash.py`／`legacy_portfolio.py`。

另對 2023 P1 正式既有 part 唯讀比對：231 日、3,696 bytes，日期與兩種報酬逐位元相同。interleaved returns SHA-256 `0201d17d0f80f00179f0f4e6bf10d48ce39321bb670f430848cec15811be6d63`。下方歷史紀錄的 hash 採不同 bytes 排列；兩者不混用。`bitwise-final.json` 保存來源 part 路徑與 container SHA-256。

指定全套命令：

```bash
PYTHONPATH=V6:V6/research_lab/tests:. /home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python -m pytest V6/research_lab/tests tests -q
```

實際 **474 passed、1 skipped、4 warnings／90.94s**。skip 為既有案例；warnings 為既有 stale-audit mutation 的 pandas API deprecation。

## 2023 折 C0+ 實際冒煙

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=V6:V6/research_lab/tests:. /home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python \
  .artifacts/p2/job-run-mwnpp2vy-e42792/smoke-final-timed.py
```

wrapper 呼叫既有 CLI：`--mode p2 --scores-dir /home/frank/projects/MarketMamba/.artifacts/p1/full-scores-ace2eef --out .artifacts/p2/job-run-mwnpp2vy-e42792/smoke-final.json --primary-only --fold 2023 --system c0plus --run`。不重訓；第一次量測與先前全套保留為歷史 logs；本節數值與 hashes 全部對應最終程式。wrapper 只量測 `checked_cell` 耗時與 Linux ru_maxrss，不修改實作。

**109.417992s CLI／110.895806s 含啟動；1 part、2 COMPUTED cells、0 ERROR、grid_complete=false**。主要格 2023-01-03～2023-12-06、221 returns；P1 reference 到 12-20／231 returns，多出的 10 日列為未配對尾端。策略與前次一致，改變的是主要基準。

| 指標 | 實際值 |
|---|---:|
| 策略區間／年化淨報酬 | 25.522813%／29.589746% |
| 比例成本基準區間報酬（主要） | 33.948999% |
| 主要日均淨超額 | -0.000326306791 |
| 基準也收最低費用區間報酬（僅敏感度） | 13.862428% |
| 敏感度日均淨超額（不進確認性） | +0.000404358990 |
| 策略 Sharpe-like／最大回撤 | 3.169065／5.064570% |
| 年換手／gross-minus-net drag | 4.727433×／0.043674813 初始 NAV |
| 最低費用完整佣金／增額佔總成本 | 27.046362%／17.927639% |
| 平均／最大／終端持股數 | 49.549550／50／0 |

主要基準最低費用=0，blocked=4、stale=56、unresolved=1；不捏造可成交性。H-P2a CI [-0.000923187411,0.000379585805]、p_holm=.851148851；H-P2b CI [.000265073302,.000820580200]、p_holm=.001998002。**兩者都缺 2021／2022 且 primary-only/subset，均 INCOMPLETE_ENGINEERING_ONLY／未證實**。敏感度呈現的正超額不升格；已確認基準最低費用的評估影響、相同分數、signal/entry timing、universe 與尾端差異，既有 PIT/provenance 限制仍在。

## 完整執行估時與記憶體

`smoke-timing-final.json` 實測 P2 cell（含基準敏感度）9.182894s、P1 reference 11.158765s，其餘載入／hash／strata／出版／啟動合計 90.554148s。以下只是以 **2023/C0+ 單格實測外推**，未啟動完整網格或多程序研究。

- 單程序完整 180 cells：90.554 + 12×(14×9.182894+11.158765) = **1,767.186s，約 29.45 分鐘**。一次共用輸入開銷。
- 以 `--fold YEAR --system SYSTEM` 拆成 12 個獨立 job、不同 `--out` namespace，**最多 3 個程序同時、每程序單執行緒**，含每 job 重複載入成本：(12×90.554+12×(14×9.182894+11.158765))/3 = **921.094s，約 15.35 分鐘**的理想排程估計。可依 6 cores 分工，但此機記憶體先限制並行數。
- 每程序峰值 RSS **2,757,720 KiB＝2.629967 GiB**。加 20% 裕度是 **3.155960 GiB／程序**，三程序約 9.47 GiB，另保留 2 GiB 給系統，共約 11.47 GiB。六程序光實測峰值已 15.78 GiB，超過約 12GB 預算；不以六核推定可六程序。
- 此為第一個小型實測的線性估計；其他年份／系統、top_k、再平衡、費用與 I/O contention、完整 results 累積 RSS 都尚未實測。不是完成時間或記憶體上界；完整執行前仍需主控監測。分割 job 各自永久 grid_complete=false；沒有實作跨 run_hash 確認性彙整器，不能把子集冒煙當完整研究。

## Artifact／runtime 與 handoff

`smoke-final.json` SHA-256 `77acb7d28f6337345c0552d1ef08a8cec261e3db2d5aead1d711212679ee6394`；part、輸入與所有 implementation hashes 在 report/part；`verification-final.json` 彙整本輪 artifact identities、驗證與限制。價格 hash `bef2cd29bad60448cb6ea029788190deabb9286804c35d5128ac081d97a392de`；scores `9ea91f369ea8105dd5652cb18bd270c0a81e259a9ce1587fc8aa45822a51ea21`；market value `665431f4aa31131bdfc70f08d54469f1e47d929afc9c35f592419bd39e8acb09`。價格 container identity 與更早 P1 原始 run 不同；本輪只聲稱指定 2023 逐日報酬比較一致。

Python 3.12.3 pilot09 venv、CPU、單執行緒，MARKETMAMBA_DATA_ROOT unset；本 worktree Data 普通目錄，實際唯讀資料 realpath `/mnt/d/Desktop/work/ProjectForMe/MarketMamba/Data/processed_v6`。未讀 .env、未碰 protected V6/models／formal Data、未 GPU／acquisition／production／scheduler／wakeups／recurring jobs／新增 agents。

Completed：七項 bounded 修正。Changed：engine/evaluator/P2 tests、本報告、Current State／Authority Map／Worktree Register 與 Portfolio／Research domain notes。Decisions：以修訂 1 為 authority，既有 P1／歷史 artifacts 不 supersede。Evidence：上述 RED/GREEN／指定全套／實際冒煙與 hashes。Remaining：完整 168 格＋12 references、跨折確認性研究、獨立審查與 Human acceptance。Blockers：本輪工程 N/A；完整證據尚未執行。Next recommended action：主控獨立檢查 diff/artifacts 後決定完整執行。Project memory updated：是。Authority map updated：是。Git status：起始 clean，目前僅核准範圍 unstaged 修改；依禁止 commit 指示保留 dirty，不能宣稱 clean。Commit／Push：均無。Worktree：沿用上述 detached worker，保留未提交成果。Runtime/environment：如上。knowledge_update: required（基準成本、網格與 frozen gate 語義變更），治理結果見 verification。

治理實測：`check_knowledge_impact.py` 已執行並依語義變更更新 maps/domain notes；`knowledge_health.py --ci` exit 0、Overall WARNING，Project Map／Agent Guides／Domains／Runtime Freshness／Generated Index PASS，Governance／Authority Drift／Legacy Docs WARNING，無 structural FAIL。syntax compileall、link/reference 與 `git diff --check` 通過。依 code-review-and-quality 五軸自行檢查，未呼叫額外代理；獨立審查留主控。

---

# 修訂前歷史工程紀錄（以下不是現行網格／成本 authority）

# P2 低換手引擎：bounded 工程驗證

2026-11-19；job `job-run-mwnp15vx-b42d8c`，builder，base `fec03f7f19fa71fa5a4876bd02cfc59f23955cff`。
UNMERGED / NOT_DEPLOYED；E2 RESEARCH_ONLY；完成不等於 Human acceptance。

## 範圍與重用

依 [frozen P2 spec](../tasks/spec-lowturnover-p2-20261119.md)，直接擴充 [portfolio.py](../V6/research_lab/portfolio.py) 與 [portfolio_eval.py](../V6/research_lab/portfolio_eval.py)。沿用既有 `_simulate` 的 signal-close／next-observed-open schedule、缺價封鎖、terminal open liquidation、60 次成本二分求解、`_weights` 等權／score 權重與 P1 fold×system atomic parts／hash-bound resume／immutable ERROR retry。未另建回測框架、worktree、資料／訓練流程；P2 隔離 namespace 是 spec 明定的產物邊界，不能寫回 frozen P1 namespace。

初始 cash=NAV=1。`old`、`target` 與 `delta` 都是初始 NAV 單位的價值，`delta` 已包含當時 NAV 漂移；TWD 成交值 = abs(delta) × initial_capital，等價於 abs(權重變動) × 當時 NAV × initial_capital。佣金在此單位為 max(min_fee / initial_capital, discounted_commission × abs(delta))，逐筆算、再加賣出稅與滑價。不能重複乘 NAV。`min_fee=0` 保留舊浮點成本路徑；未發生委託時不因成本求解而創造最低費用。委託門檻沿用既有 abs(delta)>1e-13。最低費用啟用時要求 initial_capital 至少 TWD 10,000，是明示的研究輸入 guard，不代表券商最低開戶額或整股可執行性。

`hold_buffer_rank=None` 或 top_k 走原始選股路徑。較大時先從 signal-day、符合原 liquidity／scored-universe 條件的穩定排名中保留既有持股，再以未持有最高排名補足 top_k；同分以 code 排序。沒有 entry-day／future ranks。blocked positions 延續原引擎凍結語義，可能使實際總持股數超過目標，不虛構可成交性。

`--mode p2` 啟用固定主要格與每 fold×system 972 格描述性網格，六折兩系統共 11,664 格，另計 12 個同分數 P1 reference；LightGBM 僅描述性。預設仍為 P1。P2 無額外 market-cap strata 網格。每個 part 保存 input/spec/implementation SHA-256、source commit、run hash、payload hash、fold/system；斷點以完整 fold×system 為單位，尚未發布的該 part 需重算。H-P2a/b 使用 20 日 circular blocks、1000 samples、95% CI；one-sided centered-bootstrap p values（plus-one）做 Holm family=2 校正。H-P2b 配對共同日期，明列尾端排除日期，不補零、不接跨折 NAV。primary-only 或缺乾淨折支援均未證實。

## 實際測試

指定命令：

```bash
PYTHONPATH=V6:V6/research_lab/tests:. /home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python -m pytest V6/research_lab/tests tests -q
```

**462 passed、1 skipped、4 warnings，114.84s**。skip 為原有案例；warnings 是既有 stale-audit mutation 的 pandas deprecation。P2 focused tests **16 passed / 3.11s**；先寫測試的 RED 為 **13 failed / 3.60s**。其中包含 default daily/trades exact equality、無 phantom orders、手算成本、NAV drift／capital 換算、hash resume／corruption／changed-input 拒絕、Holm 與 partial-support gate。完整 logs 與 mutation driver 存於本 worktree ignored `.artifacts/p2/job-run-mwnp15vx-b42d8c/`。

四個變異在獨立 Python process 內以 `exec(compile(...), module.__dict__)` 暫時替換，不寫回 source；每個指定測試 exit=1，驗證後 source SHA 未變：

| Mutation | 實際失敗輸出摘要 | 結果 |
|---|---|---|
| 緩衝忽略 | `E AssertionError: assert {'A', 'B'} == {'A'}` | `1 failed in 1.29s` |
| 最低費用忽略 | `E assert 1.0 == 0.99996 ± 1.0e-12` | `1 failed in 1.44s` |
| 資金換算錯誤（最低費用除錯資金） | `E assert 1.999969999999993 == 1.99994 ± 1.0e-12` | `1 failed in 1.37s` |
| 使用未來 entry-day 排名 | `E Right contains 2 more items ... 'date': '2024-01-04', 'signal': '2024-01-03', 'code': 'A'` | `1 failed in 2.67s` |

完整輸出為 `buffer_ignored.log`、`minimum_ignored.log`、`capital_wrong.log`、`future_ranks.log`，規格及 source hash 在 `mutations.json`。

## 位元一致證明

2023／c0plus，主要格 top50／10 sessions／DV 2e7／discount1／min_fee0／10bp／raw，以唯讀 `.artifacts/p1/full-scores-ace2eef/rank_average.parquet` 重算。逐日比對指定舊 part `/home/frank/projects/MarketMamba/.artifacts/p1/eval-full-ace2eef/part-2023-00cade6c97b7f1d1.json`（檔案 SHA-256 `cd9c56df4dd91733942737968eac064a24b8e5e6ae2f8e73ca83d2a33aa4df13`）。日期逐筆相同；231 日 net＋benchmark、462 個 IEEE754 binary64、3,696 bytes 完全一致，兩邊報酬位元 SHA-256 均為：

`359ec9f2f53fdac2b4498223f1022dd173604d1dd03b49fa0ffc154fa7b42b00`

涵蓋 2023-01-03～2023-12-20，P1 total return 9.794800310309792%。此證明針對 daily returns；Config 新欄位與診斷使整份新 JSON/hash 合理不同。舊 P1 報告價格 container SHA 為 `5df865ee9d57acf4d77de4e716ec3e2dd94dc8afae1869ec8adfdcb0bd10fc14`；本次現有唯讀容器已為下列新 hash，因此不聲稱完整輸入檔同一 identity，也不推測哪些其他期間變動。2023 指定格的實際報酬一致證明如上。`compare_p1.py`／`bitwise.json` 保存重現方法與結果。

## 2023／C0+ 主要格冒煙

```bash
PYTHONPATH=V6:V6/research_lab/tests:. /home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python -m research_lab.portfolio_eval \
  --mode p2 --scores-dir /home/frank/projects/MarketMamba/.artifacts/p1/full-scores-ace2eef \
  --out .artifacts/p2/job-run-mwnp15vx-b42d8c/smoke.json \
  --primary-only --fold 2023 --system c0plus --run
```

實際 **118.956s**，1 part／2 COMPUTED cells（P2 與 P1 reference）／0 ERROR。P2 2023-01-03～2023-12-06，221 returns；P1 reference 另有 10 個尾端報酬日，H-P2b 明列排除。

| 指標 | P2 單折 |
|---|---:|
| 區間淨報酬 | 25.522813% |
| 年化淨報酬 | 29.589746% |
| 同宇宙基準區間淨報酬 | 13.862428% |
| 日均淨超額 | 0.000404359 |
| Sharpe-like | 3.169065 |
| 最大回撤 | 5.064570% |
| 年換手 | 4.727433× |
| gross-minus-net drag | 0.043674813 初始 NAV |
| 最低費用佔總成本 | 27.046362% |
| 最低費用增額佔總成本 | 17.927639% |
| 持股平均／最大／terminal | 49.54955／50／0 |

「最低費用佔總成本」是 floor-binding 委託的**完整佣金**／總佣金＋稅＋滑價；另外的 uplift 才是最低費用額外增加部分。benchmark 也逐委託收費，其最低費用為 0.18634 初始 NAV，佔 benchmark 總成本 **90.809766%**，增額比 **87.752217%**；其 blocked=4、stale position sessions=56、terminal unresolved=1。策略改善不能只從這個淨對淨超額推論：廣泛等權基準在 TWD 1m 下因小額委託承受很高的最低費用，且不是可驗證的整股策略。策略 terminal unresolved=0。已檢查 signal/entry timing、同分數 identity、future-rank mutation、window／tail／universe 與基準成本；NON_PIT 市值、PIT/provenance／下市覆蓋限制仍在。

單折 H-P2a CI [-0.000191415, 0.001112998]，H-P2b CI [0.000265073, 0.000820580]；缺 2021／2022，兩者均 `INCOMPLETE_ENGINEERING_ONLY`／未證實，不能因單折 H-P2b 漂亮而升格。

## 輸入、環境與限制

Python 3.12.3，pilot09 venv，CPU，MARKETMAMBA_DATA_ROOT unset；本 worktree `Data/` 是普通目錄，實際 read-only data root realpath 為 `/mnt/d/Desktop/work/ProjectForMe/MarketMamba/Data/processed_v6`。原始 `.env` 未讀取、無 GPU／網路／資料 acquisition／protected model／production／scheduler 改動。source commit 與 package/environment identity 在 `smoke.json` provenance。

- scores SHA-256：`9ea91f369ea8105dd5652cb18bd270c0a81e259a9ce1587fc8aa45822a51ea21`。
- score manifest SHA-256：`dc3b9b3ad9c2c27a5366440c651f5f6baa7daaec9854368a0e944874890be9d8`。
- prices SHA-256：`bef2cd29bad60448cb6ea029788190deabb9286804c35d5128ac081d97a392de`。
- market value SHA-256：`665431f4aa31131bdfc70f08d54469f1e47d929afc9c35f592419bd39e8acb09`。
- smoke report SHA-256：`304575682562caa572a5e0755b2ddb6eb78fd24fb97ef5595d7ce205f3372731`。

P1 全部限制保留：無 adj_close、下市覆蓋未驗證、NON_PIT 市值、無漲跌停旗標、observed calendar、simulation labels。最低費用只是未驗證研究假設，fractional shares／整股／零股未建模，資金不代表使用者實際資產。完整 P2 網格、跨折確認性研究與 Human acceptance 未執行。單 part 可能耗時很長，尚未發布的 fold×system 不具 cell 級恢復。

## 續跑、耗時估計與治理檢查

用相同參數加 `--retry-errors` 實際再次執行，得到 `SKIP_HASH_MATCH`（1 part，0 cell 重算），保留原始報告，另存 `smoke-retry-000001.json`，耗時 **87.119s**。該 part 未寫回，最後修改時間早於初次報告；payload 與初次報告的 results／primary／P1 reference 完全相符。重試報告 SHA-256 `87127c888e5c54b64405e13bf20ab98b777b1a5c2c5ae4d78d0f904857c31a64`。這是實際正常續跑證據，不聲稱本輪另做了 SIGKILL／斷電試驗；P1 既有 kill/restart tests 已在指定全套中執行，P2 特有模式另有 hash resume／corruption regression。

完整網格估算：首次 118.956s 減去無重算續跑的 87.119s 約 31.837s／2 cells。若沿用此均速，(11,664+12 references) × 15.918s ≈ **51.6 小時**單程序 CPU，另加一次輸入驗證與寫出。這只是本機初估，首次與續跑負載不同（首次同時跑測試），不同 k／rebalance／min_fee 的速度也不同；不作完成時間承諾。若把首次全部開銷都逐 cell 線性外推，保守上界估算約 **193 小時**。舊 P1 全網格實測約 7.36h，但其 cells／成本路徑不同，不可拿來承諾 P2。未啟動完整網格、背景常駐程序或任何 recurring job。主控應先根據資源時窗安排執行，長 fold×system 中斷會重做該 part。

`py_compile`、`git diff --check` 通過；`check_knowledge_impact.py` 為 WARNING／REVIEW_REQUIRED，已人工檢查並更新 Current State、Authority Map、Portfolio／Research domain note；GOVERNANCE artifact／authority 規則不變。`knowledge_health.py --ci` exit 0，Overall WARNING（既有 Governance／Authority Drift／Legacy Docs），Project Map／Agent Guides／Domains／Runtime Freshness／Generated Index 均 PASS，無 structural FAIL。此 change 不 supersede canonical research evidence。

## Handoff

- Completed：P2 實作、四項 in-memory mutation、預設位元證明、指定 full tests、2023/C0+ 冒煙與正常續跑。
- Changed：兩個 implementation、`V6/research_lab/tests/test_p2.py`、本報告、Current State／Authority Map／Portfolio／Research，共 8 個 paths。
- Decisions：重用 P1 accounting／durability；TWD guard=10,000；共同日期配對、minimum-fee full commission 與 uplift 分列，所有額外方法均明示。
- Evidence：ignored `.artifacts/p2/job-run-mwnp15vx-b42d8c/` 的 logs、JSON、scripts 與 hashes。
- Remaining：完整 6-fold×2-system 網格、跨折確認性結論、獨立審查與 Human acceptance。
- Blockers：bounded 工程範圍無；資料／PIT／交易限制如上，完整研究不是已完成。
- Next recommended action：審查 diff／verification artifacts；由主控安排完整執行，禁止從單折升格。
- Project memory updated：Current State、Portfolio／Research notes；Authority map updated：YES。
- Git status：起始 clean detached HEAD，結束保留 6 modified＋2 untracked，無 staged；Human 明令不 commit，故不宣稱 clean。
- Commit／Push／Merge：未執行。
- Worktree：`/home/frank/projects/MarketMamba/.worktrees/ruflo-workers/run-mwnp15vx-b42d8c/builder`，既有隔離 worktree，未建立／移除其他 worktrees。
- Runtime/environment：上述 Python 3.12.3／CPU／data realpath；無 production／scheduler／GPU／外部 acquisition。
