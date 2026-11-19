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
