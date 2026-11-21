# P4a Colab 工程交接 — 2026-11-21

UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED；DRAFT研究契約；本報告不是Human acceptance，沒有研究結果。Source base `e4f21c3747124f4c6cc46a873eae891697910067`，既有detached工作樹 `.worktrees/ruflo-workers/run-mwqd0vuy-7d9826/builder_p4a`，起始clean，依明確指示不commit／push／merge／新增agents。

## Changed / Decisions

沿用 `V6/research_lab/seq_runner.py` 的 `train_one`、SeqData／ensure_rankgauss，以及P3的checked_splits／require_same_rows；P3 collect綁定其凍結全域預算／GPU准入，不適合直接接受P4的CU預算，因此只新增P4薄adapter，沒有重造訓練loop。`seq_models.py`沒有GRU，新增width64兩層nn.GRU並共用embedding／RMSNorm／雙head；訓練器新增可選observer，原呼叫預設不變。P3凍結配方、spec、ledger／admission及completed artifacts均不修改。因既有provenance會hash全部research_lab source，新增檔案改變新checkout的source identity；既有P3必須保留原pinned source，不能重標研究結果。

新增 `p4_bundle.py` 承載source/data/lock封裝與標準庫safe-extract，`p4_colab.py`承載P4獨立Drive帳本／speed-test／durable fit。既有P3工具不具備這兩個跨runtime介面；沿用tasks、docs/operations、environments、reports與既有test tree，沒有新增top-level directory／平行系統／worktree。新增 `environments/p4a-colab.requirements.lock.txt` 是因既有2026.04 lock屬舊E1，不能代表本次指定環境；它與 `/home/frank/.venvs/marketmamba-gpu.requirements.lock.txt` 位元一致，SHA-256 `2c39a3fb9567d1f9fc8ee83369c7b6c90cd19a8e679e32e4c8a6c1d75c5782c9`。

Bundle包含9個原始matrix檔、P1 rank_average及manifest、source commit／dirty payload hashes、原樣exact lock、size/SHA inventory；不含rankgauss。解壓只接受regular file，拒絕絕對／穿越／symlink／hardlink／duplicate及SHA不符，不覆寫destination。derived cache以既有ensure_rankgauss重建，另驗完整input/source/output hash與tag。

安裝先查 `archive/notebooks/V5_5_Training.py` Cell1（舊1.6.0／2.3.0 wheels）及 `archive/MarketMamba_V3.ipynb` Cell1（Drive wheel），採其預編譯wheel方法，但用exact lock內的2.10.0+cu128／2.3.2.post1／1.7.0及URL SHA。只產生安裝指令，無網路／安裝；only-binary／cp312／ABI／版本不符STOP，不發明編譯流程。

Speed-test首折seed17三臂各2epoch×100步，逐步秒數／epoch秒數／GPU型號／溫度／時脈／功率／總wall JSON，ENGINEERING_SPEED_ONLY。正式DRAFT拒絕；凍結後依bundle完整fold×arm×seed執行。fit每epoch保留model/opt/sched/RNG/best_sd generation及latest/previous/best，完成後另存BEST權重。先local serialize/fsync/hash，再Drive exclusive copy/fsync/reverify，最後atomic pointer；complete只在predictions／metrics／checkpoints持久驗證後發布。完成fit重驗contract/seed/matrix/source/environment與全部output hashes才跳過；partial不自動resume／升格orphan，未知中斷保守扣到restart並STOP。

預算根目錄跨speed/formal共用，CU採Human確認的保守費率估算而非provider實扣；wall-hours及CU雙上限，剩餘完整grid與rankgauss stages加至少25%餘裕，執行中再驗。Bundle／runtime驗證及rankgauss前處理都有獨立START/END扣帳，零／未知費率與預估拒絕；上傳、pip及人工閒置須Human另對照Colab UI核計。Guard wait不計active訓練扣帳；Colab溫度>=85°C直接STOP，不等待／暫停，因此wait=0。缺／回退／corrupt ledger或head一律STOP，無自動clear/rebuild介面。

## Evidence

CPU Python3.12.3，`/home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python`，torch2.15.0+cpu、CUDA=False，`MARKETMAMBA_DATA_ROOT` unset；本worktree `Data/` 為普通目錄，realpath即此worktree/Data。既有canonical開發資料 `/home/frank/projects/MarketMamba/Data/processed_v6` 解析至 `/mnt/d/Desktop/work/ProjectForMe/MarketMamba/Data/processed_v6`；本次沒有讀寫formal matrix／價格／模型payload。新測試的matrix與bundle均合成於pytest `/tmp`；既有全套測試會建立並清理自己的合成P3 namespace，不代表碰觸已完成P3資料。

RED：初始P4 module缺少，pytest collection error；加入實作後budget案例11pass/1fail，定位為測試算式9.625小時未超10，改用真正超額10.125小時。新增契約／整grid gate先2fail（缺helper）再GREEN。SHA變異測試先驗損毀被拒，再停用digest比較，損毀回歸的raises斷言確實失敗。補強前一輪P4 focused輸出：

```text
.........................                                                [100%]
25 passed in 7.32s
```

第一次全套（執行時仍修改source）實際輸出：

```text
FAILED V6/research_lab/tests/test_p3.py::test_revision2_p3_main_complete_revalidated_cpu_fixture[False]
1 failed, 598 passed, 1 skipped, 4 warnings in 161.02s (0:02:41)
RuntimeError: STOP P3 admission changed during evaluation
```

根因是本worker在測試期間修改P4 `.py`，P3前後hash保護如預期拒絕，沒有放寬或修改該gate。已固定source重跑；最終完整輸出見報告末段。完整命令：

```bash
PYTHONPATH=V6:V6/research_lab/tests:. /home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python -m pytest V6/research_lab/tests tests -q
```

Knowledge-impact逐項：semantics YES（P4新增介面／GRU）；authority NO（只有DRAFT candidate）；subsystem boundary YES（bundle／Colab薄adapter）；active runtime NO；correctness gate YES（P4新guard，P3不變）；canonical artifact NO（未接受候選）；deprecation NO；next decision YES（凍結／GPU預算）。因此同change更新Current_State、Authority_Map、Training domain與Worktree_Register。Impact工具對新增research_lab多為DEFAULT，不能取代人工YES判斷。

## Remaining / Blockers / Next recommended action

未驗證：真實大型bundle／formal matrices、實際GPU速度／熱／kernel數值、Colab版本與wheel可用性／安裝／網路、Drive FUSE fsync／斷線／持久性、provider實扣CU、正式研究grid／portfolio evaluation／任何結論。CPU stub不能當官方Mamba kernel證據；CPU round-trip沒有宣稱任意不可信pickle安全。只有可信、驗hash的本地payload才可載入torch checkpoint。

DRAFT需Human／Claude決定等預算調參family、混合描述臂、P3缺2026三次是否獨立補跑及P4完整config／CU費率／總額。STOP後的人工recovery／rebuild未提供，必須保存inventory後明確修復，不能刪檔／換root重置。獨立review依本輪禁止新增agents而未執行；本報告只有作者自查與executable evidence。下一步為主控獨立審查diff／logs，再現場GPU／wheel／Drive驗證及研究凍結。

Project memory updated：YES；Authority map updated：YES，僅candidate事實與限制。Git status：起始clean，結束依no-commit保留核准範圍dirty/untracked，非clean；Commit：N/A；Push：未執行；Worktree：沿用；Runtime/environment：上述CPU環境。工程完成不等於Human acceptance。

## 最終候選補強與變更路徑

額外離線自含性檢查曾因pytest-current移動symlink選錯fixture路徑而失敗，改用固定realpath後，辨識出真正的正常Python bytecode cache會造成inventory誤拒問題；新增失敗測試後，runtime verify僅辨識對應受hash保護source的`__pycache__/*.pyc`，仍拒絕額外`.py`與所有links；tar輸入本身仍只依exact manifest。操作命令另設定`PYTHONDONTWRITEBYTECODE=1`。最新focused **31 passed in 11.69s**。

CPU orchestration測試以GPU／mount／fit邊界替身，實際走pack/extract、rankgauss、START/END帳本、exclusive publication及completed-fit重啟。驗證setup與rankgauss都有獨立扣帳，第二次仍驗輸入但不重訓已完成三個fit。另新增4組RED→GREEN證明零／負／NaN費率或零時間估計不能入場。這不證明實際mount或Mamba kernel行為。

隔離子程序從 `/tmp/pytest-of-frank/pytest-12/test_cpu_orchestration_charges0/unpacked/source` 載入bundle source，cwd=`/tmp`，CLI `--help` exit0；正式CLI CPU執行 exit1，實際末行：

```text
RuntimeError: CUDA required; CPU runs cannot launch P4 speed/formal training
```

合成bundle逐檔inventory **239 files**，bundle-manifest **40,331 bytes**，SHA-256 `a938d96cedd615b52604a689ccaa91f791eb12bef44c7699a4b593282d779e60`，source_commit仍為本輪base，但payload含未提交source hashes；不是formal data bundle或研究evidence。GRU 48features/width64/2layers實際參數量 **53,250**；官方Mamba2參數量／GPU效能尚未現場量測。

Changed paths（14個，均本次核准範圍）：

- `V6/research_lab/p4_bundle.py`
- `V6/research_lab/p4_colab.py`
- `V6/research_lab/seq_models.py`
- `V6/research_lab/seq_runner.py`
- `V6/research_lab/tests/test_p4.py`
- `environments/p4a-colab.requirements.lock.txt`
- `tasks/p4a-colab-contract-20261121.json`
- `tasks/spec-p4a-colab-20261121.md`
- `docs/operations/colab-p4.md`
- `reports/v7_p4a_colab_engineering.md`
- `knowledge/00_Project_Map/Current_State.md`：新增P4候選狀態、限制、下一步，保留P3結果。
- `knowledge/00_Project_Map/Authority_Map.md`：新增DRAFT／UNMERGED邊界與獨立lock，不supersede歷史authority。
- `knowledge/00_Project_Map/Worktree_Register.md`：記錄既有worker目的／base／owner／保留條件。
- `knowledge/01_Domains/Training.md`：記錄P4bundle／checkpoint／預算語義與限制。

所有8項knowledge-impact判斷均已在上方列出，memory更新為同change必要內容；不是NO_KNOWLEDGE_CHANGE_REQUIRED。既有Runtime freshness沒有新增或刷新外部observation。

## 最終驗證（固定source）

指定全套命令exit0，既有577個通過案例＋31個P4案例＝608 passed；1 skipped及4個Pandas deprecation warnings保留，沒有停用或修改既有測試。實際完整stdout/stderr如下：

```text
........................................................................ [ 11%]
........................................................................ [ 23%]
........................................................................ [ 35%]
........................................................................ [ 47%]
........................................................................ [ 59%]
........................................................................ [ 70%]
........................................................................ [ 82%]
...........................................s............................ [ 94%]
.................................                                        [100%]
=============================== warnings summary ===============================
test_stale_audit.py::test_mutation_controls[sessions = sorted(set(pd.to_datetime(trading_dates).strftime("%Y-%m-%d")))-sessions = sorted(d for d in daily if pd.Timestamp(d).dayofweek < 5)]
test_stale_audit.py::test_mutation_controls[sessions = sorted(set(pd.to_datetime(trading_dates).strftime("%Y-%m-%d")))-sessions = sorted(d for d in daily if pd.Timestamp(d).dayofweek < 5)]
test_stale_audit.py::test_mutation_controls[sessions = sorted(set(pd.to_datetime(trading_dates).strftime("%Y-%m-%d")))-sessions = sorted(d for d in daily if pd.Timestamp(d).dayofweek < 5)]
test_stale_audit.py::test_mutation_controls[sessions = sorted(set(pd.to_datetime(trading_dates).strftime("%Y-%m-%d")))-sessions = sorted(d for d in daily if pd.Timestamp(d).dayofweek < 5)]
  /tmp/pytest-of-frank/pytest-13/test_mutation_controls_session0/mutant.py:93: Pandas4Warning: Timestamp.dayofweek is deprecated and will be removed in a future version. Use Timestamp.day_of_week instead.
    sessions = sorted(d for d in daily if pd.Timestamp(d).dayofweek < 5)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
608 passed, 1 skipped, 4 warnings in 164.16s (0:02:44)
```

另外6項手動合成反證（`/tmp/p4-pointer-checks.json`）：checkpoint complete round-trip、corrupt pointer拒絕、missing pointer拒絕、matrix mismatch拒絕、seed isolation、orphan保留而不升格，6/6通過。全部在獨立tmp副本上執行，沒有修改歷史artifact。

最終knowledge-impact：WARNING／14 changed paths；已人工檢查提示的Operations及GOVERNANCE，operation／scheduler／artifact治理classification未改，P4新語義記錄於Training、Current State、Authority Map與Worktree Register，不需變更Operations／GOVERNANCE。不是無知識影響的acknowledgment。`knowledge_health.py --ci --as-of 2026-11-21` exit0，實際輸出：

```text
KNOWLEDGE HEALTH

Project Map        PASS
Agent Guides       PASS
Domains            PASS
Governance         WARNING
Authority Drift    WARNING
Runtime Freshness  PASS
Generated Index    PASS
Legacy Docs        WARNING

Overall: WARNING
```

syntax `py_compile`、JSON parse／executable contract檢查、25個P4相關link references、原始／repo exact lock逐位元SHA比對、`git diff --check`均通過。最後仍為detached HEAD、6個tracked修改及8個untracked新檔、無staged，因no-commit要求保留dirty，未宣稱clean。沒有commit／push／merge／deploy／scheduler／GPU／Colab／network／正式資料或權重變更。

受核對source/config/doc identity（GENERATED inventory `/tmp/p4-source-hashes.json`）：

| Path | Bytes | SHA-256 |
|---|---:|---|
| `V6/research_lab/p4_bundle.py` | 9303 | `d96ea2bd1d45bc553c8766bf1741a752318681fd55f0147b5c0e50216084cce9` |
| `V6/research_lab/p4_colab.py` | 22942 | `5d7854a4b86dd32840dae18d44aa0aac3e9946c9f01c9fa95e51af204c175785` |
| `V6/research_lab/seq_models.py` | 7341 | `54361cd3fb5bceafdba8d3945bc6ae4f8d2684c42b3b70f99211a39a001eea74` |
| `V6/research_lab/seq_runner.py` | 21984 | `dbeb048ca0015a04c92191d827000f90ea9f22da17c3fb60023e08b023052703` |
| `V6/research_lab/tests/test_p4.py` | 14538 | `b1ccd571b3134101eee2d7300ec1d40883d3fe7a1d7ce03384f5becf7ca8a1f2` |
| `tasks/p4a-colab-contract-20261121.json` | 2732 | `991940f72d7a83b856018d74854fdd3e2e04a4a2f80f89aa532036e237bdc172` |
| `tasks/spec-p4a-colab-20261121.md` | 3915 | `1d78b63a403576764bf848db489c805edde4a9177869131e2e0716b63853d987` |
| `docs/operations/colab-p4.md` | 8193 | `9a1862bce999bdb83aa4106a4b5a07fe2cc18587baff36b1850e47c8876dc3c8` |
| `environments/p4a-colab.requirements.lock.txt` | 2659 | `2c39a3fb9567d1f9fc8ee83369c7b6c90cd19a8e679e32e4c8a6c1d75c5782c9` |


## 審查修補 — job-run-mwqe7o12-d33ef8（2026-11-21）

本節追加本次修補事實；上文屬原候選歷史，不以其未提交／cache 例外敘述代表現況。起始 detached HEAD `8cb61c48b2b302338600a91b3ad59dc350d09a6c`、clean；原工程候選已提交，本次依 Human 指令不 commit／push／merge。既有 worktree `fixer_p4a`，UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED，仍無正式 P4 結果。

Changed：`p4_bundle.py`、`p4_colab.py`、`tests/test_p4.py`、P4 DRAFT spec、Colab operations guide、Current State／Authority Map／Worktree Register／Training 與本報告。`seq_runner.py` 不需修改：沿用既有 log callback，在 P4 speed 呼叫時注入靜默 logger；P3 預設行為與兩個凍結檔維持原 bytes。

Decisions：speed 只發布 timing／telemetry JSON、完成標記與總 manifest，完整研究契約改記 hash，epoch state 不持久化，不做 eval predictions，不保存 IC／scores／metrics／research_conclusion。訓練器內部 validation／probe 計算仍包含於 epoch 耗時，但其值不進輸出或 stdout。正式 fit schema 要求 BEST tensor 與 best pointer 一致、latest／previous／best checkpoint 的 hash／contract／epoch／必要 state keys／順序、metrics，以及 scores 的 finite／行數／reference rows／labels；沿用 P3 require_same_rows。原候選不合格完成檔不能自動升格或遷移。

Bundle 先逐一掃描 header，sparse／bytecode caches 直接拒絕；64 GiB 單檔、256 GiB 總量、10,000 成員與4 MiB manifest上限；header size 逐檔等於 manifest，總量等於宣告和加 manifest 大小，目的地在預檢後才建立。pack 排除 .pyc／.pyo／__pycache__，verify 不再有 executable-cache derived 例外。可信 bootstrap 在匯入 source 前清理 caches，設定 PYTHONDONTWRITEBYTECODE=1 與 sys.dont_write_bytecode；命令指南包含同一步驟。rankgauss 三個 derived 檔仍依既有 input／output／tag 驗證重用。

DRAFT 加入 GRU 53,250／Mamba2 60,706 的審查 CPU 建構數字及初始化／residual pre-norm／gated norm／dropout差異，並保留 OPEN 的搜尋、family（m=3 或階層式選項）、完整性、預算、確認性期間決策。P3 結果已知、2021–2025 已曝光，只作描述性；可確認路徑須事前規則及新資料／前瞻期間。沒有替 Human 凍結決策，沒有修改 machine contract status 或門檻。指南增加版本／ABI mismatch STOP／回報樹，禁止自行升降級。

Evidence — 首次新增實作案例 RED（未修程式時）：

```text
8 failed, 30 passed in 10.72s
FAILED test_cpu_orchestration_charges_setup_and_resumes_completed_fits
FAILED test_bundle_runtime_bytecode_cache_refused
FAILED test_empty_completed_fit_rejected
FAILED test_archive_resource_preflight_before_destination[huge]
FAILED test_archive_resource_preflight_before_destination[total]
FAILED test_archive_resource_preflight_before_destination[count]
FAILED test_forged_timestamp_pyc_removed_before_import
FAILED test_bundle_excludes_and_extract_rejects_executable_cache
```

初次 sparse 測試使用的舊式 header 未正確重現審查案例；改為真正 PAX sparse member 後，對 base `8cb61c4` 做純記憶體 replay RED，確認 huge／sparse／file／total／count 五項均進入被攔截的 copy。1 TiB 案例壓縮檔為 **291 bytes**，logical header／manifest宣告 **1,099,511,627,776 bytes**，沒有真的寫入巨大檔案：

```text
RuntimeError: TEST_ABORT: extract reached payload copy
9 failed, 44 deselected in 3.59s
IN_MEMORY baseline pytest_exit=1
```

完成 checkpoint 必要 state／pointer 順序補強的先行 RED：

```text
FAILED test_completed_fit_required_schema[checkpoint]
FAILED test_completed_fit_required_schema[order]
2 failed, 51 deselected in 1.52s
```

文件內容檢查以 git show 的 base 原文進行記憶體 replay（文件更新後補作原文 RED，不宣稱先於文件撰寫）：

```text
FAILED test_draft_documents_disclose_open_decisions_and_exposure
FAILED test_operations_require_abi_stop_and_preimport_cleanup
FAILED test_p4_knowledge_records_committed_candidate_and_uncommitted_repair
3 failed, 50 deselected in 0.06s
```

最終 focused GREEN：

```text
.....................................................                    [100%]
53 passed in 10.24s
```

四項變異均只在 Python 記憶體替換 source／函式，沒有改磁碟 source 或 P3 gate；每個 pytest exit=1，外層 harness 確認預期失敗才 exit0：

```text
IN_MEMORY sparse pytest_exit=1  # RuntimeError TEST_ABORT，1 failed／52 deselected
IN_MEMORY schema pytest_exit=1  # DID NOT RAISE ValueError，1 failed／52 deselected
IN_MEMORY pyc pytest_exit=1     # imported value=999，1 failed／52 deselected
IN_MEMORY logger pytest_exit=1  # stdout val_ic5 0.987654321，1 failed／52 deselected
```

可核對的 TEMPORARY evidence：`/tmp/p4a-red.log`、`/tmp/p4a-baseline-red.log`、`/tmp/p4a-schema-red.log`、`/tmp/p4a-docs-red.log`、`/tmp/p4a-focused-final.log`、`/tmp/p4a-mut-{sparse,schema,pyc,logger}.log`。純記憶體 harness 為 `/tmp/p4a-inmemory-check.py`；pytest 合成資料在其隔離 tmp namespace，可重建，沒有讀取正式 matrix／權重。

Knowledge-impact：semantics YES（P4速度輸出／schema／cache）；authority NO；subsystem boundary NO；active runtime NO；correctness gate YES（P4安全檢查補強，不改P3）；canonical artifact NO；deprecation NO；next decision YES（獨立審查後才決定現場測試及研究凍結）。同 change 更新 Current State、Authority Map、Worktree Register 與 Training。已檢查 Operations／GOVERNANCE：production operation、scheduler 與治理規則本身沒有變更，不需改寫。沒有新增／刷新 runtime observation。

Remaining／Blockers：官方 CUDA forward/backward／分數位元一致性、安裝與 ABI、GPU速度／熱、Drive FUSE fsync／rename／斷線持久性、大型真實 bundle／formal matrix、provider實扣CU仍 UNVERIFIED。完成 schema 只載入來源可信且hash驗過的 checkpoint；不宣稱任意不可信 pickle 安全。研究搜尋／family／完整性／預算與新確認性期間仍 OPEN，正式 P4 仍 NO_GO；本次工程完成不等於 Human acceptance。沒有 GPU／網路／資料 acquisition／Colab／正式 Data／V6/models／production／scheduler 變更，也沒有新增 agents。

Next recommended action：由獨立審查者核對本 patch 與 RED／GREEN／mutation 證據，Human 決定是否授權 Colab 環境／GPU／Drive驗證；正式研究須另外凍結 OPEN 選項。本輪不執行上述現場工作。

Runtime/environment：WSL，Python3.12.3，pilot09 venv `/home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python`，torch2.15.0+cpu，CPU synthetic tests；MARKETMAMBA_DATA_ROOT unset，本工作樹 Data 為普通目錄，其 realpath 為工作樹/Data。未移動 runtime／資料。Project memory updated：YES；Authority map updated：YES；Commit：N/A（禁止）；Push：未執行；Worktree：沿用，修補未提交。


### 本輪最終完整回歸（固定 source）

指定命令 exit0：

```bash
PYTHONPATH=V6:V6/research_lab/tests:. /home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python -m pytest V6/research_lab/tests tests -q
```

實際完整 stdout／stderr（原始 log `/tmp/p4a-full-final.log`）：

```text
........................................................................ [ 11%]
........................................................................ [ 22%]
........................................................................ [ 34%]
........................................................................ [ 45%]
........................................................................ [ 57%]
........................................................................ [ 68%]
........................................................................ [ 79%]
.................................................................s...... [ 91%]
.......................................................                  [100%]
=============================== warnings summary ===============================
test_stale_audit.py::test_mutation_controls[sessions = sorted(set(pd.to_datetime(trading_dates).strftime("%Y-%m-%d")))-sessions = sorted(d for d in daily if pd.Timestamp(d).dayofweek < 5)]
test_stale_audit.py::test_mutation_controls[sessions = sorted(set(pd.to_datetime(trading_dates).strftime("%Y-%m-%d")))-sessions = sorted(d for d in daily if pd.Timestamp(d).dayofweek < 5)]
test_stale_audit.py::test_mutation_controls[sessions = sorted(set(pd.to_datetime(trading_dates).strftime("%Y-%m-%d")))-sessions = sorted(d for d in daily if pd.Timestamp(d).dayofweek < 5)]
test_stale_audit.py::test_mutation_controls[sessions = sorted(set(pd.to_datetime(trading_dates).strftime("%Y-%m-%d")))-sessions = sorted(d for d in daily if pd.Timestamp(d).dayofweek < 5)]
  /tmp/pytest-of-frank/pytest-26/test_mutation_controls_session0/mutant.py:93: Pandas4Warning: Timestamp.dayofweek is deprecated and will be removed in a future version. Use Timestamp.day_of_week instead.
    sessions = sorted(d for d in daily if pd.Timestamp(d).dayofweek < 5)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
630 passed, 1 skipped, 4 warnings in 161.97s (0:02:41)
```

630 = 原候選608 + 新增22個通過案例；原P4 bytecode允許案例依修訂安全語義改為拒絕，原publication fixture改用合格speed schema，沒有刪測試或放寬gate。既有skip及4個Pandas warnings保留。AST syntax 3 files、JSON DRAFT schema/lock identity、490個本地Markdown references、兩個P3凍結檔diff 0 bytes及git diff --check通過。本輪沒有獨立代理審查；僅作者自查與可執行驗證，不把它稱為independent review。

最終knowledge-impact：WARNING，10 changed paths，已同change處理相關知識語義；health `--ci --as-of 2026-11-21` exit0，沒有structural FAIL：

```text
KNOWLEDGE HEALTH

Project Map        PASS
Agent Guides       PASS
Domains            PASS
Governance         WARNING
Authority Drift    WARNING
Runtime Freshness  PASS
Generated Index    PASS
Legacy Docs        WARNING

Overall: WARNING
```

受驗source與完整回歸log identity：

```json
{
  "V6/research_lab/p4_bundle.py": {
    "bytes": 10886,
    "sha256": "34f1088017f530d5db3309dbfbf74990e697f5a25adf28975999dd7097161541"
  },
  "V6/research_lab/p4_colab.py": {
    "bytes": 27352,
    "sha256": "1e7414dd3c58c5099a925b9a0c7010cd98610d93c3640aa4eab5a5b3d60ef2cd"
  },
  "V6/research_lab/tests/test_p4.py": {
    "bytes": 23538,
    "sha256": "4daec415d569c0b5fcacd0777e6d0195af113ab6c8883370a162ea7e5072c5d1"
  },
  "/tmp/p4a-full-final.log": {
    "bytes": 1963,
    "sha256": "f8f6cb94c12bbbf1096b1f4fa4f4c4949fb0a81fba66ad14ab0768fbd8023145"
  }
}
```

最終 Git status：detached HEAD `8cb61c4`；10 tracked modified、0 staged、0 untracked。依 no-commit 指示保留 dirty patch，沒有宣稱 clean。Commit／Push／Merge／Deploy：均未執行。TEMPORARY inventory `/tmp/p4a-repair-identities.json` 保存本次changed paths與logs SHA，可由本節的source／command重建；不是formal研究artifact。

## 第二輪獨立複審 findings 修補 — base 05cb840／2026-11-21

本節為本輪最新工程狀態，前節測試／hash／提交描述屬其各輪歷史證據，不代表本輪 source。候選與第一輪修補已分別保存為 `8cb61c4`、`05cb840`。本輪 `job-run-mwqew35j-886363` builder 沿用既有 `fixer2_p4a` detached HEAD `05cb840b5f4309844794d30bd4762abbcde49a13`，起始 clean。依 Human EXECUTE_APPROVED 限制，不 commit／push／merge／新增代理，不做 GPU／網路／Colab／data acquisition／production／scheduler／deployment／wakeups。第二輪 patch 尚待後續獨立 commit（hash 未產生），始終 UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED。

Changed：`V6/research_lab/p4_bundle.py`、`V6/research_lab/tests/test_p4.py`、`docs/operations/colab-p4.md`、本工程報告、`knowledge/00_Project_Map/{Current_State,Authority_Map,Worktree_Register}.md`、`knowledge/01_Domains/Training.md`。沿用現有實作與測試／文件樹，不新增 implementation tree、artifact tree 或 worktree。禁止修改的 `rolling_seq_scores.py`／`portfolio_eval.py` 保持原位元。

Decisions：

- 先於 tarfile.open，以 gzip bounded reader／TarInfo.frombuf 檢查 physical header，不觸發 PAX／GNU 自動展開。單次解壓請求最多1 MiB、extended payload最多64 KiB、physical headers最多20,000、metadata overhead最多32 MiB；連續 extended headers 拒絕，避免遞迴解析。各次掃描全域上限256 GiB＋32 MiB，manifest 出現後立即收緊為宣告總和＋4 MiB＋32 MiB；manifest 在末尾時回溯核對此前讀取，仍先受全域上限與metadata上限約束。先physical掃描，再tarfile串流 inventory，最後串流寫入；同一開啟archive inode，無全量RAM或暫存解壓副本，但增加三次gzip掃描成本。真實大型bundle效能未知。
- Linux 目錄描述符從 `/` 逐層 O_DIRECTORY|O_NOFOLLOW 開啟 ancestors／destination／payload directories，mkdir 使用 dir_fd；輸出使用 O_NOFOLLOW|O_EXCL。中途 symlink 替換會拒絕且不跟隨；失敗保留partial destination供盤點，不自動刪除／重試。
- pack／extract／verify 共用 bundle 路徑白名單：兩個既有 source Python 子樹（無tests/cache）、三個指定文件、指定fold年份／九個檔名、兩個reference、lock、manifest。任何位置的 sitecustomize.py／usercustomize.py／conftest.py／setup.py／*.pth／*.so／*.pyd 均拒絕，manifest 一致也不授權。原泛用 `source/a` 測試fixture改為合法 `source/V6/research_lab/a.py`，保留其round-trip／hash反證目的。
- 外部 archive SHA 由 Human 在可信環境記錄並於 Colab 重新核對；SHA 失守則整個 bundle 不可信。Archive內自報manifest不能證明來源，bootstrap需獨立核對；白名單不能證明惡意但受信SHA的Python source安全。可信本機帳號／parent權限仍屬前提；descriptor不是防任意同權限程序搬移已開啟目錄、改寫archive inode／Python程序的sandbox。
- 三份project map如實區分 `8cb61c4`／`05cb840` 與本輪no-commit patch；測試只要求UNMERGED／NOT_ACCEPTED／NOT_DEPLOYED，不鎖提交狀態。Current State 的P4區段只保留本輪一組最終全套數字。沒有替Human凍結五項OPEN研究決策或授予Colab acceptance。

Evidence — 先新增測試再修實作，實際 RED（原實作）：

```text
RuntimeError: REVIEW_SENTINEL oversized decompression read
AssertionError: outside/V6/research_lab/a.py exists
Failed: DID NOT RAISE ValueError  # matching manifest startup/non-allowlisted paths
13 failed, 53 deselected in 1.89s
```

初版修補再新增「manifest-first立即收緊」回歸，先RED後修正：

```text
assert 20480 <= 2560
1 failed, 68 deselected in 1.43s
```

最初巨大PAX案例使用同一GNU base-256 header／1 TiB宣告，之後補齊gzip合法零尾端，固定為審查指定的81 bytes。Directory swap由一處擴充為目的根／source中間層／research_lab末層三處；GREEN實際：

```text
PAX archive_bytes=81 requested_sizes=[512] destination_exists=False
RACE outside_written=False
RACE outside_written=False
RACE outside_written=False
15 passed, 53 deselected in 1.41s
```

補上manifest累計回歸後focused：`69 passed in 10.45s`。四項純記憶體變異，不修改磁碟source；實際執行新測試，各自pytest exit1，外層harness僅在預期失敗時exit0：

```text
IN_MEMORY pax pytest_exit=1 KILLED=True       # 恢復HEAD舊extract，巨大讀取sentinel
IN_MEMORY race pytest_exit=1 KILLED=True      # 恢復字串路徑寫入，outside出現檔案
IN_MEMORY allowlist pytest_exit=1 KILLED=True # 停用白名單，DID NOT RAISE
IN_MEMORY budget pytest_exit=1 KILLED=True    # 移除manifest收緊，20480 > 2560
```

TEMPORARY／可重建證據：`/tmp/p4a-round2-{red,budget-red,focused,full}.log`、`/tmp/p4a-round2-mut-{pax,race,allowlist,budget}.log` 與 `/tmp/p4a-round2-mutate.py`。Inputs全部為pytest隔離合成bundle／matrix，不讀寫formal Data或V6/models。測試用Python3.12.3：`/home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python`；CPU torch2.15.0+cpu，MARKETMAMBA_DATA_ROOT unset，Data為普通目錄，realpath為本worktree/Data。

Remaining／Blockers：真實Colab wheel／kernel／GPU、Drive FUSE fsync／斷線／持久性、provider CU、大型bundle速度與實際研究grid均未驗證。五項研究決策仍OPEN／契約DRAFT；正式P4依然NO_GO。CPU回歸不是Human acceptance。依禁止新增agents，本輪只有作者五軸自查及executable evidence，未執行獨立複審。Next recommended action：主控獨立複審patch與證據，再由Human決定現場環境驗證／研究凍結。

Knowledge-impact：semantics YES（bundle接收／資源與路徑gate）、authority NO、subsystem boundary NO、active runtime NO、correctness gate YES、canonical artifact NO、deprecation NO、major blocker／next decision NO。Project memory updated YES；Authority map updated YES（僅候選事實／安全與信任邊界，不升格authority）；Training domain與Worktree Register同change更新。Git status：起始clean，結束依no-commit保留八個指定檔案unstaged dirty、無既有dirty/untracked；Commit N/A，Push未執行，Worktree沿用既有fixer2_p4a，Runtime/environment上述CPU環境。

本輪最終完整測試命令（固定 source，停寫 bytecode／pytest cache）：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=V6:V6/research_lab/tests:. /home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python -m pytest V6/research_lab/tests tests -q -p no:cacheprovider
```

實際完整 stdout／stderr（exit 0）：

```text
........................................................................ [ 11%]
........................................................................ [ 22%]
........................................................................ [ 33%]
........................................................................ [ 44%]
........................................................................ [ 55%]
........................................................................ [ 66%]
........................................................................ [ 77%]
........................................................................ [ 89%]
.........s.............................................................  [100%]
=============================== warnings summary ===============================
test_stale_audit.py::test_mutation_controls[sessions = sorted(set(pd.to_datetime(trading_dates).strftime("%Y-%m-%d")))-sessions = sorted(d for d in daily if pd.Timestamp(d).dayofweek < 5)]
test_stale_audit.py::test_mutation_controls[sessions = sorted(set(pd.to_datetime(trading_dates).strftime("%Y-%m-%d")))-sessions = sorted(d for d in daily if pd.Timestamp(d).dayofweek < 5)]
test_stale_audit.py::test_mutation_controls[sessions = sorted(set(pd.to_datetime(trading_dates).strftime("%Y-%m-%d")))-sessions = sorted(d for d in daily if pd.Timestamp(d).dayofweek < 5)]
test_stale_audit.py::test_mutation_controls[sessions = sorted(set(pd.to_datetime(trading_dates).strftime("%Y-%m-%d")))-sessions = sorted(d for d in daily if pd.Timestamp(d).dayofweek < 5)]
  /tmp/pytest-of-frank/pytest-38/test_mutation_controls_session0/mutant.py:93: Pandas4Warning: Timestamp.dayofweek is deprecated and will be removed in a future version. Use Timestamp.day_of_week instead.
    sessions = sorted(d for d in daily if pd.Timestamp(d).dayofweek < 5)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
646 passed, 1 skipped, 4 warnings in 161.32s (0:02:41)
```

646 = 原630 + 本輪16個新案例；P4共69個。既有1 skip與4個Timestamp.dayofweek棄用warning保留，未刪測試或放寬研究gate。

Knowledge-impact checker exit0／WARNING，8 changed paths；已覆核Operations與GOVERNANCE，既有operation／scheduler／治理政策不變，不需改寫。語義已在Current State、Authority Map、Training、Worktree Register同change更新。AST syntax 2 files、DRAFT JSON／lock SHA、487個變更Markdown本地references（missing=[]）、git diff --check通過。對 `e4f21c3` 基準，rolling_seq_scores.py 50,048 bytes與portfolio_eval.py 50,314 bytes均diff 0 bytes。

Knowledge health `--ci --as-of 2026-11-21` exit0，沒有structural FAIL，實際輸出：

```text
KNOWLEDGE HEALTH

Project Map        PASS
Agent Guides       PASS
Domains            PASS
Governance         WARNING
Authority Drift    WARNING
Runtime Freshness  PASS
Generated Index    PASS
Legacy Docs        WARNING

Overall: WARNING
```

本輪受驗source／TEMPORARY驗證log／變異harness identities（不是formal artifacts）：

```json
{
  "V6/research_lab/p4_bundle.py": {
    "bytes": 18220,
    "sha256": "a9405454986e3b285d391da8abee5d3225b997d5d3dfa85e04c383de2841b97a"
  },
  "V6/research_lab/tests/test_p4.py": {
    "bytes": 27904,
    "sha256": "5496f4ff08944c70bcd31facb994e25071c3ea2883614c3164390aa80bfadf9f"
  },
  "/tmp/p4a-round2-full.log": {
    "bytes": 1963,
    "sha256": "085a6c59c0b263281f4509a7f971fd00bbbe54ddbf251ba8c1fd1ebaeeca91b8"
  },
  "/tmp/p4a-round2-red.log": {
    "bytes": 17468,
    "sha256": "ca1f7ea1047c07d83002764af3fb5162adc35a43c9193a4a38cd5db93654af6d"
  },
  "/tmp/p4a-round2-budget-red.log": {
    "bytes": 1933,
    "sha256": "ad103f109a06c140e6ce2963c38942a28a377772dddc9f4a46fbdecc426bf6c8"
  },
  "/tmp/p4a-round2-mutate.py": {
    "bytes": 1765,
    "sha256": "2474a0803a410a3e497ab6f2009a966576c3f1c24c95532f978620c5ba0e6b6c"
  },
  "/tmp/p4a-round2-mut-pax.log": {
    "bytes": 3482,
    "sha256": "2224650552f31d22b4b73f4ed3ded3dd90b87d127e5cb87eb8d80ec174a05573"
  },
  "/tmp/p4a-round2-mut-race.log": {
    "bytes": 2428,
    "sha256": "a736a1f29cdad64c0e5624b50e9fbbdc9caafdb602a3c4bf6aecc7cdb7bf8c24"
  },
  "/tmp/p4a-round2-mut-allowlist.log": {
    "bytes": 1384,
    "sha256": "3c6bcb088383fee78e2ea70c5c2f0a9e8a3a6fbf8eb54693b0f9cc7a7c1187eb"
  },
  "/tmp/p4a-round2-mut-budget.log": {
    "bytes": 1961,
    "sha256": "62635e4cae5fb36c705d18d1a36411ca0ce78529e87d08ed0fd4325732583b29"
  }
}
```


## 提交狀態更新（第三輪複審後）

第二輪修補已提交為 `a76e98d`；前文中「未提交／尚無 hash」的敘述為各輪歷史紀錄。第三輪（防禦性程式碼）複審：speed-test GO、正式 P4a NO_GO；剩餘 REQUIRED：`tests/test_p4.py` 的 PAX 測試無法單獨隔離 extended-header 大小上限（停用 `p4_bundle.py:85` 檢查後仍通過，被累計上限掩蓋），須在正式凍結前補上隔離測試。複審 worker 因 Codex 資安分類器拒絕『自行構造新攻擊載荷』的版本，改為不撰寫攻擊載荷的防禦性審查；gzip 炸彈、別名覆寫等新攻擊未經獨立嘗試。
