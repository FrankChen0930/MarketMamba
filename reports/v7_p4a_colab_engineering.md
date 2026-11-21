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
