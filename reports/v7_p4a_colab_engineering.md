## P4a 進度兩項 REQUIRED 修補 — 2026-11-22（job-run-mwrpdfba-d4ad9d）

本輪 base/HEAD `5156087da0e2aaba5d0cf703db3cdd67200cfbe9`，既有 detached worktree `fixer_p4a_progress`。範圍只處理複審的接續 ETA 與 cached-fit 身分；UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED，依 Human 指示不 commit／push／merge。沿用既有 runner、tests、工程報告與 knowledge 結構，沒有新 implementation／artifact tree。所有產物僅 CPU 合成 ENGINEERING_OBSERVATION；完成不等於 Human acceptance。

### 修正與決策

- `Progress` 在 identity／hash／pointer 驗證完成後，以 checkpoint 的 epoch × steps_per_epoch 設定本次起點並立即輸出／寫快照。累計 step 仍顯示原訓練進度；吞吐量分母只用本次新完成的 steps。至少 10 steps 且啟動耗時至少 1 秒才外推，否則 JSON null、TXT「預估中」。複審案例 3900 → 4000／5000、10 秒，修正後 ETA = 10 × (5000−4000) ÷ (4000−3900) = **100 秒**。
- 佇列目前 fit 優先使用上述 ETA，後續 fit 用本次已完成 fit 的固定實測秒數均值；目前 fit 尚無可靠 step 樣本時，才用該實測均值減本 fit elapsed 作粗估。沒有完成樣本時，除唯一剩餘的目前 fit 可用自身可靠 ETA，佇列維持 null。完成 fit 實測至少 1 秒才納入均值；不以後一個 fit 的 elapsed 重算前一個，也不使用歷史帳本時間。cached skips 不納入樣本。Q5 仍為額外未知，期間兩個 ETA 均 null；沒有對完整 Q5 完成時間作保證。
- cached 分支先更新目前 fit 編號、Q、arm／year／seed／window／lr，清除前一個 fit 的 loss／IC／checkpoint／step；不啟動訓練計時，finish 後進度行與 JSON／TXT 都是「已完成（略過）」。正常 context exit 保留此標記。全部快取與 mixed skip/train 都逐 fit 驗證；最後 fit 完成時 fit／queue ETA = 0，訓練完成顯示「已完成」。
- 只改觀測：不改 fit 內容／順序、配方、checkpoint authority、完成 gate、帳本、訓練模型、RNG 或研究規則。不需要新增依賴；requirements lock bytes 保持原樣。Human requirements 第 7 節同步描述新候選行為；第 5 節的需求未放寬。

### RED → GREEN 與失敗診斷

新增回歸 RED：`8 failed, 10 passed in 50.25s`（`/tmp/p4-required-red.log`）；實際數值為 `assert 2.5 == 100.0`，全快取 `fit_index=0`，resume 初始 epoch=0，均命中複審問題。另補正常退出保留略過狀態的 RED：`1 failed in 1.69s`（`/tmp/p4-final-skip-red.log`）。

首次 GREEN `2 failed, 16 passed in 51.19s`：既有變異 fixture 搜尋已移除的 ETA 算式；混合案例將仍有 complete.json 的 fit 強制當作缺失，觸發正確的 FileExistsError。修正 fixture 的變異注入點，並將一個本機合成 fit 移出 run、保留於測試暫存目錄，再實際重建；沒有放寬 immutable completed-fit gate。focused 結果 `19 passed in 57.29s`（`/tmp/p4-required-green2.log`）。之後再強化 skip 模擬耗時 5 秒仍不得進入樣本的斷言，納入最終全套。

相鄰測試實際在 epoch 2 的第 5 步中斷 CPU GRU：重啟恢復已提交 epoch 1／step 12，捨棄未提交 5 steps；fit 編號／身分保持相同，續跑 step 24 → 36，第一個可靠 ETA = 1.2 秒。最後訓練及略過 fit 均驗 ETA=0 與完成狀態；64-fit 全快取驗每筆 snapshot 及最終持久 JSON，mixed 案例驗第 2 fit 訓練、其餘略過，均無前筆身分殘留。

### 記憶體變異反證

只在子程序記憶體 compile 變異，不修改 repo source；以 pytest 實際執行新增測試，四項均因預期 assertion 失敗而 exit 1：

```text
historical_steps exit=1; 1 failed in 0.14s
insufficient_sample exit=1; 1 failed in 0.18s
recomputed_duration exit=1; 1 failed in 0.19s
skip_identity exit=1; 1 failed in 40.38s
```

對應變異：將 `/measured_steps` 改回 `/step`；10 steps 門檻改為 1；完成 fit 均值改成目前 elapsed；移除 cached 分支的 start_fit 身分更新。第四項重現 completed_fits=64、fit_index=0 與識別欄位未知。首次從 `/tmp` 啟動工具缺 repo import path，得到 ModuleNotFoundError，不算 mutation 證據；補正 `PYTHONPATH=.:V6` 才取得以上結果。暫存腳本 `/tmp/p4_required_mutants.py`、logs `/tmp/p4-mutant-*.log` 皆 TEMPORARY，非完成／研究 authority。

### 最終驗證

完整 CPU 指令（未使用網路／GPU／Colab，未執行 pip check）：

```bash
PYTHONPATH=V6:/tmp/p4-notebook-deps OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' /home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python -m pytest V6/research_lab/tests tests -q -p no:cacheprovider
```

實際輸出（`/tmp/p4-required-full.log`）：

```text
745 passed, 1 skipped, 5 warnings in 273.22s (0:04:33)
```

唯一 skip 是既有 CUDA 測試；warnings 為刻意 duplicate ZIP fixture 及既有 Pandas4 deprecation。沒有停用測試或降低門檻。使用既存 `/tmp/p4-notebook-deps`，無安裝套件；nbformat 解析／驗證 PASS，8 個 notebook code cells 均 compile PASS，未重新產生真實資料 bundle 或上傳交付。既有權重／預測／RNG 回歸納入全套；本輪不宣稱另有獨立 GPU 比較。

與 HEAD 比對：64-fit 內容與順序相同。SHA-256(`json.dumps(queue,sort_keys=True,separators=(',',':'))`) = `4dc90ec0ada6c6d34b43c429d359424c502634e502c6856c6774bf138de7ba23`，與複審相同。下列逐檔 SHA-256 綁定本輪 source／測試（syntax compile 通過），以及未改的保護／契約 bytes：

| 路徑 | SHA-256 | 與 HEAD |
|---|---|---|
| V6/research_lab/p4_colab.py | 63c6a9e123102f3e1b5dc36b3eefa4751d76ec2c929f078851d8fc21cb60ab32 | 已修補 |
| V6/research_lab/tests/test_p4_progress.py | 3c432e901ad1f89fb492e39b30827cb1228b3d38708e197f4036fddc04d2f1fc | 已擴充 |
| V6/research_lab/tests/test_p4_full.py | 43c82977e09eea59e4464e03d39c31f8816f77d4114d670b39bbf529e0cd9857 | 已擴充 |
| V6/research_lab/rolling_seq_scores.py | 5d10d52b1d423c73fa518535f01e80b2760a0d3087036b9b778491d13942aefd | 相同 |
| V6/research_lab/portfolio_eval.py | 3e46e07277c5d13fe6b24608b7e7dc20648e9558be322ad414d2f82a60c4a30c | 相同 |
| tasks/p4a-colab-contract-20261121.json | 7d5813bc21644817ccc991c97049f2c65efe88a9f4c18903b42038337b328c41 | 相同 |
| environments/p4a-colab-runtime.json | b45380920c48c5548ef8880f63f4a351a3de7ff5d6b9f2abef50ebf87e763194 | 相同 |
| environments/p4a-colab.requirements.lock.txt | 2c39a3fb9567d1f9fc8ee83369c7b6c90cd19a8e679e32e4c8a6c1d75c5782c9 | 相同 |

### Knowledge gate、限制與交接

人工 knowledge-impact：Semantics=YES（進度觀測）、Major blocker/next decision=YES（修補候選待獨立複審）；Authority／Subsystem boundary／Active runtime state／Correctness gate／新增或 supersede canonical artifact／Deprecate=NO。正常測試後執行 check_knowledge_impact.py，工具回報 PASS、三個程式／測試皆 DEFAULT，無自動更新；人工仍更新 Current State／Authority Map／Training／Human requirements。沒有 primary authority supersession，沒有重標歷史 artifacts，也沒有更新外部 runtime observed_at。

文件更新後再次執行 impact checker：WARNING，`canonical-knowledge` 規則要求 REVIEW_REQUIRED `knowledge/GOVERNANCE.md`；已核對 completion／severity／supersession 規則，這次不改治理政策，無需修改該檔。這是人工已完成的 review，不是假稱工具自動清除 warning。

`knowledge_health.py --ci` exit 0，實際輸出：

```text
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

無 structural FAIL；Governance 檢查 18 authority refs／807 Markdown links／53 JSON／3 runtime observations，warnings 為 8 個 AUTHORITY_REPIN_CANDIDATE；Authority Drift 亦含 repin 提醒，Legacy Docs 為 README／OVERVIEW／PROJECT 的舊架構／特徵／排程敘述漂移。本輪不自動 repin 或修改這些範圍外文件。`git diff --check` 通過，無 staged／untracked 檔案，8 個已追蹤檔案修改皆屬核准範圍；工作樹非 clean 是明確 no-commit 要求。依 code-review-and-quality 自查 correctness／邊界／可讀性／安全／成本，沒有引入額外依賴或放寬 gate；這不是獨立複審。

Runtime：WSL-native 既有任務 worktree，Python 3.12.3；沿用 `/home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python` 的 torch 2.15.0+cpu／numpy 2.5.4／pytest 9.2.0，只借用環境，不引用該研究 repo 結果。`MARKETMAMBA_DATA_ROOT` 未設定；本 checkout `Data/` realpath 為 `/home/frank/projects/MarketMamba/.worktrees/ruflo-workers/run-mwrpdfba-d4ad9d/fixer_p4a_progress/Data`，測試只用 `/tmp/pytest-of-frank/` 的合成資料；未寫正式 Data／V6/models。環境不同於 Colab OBSERVED cp313／T4，CPU 通過不能推定 CUDA 及 Drive FUSE 行為。ETA 為最大 epochs／實測均值粗估，早停、不同臂／窗與 I/O 可造成誤差，Q5 額外未知。

Remaining／Next：獨立複審本 patch；通過後另案重新 deliver，需重新綁定 source identity。Colab／T4／Drive 現場仍 UNVERIFIED，不解除原 NO_GO、不等於 Human acceptance；本輪沒有 GPU、網路、Colab、資料取得、production、scheduler、通知或 wakeup。無額外 agents。Commit／Push／Merge／Deploy 均未執行；保留未 staged 的核准 patch，依 no-commit 指示工作樹預期非 clean，非遺漏提交。

# P4a Colab 工程交接 — 2026-11-21

## 2026-11-22：後續交付進度與 speed-test 獨立帳本（本輪候選）

`job-run-mwrof3ay-8d0c43` builder；既有 worktree `.worktrees/ruflo-workers/run-mwrof3ay-8d0c43/builder_p4a_progress`，detached HEAD/base `f6a173eab03332bc3c50d78885a1d865366d2518`。**UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED**，依明確 no-commit 授權保留 patch；工程完成不等於 Human acceptance。主要依據為完整閱讀的[Human 持續需求](../docs/operations/colab-human-requirements.md)第5／7節，不改正在跑的原始02_formal或其run identity。

### 變更與決策

- 沿用既有 `p4_bundle.py` notebook 產生器：訓練用 `Popen(stdout=PIPE, stderr=STDOUT, text=True, bufsize=1)`，持續逐行排空與 `print(..., flush=True)`，子程序同時 `python -u`／`PYTHONUNBUFFERED=1`。非零退出印訓練步驟、期望0、實際code與可行動下一步；中斷會終止／回收子程序。finally列出結果目錄、manifest、進度位置。第0格維持原有精簡設定；新增「目前進度摘要」格，speed僅印白名單工程欄位，文件說明單一kernel無法在訓練中另跑摘要格。
- `seq_runner.train_one` 在既有 step_end 後每50 steps及epoch checkpoint後送純觀測事件，沒有新增抽樣、模型前向／反向、指標計算或RNG呼叫；P4的Observer統一顯示fit序號／總數、Q／臂／折／seed／window／LR、epoch／累計step／loss、FORMAL限定IC5、elapsed／ETA、GPU與checkpoint時間。P3未提供Observer時維持原訓練路徑；禁止修改的P3 adapter與portfolio evaluator bytes原樣。
- P4 `Progress` 用獨立時間心跳（60秒，與steps無關），快照在 `EXPERIMENT_DIR/progress/formal/` 或 `progress/speedtest/`；先temporary寫入／flush／fsync，再原子replace。TXT繁體中文、JSON機器可讀，含run／bundle／source identity。快照是GENERATED／ENGINEERING_OBSERVATION，不是checkpoint、完成證據或研究authority。各檔原子，但兩檔不是交易；JSON及observed_at決定摘要新鮮度。寫入例外提示Drive／空間後繼續訓練。先取得ledger writer lock才啟動進度，避免第二個同帳本程序覆寫活躍快照。
- fit ETA按最大epochs與已完成steps外推，早停可能縮短；queue ETA按本次程序已完成fit的實際耗時均值外推。尚無計時樣本以null／「預估中」呈現；已驗證cached skips不捏造耗時樣本。Q5不是fit，沒有以fit計時假裝其耗時；明列額外未知，Q5期間ETA為null。這是粗估，不是完成時間保證。
- runner集中路由帳本：CLI `--drive-root` 保持正式root，speed自動使用sibling `<root>_speedtest`。Notebook固定formal=`EXPERIMENT_DIR/budget`，speed=`EXPERIMENT_DIR/budget_speedtest`，摘要／匯出採相同有效路徑；各模式跨交付延續自身帳本。原chain／identity／缺帳／thermal STOP不改，沒有自動遷移、重置或清除歷史帳本。SPEED_TEST進度不接受val_ic5，沒有保存model／history／scores的隔離例外。
- 新增 `tests/test_p4_progress.py`，擴充既有P4測試而非新增平行implementation／報告／artifact tree；真實子程序、合成CPU模型、tempfile與故障注入均在既有結構可承載。現有 `test_p4_full` 的真實runner orchestration現在先3個speed fits，再64個FORMAL fits＋Q5，最後validated skip；只有GPU／mount／實際train邊界使用CPU doubles。

### RED／GREEN 與實際證據

最初6項新回歸測試於實作前RED：`6 failed in 7.38s`（`/tmp/p4-progress-red.log`）。首次GREEN `6 passed in 8.31s`。擴充focused最初 `2 failed, 158 passed, 1 warning in 74.95s`：變異測試os.replace注入延伸至後續Ledger（測試範圍錯誤）與既有orchestration仍讀舊speed root；先定位再修正，沒有放寬實作gate。修正後針對性回歸 `11 passed in 12.92s`（`/tmp/p4-progress-fixes.log`）。另有一輪未設單執行緒的research_lab subset主動中止exit130，不用該輪聲稱通過。

測試包含：2,048行×1,024字元與stderr輸出無死鎖；child回傳7顯示摘要；READY→父程序轉印/ack→child才可退出，驗證不是收集後一次印出。原子replace前讀者仍見舊完整JSON；模擬Drive replace失敗保留舊快照並不中斷；step0時獨立心跳仍更新loss與observed_at。無樣本queue ETA=null，完成fit實耗20秒、總4 fits時下一fit開始的剩餘外推=60秒。兩個獨立模式帳本接受不同契約，刪head各自仍STOP；原共用root遇不同契約確實重現 `ledger identity/chain mismatch`。

CPU GRU兩epochs×51steps、seed17、d_model8，有／無Progress與Observer，最終權重逐tensor的numpy bytes、預測完整bytes、torch RNG state均相同；沒有以近似誤差宣稱位元一致。SPEED_TEST實際CPU訓練僅5行進度（起始＋兩次step50＋兩次epoch），stdout／TXT／JSON均不含IC。測試使用的20ms心跳僅加速工程驗證；實作預設60秒。

至少三項in-memory變異（僅記憶體compile，沒有修改磁碟implementation）實際輸出：

```text
IN_MEMORY speed_redaction KILLED=True (IC invariant assertion)
IN_MEMORY unknown_ETA KILLED=True (None invariant assertion)
IN_MEMORY nonfatal_Drive KILLED=True (unexpected exception)
```

WALLCLOCK_HEARTBEAT獨立觀測：無training steps，initial `2026-11-22T10:21:13.675095+00:00` → updated `2026-11-22T10:21:13.720515+00:00`，loss更新為.456。這是本機合成runtime證據，不更新Colab observed_at。

完整CPU回歸指令與實際輸出（TEMPORARY log `/tmp/p4-progress-full-final.log`）：

```bash
PYTHONPATH=V6:/tmp/p4-notebook-deps OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' /home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python -m pytest V6/research_lab/tests tests -q -p no:cacheprovider
```

```text
737 passed, 1 skipped, 5 warnings in 259.48s (0:04:19)
```

唯一skip為既有CUDA測試；warnings為刻意duplicate ZIP fixture與既有Pandas4棄用警告。沒有停用測試／降低threshold。完整全套期間source與測試未變更。

實際runtime：WSL Python3.12.3、torch2.15.0+cpu、CUDA=False；上述既有venv只作runtime，不取用歷史repo研究結果。nbformat5.9.2從既有離線 `/tmp/p4-notebook-deps` 載入，無安裝／連網。初始focused用既有marketmamba-gpu venv但全在CPU；正式全套採CPU-only venv。`MARKETMAMBA_DATA_ROOT=UNSET`，本worktree `Data` realpath為該worktree/Data，未讀寫formal Data／模型payload。

生成notebook通過nbformat、12cells／8code cells全compile；`/tmp/p4-progress-notebook.ipynb`，17,642 bytes，SHA256 `3683e125f2e5f8f222ce9c6e63f427314bebd668e88d6d71b668d6249676b94a`，TEMPORARY／ENGINEERING_ONLY，未打包真實ZIP、上傳或替換原交付。6個Python檔syntax PASS。逐bytes對HEAD驗證未變：`rolling_seq_scores.py` SHA `5d10d52b1d423c73fa518535f01e80b2760a0d3087036b9b778491d13942aefd`、`portfolio_eval.py` SHA `3e46e07277c5d13fe6b24608b7e7dc20648e9558be322ad414d2f82a60c4a30c`、FROZEN contract SHA `7d5813bc21644817ccc991c97049f2c65efe88a9f4c18903b42038337b328c41`、OBSERVED runtime SHA `b45380920c48c5548ef8880f63f4a351a3de7ff5d6b9f2abef50ebf87e763194`。

### Knowledge-impact、限制與交接

Gate：semantics=YES（進度及帳本路由）、next decision=YES（獨立審查後重建後續交付）；authority／subsystem boundary／active runtime／correctness thresholds／新增或supersede canonical artifact／deprecation=NO。正常tests後執行impact checker，再同change更新Current_State、Authority_Map、Training與Worktree_Register；Human requirements第7節如實標「已實作候選／CPU合成驗證／Colab UNVERIFIED」。不重標歷史artifact，不改runtime observed_at或authority pins。

`check_knowledge_impact.py`對tracked diff exit0／WARNING；新增untracked測試另以 `--changed-file` 檢查exit0／PASS（該參數覆寫預設diff，不把此單檔PASS當全diff結果）。人工更新四份knowledge；已讀GOVERNANCE及Operations，治理規則及營運子系統邊界未變，不需改該兩檔。`knowledge_health.py --ci` exit0／Overall WARNING：Project Map／Agent Guides／Domains／Runtime Freshness／Generated Index PASS；Governance／Authority Drift／Legacy Docs WARNING，無FAIL，不更新觀測或重寫歷史以清除warning。`git diff --check`、新增Markdown本地連結及protected-path檢查PASS；新增untracked測試另檢查空白與compile。全套之後只更新文件與清除新測試一處行尾空白，無實作語義變更。

最終新增進度測試（清除一處行尾空白後）：`10 passed in 9.83s`，log `/tmp/p4-progress-final-focused.log`。最終核對13paths（12modified／1untracked／0staged）、15個新增本地Markdown連結、6個Python syntax、protected bytes及 `git diff --check` PASS；工作樹依明確no-commit授權保留dirty。額外空白掃描曾誤將既有test_p4.py全檔舊空白納入，已限縮為新增檔全文及tracked新增diff，未格式化無關歷史程式。

本輪implementation／新增測試artifact identity（source commit為上述HEAD，dirty content由以下hash識別）：

| 路徑 | bytes | SHA-256 |
|---|---:|---|
| `V6/research_lab/p4_colab.py` | 49668 | `d4742a9ec41c1f1ffd69b22f36e05f7b557adcfcdc99add7b1bc08dd673e5b9b` |
| `V6/research_lab/p4_bundle.py` | 52451 | `d080faee1ec73034584645249d50e49a0946031f7569bb2ac32ac7f3f22e1ad3` |
| `V6/research_lab/seq_runner.py` | 23135 | `1f4ba6fa9d97a93a5f59413e26b68252c926f071e51dc5465098438af69b088e` |
| `V6/research_lab/tests/test_p4_progress.py` | 9177 | `28c06387a2cd876e6462b5a7a5eedf0affdef0e156dd23b73977dbe8d96cd47e` |

五軸自查：觀測不參與訓練／RNG；進度白名單及獨立namespace維持speed隔離；沿用既有P4 ownership、標準庫與輸出結構；I/O主要在心跳thread，步數輸出限50steps／epoch，不新增指標計算。沒有額外agents或獨立審查，不將自查視為acceptance。

Remaining：Colab notebook即時呈現、T4／CUDA數值可重現、Drive FUSE fsync／rename／斷線持久性、Drive長時間卡住的I/O、真正Colab中斷與子程序終止、Q5現場推論與瀏覽器下載仍UNVERIFIED；本機原子檔測試不推定Drive保證。SIGKILL／runtime消失可能留下舊快照，Human應核對observed_at。舊帳本可能已含不同契約；新版不自動修復或遷移。正在跑的02_formal本輪完全未觀測／未修改。

Completed：後續交付進度可見性與模式帳本分離工程。Changed：3實作、3測試、2operations文件、本報告及4knowledge文件，共13paths（含1個新測試）。Decisions／Evidence如上。Blockers：本機工程N/A；接受與現場驗證待辦。Next recommended action：獨立複審patch，接受後另案deliver與Colab驗證。Project memory updated：YES；Authority map updated：YES（候選邊界；研究／production authority未變）。Git status：起始clean，依no-commit保留dirty patch，0 staged；不宣稱clean。Commit：N/A（禁止）；Push：未執行；Merge／deploy：未執行。Worktree／Runtime/environment如上；沒有GPU／網路／Colab／資料取得／production／scheduler／additional agents／wakeups／recurring jobs或protected V6/models變更。

## 2026-11-22：d82d020 複審三項 REQUIRED 修補

`job-run-mwric0jj-32d745`，builder，detached HEAD/base `d82d0209cba6686c10fc1a01c78172446a6796fa`。UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED；只沿用既有worktree、P4 implementation／tests／knowledge結構，不新增流程、artifact tree或agents。本輪工程完成不構成獨立複審GO、Colab現場驗證或Human acceptance。

### Changed／Decisions

1. `p4_eval.py`：Family A/B 對所有比較系統與 C0+ 要求完整相同列集合；不一致拋 ValueError，JSON錯誤包含 common_rows、input_rows、excluded_rows與比例，不能裁列後判完整。Family C 保留共同交集，取出 raw seed scores 在該交集重新逐fold/date percentile rank，再平均seed後送原P2引擎；C0+、W60也限制同一交集，保留排除報告。原始 seed鍵／label完整性檢查仍先執行，不容忍缺seed。
2. `p4_colab.py`：fit_contract 新增有效 recipe；正式發布與續跑共用驗證，先核對 frozen queue 的queue/year/arm/seed/window/lr，recipe等於凍結配方加該fit自身window/lr。重算連續history的BEST與bad，要求8epochs或bad>=patience；拒絕早停後仍繼續的history、BEST pointer非首個最佳epoch、metrics／checkpoint history prefix不一致、預測fold/system/seed/window/lr不符。BEST權重／hash既有檢查保留。
3. 既有formal fixtures與64-fit CPU boundary doubles改為有效4epoch早停history；沒有刪測試、降低threshold或改成特例。新增滿8epoch、主格／長窗／LR各自有效配方與重算hash後仍拒絕語義損壞案例。

相容性限制：舊缺recipe或完整history證據的完成標記不再可skip；fail closed，不修改、升格或遷移歷史artifacts。研究／環境機器契約 bytes未改。後續交付須另行重建source綁定，本輪不產出真實訓練ZIP、不接觸正式Data。

### Evidence：RED → GREEN／完整回歸／反證

RED（`/tmp/p4-required-red.log`）：`5 failed, 18 deselected in 1.92s`，A/B缺列各一、Family C排名、epoch1/bad0及epoch7仍未終止各一。P4 focused最終（`/tmp/p4-required-focused-final.log`）：`150 passed, 1 warning in 62.95s`。最初focused為136passed，之後新增14個配方／語義損壞案例再跑最終150；沒有掩蓋失敗。

排名反轉合成案例實際輸出：

```text
full_support_W60=[0.8,0.6000000000000001,0.5,0.55,0.55]
common_W60=[0.6,0.7,0.5,0.6000000000000001,0.6]
code0>code1 before=True after=False
C common_rows=5; W60/C0+ input_rows=10 excluded_rows=5 excluded_fraction=0.5
A/B missing one C0+: REJECTED; neural input_rows=10 excluded_rows=1; C0+ input_rows=9
```

三項mutants只在獨立CPU程序的記憶體compile／exec module，不改repo source。Runner `/tmp/p4-required-mutate.py`；logs `/tmp/p4-required-mut-{rows,ranks,termination}.log`：

```text
IN_MEMORY rows pytest_exit=1 KILLED=True
IN_MEMORY ranks pytest_exit=1 KILLED=True
IN_MEMORY termination pytest_exit=1 KILLED=True
```

分別移除A/B列一致gate、回退至全支援seed排名、移除訓練終止gate，對應新測試因實際assertion失敗被殺死（不是collection/import錯誤）。所有logs／pytest合成artifacts屬TEMPORARY／ENGINEERING_ONLY；合成輸入由版本化test fixture決定，不提供研究或交易evidence。

完整指定回歸（受測source期間固定；`/tmp/p4-required-full.log`，exit0）：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/tmp/p4-notebook-deps:V6:V6/research_lab/tests:. \
/home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python -m pytest V6/research_lab/tests tests -q -p no:cacheprovider
```

```text
727 passed, 1 skipped, 5 warnings in 248.41s (0:04:08)
```

唯一skip為既有CUDA測試；nbformat透過既有離線 `/tmp/p4-notebook-deps` 實際執行。Warnings為duplicate ZIP fixture及既有Pandas Timestamp.dayofweek棄用警告。Python3.12.3／torch2.15.0+cpu，GPU隱藏，無網路或套件安裝。MARKETMAMBA_DATA_ROOT unset；Data存在且realpath為本worktree `/home/frank/projects/MarketMamba/.worktrees/ruflo-workers/run-mwric0jj-32d745/fixer_p4a_full/Data`，未用正式資料訓練、未寫formal Data。

受驗source identity（syntax AST parse PASS）：

| Path | bytes | SHA-256 |
|---|---:|---|
| `V6/research_lab/p4_eval.py` | 12099 | `c8e787f17e7818eec939ed0740285482af02f28435cee22a3d96b3a3f727c164` |
| `V6/research_lab/p4_colab.py` | 42230 | `8c5833ece2ecfd720c593479ee81cf99b392d3ca15aa786753c6715ffc4c8526` |
| `V6/research_lab/tests/test_p4.py` | 50512 | `8080e0c1ea0401bf053d6b2587c945212b041082d13ed8aafeb100e2a2fe1bac` |
| `V6/research_lab/tests/test_p4_full.py` | 23477 | `6184edb172b610241735a0fd624a76f7b28c0964c52368407a404386fddae6d5` |

逐bytes確認 `rolling_seq_scores.py` SHA `5d10d52b1d423c73fa518535f01e80b2760a0d3087036b9b778491d13942aefd`、`portfolio_eval.py` SHA `3e46e07277c5d13fe6b24608b7e7dc20648e9558be322ad414d2f82a60c4a30c`、FROZEN契約SHA `7d5813bc21644817ccc991c97049f2c65efe88a9f4c18903b42038337b328c41`、runtime契約SHA `b45380920c48c5548ef8880f63f4a351a3de7ff5d6b9f2abef50ebf87e763194` 均與HEAD相同。無V6/models／formal Data／scheduler／production diff。

### Knowledge-impact／限制與交接

逐項gate：semantics=YES、correctness gate=YES（補強契約執行，不降標）、next decision=YES（修補後待獨立複審）；authority／subsystem boundary／active runtime／new or superseded canonical artifact／deprecation=NO。正常tests後執行check_knowledge_impact.py初次PASS，工具未匹配research_lab專屬規則，仍由人工更新Current_State、Authority_Map、Training及Worktree_Register。保留舊歷史evidence；未改runtime observed_at、authority pins、generated indexes或治理規則。

最終 `check_knowledge_impact.py` exit0／WARNING（canonical knowledge與worktree責任需人工review）；已讀GOVERNANCE與Operations，治理規則及營運子系統邊界未變，不需改該兩檔。`knowledge_health.py --ci` exit0／Overall WARNING，Governance／Authority Drift／Legacy Docs為WARNING；Project Map／Agent Guides／Domains／Runtime Freshness／Generated Index均PASS、無FAIL。`git diff --check` PASS；新增Markdown連結／reference已核對，受保護路徑無diff。依no-commit指示保留dirty patch，不宣稱clean。

Code-review-and-quality五軸自查：正確性由RED/GREEN與mutation證明；沿用P4邊界與P2引擎，新增兩個小validator而非重寫引擎；未新增外部I/O／依賴／服務；seed排名額外重算限Family C，checkpoint history最多8epochs。沒有獨立審查者，不將自查當Human acceptance。

Remaining／Blockers：本輪本機工程無阻擋；Colab／T4／Drive FUSE／真正SIGKILL現場行為仍UNVERIFIED，無GPU或研究結果、無真實交付重打包／上傳。Next recommended action：獨立複審patch與實際logs，接受後另行重建綁定新source的交付與現場驗證。

Completed：三項REQUIRED工程修補與CPU驗證。Changed：2實作、2測試、本報告、Current_State／Authority_Map／Training／Worktree_Register，共9 tracked paths。Decisions／Evidence如上。Project memory updated：YES；Authority map updated：YES（候選工程邊界，研究／production authority未變）。Git status：起始clean，結束依no-commit保留9 modified、0 staged／untracked。Commit：N/A（明確禁止）；Push／Merge：未執行。Worktree：沿用上述既有worktree、HEAD d82d020；未新增或移除。Runtime/environment如上；無GPU／網路／Colab／data acquisition／additional agents／wakeup／recurring jobs／production／scheduler／protected models變更。

## 2026-11-22：完整固定佇列 builder（本輪候選）

`job-run-mwrgy396-cf1213`，base/HEAD `f333421fafdcb02f716b16376ff65717021edfa4`；UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED。本節取代下文舊P4a的DRAFT、時間/CU caps、25%reserve、completed-fit-only與每次內含矩陣交付的目前操作說明；保留歷史evidence，不修改P3結果或runtime observed_at。凍結日期2026-11-22，在任何P4a結果出現前；沒有研究結果。

### 變更與決策

- 擴充既有 `p4_bundle.py`：`dataset-manifest` CLI（14檔逐檔size/SHA、無links、指定X identity）；Drive→SSD準備與fold links；共享rank-gauss及tag/hash/implementation identity回存Drive；matrix_v1交付ZIP只有source/契約/環境/資料manifest/C0+鍵檔。保留舊`pack`格式的回歸相容測試，新操作使用matrix_v1。
- 打包來源解析後限制於指定fold根目錄；ZIP實際邏輯大小先對manifest逐檔比對，再限制2000倍壓縮比（可容納合法零矩陣）；隔離PAX-specific64KiB邊界測試。Notebook自動綁ZIP size/SHA、預設FORMAL、設定RUN_DIR/DATA_DIR、Drive wheel優先與舊目錄回退、無pip check、無時數/CU設定；末尾下載scores/metrics/checkpoints/ledger。
- `seq_runner.py` 新增opt-in `resume_state`，P3未提供時保持原訓練行為。15% warmup/cosine抽成等值可測函式，AdamW/clip1.0/patience/BEST不改；保存/恢復numpy全域、batch generator、torch CPU/CUDA RNG與optimizer/scheduler/history。
- `p4_colab.py`：完整epoch validation後本機serialize/fsync/hash→Drive copy/fsync/re-read hash→atomic pointer；保留latest/previous/best與未選orphan。缺pointer但有orphan或corrupt/mismatch均fail closed，不自動升格；中斷epoch重做，完成fit逐檔/schema/BEST驗證後skip。完成manifest、scores與BEST發布使用原子commit記錄／temporary rename。
- P4專用record-only帳本，無時間/CU閘門與25%reserve；舊cap helper僅保留相容測試，不在新runner路徑呼叫。逐次嘗試記wall秒與GPU/溫度/時脈摘要，首筆telemetry與完整epoch事件持久化。未知中斷區間到重啟保守記錄，不冒稱活躍訓練時間。85°C硬體STOP先持久寫入，process loss也不會由record-only恢復清除。
- 凍結機器契約、OBSERVED cp313環境hash與規格：Q1/Q2主格36、Q3回看24、Q4 LR4，共64fits，再Q5三折crossstock seed17、R10驗證/評估排列推論。Q5比較固定BEST與mamba2，輸出每日Spearman/IC變化/top50訊號名單重疊（不等同有交易限制的實際成交持股）、IC標準差與順序敏感判定。
- 新增 `p4_eval.py`，原因是P3 evaluator不可修改；重用原P2公開函式、volume provenance與portfolio引擎。A/B/C分別Holm3/2/4、block20/1000；Family C使用所有比較系統/W60/C0+同一交集，明報排除比例；完整下載run要求64fit且驗證完整manifest與輸出，缺seed/列/label不符拒絕。歷史門檻通過不宣稱獨立確認，P3已曝光replication與C0+調參不對稱明列；P4d另案OPEN。
- 規格外推T4：主格11.17h、長窗19.24h（線性窗長假設）、LR1.04h，共約31.45h純訓練步；另加validation/inference/I/O/編譯/Q5。不是完整批現場量測，無時間限制。

### RED／GREEN與反證

初始新測試 `6 failed in 9.38s`：manifest/preparation、frozen queue、warmup、精確resume、record-only ledger、fold root escape；修正後 `6 passed in 6.16s`。評估／排列測試RED為 `2 failed, 7 passed`；ZIP ratio RED為 `1 failed, 11 passed`。之後CPU完整佇列/驗證skip與dropout resume等focused `129 passed`；telemetry漏記先RED `1 failed` 再補實作。硬體STOP在END前process-loss情境亦先RED，修正後兩項focused `2 passed in 1.48s`。

測試以CPU合成matrix、GRU與注入causal mixer的crossstock完成：epoch2中斷/接續與一次跑完的模型權重逐tensor `torch.equal`、預測 `np.array_equal`；額外dropout=.4的crossstock亦位元一致。完整64fit/Q5採CPU boundary doubles，重跑沒有再次fit或Q5；不是64次真實模型訓練。另有真實P2引擎小型合成比較，P4與未改P3公開路徑的逐日報酬完全相同。來源/矩陣/seed不符、壞pointer、缺pointer/orphan、partial copy均拒絕。

5項in-memory mutation（不改repo檔）全部killed，輸出：

```text
IN_MEMORY_MUTATION warmup pytest_exit=1 KILLED=True
IN_MEMORY_MUTATION rng pytest_exit=1 KILLED=True
IN_MEMORY_MUTATION dataset pytest_exit=1 KILLED=True
IN_MEMORY_MUTATION identity pytest_exit=1 KILLED=True
IN_MEMORY_MUTATION labels pytest_exit=1 KILLED=True
```

第一次mutation harness因`/tmp`腳本缺repo root sys.path，為collector/import錯誤（exit4），沒有算killed；修正harness路徑後才得到上列有效exit1。腳本與輸出是TEMPORARY `/tmp/p4_mutations.py`、`/tmp/p4-mut-*.log`。

首次全套 `1 failed, 705 passed, 1 skipped, 5 warnings in 288.36s`；失敗在P3 `test_revision2_revalidate_all_24_fits[environment]` 的初始source/environment重驗，單獨重跑 `1 passed in 3.08s`。當時全套期間仍改動被P3 provenance hash涵蓋的source，可能觸發身分保護；未修改P3 gate或停用測試。其後停止改source，再完整重跑，以最終輸出為準。

最終完整輸出（`/tmp/p4-full-final.log`，exit0）：

```text
708 passed, 1 skipped, 5 warnings in 212.59s (0:03:32)
```

唯一skip為既有CUDA測試；warnings為刻意duplicate ZIP fixture及既有Pandas4棄用警告，沒有停用測試或放寬門檻。最終全套期間source/契約凍結不改。

驗證命令：

```bash
PYTHONPATH=V6:/tmp/p4-notebook-deps OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''  /home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python -m pytest V6/research_lab/tests tests -q -p no:cacheprovider
python3 tools/project_knowledge/check_knowledge_impact.py
python3 tools/project_knowledge/generate_indexes.py
python3 tools/project_knowledge/knowledge_health.py --ci
git diff --check
```

實際CPU runtime：Python3.12.3、numpy2.5.4、pandas3.1.1、torch2.15.0+cpu；nbformat從既有離線 `/tmp/p4-notebook-deps` 載入，未安裝套件或連網。這不是Colab環境。初始部分focused使用既有marketmamba-gpu venv的torch2.10.0+cu128，但全部CPU，未有可用GPU。`MARKETMAMBA_DATA_ROOT`未設定，此worktree的Data不存在；沒有寫正式資料。

已驗syntax6Python檔、nbformat及10格notebook（含全部code cells compile）、FROZEN schema與runtime SHA綁定、synthetic archive逐檔hash及metadata；逐bytes確認以下與HEAD相同：

```text
rolling_seq_scores.py 50048 bytes SHA256 5d10d52b1d423c73fa518535f01e80b2760a0d3087036b9b778491d13942aefd
portfolio_eval.py 50314 bytes SHA256 3e46e07277c5d13fe6b24608b7e7dc20648e9558be322ad414d2f82a60c4a30c
environments/p4a-colab-runtime.json 2350 bytes SHA256 b45380920c48c5548ef8880f63f4a351a3de7ff5d6b9f2abef50ebf87e763194
```

Knowledge-impact：semantics/correctness gate/canonical contract/next decision/worktree responsibility YES，active external runtime/production authority NO。已同步Current State、Authority Map、Worktree Register、Research domain；generated-index核對PASS。knowledge health無FAIL，Governance/Authority Drift為8項AUTHORITY_REPIN_CANDIDATE，Legacy Docs為README/OVERVIEW/PROJECT舊敘述WARNING；不在本任務擅自repin歷史authority或改production。

### Changed paths（完整）

```text
V6/research_lab/p4_bundle.py
V6/research_lab/p4_colab.py
V6/research_lab/p4_eval.py
V6/research_lab/seq_runner.py
V6/research_lab/tests/test_p4.py
V6/research_lab/tests/test_p4_full.py
tasks/p4a-colab-contract-20261121.json
tasks/spec-p4a-colab-20261121.md
docs/operations/colab-p4.md
reports/v7_p4a_colab_engineering.md
knowledge/00_Project_Map/Current_State.md
knowledge/00_Project_Map/Authority_Map.md
knowledge/00_Project_Map/Worktree_Register.md
knowledge/01_Domains/Research.md
```

### 限制與交接

Colab現場的Python3.13整批、T4長窗OOM與效能、額外Triton編譯、wheel安裝/備援編譯、Drive FUSE的fsync/rename/斷線持久性、真正hard-kill/新runtime、真實矩陣14檔與上傳、Q5 GPU推論與瀏覽器自動下載全部 **UNVERIFIED**。本輪未對Human的Windows upload資料夾實際產生manifest，也未製作可訓練的真實兩檔，只測合成fixtures；可依操作指南執行CLI。CPU中斷測試是epoch-boundary KeyboardInterrupt／重新建構物件；publication故障用注入失敗，不冒稱OS SIGKILL或FUSE斷線現場證據。遺失/損壞pointer、未完整ledger提交仍fail closed而需人工診斷；不承諾任意損毀後全自動恢復。

Q5 top50是訊號排名名單，未引入未交付的OHLCV或假造tradability；完整交易持股重疊需另以P2引擎與價格評估。rank-gauss cache tag/identity不符重建，cache可重建；歷史checkpoint/orphan不刪。telemetry在首筆/epoch/end持久化，硬kill可能遺失該epoch尚未commit的後續樣本，但保留首筆與上次完整epoch，未知不補零。一次性cache與指標的Drive現場耐久性仍待驗。

- Completed：本輪離線工程候選、凍結規格與測試；不是Human acceptance。
- Changed：p4_bundle/p4_colab/seq_runner、新p4_eval、test_p4與新test_p4_full、研究契約/規格、操作指南、本報告與四份knowledge；無protected路徑變更。
- Decisions：固定64fit/Q5、record-only、epoch resume、共享dataset、獨立P4 evaluator；保留歷史/前瞻evidence界線。
- Evidence：上述RED/GREEN、位元一致、5個mutation、syntax/hash/schema/knowledge與完整輸出。
- Remaining：真實manifest/交付與Colab現場驗證、P4d、獨立審查及Human acceptance。
- Blockers：本機無正式Data/Colab現場；任何corrupt/mismatch仍STOP，不放寬gate。
- Next recommended action：獨立review patch，Human一次上傳matrix_v1，產生新兩檔後做現場驗證。
- Project memory updated：YES；Authority map updated：YES（不改production）。
- Git status：起始clean detached，結束保留本輪dirty/untracked patch；不能同時遵守no-commit與clean-worktree，不stage、不丟棄patch。
- Commit：NONE（明確禁止）；Push：NONE；Worktree：既有builder_p4a_full，未新增/移除worktree。
- Runtime/environment：上述CPU、Data未掛載；無GPU/網路/Colab/production/scheduler/付費成本，無additional agents、無wakeups/recurring jobs。

---

以下為歷史工程紀錄（由上節更新本輪目前狀態，不重標歷史evidence）。

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

## Colab 操作修訂：外部 SHA 可選／Drive wheels／獨立環境契約（2026-11-21）

本節為 `job-run-mwqi30jf-aa0d7f` builder 的未提交候選，base/HEAD `36aa27da6e26c1877332c9cdea927e6409c3d105`，UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED。Human 明確核准本 bounded task，並禁止 commit/push/merge／GPU／網路／Colab／新增 agents；本節不構成 Human acceptance，也沒有研究結果。

Human 背景：第一次 Colab 因 Python 非3.12被原 gate STOP；過去曾手動切換 runtime 版本，再從 `MyDrive/Mamba_Wheels_T4` 安裝 2026-03-04 建立的 cp312/linux_x86_64 mamba_ssm2.3.0／causal_conv1d1.6.0。這些是 Human 提供的歷史檔案資訊，不是本輪觀察到的已安裝版本；實際 Colab torch／CUDA／套件組合 UNKNOWN。Human pip list 尚待提供，沒有以猜測版本填契約。

### Completed／Changed／Decisions

- `p4_bundle.py`：CLI `--sha256` 可省略，API／CLI 都計算並印 archive SHA；提供時仍在寫入前嚴格核對。內部 manifest 逐檔 size/SHA 不變；原資源限制、dir_fd／O_NOFOLLOW／O_EXCL 及白名單全保留，白名單只增指定 Colab runtime JSON。指南不再要求貼 archive 或 bootstrap SHA。本節依 Human 決策 supersede 舊輪次「外部 SHA 必填」策略，不重標歷史證據；AGENTS.md 不改。
- `installation_commands(root, wheel_dir=...)` 與 CLI `install-plan --wheel-dir`：列出實際 wheel 檔名、版本、cp／ABI／平台標籤及 SHA，與目前 Python 支援標籤核對；缺少、多版本或不相容 STOP。只產生 `pip install --no-deps --no-index --find-links <dir> <兩個wheel路徑>` 及 `pip check`，不執行安裝、不升降級 torch 或其餘套件。保留原完整 lock 安裝路徑。
- 新 `environments/p4a-colab-runtime.json`：TBD／UNOBSERVED，observed_at=null，與原樣本機 P3 lock 分開。新增檔案原因：現有 P3 lock 不能記錄不同且未知的 Colab runtime；沿用既有 environments，不新增目錄。Bundle 逐檔 size/SHA 自動包含其 identity。正式模式要求 OBSERVED、非空 observed_at、runtime 及列出套件完全相符，必要核心套件不能省略；TBD／缺套件／版本不符 STOP。原研究 DRAFT gate 不變。
- `p4_colab.py`：speed-test 對環境版本差異繼續，stdout／run binding／各 speed JSON 保存 `environment_mismatch` 每項 expected／actual 與「結果不可與正式環境比較」；CUDA 缺失仍一律 STOP。正式模式以獨立 Colab 契約為準，本機 lock 差異另記 `local_lock_mismatch`，不拿 P3 版本取代 Colab authority。P4a 三臂同一環境 identity 才可互比，與 P3 跨環境比較須明列環境差異。
- 同步 DRAFT machine contract／spec、操作指南、Current State／Authority Map／Training domain／Worktree Register；`rolling_seq_scores.py`、`portfolio_eval.py` 與原 lock bytes 不變。V6/models、formal Data、production／scheduler 未變。

### Evidence

CPU 合成資料；Python3.12.3，`/home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python`，torch2.15.0+cpu／CUDA=False；MARKETMAMBA_DATA_ROOT unset，`Data/` 為本 worktree 普通目錄，非 canonical formal Data realpath。本輪測試 artifacts 在 pytest 隔離暫存 namespace，未使用真實 folds、Drive wheel bytes 或 GPU kernel。

RED：新增八項行為案例 **8 failed／69 deselected（3.31s）**，原因為 CLI 強制 SHA、wheel-dir API 缺失、環境 snapshot／差異記錄 API 缺失。Speed JSON 欄位補測另 **1 failed／77 deselected（1.27s）**。第一次 focused 為76pass／1fail，定位新fixture共享 packages 字典，修改預期時也改了模擬觀測；分離字典後通過，未放寬生產 gate。補足內部 hash／runtime／缺套件／CUDA 回歸後 focused：**86 passed（11.12s）**。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=V6:V6/research_lab/tests:. /home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python -m pytest V6/research_lab/tests/test_p4.py -q -p no:cacheprovider
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=V6:V6/research_lab/tests:. /home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python -m pytest V6/research_lab/tests tests -q -p no:cacheprovider
```

三項純記憶體變異，未改磁碟 source，各以 DID NOT RAISE 的預期斷言失敗證明被測試殺死：

```text
IN_MEMORY wheel pytest_exit=1 KILLED=True     # 停用 wheel tag gate
IN_MEMORY formal pytest_exit=1 KILLED=True    # 停用正式環境 mismatch STOP
IN_MEMORY manifest pytest_exit=1 KILLED=True  # 停用 extract 後逐檔驗證
```

TEMPORARY／可重建 logs：`/tmp/p4-colab-fix-{red,report-red,focused,full,impact,health}.log`、`/tmp/p4-colab-fix-mut-{wheel,formal,manifest}.log`，harness `/tmp/p4-colab-fix-mutate.py`。語法：3個Python source AST、2份JSON解析、5個指南Python儲存格解析通過；CLI help 確認 `--wheel-dir`。禁改兩支腳本與原lock逐位元比對HEAD一致。

Knowledge-impact：semantics YES、candidate authority YES（Colab獨立契約）、subsystem boundary NO、active runtime NO、correctness gate YES（speed mismatch與正式契約分離）、new canonical candidate artifact YES、deprecate NO、blocker/next decision YES（待pip list／自檢）。工具影響分析 PASS，但規則未涵蓋部分新模組；人工判斷仍更新四份knowledge。knowledge_health.py --ci exit0／Overall WARNING；Project Map／Agent Guides／Domains／Runtime Freshness／Generated Index均PASS，Project Map broken_links=0；既有治理及authority repin／legacy docs warnings沒有被繞過或改成PASS。

### Remaining／Blockers／Next recommended action

Human pip list、完整 runtime 自檢與observed_at尚未收到，故正式 Colab 契約仍TBD，正式模式仍STOP；研究契約仍DRAFT。獨立審查、真實 wheel 安裝、torch/CUDA ABI／GPU kernel、Drive durability、完整大型 bundle 成本均未驗證。先前報告的PAX isolated gate測試審查待辦不在本任務範圍，本輪沒有宣稱解決。下一步為獨立審查此patch；Human提供現場輸出後填實際契約、重新pack；無訓練或研究結論。

Project memory updated：Current_State.md／Training.md／Worktree_Register.md；Authority map updated：YES。Git status：起始clean，依no-commit保留本task dirty patch，未stage。Commit：N/A（禁止）；Push：未執行；Worktree：沿用builder_p4a_colab_fix，未建立或移除；Runtime/environment：WSL CPU synthetic only；GPU／網路／Colab／部署／排程／agents：0。

本輪完整 CPU suite 最終實際輸出：**663 passed／1 skipped／4 warnings in 162.43s（0:02:42）**，exit0；比原646增加17個P4案例（69→86），既有skip與Timestamp.dayofweek棄用warnings保留。未刪除／停用測試或放寬門檻。`git diff --check` exit0。Knowledge health結構檢查exit0／WARNING，沒有structural FAIL。

本輪受驗 source／契約與 TEMPORARY logs identities（不是研究 artifacts）：

```json
{
  "V6/research_lab/p4_bundle.py": {
    "bytes": 19883,
    "sha256": "1c8fb328ded61ec80b3f83e7d9986e0ea4d7fe211632f47ad6ef812767e4c18e"
  },
  "V6/research_lab/p4_colab.py": {
    "bytes": 30599,
    "sha256": "3f2df59f98f23631f1a0740a2a2d9366c2add695b3f73064e880ddc13d3eb235"
  },
  "V6/research_lab/tests/test_p4.py": {
    "bytes": 35154,
    "sha256": "fdde4ff6326034797265ee6d7d7d3ee7e8451a9ad2fe698e8ca9b5cee758b0ee"
  },
  "environments/p4a-colab-runtime.json": {
    "bytes": 1177,
    "sha256": "97997abe162040372cac33afd8ac85ff71abcb27b3a406ef98461b0656223f36"
  },
  "tasks/p4a-colab-contract-20261121.json": {
    "bytes": 3291,
    "sha256": "d779adf0c876465777a8afec5c500c52d278fe95cdc9a76dd79724270854ef6f"
  },
  "docs/operations/colab-p4.md": {
    "bytes": 14567,
    "sha256": "f4409292cbe6d85cec8a2c88b12303787566e594a10e2eb60f234785473c881a"
  },
  "/tmp/p4-colab-fix-red.log": {
    "bytes": 13273,
    "sha256": "fd1dcf59af356196fc6dde0fa71bcf39ec4ab71d12d74c6e8c9f35b883da762e"
  },
  "/tmp/p4-colab-fix-report-red.log": {
    "bytes": 996,
    "sha256": "dd18ecf7b3156966b3644e5e8a140694e689cc732fb98ed35c56e19c69693679"
  },
  "/tmp/p4-colab-fix-focused.log": {
    "bytes": 180,
    "sha256": "8384929b8fdcba56f376addc5effdc48056bbb5992b86a04bc7590eac9e9cbd3"
  },
  "/tmp/p4-colab-fix-full.log": {
    "bytes": 2043,
    "sha256": "f2fe1e67530534312594e1ac429bd2afb6aebc173cb4a325d375f89ebb34b22f"
  },
  "/tmp/p4-colab-fix-mutate.py": {
    "bytes": 1013,
    "sha256": "bf9accc60330318277e0956ea8811d538451b1eb31640f1e3a6a1c3d1628abc3"
  },
  "/tmp/p4-colab-fix-mut-wheel.log": {
    "bytes": 2180,
    "sha256": "d28816f90d7d1b915ad70498597bc93cdd0fbd874c6ae253b757503ddd63df6f"
  },
  "/tmp/p4-colab-fix-mut-formal.log": {
    "bytes": 2665,
    "sha256": "128bd264a20d157cd41bd2b3fa25b0c29162dfc331d19e8231d1d3a444bac41d"
  },
  "/tmp/p4-colab-fix-mut-manifest.log": {
    "bytes": 1490,
    "sha256": "ff5a8ca4793fa616942736d22d112dc9a692c943c88b41ce377b2ee9f2ad5fe3"
  }
}
```

本輪 changed paths 精確清單（11 modified＋1 untracked，全部未stage）：

```text
V6/research_lab/p4_bundle.py
V6/research_lab/p4_colab.py
V6/research_lab/tests/test_p4.py
docs/operations/colab-p4.md
environments/p4a-colab-runtime.json
tasks/p4a-colab-contract-20261121.json
tasks/spec-p4a-colab-20261121.md
reports/v7_p4a_colab_engineering.md
knowledge/00_Project_Map/Current_State.md
knowledge/00_Project_Map/Authority_Map.md
knowledge/00_Project_Map/Worktree_Register.md
knowledge/01_Domains/Training.md
```

## P4a 兩檔交付工程候選 — 2026-11-22

本節是 `job-run-mwqr9yiu-8c464e` builder 的未提交 patch，base/HEAD `74795a4a56f25585000d26d281e305381de6230a`；UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED，工程完成不代表 Human acceptance。以下取代舊手動 bootstrap 上傳主流程，不改寫歷史結果／歷史驗證紀錄。使用者本輪明確限定不 commit/push/merge/GPU/network/Colab/agents，全部遵守。

### 變更與決策

- 沿用既有 `V6/research_lab/p4_bundle.py`；新增 `deliver --folds-root ... --years ... --p1 ... --lock ... --output-dir ...` 一次發布 `p4a-bundle.zip`、`p4a-colab.ipynb`，拒絕既有輸出。既有 `pack` 依 `.zip` 副檔名使用真正 ZIP／ZIP64，其餘 tar.gz 保留。既有 source／matrix／P1／lock／runtime contract／manifest 內容邊界不變，不另造平行implementation／頂層artifact tree；僅在pytest隔離目錄產生合成兩檔。
- ZIP 中相同 `(bytes,SHA-256)` 的跨fold `X.npy` 只存第一份；manifest `files` 是完整邏輯檔案identity，`aliases` 指向實體成員，不允許鏈結、缺target、非X或identity不同。解壓重新串流讀取target，還原每fold的一般檔案，逐份核對大小／SHA。打包端只對fold input解析symlink，其餘輸入沿用regular要求；解壓端ZIP與tar均拒絕links／危險路徑／非白名單／cache，沿用descriptor安全輸出。
- ZIP inventory、manifest大小、單檔64GiB、邏輯還原總量256GiB、10,000成員上限在建立destination前核對；payload以1MiB串流分塊讀寫並核hash，partial失敗保留盤點，不自動重試。格式偵測曾移動檔案游標導致tar回歸失敗，已定位並補seek(0)，既有tar測試恢復通過。
- Notebook第0格集中所有Human預算／路徑選擇，預設SPEED_TEST、6小時、100CU、10CU/hour、每fit0.5小時、25%餘裕。先找/content再找Drive inputs，無檔明確STOP；標準庫只讀固定bootstrap及manifest並限制大小／類型、驗bootstrap內部hash，再完整安全解壓。無手貼SHA／另傳bootstrap。內部hash只能證bytes一致，不能自證來源；兩檔都必須可信。
- 環境先自檢Python／torch／CUDA／GPU／套件；符合契約略過wheel安裝，否則沿用既有no-deps／no-index／find-links計畫。補cp313正向與cp312不符STOP；CUDA不可用一律STOP。兩wheel均缺才執行官方PyPI sdist編譯格，核sdist SHA、mamba所有C++17旗標改C++20、MAMBA_FORCE_BUILD=TRUE、MAX_JOBS=4、T4 sm75，輸出Drive；安裝／略過後皆執行Mamba2小型前後向有限值冒煙及pip check。
- SPEED_TEST沿用既有三臂速度輸出隔離，不產IC／績效；FORMAL沿用validate_contract拒絕DRAFT，未改研究threshold。JSON／run manifest直接寫Drive，摘要秒/epoch、GPU、溫度／時脈，下載格匯出JSON與ledger。重連比對同一zip manifest，沿用既有completed-fit resume及budget root，不重置帳本；未知中斷仍扣帳STOP。
- `docs/operations/colab-p4.md` 主流程改為兩檔，舊tar／逐格／cp312段落明列備援歷史範例；目前版本以既有OBSERVED runtime契約為準。Current_State、Authority_Map、Training與Worktree_Register同步更新；沒有改runtime observed_at／machine契約／本機P3 lock或production authority。

### CPU驗證與實際輸出

初始RED（實作前，原輸出保留於 `/tmp/p4-delivery-red.log`）：`7 failed, 1 passed, 86 deselected in 1.99s`，包括缺ZIP支援／fold symlink拒絕，以及當時CPU環境缺nbformat。補離線nbformat後，另以HEAD原始p4_bundle source在記憶體重跑三個新行為測試確認baseline RED（不改工作樹source）。

測試Python為 `/home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python`，Python3.12.3、torch2.15.0+cpu、CUDA=False，`MARKETMAMBA_DATA_ROOT` unset；Data是本worktree普通目錄，realpath `/home/frank/projects/MarketMamba/.worktrees/ruflo-workers/run-mwqr9yiu-8c464e/builder_p4a_delivery/Data`，未讀寫formal data作本輪實驗。nbformat5.9.2／traitlets／jupyter_core／fastjsonschema從既有Windows Anaconda純Python套件唯讀複製到 `/tmp/p4-notebook-deps`；jsonschema等從本機pip cache解壓，沿用venv的platformdirs，未連網、未變更既有環境。此暫存namespace可重建，不是新增canonical environment。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/tmp/p4-notebook-deps:V6:V6/research_lab/tests:. /home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python -m pytest V6/research_lab/tests/test_p4.py -q -p no:cacheprovider
```

```text
107 passed, 1 warning in 19.55s
nbformat 5.9.2 valid; code cells 7
syntax_files=2 nbformat_version=5.9.2 code_cells=7
IN_MEMORY zip pytest_exit=1 KILLED=True
IN_MEMORY dedup pytest_exit=1 KILLED=True
IN_MEMORY draft pytest_exit=1 KILLED=True
```

三項變異均只在獨立Python程序的記憶體替換函式：ZIP停用regular/link檢查，由具有合法manifest的symlink regression捕捉；清空dedup aliases，由ZIP只存一份及正確對應測試捕捉；停用DRAFT狀態拒絕，由既有正式契約測試捕捉。source檔從未寫入mutant。Logs為 `/tmp/p4-delivery-mut-{zip,dedup,draft}.log`，runner `/tmp/p4-delivery-mutate.py`。

測試涵蓋真正ZIP、兩fold共用symlink X、去重實體成員數與alias、解壓九檔逐位元等於原始輸入、兩檔CLI、重複輸出拒絕、標準庫bootstrap與同root重連、每cell compile／nbformat JSON、預算設定合法、實際CLI parser、已裝環境跳過pip、雙缺wheel備援分支與no-CUDA無動作。ZIP惡意路徑／symlink／cache／非白名單／duplicate／missing／超限／偽造alias／corrupt payload均有CPU回歸；既有tar惡意成員及完整runner隔離測試保留。

第一輪完整測試（最終三個環境分支測試加入前、尚未載入離線nbformat）：`680 passed, 2 skipped, 5 warnings in 248.97s (0:04:08)`。兩skip為既有GPU smoke與當輪缺nbformat；其後P4 focused已包含nbformat並全過。

補充baseline RED：`3 failed, 104 deselected in 4.73s`、`BASELINE_RED source=74795a4 pytest_exit=1`，分別因fold symlink不支援、缺colab_notebook、CLI不認deliver；log `/tmp/p4-delivery-baseline-red.log`。

### Artifact identity 與限制

合成fixture（不是可訓練真實資料、不可作研究）例：`/tmp/pytest-of-frank/pytest-69/test_deliver_cli_synthetic_two0/delivery/` 的ZIP為860688 bytes，SHA-256 `ac3749b9c36f3394162893ed8b2bbc61a2f7d9f2c3833fcb079cb2015eef519f`；notebook為11783 bytes，SHA-256 `052f99d9b4db918f79633d960b146f1b48053105ba0ed0dcd2ec30d4f55e8703`。Input只有合成原始fold／P1 identity，contract/lock/runtime identity為本checkout；ZIP metadata記錄source commit、未提交狀態、輸入realpath、環境與ENGINEERING_ONLY。沒有真實大型fold打包、矩陣訓練或模型結果；正式兩檔應在審查後以真實已核准輸入重新deliver，不能拿合成ZIP訓練。

本輪source identities：

| Path | bytes | SHA-256 |
|---|---:|---|
| V6/research_lab/p4_bundle.py | 38021 | ec103a139c4c2383ed52687957f40c3954dfd87d540b217be162bf1c06b386c0 |
| V6/research_lab/tests/test_p4.py | 44537 | 08992fc4ba17272e624dfd28f4b2a313b787299a8d9fbcd15408f1dc4128b87a |
| docs/operations/colab-p4.md | 18751 | 13f87a965b569cf3b8ad4db79aadbffc4b3071c2cf8d2a69c4cd50a7f4d22137 |
| environments/p4a-colab-runtime.json（未改） | 2350 | b45380920c48c5548ef8880f63f4a351a3de7ff5d6b9f2abef50ebf87e763194 |
| environments/p4a-colab.requirements.lock.txt（未改） | 2659 | 2c39a3fb9567d1f9fc8ee83369c7b6c90cd19a8e679e32e4c8a6c1d75c5782c9 |
| tasks/p4a-colab-contract-20261121.json（未改） | 3291 | d779adf0c876465777a8afec5c500c52d278fe95cdc9a76dd79724270854ef6f |

Colab現場的ZIP解壓／大型資料磁碟與I/O／編譯儲存格／sdist現場內容／Drive掛載、fsync、atomic rename、斷線持久性與download皆 **UNVERIFIED**。Human既有T4輪子與小型Mamba2成功是OBSERVED_BY_HUMAN_PASTE，不是本worker執行、不是專案official GPU smoke。編譯時間估計30–90分鐘可能超過，需另預留CU；runner ledger不含上傳、安裝／編譯、閒置，不是Google billing API。不提供partial optimizer自動resume或清除STOP介面。JSON下載不含正式權重，不能聲稱完整formal run備份。

Knowledge-impact人工逐項：semantics=YES（交付／ZIP／啟動UX）；authority=NO（沒有研究或production提升）；subsystem boundary=NO；active runtime state=NO；correctness gate=YES（新增ZIP gate，不降低舊門檻）；new/superseded canonical artifact=NO（generated ZIP與notebook）；deprecation=NO（tar保留備援）；major blocker/next decision=YES（Human OBSERVED環境已存在，下一步改為審查與現場ZIP／編譯／Drive驗證）。已同change更新Current_State／Authority_Map／Training／Worktree_Register。`check_knowledge_impact.py` PASS；`knowledge_health.py --ci` exit0，Overall WARNING（Governance、Authority Drift、Legacy Docs），Project Map／Agent Guides／Domains／Runtime Freshness／Generated Index PASS，沒有structural FAIL。未自行消除既有歷史／authority warnings或放寬gate。

### 最終完整回歸（含離線nbformat與全部新增測試）

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/tmp/p4-notebook-deps:V6:V6/research_lab/tests:. /home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python -m pytest V6/research_lab/tests tests -q -p no:cacheprovider
```

實際stdout（`/tmp/p4-delivery-full-final.log`；exit0）：

```text
........................................................................ [ 10%]
........................................................................ [ 21%]
........................................................................ [ 31%]
........................................................................ [ 42%]
........................................................................ [ 52%]
........................................................................ [ 63%]
........................................................................ [ 73%]
........................................................................ [ 84%]
...............................................s........................ [ 94%]
.....................................                                    [100%]
=============================== warnings summary ===============================
research_lab/tests/test_p4.py::test_zip_inventory_and_payload_gates[duplicate]
  /usr/lib/python3.12/zipfile/__init__.py:1620: UserWarning: Duplicate name: 'folds/fold_2021/X.npy'
    return self._open_to_write(zinfo, force_zip64=force_zip64)

test_stale_audit.py::test_mutation_controls[sessions = sorted(set(pd.to_datetime(trading_dates).strftime("%Y-%m-%d")))-sessions = sorted(d for d in daily if pd.Timestamp(d).dayofweek < 5)]
test_stale_audit.py::test_mutation_controls[sessions = sorted(set(pd.to_datetime(trading_dates).strftime("%Y-%m-%d")))-sessions = sorted(d for d in daily if pd.Timestamp(d).dayofweek < 5)]
test_stale_audit.py::test_mutation_controls[sessions = sorted(set(pd.to_datetime(trading_dates).strftime("%Y-%m-%d")))-sessions = sorted(d for d in daily if pd.Timestamp(d).dayofweek < 5)]
test_stale_audit.py::test_mutation_controls[sessions = sorted(set(pd.to_datetime(trading_dates).strftime("%Y-%m-%d")))-sessions = sorted(d for d in daily if pd.Timestamp(d).dayofweek < 5)]
  /tmp/pytest-of-frank/pytest-70/test_mutation_controls_session0/mutant.py:93: Pandas4Warning: Timestamp.dayofweek is deprecated and will be removed in a future version. Use Timestamp.day_of_week instead.
    sessions = sorted(d for d in daily if pd.Timestamp(d).dayofweek < 5)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
684 passed, 1 skipped, 5 warnings in 230.28s (0:03:50)
```

唯一skip為既有需CUDA測試；本輪新增nbformat測試實際執行。Warnings為既有Pandas4與刻意建立duplicate ZIP成員的fixture警告，沒有刪測試或降低門檻。

### 固定交接欄位

- Completed：兩檔交付產生器、ZIP內容去重／安全解壓、notebook及cp313／備援指南／CPU驗證完成；不等於Human acceptance。
- Changed：`V6/research_lab/p4_bundle.py`、`V6/research_lab/tests/test_p4.py`、`docs/operations/colab-p4.md`、本報告、`knowledge/00_Project_Map/{Current_State,Authority_Map,Worktree_Register}.md`、`knowledge/01_Domains/Training.md`，共8個modified tracked paths；無新增untracked／staged。
- Decisions：沿用既有p4_bundle／p4_colab；tar保留備援；不改rolling_seq_scores.py／portfolio_eval.py／研究契約／environment lock／runtime observation；FORMAL DRAFT仍STOP。
- Evidence：初始RED 7fail、離線依賴補齊後baseline RED 3fail；P4 GREEN107pass；完整684pass/1skip/5warning；三個in-memory mutants全killed；nbformat5.9.2解析7code cells；syntax／逐檔size-SHA／知識links／git diff --check已驗。
- Remaining：真實大型bundle打包、Colab現場ZIP／編譯／Drive／download UNVERIFIED，無研究或GPU結果；正式研究需freeze。
- Blockers：本輪CPU工程無阻擋；現場驗證與Human acceptance不在本輪執行範圍。
- Next recommended action：獨立審查本patch，核准真實輸入後deliver兩檔，再另行授權現場GPU／Drive驗證；不能將合成ZIP拿去訓練。
- Project memory updated：YES，Current_State／Training／Worktree_Register。
- Authority map updated：YES，候選交付邊界；未提升研究／production authority。
- Git status：起始clean detached HEAD；結束8個modified tracked paths、0 staged／untracked；依no-commit授權刻意保留dirty，未宣稱clean。
- Commit：N/A（明確禁止）；Push：未執行；Merge：未執行。
- Worktree：既有 `.worktrees/ruflo-workers/run-mwqr9yiu-8c464e/builder_p4a_delivery`，HEAD74795a4；未新建／刪除worktree。
- Runtime/environment：WSL本機Python3.12.3／torch2.15.0+cpu，CUDA=False；Data realpath與offline nbformat namespace見上；無GPU／網路／Colab／新增agents／scheduler／production／data acquisition／protected models變更。


## 兩檔交付複審結果與已知限制（2026-11-22，第五輪複審）

複審（HEAD `684b97d`）判定：CRITICAL 0、REQUIRED 2；全套 683 passed／2 skipped（nbformat 缺失的 1 項以離線依賴補跑 2 passed）；真實產物 `p4a-bundle.zip`（840862631 bytes，SHA-256 `263beacb2e3c3b8d5e4477f0d68ff375d464204484d9b665e513caffd840d776`）與 `p4a-colab.ipynb`（11783 bytes，SHA-256 `052f99d9b4db918f79633d960b146f1b48053105ba0ed0dcd2ec30d4f55e8703`）經 notebook 同一 bootstrap／解壓路徑驗證 PASS（240 檔、2021 fold 九檔與來源一致）。

兩項 REQUIRED 的處置（主控決定，Human 的威脅模型為『自己打包、自己上傳、自己的 Colab』，且明確要求降低儀式感）：
1. 打包端對 `folds/` 輸入 `resolve(strict=True)` 後未限制解析後來源範圍（可將誤置符號連結指向的其他檔案包入）：**暫時接受為已知限制**，輸入為 Human 自己的 fold 目錄；正式 5 折打包前須補『解析結果必須落在指定 matrices 根目錄內』的檢查。
2. ZIP 缺壓縮比上限（真實檔最高約 1014，不能任意設低值）：**暫時接受為已知限制**，zip 為 Human 自己產生；正式打包前須補以邏輯大小與矩陣預期大小為基準的上限與回歸測試。
兩項皆記為 OPEN，不得在『正式 P4a』准入時省略；speed-test 交付不依賴它們被修正。複審的 GO／NO_GO 為 NO_GO，本節是主控對該結論的明示偏離，非複審通過。

## P4a Drive 分次交付路徑候選 — 2026-11-22（job-run-mwqscvk1-a3db0f）

UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED。本輪在既有 `builder_p4a_drive` worktree，base／HEAD `df78d6cb3bb706f6f4d58b94310c3d352aab8385`，起始 clean detached HEAD；依 Human EXECUTE_APPROVED 授權僅保留 patch，無 commit／push／merge／額外 agents／GPU／網路／Colab。工程完成不等於 Human acceptance。本節取代前輪操作預設與交付路徑描述，不修改歷史測試或實驗證據。

### 變更與界線

沿用 `V6/research_lab/p4_bundle.py`、原 pack／executor／測試與既有 docs／reports／knowledge 容器，不建立新 implementation、artifact tree 或 worktree。`deliver` 新增 `--run-name`（預設01_speedtest）與 `--drive-root`（預設/content/drive/MyDrive/MarketMamba）；仍輸出到 `--output-dir`，另外寫兩行 README.txt：上傳 Drive 目的資料夾、開啟 notebook 檔名。三個輸出任一已存在皆拒絕覆寫，run-name 限單層資料夾，Drive root 必須為絕對路徑。CLI 回傳三檔大小／SHA。

Notebook第0格預填 RUN_DIR、EXPERIMENT_DIR（RUN_DIR上一層）與 WHEEL_DIR。交付目錄是 `MarketMamba/p4a/<run-name>`；runner、摘要及 JSON 匯出均固定使用 `EXPERIMENT_DIR/budget`。speed 與 formal 換交付子資料夾不會改帳本根，不遷移／刪除／重置舊帳本；舊根已有消耗時仍須保留原 EXPERIMENT_DIR，帳本遷移另行盤點。WHEEL_DIR預設 `MarketMamba/wheels`，空／不存在則回退 `Mamba_Wheels_T4_new`；兩處皆缺必要wheel才編譯，輸出回 WHEEL_DIR。已裝套件符合契約仍略過安裝；部分wheel、重複版本或標籤不符維持STOP，不跨目錄拼湊wheel。

ZIP搜尋順序是 RUN_DIR → EXPERIMENT_DIR/inputs → /content；找不到時列出全部候選完整路徑。`deliver` 自動將ZIP實際bytes／SHA綁到同批notebook，不需Human手貼；殘留/content同名檔SHA不符明確WARNING並忽略，即使Drive找不到也不採用。選中的Drive版identity不符則STOP，不靜默改用其他副本。

ZIP壓縮後bytes不能與邏輯還原大小直接比較，因此採兩層檢查：先驗實際ZIP長度／SHA與notebook綁定交付identity一致；再驗manifest新增 `total_bytes` 等於完整 `files` 邏輯大小總和（含aliases），ZIP非alias實體成員的未壓縮大小則逐檔及加總比對manifest。舊manifest省略total_bytes時以files加總相容。大小不符／截斷均於bootstrap載入和解壓前STOP，提示可能上傳不完整；原白名單、資源上限、安全解壓與逐檔size／SHA驗證繼續強制。這不是來源認證，兩檔仍須可信。

SPEED_TEST預設1.0小時／15.0 CU；FORMAL兩上限無預設，未明填就在第0格STOP。保留10 CU/hour、每fit/stage0.5小時、25%reserve。全新speed三臂加一stage的剩餘計畫至少 `(3+1)*0.5*1.25=2.5小時／25 CU`，尚未含已花費，因此新小上限會觸發既有計畫STOP；本輪沒有降低估計或gate，也不自行提高預算。Human須依實際保守估計與完整剩餘計畫另行核定。編譯、上傳與閒置仍不在runner估算帳本，不能把CU稱Google實扣。

`rolling_seq_scores.py`、`portfolio_eval.py`、`p4_colab.py`、P4 DRAFT研究契約、runtime觀測時間、environment lock、formal Data、V6/models、production及scheduler不變。既有resolved fold來源範圍與ZIP壓縮比兩項OPEN限制仍OPEN，正式五折准入前仍需處理；本輪不宣稱複審GO或正式研究readiness。

### 驗證方法與證據

Python：`/home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python`，3.12.3／torch2.15.0+cpu；GPU隱藏 `CUDA_VISIBLE_DEVICES=''`，單CPU執行緒。沿用既有離線 `/tmp/p4-notebook-deps` 的nbformat5.9.2及相依套件，沒有安裝／下載／改venv。`MARKETMAMBA_DATA_ROOT` unset；Data是本worktree一般目錄，realpath `/home/frank/projects/MarketMamba/.worktrees/ruflo-workers/run-mwqscvk1-a3db0f/builder_p4a_drive/Data`，未以formal資料執行實驗。

RED：新行為測試 `6 failed, 107 deselected in 3.93s`（`/tmp/p4-drive-red.log`）；CLI `--run-name/--drive-root` 尚不存在，`1 failed, 112 deselected in 1.59s`（`/tmp/p4-drive-cli-red.log`）。GREEN focused：`113 passed, 1 warning in 16.58s`（`/tmp/p4-drive-focused-final.log`）。首次focused有一項測試誤取摘要格而非runner格，依實際失敗定位更正索引後通過，沒有停用測試。

新增測試涵蓋第0格1/15上限、FORMAL未填STOP、custom run／Drive root、README精確兩行、ZIP預期identity綁定、跨交付及speed/formal相同budget路徑、三路搜尋順序、殘留/content警告／忽略及搜尋路徑錯誤訊息、wheel首選／舊目錄回退／編譯目的地、ZIP截斷／manifest total／成員宣告大小不符STOP。既有兩次實際stdlib bootstrap／解壓／同root重連測試仍執行；nbformat解析及全部7個code cells compile實際PASS，非skip。

三個mutants僅在獨立程序記憶體覆寫colab_notebook回傳cell，不寫入source；runner `/tmp/p4-drive-mutate.py`：

```text
IN_MEMORY order pytest_exit=1 KILLED=True
IN_MEMORY budget pytest_exit=1 KILLED=True
IN_MEMORY size pytest_exit=1 KILLED=True
```

分別倒轉ZIP候選順序、將runner帳本綁到RUN_DIR、移除整個大小檢查；各由對應行為測試抓到。Logs：`/tmp/p4-drive-mut-{order,budget,size}.log`，皆TEMPORARY／ENGINEERING_ONLY，不是研究artifact。

完整回歸命令：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/tmp/p4-notebook-deps:V6:V6/research_lab/tests:. \
/home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python -m pytest V6/research_lab/tests tests -q -p no:cacheprovider
```

實際全套輸出（`/tmp/p4-drive-full.log`，exit0）：**690 passed, 1 skipped, 5 warnings in 181.88s (0:03:01)**。唯一skip為既有CUDA測試；warnings為刻意建立duplicate ZIP的fixture與既有Pandas4 Timestamp.dayofweek棄用警告，未刪測試／放寬門檻。

受驗source identity：

| Path | bytes | SHA-256 |
|---|---:|---|
| V6/research_lab/p4_bundle.py | 42549 | 391c24683f408311898d7026adf585b2cb17fd2fbc31dc1acc4966e70dc8831a |
| V6/research_lab/tests/test_p4.py | 49806 | 3a3f3abda43fbabb3b5345613b4194dbef312799e808731a7f5d8846c7a421e8 |

全套中的合成fixture為 `/tmp/pytest-of-frank/pytest-81/test_deliver_cli_synthetic_two0/delivery`，只含測試bytes，**不可拿去訓練**：ZIP862938 bytes、SHA `7cdcc506f465589d6ff4cc828ff6c2b7bdff7e51d0037a1eeba09fd16ec2fcfc`；notebook14807 bytes、SHA `72aa8da272876deab06152ec5c24ee16868ddc46ab0949668cc7338280f6395c`；README100 bytes、SHA `4cb4cb4f94097cbcd61e7873d77fd6170f29dbec62391ae742b1587693d747b9`。notebook內嵌EXPECTED_ARCHIVE另行重核等於實際ZIP identity。pack manifest保留source commit／dirty狀態、逐檔source及合成輸入identity；無model／research結果，evidence class仍ENGINEERING_ONLY。

Knowledge-impact逐項：semantics=YES（交付路徑／預設／大小檢查）；authority=NO；subsystem boundary=NO；active runtime state=NO；correctness gate=YES（新增預檢，無降標）；new/superseded canonical artifact=NO（交付檔為GENERATED）；deprecate=NO；major blocker/next decision=YES（新增小預算下計畫STOP需Human核定）。Current_State、Authority_Map、Training與Worktree_Register已同change更新。`check_knowledge_impact.py` exit0／WARNING，治理與worktree責任規則要求人工review：Operations與GOVERNANCE已讀，操作子系統／治理規則未變，不需改該兩檔。並非以NO_KNOWLEDGE_CHANGE_REQUIRED略過本輪knowledge更新。

`knowledge_health.py --ci` exit0，Overall WARNING（Governance、Authority Drift、Legacy Docs）；Project Map／Agent Guides／Domains／Runtime Freshness／Generated Index全部PASS，無structural FAIL。未修改runtime observed_at、authority-map.json或generated indexes。syntax、nbformat／每格compile、artifact identity、連結／reference與`git diff --check`驗證；受保護路徑及p4_colab／environment／P4契約相對HEAD無diff。依code-review-and-quality對本輪差異作五軸自查，沒有新增獨立審查者；後續獨立review仍必要，不宣稱Human acceptance。

### 固定交接欄位

- Completed：Drive分次交付設定、共用budget、wheel回退、ZIP預檢、CLI／README及CPU RED→GREEN完成；不等於Human acceptance。
- Changed：`V6/research_lab/p4_bundle.py`、`V6/research_lab/tests/test_p4.py`、`docs/operations/colab-p4.md`、本報告、`knowledge/00_Project_Map/{Current_State,Authority_Map,Worktree_Register}.md`、`knowledge/01_Domains/Training.md`，共8個tracked paths。
- Decisions：沿用既有產生器／executor／knowledge；大小分壓縮identity與manifest邏輯大小核對；FORMAL無預設；既有budget、DRAFT／CUDA／25%reserve gate不變。
- Evidence：RED6項＋CLI1項；focused113passed；full690passed／1skipped／5warnings；3項in-memory mutants全killed；nbformat5.9.2／7code cells compile；syntax、size／SHA、link／reference、protected-path diff及git diff --check。
- Remaining：未產出／上傳真實交付、未操作或驗證Colab／GPU／Drive／編譯；未作研究結論。正式freeze與既有兩項OPEN限制待處理。
- Blockers：本輪本機工程無阻擋；預設1小時／15CU不足既有保守全新speed計畫，會STOP，須Human核定估計／預算，不在本輪擅改。
- Next recommended action：獨立審查patch與預算，核准真實輸入後重新deliver三檔，再由Human上傳指定Drive資料夾；現場驗證另行授權。
- Project memory updated：YES（Current_State／Training／Worktree_Register）。
- Authority map updated：YES（候選操作邊界）；研究／production authority未變。
- Git status：起始clean detached HEAD；結束8個modified tracked paths、0 staged／untracked。依no-commit指示刻意保留dirty patch，未宣稱clean。
- Commit：N/A（明確禁止）；Push：未執行；Merge：未執行。
- Worktree：沿用既有 `.worktrees/ruflo-workers/run-mwqscvk1-a3db0f/builder_p4a_drive`，HEAD df78d6c；未新建／刪除worktree。
- Runtime/environment：WSL本機Python3.12.3／torch2.15.0+cpu／CUDA隱藏、MARKETMAMBA_DATA_ROOT unset；Data realpath與離線nbformat namespace如上。無GPU／網路／Colab／data acquisition／agents／scheduler／production／protected models變更。
