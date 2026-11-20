# P3 修訂 3 預算錨點工程候選 — 2026-11-20

依[凍結修訂 3](../tasks/spec-mamba-portfolio-p3-20261120.md)，job `job-run-mwoibr4r-44988a`，detached base `39320a04512c446fb75aa993690a7496e434e57b`；UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED。本節更新修訂 2 的預算完整性及恢復程序，其他准入／研究假設保留。沿用既有 adapter、test_p3、report、project map 與 `.artifacts/p3/`；`revision3-verification/` 為此 source/task 專用 GENERATED / SYNTHETIC_CPU_ENGINEERING_ONLY logs，避免覆寫先前證據；不是新的研究實驗或結果 authority。

## 錨點與發布順序

現有 hash chain 只能驗內容，無法辨識合法前綴截斷／清空。本輪新增全域 `.artifacts/p3/budget-anchor.json`，與 ledger 共用原有固定目錄 inode flock，不依賴 out-dir。每次事件先 append／flush／fsync ledger 及其目錄，再用同目錄 exclusive 暫存檔寫出 `sequence`（事件數）、`head_sha256`、`spent_seconds`、`observed_at`，fsync 檔案後 `os.replace` 原子替換，最後 fsync 目錄。spent 從包含該次新事件的完整有效帳本計算，不依賴尚未更新的 instance counter。fresh namespace 唯一可自動建立零事件錨點／空帳本；有任一舊 state 或 fit／sidecar／run_identity 時不可視為 fresh。

啟動在 recovery／STOP 清除／任何新 START 前驗證：事件數、頭 hash、spent 必須與錨點完全一致，任一不符即拒絕。這比僅檢查 spent 不回退更嚴格；ledger 已 fsync 但 anchor 尚未替換時中斷，也會 fail closed，須顯式恢復。再驗所有存活 sidecar 的 snapshot 是否是有效帳本前綴，以拒絕帳本與錨點共同回退卻留下新 fit 的狀況。sidecar 存的是發布時的三元組；payload 在 fit 計時範圍內落盤並 fsync，END 扣帳及錨點完成後才 exclusive 發布 sidecar，包含該 fit 的全部已記帳耗時。manifest 同樣記三元組與完整 ledger identity；evaluator 沿既有 revalidate_manifest 再驗 snapshot、帳本、准入與24個 completed bindings。

## 顯式重建（本輪只於合成 tmp fixtures 執行）

先停止寫入並盤點所有 namespace；不要刪 sidecar 或改寫歷史輸出。確認原因後，在 repository root 執行：

```bash
PYTHONPATH=V6 /home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python \
  -m research_lab.rolling_seq_scores --rebuild-budget \
  --reason '<已核對的損壞原因、sidecar inventory 與未知耗時限制>'
```

命令使用同一目錄鎖，逐一讀取所有 namespace 的 `fold_*.json`，取 sidecar `budget.spent_seconds` **最大值**作 `sidecar_spent_lower_bound`，不是相加，也不是只讀指定 out-dir。新帳本首事件為 REBUILT，包含非空 reason、時間、sidecar identities、下界及實際 debit；實際 debit 取 sidecar 下界、存活 anchor spent、可驗證帳本 spent／pending 中斷保守扣帳的最大值。每個仍存在的舊帳本／錨點以 `budget-rebuild-<uuid>-<原檔名>` 保留原始 bytes，REBUILT 記錄其 path／size／SHA；不是新研究結果。重建後仍扣除該 debit，可用預算為14h減debit；50500秒的合成案例即使清除 STOP 也不能再啟動。

重建會另加持久 STOP；必須檢查保留歷史、未知中斷成本後，沿既有 `--clear-stop STOP-<id>.json --reason '<檢查結果>'` 留 CLEARED。刪 STOP 檔無效。重建不替舊 fit 補造 START／COMPLETED，不授予舊輸出新的確認性／續跑 authority；需新 namespace。已盤點的 sidecar 僅按 REBUILT 綁定的 exact identity 視為舊鏈證據，其他 sidecar 仍須通過有效前綴驗證。缺 spent 的舊 sidecar、無 sidecar 的 orphan payload／無法解析的 sidecar 不會被估成0，命令拒絕，需另外人工盤點。無法解析的 anchor 也會拒絕，不能以壞值扣帳。

## 實際 RED→GREEN 與反證

命令前綴均為 `PYTHONPATH=V6:V6/research_lab/tests:. /home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python`。初始 `-m pytest V6/research_lab/tests/test_p3.py -k revision3 -q`：

```text
10 failed, 83 deselected in 2.28s
```

逐項 RED 包含 empty、valid_prefix、BUDGET_BYPASS 換 out-dir、缺 anchor／ledger、sequence／head／spent mismatch、REBUILT 及原子 fsync。初版修復捕捉到恢復物件缺 stops 初始化，保留 `green-initial.log`（1 failed / 9 passed），修正後加強存活 evidence、共同回退與重建耗盡／未知 sidecar 檢查。原 P3 全項實跑 `93 passed in 80.09s`；補強後最終 focused `16 passed, 83 deselected in 3.25s`（`green-final.log`）。最終 sidecar helper 必須顯式傳入 budget，不再允許省略後產生缺 snapshot 的 sidecar。

in-memory 反證將當前 source 讀入記憶體後變異並交 pytest 執行，不改 repository implementation；可重跑 `revision3-verification/mutation_runner.py`。以下是實際輸出，每個子程序 exit1，預期失敗不算正常套件通過：

```text
empty           1  1 failed in 0.18s
valid_prefix    1  1 failed in 0.18s
BUDGET_BYPASS   1  1 failed in 0.21s
anchor_mismatch 1  3 failed in 0.47s
missing_state   1  2 failed in 0.37s
REBUILT         1  1 failed in 0.13s
atomic          1  1 failed in 0.11s
```

移除錨點驗證會讓清空／合法前綴／換 out-dir 直接進入 `DID NOT RAISE ValueError`；遺失 state 自動補零同樣被拒絕測試抓到；REBUILT 使用最小值會出現 `assert 0.0 == 456.0`；直接覆寫錨點而不 replace／fsync 會出現 `assert 0 == 3`。錨點三種 mismatch 均使用未耗盡預算，移除驗證即 `DID NOT RAISE ValueError`，避免只靠後續耗盡 gate 偶然拒絕。完整 traceback、命令、來源及 artifact hashes 見忽略目錄 `.artifacts/p3/revision3-verification/`。

逐項 GREEN（`category_checks.py` 以實際 pytest node 重跑，exit0）：

```text
empty            0  1 passed in 1.81s
valid_prefix     0  1 passed in 2.04s
BUDGET_BYPASS    0  1 passed in 1.98s
missing_state    0  2 passed in 2.03s
anchor_mismatch  0  3 passed in 2.15s
REBUILT          0  1 passed in 1.80s
```

指定完整命令最終實際輸出：

```bash
PYTHONPATH=V6:V6/research_lab/tests:. /home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python -m pytest V6/research_lab/tests tests -q
```

```text
574 passed, 1 skipped, 4 warnings in 177.00s (0:02:56)
```

語法 AST／py_compile、artifact SHA-256 round-trip、同 change 新增 Markdown links、`git diff --check` 通過；`check_knowledge_impact.py` exit0／PASS，人工判定 budget correctness gate／恢復語義改變，已更新 Current State、Authority Map、Research domain、Worktree Register。`knowledge_health.py --ci` 的最終實際結果及所有檔案 identities 見 `revision3-verification/verification.json`，WARNING 不作通過研究 gate 的依據。

Completed：修訂3錨點／恢復／六類破壞回歸／反證。Changed：adapter、test_p3、工程報告及上述4份knowledge。Decisions：缺 state拒絕／顯式重建／最大下界扣帳／重建STOP；防疏失不防刻意清除全部證據。Evidence：上述CPU logs及verification manifest。Remaining：獨立審查、GPU smoke、24fits及Human acceptance。Blockers：GPU_SMOKE_NOT_RUN，無正式P3研究結果；本task無未完成實作項。Next recommended action：主控獨立審查diff、logs與重建邊界。Project memory updated：YES；Authority map updated：YES，僅candidate boundary。Git status：起始clean，最終7個指定tracked檔案unstaged modified，no-commit依指示不宣稱clean。Commit／Push：N/A（禁止），無merge／deploy。Worktree：既有run-mwoibr4r fixer，dirty不可安全移除。Runtime/environment：WSL／Python3.12.3 pilot09 venv／CPU，詳見範圍聲明。

## 範圍與風險

此機制**防疏失與誤操作，不防刻意同時刪除帳本、錨點與全部 sidecar**；也不是外部不可變儲存、防竄改硬體或安全審計簽章。目錄及其父路徑仍須可信且穩定；不宣稱非合作寫入者／檔案系統故障下仍可自動恢復。ledger 與 anchor 是兩次 durable publication，跨檔中斷採拒絕與顯式重建，不稱跨檔交易。遺失證據可能隱藏 sidecar 前／未完成 attempt 的成本；REBUILT 下界不代表精確 GPU 時數，STOP 檢查不能省略。無 spent 的 legacy sidecar 不能透過本命令靜默遷移。

Runtime/environment：現有 WSL-native fixer，Python3.12.3 pilot09 venv，CPU，MARKETMAMBA_DATA_ROOT unset，Data realpath 為本 checkout 的普通目錄，未當成外部 formal matrix；沒有改 Data、V6/models、正式歷史 artifacts，也沒有 GPU、acquisition、production、scheduler、wakeups、recurring jobs、additional agents、commit、push、merge、deployment。GPU_SMOKE_NOT_RUN／完整24 fits未跑／正式P3研究結果未產生；本輪工程完成不是 Human acceptance。

---

# P3 修訂 2 工程候選 — 2026-11-20

本節取代下方歷史工程紀錄中的准入、同 namespace 預算及刪除 STOP 程序。依[登記修訂 2](../tasks/spec-mamba-portfolio-p3-20261120.md)實作；job `job-run-mwnv3018-83e5c9`，detached base `a91f38e86dab0d332fc63c6dc8a6ab7361a19b99`，UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED。沿用既有 adapter、evaluator、tests、report 與 `.artifacts/p3/`；`revision2-verification/` 只隔離本輪 GENERATED CPU logs，避免覆寫 revision1 證據，不是新研究 experiment 或正式市場結果。

`rolling_seq_scores.py` 新增 `--gpu-smoke`，只有 CUDA 與指定套件環境符合時才執行官方 Mamba2 的 temporal（60 tokens）與 cross（256 tokens、d_conv=2）前向／反向；量測 finite、parameter count、gradient tensor count、peak allocated memory、GPU model、起訖溫度、elapsed seconds、output／gradient abs max。輸出 `p3-official-gpu-smoke-v2`，帶凍結設定、官方 signature 的所有預設、source／spec／interpreter／lock／installed distribution 逐檔 SHA-256。官方 `dt_limit` 中 Infinity 在 JSON 明確記為字串，建構 mixer 時還原；不改官方預設。smoke 執行器不寫 reviewed_by。raw smoke 與 reviewed artifact 均 exclusive、fsync，不接受舊版 v1 或 human_reviewed 布林。

凍結字典補 `dropout=.1`、`window=60`、`normalize=rankgauss`、`train_probe_sessions=40`；SeqData 明傳 window／normalize。gradient clip=1、15% warmup／cosine、AdamW 預設 betas、cross d_ff=4*d_model 及 FFN dropout 的 owner source（seq_runner／seq_models／cross_stock）逐檔 SHA 包含在 smoke 與 evaluator 現況重驗中。mixer_parameters 從安裝的 Mamba2 signature 讀出其餘預設，包含 ngroups、chunk_size、dt_min/max、A_init_range、rmsnorm、dt_limit 等，不以手抄子集合代替。

## 解鎖程序（本輪沒有執行 GPU 或獨立審查作業）

在指定 `~/.venvs/marketmamba-gpu` 環境，保留 `~/.venvs/marketmamba-gpu.requirements.lock.txt`，先凍結 source；套件為 torch2.10.0+cu128／mamba-ssm2.3.2.post1／causal-conv1d1.7.0。以下是後續執行程序，不是本輪 CUDA 成功證據：

```bash
PYTHONPATH=V6 ~/.venvs/marketmamba-gpu/bin/python -m research_lab.rolling_seq_scores \
  --gpu-smoke --job-id <producer-job-id> --out-dir .artifacts/p3/admission/raw-smoke.json
```

接著由另一個獨立 worker／Human 執行審查作業：核對 raw JSON、實際 logs／量測、source diff、環境 lock、逐檔 hashes 與溫控。審查者保留實際 job id 與 GO／NO_GO 結論。`--record-review` 只記錄已完成的審查，不會啟動 worker，也不會自行作出獨立判斷。在同一指定環境另執行：

```bash
PYTHONPATH=V6 ~/.venvs/marketmamba-gpu/bin/python -m research_lab.rolling_seq_scores \
  --record-review .artifacts/p3/admission/raw-smoke.json \
  --out-dir .artifacts/p3/admission/reviewed-smoke.json \
  --review-kind independent_worker --job-id <independent-review-job-id> --verdict GO
```

Human 審查改用 `--review-kind human` 與可稽核的 review id；producer 與 reviewer job id 不得相同。raw evidence 保持原樣，新的 reviewed artifact 寫入 `reviewed_by={kind,job_id,verdict,evidence_sha256}`；evidence_sha256 綁定原始 raw 檔案的實際 SHA-256，review_source 保留其 path／bytes／SHA 並於准入時重讀；另有 canonical body hash 排除 review metadata 與自身 hash，final artifact SHA 由 admission.identity 記錄。verdict 非 GO 不得准入。standing_authorization 固定為「Human 在 2026-11-19／20 的對話中授權主控自行接續本機（RTX 3060）實驗、不得上 Colab。此授權可由 Human 日後稽核；不表示 Human 已逐項審核。」這是既有授權紀錄，不是逐項 Human acceptance；本輪 No GPU 指令仍優先。

之後才以正式、不含 `--smoke` 的命令完成 24 fits：

```bash
PYTHONPATH=V6 ~/.venvs/marketmamba-gpu/bin/python -m research_lab.rolling_seq_scores \
  --folds-root <已核對的唯讀rolling矩陣根> --p1-scores-dir <已核對的P1分數目錄> \
  --out-dir .artifacts/p3/registered-scores --years 2021 2022 2023 2024 2025 2026 \
  --models mamba2 crossstock --seeds 17 29 \
  --gpu-smoke-evidence .artifacts/p3/admission/reviewed-smoke.json --run
```

`p3_main` 不信任 RESOLVED 字串：重新讀 evidence 路徑及 SHA、body/review hash、source/spec、environment/files、frozen cfg、mixer defaults；重新比對 run_identity、24 個 run_hash／sidecar／payload SHA／job keys 與帳本完成事件。評估完成發布前再次重驗准入，避免評估期間變動。全域帳本 path/SHA、spent、14h limit 任一不符，或 STOP／pending attempt 存在，皆不得 CONFIRMATORY。P1 三種子、支援列／配對日期、CI／Holm 舊閘門保留。被拒的合成／partial manifest 只能得到 INCOMPLETE_ENGINEERING_ONLY，無法提升為研究結論。

## 全域帳本與 STOP

唯一帳本為 `.artifacts/p3/budget-global.jsonl`，所有 out-dir 共用；GPU smoke 亦扣帳。每事件帶 previous_sha256／sha256。Linux／WSL 以 flock 鎖住 `.artifacts/p3` 目錄固定 inode；刪換 budget.lock 或 budget-global.jsonl 不會解除其他程序持有的鎖。目錄及其父目錄須可信且穩定；原生 Windows 不支援此 flock 路徑，會拒絕而非退回較弱鎖。

有 fit／run_identity 卻無全域帳本，或 hash chain／namespace 的 run binding／completed fit binding 不相符，即拒絕。舊版 per-namespace 帳本不會自動推算為新全域帳本；已有正式 attempts 卻缺全域歷史時須另行盤點、核對與授權遷移，禁止刪掉舊檔再把 spent 當 0。全域帳本一旦有後續事件，舊 manifest SHA 即 stale，不允許手改 SHA 升格。

錯誤／中止／溫控 STOP 先 append＋fsync STOP 事件，STOP 檔也 fsync，再 append＋fsync END。硬中斷到重啟間仍保守扣帳，先鎖 STOP 再補 INTERRUPTED END；不宣稱精確 GPU 時數。刪除 STOP 檔不清除帳本 latch。檢查根因後，只能用以下明確命令保留原因與時間：

```bash
PYTHONPATH=V6 ~/.venvs/marketmamba-gpu/bin/python -m research_lab.rolling_seq_scores \
  --clear-stop STOP-<事件名稱>.json --reason '<檢查結果與清除原因>'
```

CLEARED 事件不重置 spent，不可解除已用盡 14h 的預算限制。`collect()` 為 internal adapter，必須持有 active global-ledger attempt；直接呼叫被測試鎖住。所有正式入口在缺合格 evidence 時 fail closed。

## 驗證與限制

指定完整命令與最終實際輸出（exit 0）：

```text
PYTHONPATH=V6:V6/research_lab/tests:. /home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python -m pytest V6/research_lab/tests tests -q
558 passed, 1 skipped, 4 warnings in 191.92s (0:03:11)
```

初輪反證：`9 failed, 50 deselected in 3.95s`；修正後同選取：`9 passed, 50 deselected in 7.67s`。按任務逐項的最終 GREEN（實際 stdout，均 exit0）：

| 項目 | RED 實際失敗 | 最終 GREEN 實際輸出 |
|---|---|---|
| 1 executor smoke／獨立 review | smoke 入口不存在；raw review SHA 回歸另為 `1 failed, 82 deselected in 1.58s` | `3 passed, 80 deselected in 2.51s` |
| 2 confirmation 重驗 | 手寫 RESOLVED 被舊閘門放行；發布前竄改另為 audit RED 的一項 | `17 passed, 66 deselected in 57.16s` |
| 3 全域帳本／鎖／STOP | 初輪3項失敗；STOP檔 fsync 另為 `1 failed, 80 deselected in 1.61s` | `7 passed, 76 deselected in 2.28s` |
| 4 frozen／官方 defaults／環境 | 初輪4項失敗；Infinity 為 `1 failed, 77 deselected in 1.49s`；傳遞套件為 audit RED 的另一項 | `6 passed, 77 deselected in 1.80s` |
| 5 工程報告 | 舊手寫 JSON／刪STOP解鎖程序已明確 supersede | 新程序、local links／artifact hashes／治理檢查已核對 |

額外 audit RED：`2 failed, 1 passed, 80 deselected in 8.71s`；GREEN：`3 passed, 80 deselected in 8.82s`。最後 admission 相關回歸：`20 passed, 63 deselected in 26.00s`。完整 source 固定前曾跑第一輪全套：`556 passed, 1 skipped, 4 warnings in 159.50s`，不是最終數字。所有 raw stdout／選取命令／source hashes 保存在 `.artifacts/p3/revision2-verification/verification.json` 與同目錄 logs。

四項 in-memory mutation（未改磁碟 source）均 exit1：停用 frozen validator → `4 failed, 76 deselected in 0.15s`；停用 confirmation revalidator → `1 failed, 79 deselected in 3.15s`；停用 review validator → `1 failed, 79 deselected in 0.14s`；略過 hash-chain check（同時更新 manifest ledger SHA，排除只靠檔案 hash 擋下）→ `1 failed, 79 deselected in 1.02s`。

CPU 真實 CLI `--gpu-smoke` 拒絕為 `RuntimeError: STOP CUDA required for GPU smoke`，不產生 evidence。CPU producer boundary double 的量測為每 mixer 4160 parameters／2 gradient tensors、forward/backward finite；123456 peak bytes／60°C 為明確注入值，絕非 GPU 實測。正式 GPU 型號／顯存／溫度／耗時 N/A。

Knowledge-impact：語義、candidate correctness gate、內部入口 boundary、deprecated v1／刪檔清STOP及 next procedure 有變；Current State／Authority Map／Research／Worktree Register 同 change 更新，無 production/runtime/正式研究 authority 變更。checker exit0／WARNING，人工已檢查 Operations／GOVERNANCE，沒有該兩檔語義變更，不修改。knowledge_health.py --ci exit0／Overall WARNING：Project Map／Agent Guides／Domains／Runtime Freshness／Generated Index PASS；Governance／Authority Drift／Legacy Docs WARNING，無 structural FAIL。AST syntax、added local references、artifact SHA、git diff --check 通過。

Completed／Changed：3個source/test檔、工程報告與4個knowledge檔。Decisions：准入／預算／STOP fail closed，raw SHA 與 final reviewed SHA 分開保留；不自稱獨立審查或 Human acceptance。Remaining／Blockers：官方 CUDA smoke、真實獨立審查、正式24fits／市場結果未執行。Next recommended action：審查本 diff 與 logs，依後續任務授權執行解鎖程序。Project memory updated／Authority map updated：YES。Git status：起始clean，結束8個指定檔案 modified/unstaged，無 staged／tracked外變更。Commit／Push：N/A，依指示未執行。Worktree：既有 fixer、detached a91f38e，未保留patch前不得移除。Runtime/environment：WSL-native worker、pilot09 Python3.12.3、torch2.15.0+cpu，MARKETMAMBA_DATA_ROOT unset，Data realpath 為本 checkout 的 Data 目錄；未觀察Windows scheduler或GPU runtime。

所有新增 positive GPU-producer／24-fit fixtures 均為 CPU boundary doubles，不能作為官方 CUDA evidence；測試僅證明程式控制路徑與拒絕條件。hash chain／review metadata 提供完整性與追溯，無法取代 OS 權限、獨立審查的真實執行或 Human 稽核。沒有新 GPU 量測或正式 P3 結果；PIT／下市覆蓋／adjusted prices／兩種子規模限制保留。

---

以下修訂 1／初版紀錄是 HISTORICAL_ENGINEERING_ONLY；其中手寫 GPU JSON、human_reviewed、同 namespace budget、直接刪 STOP 指令已由上方取代。

# P3 修訂 1 bounded repair — 2026-11-20

UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED。`job-run-mwnu9iri-e7a54a`，既有 fixer worktree，detached HEAD `69dc5baaee2d8e5c35d36458335c29fb4f26b619`。本節是修訂 1 的工程狀態；下方初版紀錄的模型歧義與預算缺口已由本節取代，舊數值不重新標成新 evidence。沒有 commit、push、merge、GPU 訓練、安裝、資料 acquisition、正式 Data／V6/models／production／scheduler 變更或額外 agents。

## 變更與契約

- (2) `rolling_seq_scores.FROZEN_CFG` 明列完整 19 項有效設定，含 nested mixer_kw；`FROZEN_DEFAULT_CFG` 另明列繼承基線。`effective_config` 每次啟動逐項比對 DEFAULT_CFG 與 baseline、合併 RECIPE 與 frozen effective，回傳深複本。即使修改被 RECIPE 覆寫的 default，也拒絕。正式 `run_jobs`／`collect` 再驗 cfg 相等；smoke 只容許 epochs=1、steps_per_epoch=20，manifest 保留完整 frozen 字典與實際 parameters。注入 CPU 小型 cfg 明確標 CPU_ENGINEERING_ONLY，永遠不解鎖確認性。
- (3) H-P3a/b 明寫：CI 下界>0 且 Holm p<.05 為優於；上界<0 且 Holm p<.05 為劣於；其餘未證實。既有式本已同義，此次展開分支並增加零邊界、p=.05、CI 跨零、raw p=.03 但 Holm=.06 的回歸。
- (5) `budget.jsonl` 是同一 run namespace 的 append-only 持久帳本；每次嘗試在 preprocessing／fit／輸出驗證前 fsync START，結束（含 Exception、KeyboardInterrupt、ThermalStop）fsync END 與實測 monotonic seconds。每個 batch 使用剩餘預算 deadline；fit 返回後亦驗 deadline。啟動累計 END，達 50,400 秒即拒絕，即使人工已移除 STOP；完成 fit 的 resume 不重複扣費。單一 writer 以 POSIX flock／Windows msvcrt lock 保護，並持鎖到完成 manifest。manifest 保存 ledger identity、累計秒數與限額。
- (6) 現有 checked_splits 與真實六折一致，未修改或放寬任何 purge threshold。真實 splits 被 in-memory 污染至 eval_first-20 時仍必須拒絕。測試逐列比對 SeqData.split_rows，並透過 rolling_runner.run_fold_seed 的 run_fn 邊界檢查實際傳入的 eligible mask／outer split。C0+ 使用 outer split1；神經 fit／val 是其成熟且 finite 的子集，沒有把兩種 fit_rows 宣稱相同。
- (7) p3_main E2E 實讀兩個分數目錄、P1 pinned_reference 與價格 parquet，實跑組合 engine／9個 parts／JSON（三個確認折×三系統），檢查 resolved、unresolved、smoke、種子、參數、admission 六種 readiness。Volume provenance 僅在此合成 fixture 以測試替身隔離，該規則保留既有 P1 測試。新增 `--part-dir`（P3 專用）；預設 `<完整輸出檔名>.parts`，無副檔名 out 也安全，越界／重疊拒絕。保留既有 parts identity 驗證與 out 不覆寫。
- (8) 修訂 1 第 1 點有權取代原 no-conv：官方 `mamba_ssm.Mamba2` temporal 預設 d_conv=4，跨股雙向 mixer d_conv=2；d_state16、headdim8、expand2、d_model64。程式讀取已安裝官方 class signature，預設不是4即拒絕，不偷偷指定另一預設。manifest 的 mixer_parameters 明載實際參數；注入替身另標 caller-defined，不能冒充官方參數。官方 collect 分支已由 CPU 邊界替身測試證明可達，非永遠 STOP 的死碼。

## 持久預算的中斷限制與人工 STOP 移除程序

SIGKILL／os._exit 不會執行 finally；下次啟動看到未結 START，追加 INTERRUPTED END，計入 wall elapsed 與跨程序 monotonic elapsed 的較大者（包含停機至重啟的時間），並建立 STOP。**這是保守扣帳，非可精確還原的 GPU kernel 用量**；不把未觀察到的終止時刻捏造成實測。正常完成／可捕捉的中止則記實際 elapsed。帳本缺損、截斷、負數／非有限 debit、事件順序錯誤 fail closed。跨主機／手動改時鐘、強制斷電 durability、Windows lock 實機行為未由本輪驗證；不複製或重置帳本來規避上限。

1. Human 先確認該 namespace 無存活訓練程序，保存並檢查每個 `STOP-*.json`、budget.jsonl、溫度紀錄與 fit sidecars；處理熱／來源／契約失敗原因。kill 留下的 START 必須先由 ledger recovery 形成 INTERRUPTED END 與新 STOP，才能判斷餘額。
2. Human 在核對餘額、確認允許再嘗試後，只逐一移除已核准的精確 `STOP-*.json`。程式不自行刪 STOP；**不得刪除、截短、重建 budget.jsonl、改 run_identity，或改 namespace 重置同一研究預算**。保留 STOP 內容於既有 run 的人工審核紀錄；本輪沒有替使用者清除任何真實 STOP。
3. 同 namespace 以相同 source／input／cfg 啟動，重新核對 ledger 與 identity。任何剩餘 STOP 仍拒絕；預算耗盡不能靠移除 STOP 解鎖。部分 parquet 或 sidecar 缺一仍屬 orphan，需人工評估，不自動當作完成。

## UNRESOLVED → RESOLVED 的解鎖條件與程序

修訂已解除 **模型文字矛盾**，但沒有把本輪狀態直接升為 RESOLVED。官方 CLI／library 無 GPU smoke evidence 時在輸出／矩陣／CUDA 存取前拒絕。No GPU 授權下，本輪未產生正式 smoke evidence。

日後取得獨立 GPU 授權後，由 Human 在修訂 1 指定環境完成官方 Mamba2 forward/backward 冒煙並審核，保存獨立 `p3-official-gpu-smoke-v1` JSON。它須含 `human_reviewed: true`、`forward_finite: true`、`backward_finite: true`，`parameters` 等於完整 FROZEN_CFG，`mixer_parameters` 等於程式實際讀取的官方 temporal／cross 參數，`implementation` 等於 provenance 的逐檔 hashes，`environment` 等於 `runtime_environment('cuda')` 的完整字典。此 JSON 是人工審核證據描述，不是布林值自行證明執行：審核者須保留實際 smoke logs／前反向數值與環境 lock；本輪測試中的 synthetic JSON 不得使用於正式工作。

`--gpu-smoke-evidence <已審核JSON>` 觸發 official_admission：torch=2.10.0+cu128、mamba_ssm=2.3.2.post1、causal_conv1d=1.7.0、非空 GPU 型號、cuda device、frozen cfg、mixer、environment、implementation 任一不符即 STOP；通過才讓新 manifest 記 `contract_status: RESOLVED` 及 admission evidence 的 path／bytes／SHA-256。不得手改既有 UNRESOLVED manifest 或重標 CPU 結果。source／環境改動須重做相符審核。

確認性還須 **非smoke**、完整有效參數、frozen_effective_config、RESOLVED admission、神經 seeds17/29、C0+ seeds17/29/43且 max_train_rows500000、2021–2023 三系統完整配對且每組>=20共同報酬日、bootstrap1000。其餘只出 INCOMPLETE_ENGINEERING_ONLY／未證實。此條件不是 Human acceptance 或 trading readiness。

Windows／WSL 的每個 score manifest 必含 `provenance.environment` 的 `platform`、`system`、`release`（可辨 WSL kernel）、`python`、`executable`、`device`、`packages.torch`、`packages.mamba_ssm`、`packages.causal_conv1d`、`gpu_model`；以及 `environment_observed_at`。沒有套件記 MISSING，CPU 無 GPU 型號記 null，不捏造 Windows／GPU observation。GPU fit 另含 peak_memory_bytes／temperatures。

## 真實折唯讀驗證

外部根 `/home/frank/projects/ruflo-evaluation/pilot09/marketmamba/.local-data/matrices/rolling/`，各 splits.npy 為 7,424,055 bytes。未建 cache、未修改矩陣；使用現有 stock_idx.npy。以下為實際未抽樣 inner rows（val cap 尚未套用）：

| fold | splits SHA-256 | fit | val | eval | outer train max / eval min |
|---|---|---:|---:|---:|---|
|2021|6c70097cb2f8bbbe27d16b5bc86cacbab506d3fabddc989a7eb28a3a290979d6|2325741|185595|378547|3931 / 3952|
|2022|f55a74081ba43359edff6231aab488b0df9944c607de8489746d399e06e4bf89|2689426|192276|385419|4175 / 4196|
|2023|20a8ad5b9d963130919eafa6c9b425c14bd5cd44f60c4f8c7964d169dc199eac|3066168|193724|390606|4421 / 4442|
|2024|8e6c57aaa8a8611c31440c3e673693d40f879abd708f3615c1f8e7cb9eeec98c|3437680|202923|418138|4660 / 4681|
|2025|281442ccb444f4a60d8b94a1a53ef5ae96a4a9a3d9b5b4662fd93ac91890727d|3832931|215813|425682|4902 / 4923|
|2026|dfd3fa021b5d29447b67c65ecbc602e990cf92778828d151123c714fce57fa6b|4248703|214903|302228|5145 / 5166|

各折 inner gap=20；outer gap=21；checked_splits PASS。測試 pin 真實 splits hash，資料不存在時 explicit skip，並不宣稱其他環境已觀察到真實折。

## RED → GREEN 與本輪交接

指定完整命令（於最新source固定後執行）：

```bash
PYTHONPATH=V6:V6/research_lab/tests:. /home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python -m pytest V6/research_lab/tests tests -q
```

實際輸出：`525 passed, 1 skipped, 4 warnings in 137.80s (0:02:17)`，exit0。完整stdout/stderr保存在 `.artifacts/p3/revision1-verification/full-pytest.log`；skip為既有項目，4個Pandas dayofweek deprecation warnings未改動。環境實際torch2.15.0+cpu、mamba_ssm／causal_conv1d MISSING；不宣稱修訂指定GPU環境已安裝或已驗證。

Knowledge-impact gate：8項檢查中有效語義、correctness gate與next decision有變，已同change更新Current State／Authority Map／Research／Worktree Register。check_knowledge_impact.py exit0／WARNING（自動規則未識別P3 source）；已人工檢查其要求的Operations／GOVERNANCE，沒有operations runtime或治理規則變更，這兩檔不改。沒有supersede任何正式artifact、production authority或既有研究結論。

初輪新增測試實際 **7 failed、31 passed／15.36s**；補完後 focused **48 passed／33.31s**（之後另加兩個預算／STOP回歸，最終全套數值另列）。額外發現「移除 STOP 後已耗盡預算的 namespace 可開啟」：RED **1 failed／1.53s**（DID NOT RAISE），修正啟動累計閘門後 GREEN **1 passed／48 deselected／1.53s**。曾有一次測試期間修改來源使 end-to-end 正確拒絕 `P3 source changed during run`；停止並行 source edits 後全綠，不放寬該 gate。

最終 focused：`50 passed in 36.96s`，exit0。knowledge_health.py --ci：exit0／Overall WARNING，Project Map／Agent Guides／Domains／Runtime Freshness／Generated Index PASS，Governance／Authority Drift／Legacy Docs為WARNING，無structural FAIL。syntax AST、artifact SHA-256、diff新增連結與git diff --check通過。

in-memory mutations（未改磁碟 source，各 exit1）：config 1 failed；decision 2 failed／5 passed；budget 1 failed；real purge 1 failed；baseline p3_main 1 failed；mixer 1 failed；environment 1 failed。輸出位於 `.artifacts/p3/revision1-verification/RED-*.log` 與 mutations.json。config 直接修改 DEFAULT_CFG.weight_decay=.2，即在有效字典建構時被 frozen gate 拒絕；判定 mutation 移除 Holm p 條件，purge mutation 移除檢查，budget mutation 清空帳本，main mutation 執行 HEAD 初版函數，均被測試抓到。

本輪 runtime 是 WSL-native fixer checkout、Python3.12.3 pilot09 venv、CPU；MARKETMAMBA_DATA_ROOT unset；工作樹 Data/ 為本機目錄（非外部矩陣 realpath）。正式資料未更動；沒有觀察 Windows 操作環境／GPU 型號／溫度。本輪新 evidence 只限 SYNTHETIC_CPU_ENGINEERING_ONLY／唯讀 metadata engineering evidence，不產生市場結論。

Completed／Changed：修訂項2/3/5/6/7/8 source、tests、report與knowledge。Decisions：官方卷積配方；保守中斷扣帳；STOP人工清除；缺smoke不解鎖。Remaining／Blockers：正式GPU smoke、Windows實機、完整24fits與正式P3結果未執行。Next recommended action：獨立審查本diff與實際logs，再由Human另行決定GPU工作。Project memory／Authority map updated：同change。Git status：起始clean，依no-commit留指定檔案unstaged；Commit／Push：N/A。Worktree：既有fixer，不安全移除未保留patch的dirty工作樹。Runtime/environment：上述WSL/CPU。工程完成不等於Human acceptance。

## 初版工程紀錄（修訂 1 前，HISTORICAL_ENGINEERING_ONLY）

以下保留初版當時的歧義、限制與驗證數字；不是目前 blocker 或指令。

# P3 bounded builder candidate — 2026-11-20

UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED。Detached base `06bc0c73a44d6bb725b36f136c7c953d4d199e14`，job `job-run-mwnteeta-4dea80`，既有 builder worktree。只實作本輪 P3 工程候選；沒有 commit、push、merge、GPU、Colab、資料 acquisition、production 或 scheduler 變更。

## 完成範圍與 STOP

新增 `V6/research_lab/rolling_seq_scores.py`：相容的分數欄位、逐 fit exclusive parquet／hash sidecar、run identity、輸入 SHA-256、來源／環境 identity、GPU 型號／peak allocation 欄位、每 batch 溫度紀錄、fit-level resume、逐日截面 percentile rank 種子平均。預設 dry-run。矩陣輸入只以 allowlist symlink 與 `np.load(..., mmap_mode='r')` 引用到 `.artifacts/p3/work/`，不引用外部 rank-gauss cache；derived cache 另以輸入／implementation／輸出 hashes 綁定。Symlink 本身不是 OS 存取控制：唯讀保證是此流程只讀輸入的程式行為，未變更外部權限。

正式 CLI 與無注入 mixer 的 library entry 均 **STOP_MODEL_CONTRACT_AMBIGUITY**。Frozen spec 同時要求「官方 mamba_ssm.Mamba2」與「no-conv」，但歷史 recipe 沒有指定 `d_conv`，而 `CrossStockRanker` 跨股 mixer 明確覆寫 `d_conv=2`。既有 `V6/experimental/v7_corrected_e5_model.py::Mamba2NoConv` 是另一路注入官方 SSD 的實作，不能靜默替代官方 Mamba2 class。未修改登記規格，也未新增可繞過 gate 的 CLI 參數。`run_jobs`、`collect` 目前僅允許明確注入 mixer 的工程驗證，**不是正式可訓練的 P3 交付**；解除 gate 與產生確認性分數尚未完成。

使用者本輪總限制含 `No GPU`，工作項 5 卻要求 GPU 冒煙。採限制較嚴的解讀：GPU 冒煙 NOT_RUN，GPU 型號／顯存／實際溫度 N/A。指定 pilot09 venv 是 Python3.12.3，實際 `mamba-ssm` package metadata 為 MISSING；沒有安裝或下載依賴。CPU injected mixer 的數字不可冒充官方 Mamba2 性能。

溫控透過可注入讀值與 sleep：低於 80°C 正常；達 80°C 暫停，到 72°C 以下才續；達 85°C／讀值未知／runtime deadline 即 latched STOP。預設使用 `nvidia-smi --id=0`；CUDA device 0 固定，不自行選別張卡。每次 windows 載入（訓練、驗證、推論）前檢查；單個正在執行的 kernel 中不輪詢。14 小時預算計入同 namespace 已完成 fit wall time 與本次 process wall time；**失敗／被 kill 的未完成 fit 時間未完整跨重啟累計**，不宣稱可靠的跨程序 GPU 時數會計。只支援完成 fit 續跑，不支援 mid-fit checkpoint；partial payload 缺 sidecar 時保留 orphan 並停止，使用新 namespace。

`portfolio_eval.py --mode p3` 使用獨立分支，P1／P2 原本 evaluator 與預設行為不變。三系統共用固定 P2 主要格（raw、top50、20 sessions、buffer100、discount0.6、min fee20、capital1e6、benchmark proportional only）。c0plus 從 P1 manifest/hash 驗證的 rank_average 讀取，其餘從 P3 讀取，每 fold 逐一檢查 keys 及 label 完全相同。輸出各系統逐折逐日淨報酬、描述性對基準淨超額、逐折 IC paired HAC／block bootstrap、H-P3a/b 共同日期差與 Holm family2。缺折／種子／正式 contract admission 或不是 1000 bootstrap 時不確認。沒有重新擬合 C0+。

新增檔案的理由：既有 rolling_scores 只支援 c0plus.run；seq_runner 的 result／scores 不具有 P1 parquet、完整輸入 hash 與 immutable fit contract，且會在矩陣旁產生 cache。因此新增薄 adapter，沿用現有 research_lab、reports、.artifacts，不新增 top-level tree 或 worktree。

## 偵察與重現設定

- `seq_runner.SeqData`：預設 60-session 同股完整觀測視窗；`split_rows` 的 eval 為 split2 & eligible，不因 y5/y10 缺失縮減評估列，與 P1 `matrix_baselines.Context.eval_idx` 一致。P3 在訓練前比對 P1 keys／labels。
- `rolling_runner.run_fold_seed`／`rolling_scores.collect`：使用 `SeqData(normalize=None)` eligibility 傳給 C0+，沒有神經網路訓練。outer purge 實際 owner 是 `rolling_folds.make_fold_splits`，train session `< eval_first - 20`；P3 對全部 split1 逐列驗證，不能只看被內層 fit 篩掉後的列。
- 神經網路內層：最後126個 eligible train sessions 作 validation，fit_last 為 `train_dates[-val_dates-20]`，fit／val 都須 finite y5 & y10。保留預設 validation cap80,000，固定 RNG0；eval 永不抽樣。預設 AdamW weight_decay0.01、warmup15%、cosine、gradient clip1、patience3、BEST by val IC5。若無有效最佳驗證狀態，P3 拒絕 LAST fallback。8 epochs 是上限，既有 patience3 可提前停止。
- `train_one`：傳入 `cfg={**DEFAULT_CFG, **RECIPE}`，其中 d_model64、n_layers2、mixer_kw(d_state16,headdim8,expand2)、lr3e-4、epochs8、steps_per_epoch400、dates_per_batch8、stocks_per_date256、listnet_w0、n_cross1。pure MSE targets 為逐日 centred rank，head5:head10=1:0.5。模型 `mamba2`／`crossstock`，seeds17/29；crossstock 訓練 stock token 不重複，推論使用既有 target256／n_perm2／seed0 分塊。
- `ensure_rankgauss`：每 session／feature 截面 rank-gauss；macro_columns 以固定 seed0 抽60 sessions 判定日內常數並歸零；其餘常數截面轉換也是0。這是歷史演算法，不能據此宣稱總經欄位有人工 frozen schema。P3 只改快取目的地／hash gate，不修改歷史轉換邏輯。
- pilot09 唯讀根：`/home/frank/projects/ruflo-evaluation/pilot09/marketmamba/.local-data/`。`local_b8/E5_seq_2fc9d1789f/result_mamba2_s17.json` 與 `E5_seq_048a4212ad/result_crossstock_s17.json` 的實際 config 是上述8×256、8×400、純MSE，後者 n_cross1。`E5_seq_1f512a7e3e` 是16日／n_cross1；`E5_seq_c66fc390d0` 是8日／n_cross2，不能混用。
- `colab_b8` 到 `colab_b8v8/b8_seq_baselines.ipynb` 實際執行 cell 皆呼叫 `sr.run_experiment(... kinds=KINDS, seeds=seeds, cfg=cfg, device='cuda', tag=...)`；smoke 為1 epoch×20步、8日×128檔、baseline50k。v3–v8 非smoke傳 epochs／steps，v1/v2 非smoke傳空 cfg；v6–v8 另由 RUN_TRAINING 控制。v7加入 rank-gauss 診斷，v8延續。僅靠 notebook override 不能重現本次8日設定，還必須明傳 dates_per_batch8／stocks256／MSE／n_cross1。local_b8 中找到的是結果與scores／daily_ic，沒有保存獨立 local launch .py/.sh；不捏造 shell invocation。偵察 identity 見忽略目錄 `builder-verification/reconnaissance.json`。

可重現的預檢命令（不訓練）：

```bash
PYTHONPATH=V6 /home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python -m research_lab.rolling_seq_scores \
  --folds-root /home/frank/projects/ruflo-evaluation/pilot09/marketmamba/.local-data/matrices/rolling \
  --p1-scores-dir /home/frank/projects/MarketMamba/.artifacts/p1/full-scores-ace2eef \
  --years 2023 --models mamba2 crossstock --seeds 17 --smoke \
  --out-dir .artifacts/p3/planned-smoke
```

folds-root 已對照 P1 manifest 的 input keys；本輪未讀取／重算完整矩陣 hash，也未聲稱正式矩陣已通過訓練准入。main worktree 有 P1 full-scores-ace2eef，本 worker 沒有隱式跨 checkout fallback。加 `--run` 會在建立輸出／讀矩陣／初始化 GPU 前停止並說明模型 contract 衝突。

## 驗證與反證

指定命令最終結果：**493 passed、1 skipped、4 warnings／108.28s**（首輪491 passed／114.80s之後新增best／resume回歸）。既有 skip 與 Pandas dayofweek deprecation 警告未修改。focused P3：18 passed／5.53s。syntax compile、`git diff --check` 通過；knowledge-impact PASS（自動規則未識別新 P3 語義，但人工仍更新 Current State／Authority Map／Research domain）。新增測試包含 immutable resume／輸入變更、work cache isolation、no-best拒絕、合成三系統 evaluator／part reuse／不完整 confirmation gate，以及兩個注入 mixer 的 CPU adapter 路徑。

五個 in-memory mutation 沒有改 repository source，每次單獨 pytest，均 exit1：

```text
rows     test_p3_exact_p1_rows                  1 failed in 0.05s
pit      test_p3_rank_average_pit               1 failed in 0.06s
purge    test_p3_training_purge_maturity        1 failed in 0.03s
pairing  test_p3_common_dates                  1 failed in 0.03s
thermal  test_p3_temperature_hysteresis_stop    1 failed in 0.03s
```

對應壞行為：no-op row gate、rank group移除date、直接跳過outer purge gate、union date並補0、no-op thermal gate。`mutation-*.log` 保存完整失敗 traceback，`mutations.json` 保存實際 exit code；沒有把預期失敗算成通過的測試。

本地 `.artifacts/p3/builder-verification/` 是 GENERATED／SYNTHETIC_CPU_ENGINEERING_ONLY。首輪 `cpu-injected-scores` validation cap120 導致不足10檔／日，出現 best=-9，已保留為不合格工程觀察，不能使用於模型／選模結論。修正 adapter 後重跑 `complete-validation/`，改為保留完整240 validation rows。注入 `nn.Linear` 非官方Mamba，2日×12檔、d_model16、1層、1epoch×2步；用固定60°C假讀值驗證 guard hook。實際 GPU 溫度與顯存皆未量測。實際 wall time：mamba2 adapter **2.1513s**、crossstock adapter **0.0966s**；各984 fit／240 validation／240 evaluation rows。三系統合成 portfolio **1.7584s**、各61 return days，H-P3a/b 的 p_holm=1、INCOMPLETE_ENGINEERING_ONLY。兩個 adapter 的首次初始化成本不同，不比較其速度。完整摘要見 `complete-validation/smoke-summary.json`，不是正式市場分數。

## 完整24次擬合估時與限制

沒有合格的本機 GPU 冒煙，**不能提供基於本輪實測的24 fits GPU估時**。規格既有10–25分／fit 僅推出4–10 GPU小時的預算先驗，非新測量；14小時上限不變。歷史local_b8 seed17（不同矩陣／非本次GPU確認）mamba2八epoch合計約21.21分、crossstock約20.52分，不能據此替代本輪外推。若日後合格smoke各20步取得完整job耗時 Tm/Tc 秒，粗略保守估計為 `12 × 160 × (Tm + Tc) / 3600` 小時；它把一次性的rank-gauss／評分／啟動也放大160倍，只可作保守包絡。較合理需獨立記錄每步訓練、每epoch驗證與一次性前處理，再用 `12 × [3200(t_m+t_c) + 8(v_m+v_c) + p_m+p_c+q_m+q_c]` 秒估算；其中 t是訓練每步、v是每epoch驗證、p/q是前處理／最終推論。

限制保留 P1/P2：模擬標籤、survivorship、無 adj_close、未驗證下市及PIT市值、觀測日曆；兩種子對C0+三種子，不升格成交易或production evidence。溫控單步輪詢不是NVIDIA長期訓練認證。未執行真實2023分數生成、完整24 fits、正式P3組合評估、GPU性能／估時或Human acceptance。

## 修訂 3 補丁（2026-11-20，主控依兩位審查共同發現修正）
兩位審查員獨立指出：重建預算時，REBUILT 事件與 STOP 檔之間崩潰會讓重啟直接放行。修正：REBUILT 事件本身記錄 `stop_name` 並在 `budget_state` 中視為 STOP 鎖（STOP 檔僅為輔助證據）；重建時把舊鏈與現存 STOP 檔的未解 STOP 寫入 `inherited_stops` 帶入新鏈（清除 REBUILT 的 STOP 不會清除繼承來的 STOP）；`--rebuild-budget` 在帳本、錨點與 sidecar 彼此一致（健康）時拒絕執行。三個回歸測試在修正前皆失敗（3 failed），修正後 102 passed（test_p3.py）。未修的 MINOR（審查 Claude）：損壞帳本的可驗前綴未納入下界、fresh 判定未計入 STOP／重建備份檔、orphan payload 需人工盤點；預算不會被少扣到低於 sidecar 與錨點的最大值。
