# MarketMamba — Codex 專案指引

> 最後更新：2026-09-20
> 本檔只放長期規則、目前權威狀態與工作入口。完整脈絡在 `obsidian_note/🏠 Home.md`；機器可驗證的契約與數字以 `research/`、`docs/`、manifest 和 tests 為準。

## 開工順序

1. 讀本檔。
2. 讀 `obsidian_note/01 系統現況/現況整理.md`。
3. 依任務選讀：
   - 資料／PIT／labels：`03 架構筆記/V7 資料處理與 Correctness 邊界.md`
   - Colab／訓練／續跑：`03 架構筆記/Colab 訓練與續跑標準.md`
   - 刪除／空間：`07 決策與教訓/2026-09-20 空間整理.md`
4. 讀實際分支的 `git status`、相關程式、tests 與 contract；文件不能取代現況檢查。
5. 引用研究數字時，回到產生該數字的 JSON／Parquet／report，不從聊天記憶抄數字。

## 使用者的協作與開發習慣

- 一律以繁體中文溝通。
- 改程式前先提出具體計畫：改哪些檔、為什麼、驗證方式；取得核准後可自主執行完整工作窗，不要每個小步驟重問。
- 診斷先行、一次只改一個可歸因變因；正式實作應一次到位，避免反覆重訓與浪費 Colab。
- 預估時間要以實際執行紀錄校正。工作可能很快完成時，不要虛報數小時；仍應把相鄰、已核准且安全的驗證做完。
- 結果異常地好時主動懷疑 leakage、PIT、universe、label timing、source provenance、selection bias 與 evaluation window。
- 誠實呈現 STOP、限制與未知，不為了繼續流程放寬 gate。
- 使用者通常親自跑 Colab；Codex 負責準備 bundle/notebook、續跑、安全落盤、診斷與可直接操作的說明。
- 除非明確授權，不 push、deploy、啟用排程、啟動 GPU、購買資料、增加通知對象或變更 production authority。
- Git 只 stage 指定檔案，禁止 `git add -A`。保留所有既有 dirty changes，不以 reset/checkout 覆蓋。
- 若使用量監控已被要求：只有 weekly 使用量超過 95% 且該工作窗已有明確授權時，才可消耗一次 reset；每次 reset 都要獨立授權。
- 長任務持續提供簡短進度，但不以進度訊息取代最終自包含摘要。

## 不可破壞的邊界

- V6.1／既有 dashboard 是 production 紅線。新功能採 additive、shadow、feature flag；不可順手改現有 endpoint、排程或資料流。
- `V6/models/` 與任何正式模型權重不可修改或刪除，除非使用者逐次明確授權。
- legacy A/B/C artifacts 是歷史研究證據，不修改、不重標為可交易證據。
- V7 strict Phase 0 目前仍為 **STOP**。
- historical-simulation readiness 為 **PASS**，只代表研究模擬可用，不代表 verified-live。
- corrected verified executable labels：5d=0、10d=0。
- historical simulated labels：5d=6,889,229、10d=6,790,554。
- 不得調低 zero-tolerance、acceptance threshold，或為單一案例加特例。
- 所有 execution、tradability、OHLCV、universe 與 fundamentals 證據 fail closed；來源不合格即不可推定可交易。
- Line Notify 已停止服務，不新增或建議 Line Notify。

## 目前分層架構

### Production／operations

- Active production path：Windows Scheduler → `v62_daily.bat --fetch-only` → 多來源抓取 → mutable Parquet + logs。
- 完整 V6.2 inference-to-publication 已實作但目前停用。
- Phase A Shadow Operational Ledger 位於 `feature/production-phase-a-shadow-ledger`：
  - SQLite WAL control plane
  - run / attempt / stage / artifact / check
  - fencing、transactional stage commit、atomic current-state projection
  - credential/path redaction
  - 僅 shadow observe fetch-only；不改既有 success decision
- 後續順序：provider certification → incidents → resumable orchestration → atomic publication/UI → 經核准的 V7 forward paper。

### V7 research／correctness

- Canonical baseline：`E5-PIT-Clean-v1`。
- 48 features；固定順序；causal scaling。
- no industry neutralization、no graph。
- verified-lifecycle historical universe。
- P0/P1/P2 frozen execution proxy policy；source provenance eligibility gate。
- strict Phase 0 STOP 與 historical-simulation PASS 必須分開表達。
- E1：Rolling-Origin Unchanged-Contract Refresh，固定架構與訓練 contract，只更新 split／rolling-origin 訓練。
- E1 正式 seeds：17、29；precision fp32；不做模型搜尋。

## V7 資料處理規則

資料流的 canonical 順序：

1. 官方／允許來源快照與 source registry。
2. security master lifecycle：上市、下市、轉板、暫停、恢復皆為時間區間，不以今日名單回填歷史。
3. PIT fundamentals/revenue：只能在 publication time 之後可見。
4. OHLCV provenance eligibility：mixed-source 或來源不合格 observation 排除。
5. verified-lifecycle universe：以該日可知狀態決定 membership。
6. labels：
   - signal 在 session t 收盤後形成；
   - entry 是下一個固定 calendar session open，不能往後 roll；
   - exit 是第 h 個 holding session close，entry session 算第 1 日；
   - holding interval 每一日都要有 provenance-eligible observation 且不可 suspended；
   - return = exit_close / entry_open - 1。
7. feature matrix：48 維 order 取自 `feature-manifest.json`；cross-sectional scaling 必須 per-session causal；macro 使用 causal expanding。
8. manifest/hash/complete marker 全數通過後才可把資料或矩陣標為 complete。

Authoritative 本機資料在 `Data/`。不要因空間不足刪除；大型重複矩陣、runtime、bundle 才是優先清理對象。

## Colab 參考檔

主要正式參考：

- E1 bundle：`deliveries/V7-E1-Rolling-Origin-Refresh-Colab-Bundle.zip`
- E1 notebook source：`.worktrees/v7-corrected-e5-2026-diagnostic/notebooks/v7_e1_rolling_origin_refresh_colab.ipynb`
- 已驗證過三種子續跑的舊參考：`deliveries/V7-Confirmation-20260914/V7-三種子確認與續跑.ipynb`
- Corrected E5 lifecycle：`V6/experimental/v7_corrected_e5_lifecycle.py`
- Corrected E5 trainer：`V6/experimental/v7_corrected_e5_train.py`
- E1 runner/checkpoint：`V6/experimental/v7_e1_run.py`、`v7_e1_checkpoint.py`
- Exact lock：`environments/v7-colab-2026.04/requirements.lock.txt`

先以目前 branch 內實際存在的版本為準；不要從舊 ZIP 複製整套程式覆蓋新 contract。

## 所有訓練 notebook／runner 必須具備

1. **自包含與可驗證**：bundle manifest、逐檔 size/SHA-256、safe ZIP paths、包內 verifier 可獨立啟動；錯誤要顯示真正 child traceback。
2. **環境鎖定**：記錄 Python、torch、CUDA、GPU、precision、requirements identity、contract identity、source commit。
3. **矩陣 build-once/reuse**：
   - 矩陣完成後持久化至 Drive；
   - 新 runtime 先以 data/feature/label/universe/contract identity 驗證；
   - identity 未變就 reuse，不可只因 runtime 重開重建；
   - partial 目錄不能被當成 complete。
4. **斷線續跑**：
   - 每個 seed／stage 使用獨立 namespace；
   - checkpoint 先 local serialize + fsync + hash，再複製 Drive、重驗，最後 atomic 更新 pointer；
   - 至少保留 latest、previous、best；
   - contract/matrix hash 不符時拒絕續跑；
   - validated orphan 可以恢復 latest，但不得自動覆寫 best。
5. **正確模型選擇**：BEST-not-LAST；selection metric 與 secondary metric 必須依 frozen contract。
6. **可觀測性**：epoch、batch、global step、phase、loss、IC5/IC10、learning rate、elapsed/ETA、GPU/CPU/RAM、checkpoint 時間與 throughput 要可見並落盤。
7. **冪等 stages**：完成 stage 驗證 artifact 後 reuse；不完整 stage 才 resume/retry；不可用「檔案存在」當唯一完成條件。
8. **資源安全**：預先估 RAM/disk；大型矩陣避免重複解壓/複製；OOM/斷線仍須保留已驗證 checkpoint。
9. **輸出契約**：predictions、metrics、telemetry、logs、manifests、evaluation report 各有固定位置與 hash；seed 間不得互相覆蓋。
10. **評估與 gate**：evaluation window、purge、horizon maturity、paired tests、acceptance threshold 皆凍結；結果無論好壞不改 strict Phase 0。
11. **釋放 GPU**：只有 durable artifacts、checkpoint、log、evaluation 均確認寫入 Drive 後，才執行 runtime unassign。
12. **測試**：CPU smoke、checkpoint round-trip、kill/restart、corrupt pointer、contract mismatch、matrix mismatch、bundle-from-inside、full regression 都要覆蓋。

## Git／測試／分支

- 主工作樹可能長期有使用者變更；先看 `git status --short --branch`。
- Correctness、E1、production Phase A 使用獨立 worktree。不要因整理目錄刪 branch；移除 worktree 前先確認 clean 且提交已包含於後續分支。
- 目前重要分支：
  - `feature/v7-e1-rolling-origin-refresh`
  - `feature/production-phase-a-shadow-ledger`
  - `fix/v7-historical-pit-reconstruction`
- V7 experimental full tests（有環境時）：
  ```bash
  python -m unittest discover -s V6/experimental -p '*_test.py'
  ```
- Runtime/venv 可由 lock 重建，不是 authoritative artifact。只在需要執行 tests 時建立，避免長期保留多份 6–9GB 環境。

## 儲存與清理政策

### 永久保留

- Git source/history、contracts、tests、manifests、reports。
- `Data/` authoritative/PIT/provenance/universe/labels/feature cache。
- 正式 predictions、必要 metrics/log、仍需續跑的 checkpoints。
- legacy A/B/C evidence。
- 最新可重現的 Colab bundle 與已知可靠續跑 notebook。

### 可刪／可重建

- `.venv`、`.runtimes`、test/smoke artifacts。
- 已被新 bundle 取代的舊大型 ZIP。
- 已持久化且 identity 未變的重複 matrix copy。
- completed branch 的 clean worktree checkout（branch/history 保留）。
- download `Zone.Identifier`、temporary extraction、partial、stale cache。

刪除前先產生 inventory、保留仍被 manifest 引用的 evidence、記錄釋放空間與重建方式。WSL 內刪檔後 Windows C 槽不一定立即縮小；需要時另做 WSL VHD compact。

## Current Status（2026-09-20）

- Correctness：strict Phase 0 STOP；historical simulation PASS。
- Corrected E5 diagnostics 已完成，E1 rolling-origin 實驗 bundle 已準備。
- E1 Colab bundle 修正提交：`044eef7`；4,558 files；包內 verifier 與 223 tests 通過。
- Production roadmap Phase A Shadow Operational Ledger 已實作於獨立分支；尚未啟用排程／deploy。
- 主工作樹有既有未提交 V7/UI 變更，必須保留。
- 下一個研究動作：由使用者在 Colab 執行／續跑 E1，帶回 evaluation artifacts。
- 下一個系統動作：Phase A shadow replay 驗證；不得越級啟用 full V6.2 或 V7 live。
