# P4b CPU 工程候選（2026-11-24 對話凍結）

狀態：**FROZEN 規則／CPU_ENGINEERING_ONLY／E2 RESEARCH_ONLY／UNMERGED／NOT_DEPLOYED／NOT_ACCEPTED**。Builder 自查不是獨立複審；完成不等於 Human acceptance。既有 P4a NO_GO／64-fit 完整性與 production authority 不變。

## 建立前盤點、範圍與決定

起始 detached HEAD `bc549b80703e97028602afb6e3186db5012764c6`、clean；沿用指派 `job-run-mwuunk1u-38d24c/builder_p4b`，不新增 worktree。已讀 AGENTS／CLAUDE、Current State 開頭凍結表、Authority hierarchy、P4b 全文、P4a machine contract、p4_eval、portfolio_eval、significance、p4_colab 的 resume_fit／BEST／ledger，以及 Research／Training／Data／Labels／Universe domain 與 ml-leakage-audit。

既有 P4a evaluator 的 family／64-fit gate 不容納 P4b 的獨立36-fit准入，且不能改其研究身分；因此在同一 `V6/research_lab/` 新增單一薄 adapter 與既有 tests/ 下的測試，重用 common_rows／averaged_scores／portfolio_rows／p3_pair／paired_ic_test／holm_adjust／resume_fit。既有 P4a 工程報告很長且主要描述 Colab 事故，因此沿用 reports/ 容器新增本短報告，避免把新混合誤列為 P4a 結果。不建立新 framework、top-level tree、backup 或獨立 memory。

重大選擇及替代：

- 固定36-fit loader，完整驗證64-job queue/binding但只准入Q1/Q2；替代放寬原64-fit gate會錯誤升格 run，因此不採。固定 run_hash、recipe、rankgauss identity、SOURCE_UPGRADE bytes／source maps、ledger generations/head/完成記錄及逐fit resume_fit（BEST/history/checkpoints/scores/reference）全部 fail closed；不開 Ledger writer、不恢復／修改帳本。
- 種子支援完全一致後，四臂同一全局 S、裁 raw seeds 後 rank，神經用同源 averaged_scores；C0+三seed做同義薄 adapter，固定seed排序消除輸入排序導致浮點ties漂移。B2保留外層rank，三個 .5/.5，無替代權重或擇優搜尋。
- D由P1 manifest已凍結的dates.npy／逐折splits.npy（split=2）生成並核bytes/hash，與輸入日期聯集比對；替代只用交集或聯集會漏掉全臂同日消失，不採。整日／折缺失STOP；非空但每日<95%或<50保留明細、不得提名。NaN label保留，IC診斷僅共同成熟列。
- 主要組合走同源portfolio_rows／P2全部Config；為取得summary省略的trades/daily持倉資訊，診斷另跑同一engine（只把其不供推論的bootstrap draws設1），以實際fills重播數量及marks，逐日核對weights+cash=1，不重做成交決策。正式統計始終block20／1000／seed17、centered雙邊；只2021–2023 Holm3，六折／近期／全部池完整揭露。空持股集合分母UNKNOWN，常數相關UNKNOWN，不填0以通gate。
- CLI將 `--rehearse-2021` 與 `--evaluate` 分開；全量入口要求事先凍結price/market_value size/hash、volume evidence及完整Config snapshot，缺少即STOP。這次只跑排練；未讀取正式價格／市值資料，未產生績效／候選結論。

## 凍結與身分

Human對話紀錄完整附於 [規格](../tasks/spec-p4b-blend-20261124.md) 與 [彙整表](../knowledge/00_Project_Map/Current_State.md#三份draft-human凍結決定彙整)；批准者Human、日期2026-11-24，確切UTC／台北時間UNKNOWN，無數位簽章。原始規則 SHA `8d4f194730c2a147b42afe69750f722fccab1d0111e721d2a1f279df369f53c2`，只改狀態並附紀錄、凍結後規則未改。前瞻及P4d仍DRAFT。

- P4a來源為Human下載解壓的主checkout `.artifacts/p4/p4a-results-3/run-6fbaf815b0716a086a794dfbc12a21f67688263d116469e272e57783448c3730`。來源ZIP SHA `b9060b740d0e94db7ae12e483ac20a095f0a7b3fce7e1ef1e13f1fa660d12116` 是 **Human提供，這次未重新開ZIP驗證**；逐fit解壓後bytes均由resume_fit核對。
- P1原manifest SHA `dc3b9b3ad9c2c27a5366440c651f5f6baa7daaec9854368a0e944874890be9d8`；18份三seed原分數逐一核對，僅選c0plus，不用cached rank_average分數。date/split歷史矩陣唯讀，非讀取blind／formal Data。
- 來源升級audit SHA `a77a884005b942279429acede859de06db707d96ee8b73ae342edee3fa6967ca`，固定原run binding、來源from/to maps、ledger事件與hash chain；不要求把新分析程式冒充舊訓練source。
- 所有逐fit output／P1 seed／split／日期及程式bytes、完整Config、環境與輸出hash皆在忽略Git的 `.artifacts/p4b/rehearsal-2021-final/manifest.json`；此manifest及程式身分另補下方，不宣稱文件metadata的新增使原規格hash仍相同。
- 價格／市值snapshot尚未提供或驗證，因此本輪沒有價格identity；全量評估前必須先產生並保存snapshot，不能以可變cache直接執行。這是剩餘輸入凍結／現場驗證項，不能稱全量已驗收。

## 實際驗證與限制

初始RED：15 failed（新模組尚不存在）。首版GREEN前抓到輸入shuffle後C0+三seed浮點累加順序改變ties；先保留失敗，再固定seed→keys順序，原15項通過。持股診斷新增前RED為1 failed／17 passed（缺holding_diagnostics）；補完後18 passed。上述是作者工程循環，非獨立驗證。

真實排練先驗所有36 fits及18份P1原分數，僅保留2021到記憶體計算；2021 **378,547 common rows**、所有每日95%／≥50 gates通過。第一次manifest `b9b8be232585622adae052be8d9050946a8f832b8115d7d72346d0c71adefdba` 保留；CLI／來源升級pin／環境身分補完後以fresh namespace重驗，最終manifest與實際測試結果見下方。沒有全量研究、重訓、GPU、雲端Drive、Colab、取得資料、排程、wakeup、額外agents、commit／push／merge／deploy。

已曝光歷史只能描述性；C0+調參且三seed、神經固定配方兩seed，不對稱偏向C0+。NaN成熟與共同支援處理不認證source PIT、財報可知時間或universe。raw股價無股利、非PIT市值、下市覆蓋、停牌／實際成交及流動性仍未驗證；縮小共同宇宙仍可能有selection bias。2026部分年度、pool blocks可能跨折、1000 draws精度有限。低秩相關不代表有增益。沒有任何獨立確認、可交易或架構普遍優劣宣稱。

仍未驗證：全六折真實混合／組合／統計／candidate輸出、真實價格及volume snapshot、實際停牌／缺價持倉replay、端到端evaluate CLI、正式ZIP transport完整性、獨立複審與Human acceptance。不得以本次支持率通過推定其餘皆通過。

## 工程身分補錄與實際focused證據

最終adapter SHA `064e8b8404bc0aefbb60598471c58ac78a71e952be203c5dff8126dfd10b5823`；tests SHA `542e239e09a123c44a4348e3a56924e253aa19bb44c145d8034bd53a0aa4b056`。完整code bytes（research_lab及experimental程式／兩契約，共269檔）identity manifest在 `.artifacts/p4b/engineering/identity.json`，SHA `3a9b6d8bd48d9218927d53d2d82aed6c507b792af71a4d1774c5e4554f16c276`。最終排練manifest SHA `78090080bcbc3c4d4a2e5564a130fe3ee0a44310658ebdb52fbda4e5825122f4`，內含逐輸入hash與Config；只在排練後增加測試，adapter bytes與排練相同。

P4b focused命令：`PYTHONPATH=V6 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 /home/frank/.venvs/marketmamba-gpu/bin/python -m pytest -q -s V6/research_lab/tests/test_p4b.py`；實際 **29 passed in 14.93s**。log `.artifacts/p4b/engineering/focused-final.log`，9項assertion-killed：weights／outer_rank／raw_logits／nan_drop／date_denominator／gate_success／missing_fit／rank_before_intersection／ignore_holm；不是用import或環境錯誤計為killed。測試含新上市支援、尾NaN、整日／整折／單臂消失、seed不等、非有限值／標籤錯配、原P4同源一致性、B退化相同持股及日報酬、Holm手算／雙邊符號反轉／重複日、36-fit metadata及ledger/head/hash損壞／唯讀。

Knowledge unittest：23 tests OK；health `--ci --as-of 2026-11-24` exit0、Overall WARNING（Governance／Authority Drift／Legacy Docs），無structural FAIL。impact工具source/report focused PASS；人工gate判斷凍結／navigation／blocker改變，故同步三knowledge與Worktree Register；Operations／GOVERNANCE規則無更動。

## 最終檢查與未解除的整合阻擋


最終驗證補錄（前次identity／排練產物皆保留、不覆寫）：每日同成熟mask實際IC、實際交易成本／gross/net揭露及執行時git身分補齊後，adapter SHA `233f08fb7462bf3c5375772b34fa0f7f20bc85a4c44edb99a93f1424bdd5c399`，tests SHA `3a242d019a322a6d25a4aa6bcb9fbfadf4ba21bc8cb15231b403630989fde793`；最終269檔code／契約identity manifest `.artifacts/p4b/engineering/identity-verified.json` SHA `21b33e7a4a7bca49ca375e07bba8f570f7f71d822dfbf819e4956a85c5c931d3`（包含此補錄前spec bytes）。最終真實排練 `.artifacts/p4b/rehearsal-2021-verified/manifest.json` SHA `0dcd0477c5318edfb764bf646585d08fd132d5f3616231393e045cc54ace35e4`；36 fits／18份P1 seeds／ledger769 events／2021共378,547列／244日期，976每日gate均通過，min retention=1.0、min names=1493，所有8份分數parquet hashes與前次完全相同。

最終focused **30 passed in 14.60s**、9項assertion-killed mutation。全套research_lab實際 **28 failed, 1153 passed, 6 skipped, 4 warnings in 384.86s**，28項同為`p4_source_compat.declaration`拒絕未審查新增`p4b_blend.py`；已獨立單測重現1 failed。保留gate、不修改P4a來源相容白名單或原64-fit gate，故**驗收STOP／未全套通過**，不是宣稱既有baseline缺陷；新分析模組造成source inventory整合阻擋，需獨立數值中立審查及明確來源打包範圍決策。未執行Drive／Colab或建立升級交付。全量真實評估／price-volume snapshot／獨立複審／Human acceptance仍待。

完整回歸命令：`PYTHONPATH=V6 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/frank/.venvs/marketmamba-gpu/bin/python -m pytest -q V6/research_lab/tests`；完整輸出保存在`.artifacts/p4b/engineering/full-suite.log`，28項失敗只有一種error signature。此全套collection當時1187項，後加2個focused驗證（IC與額外mutation）後本地collection1189；不把後來focused測試混稱全套當時已跑。所有執行皆CPU，既有訓練相關測試只用合成fixtures／替身，沒有真實研究重訓。

規格原凍結規則bytes（排除核准狀態行／附加metadata）與HEAD逐字比較通過；protected rolling_seq_scores／portfolio_eval與HEAD bytes一致。AST兩檔、674個本機Markdown references、8個輸出parquet hashes與前次一致、git diff --check通過。工程identity manifest保留各時點bytes，不覆寫前版manifest。無staged、5份既有文件modified＋3份新增untracked；artifact為ignored。依no-commit未做到clean，沒有commit／push／merge。

後續Human用法：唯讀排練可用`PYTHONPATH=V6 CUDA_VISIBLE_DEVICES='' /home/frank/.venvs/marketmamba-gpu/bin/python -m research_lab.p4b_blend --run-dir <指定run-6fbaf815…> --c0plus-dir <指定P1目錄> --output-dir .artifacts/p4b/<fresh-id> --rehearse-2021`。全量改`--evaluate --data-root <事先凍結的snapshot目錄> --snapshot-manifest <事先保存的JSON>`；JSON需`files`中的prices_raw.parquet／market_value_raw.parquet各bytes/sha256、`volume_units`完整同源證據及`primary_config`完整全欄JSON。這次沒有產生此snapshot或全量結果，不能略過。


## 本機評估模組與 Colab bundle 來源邊界修補（2026-11-24）

本輪 `job-run-mwuv96vj-55da06/builder_bundle_exclude`，沿用既有 detached HEAD `8dc5cc961f520272cd94281b8e104347cdecd034`、起始 clean；**UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED / CPU_ENGINEERING_ONLY**。上一節 28 failed 為原執行歷史，不改寫其 log 或原研究 artifacts。原 log 不在本 worktree，已唯讀核對主 checkout 同路徑 `.artifacts/p4b/engineering/full-suite.log`，確為 `28 failed, 1153 passed, 6 skipped, 4 warnings in 384.86s`，共同 error 為未審查新增 p4b_blend.py；本輪新 log 使用 `bundle-exclude-*` 名稱，不覆寫原證據。

重大選擇與替代：

| 方案 | 決定與理由 |
|---|---|
| 一般顯式 denylist | 單純排除不足；若未綁宣告與必要來源保護，可能悄悄漏掉訓練檔 |
| 本機專用精確路徑契約清單 | **採用**；唯一 `V6/research_lab/p4b_blend.py`，固定 SHA pin、共用 collector、宣告／採用精確比對及負向測試，最小可稽核邊界；未來逐項審查擴充，不自動納入同名類型 |
| 依目錄排除／搬檔 | 涉及移動既有評估入口與 imports，擴大改動；整目錄排除也可能隱藏必要來源，未採 |
| 依命名排除（例如 p4b_*） | 新檔可意外逃過差異 gate，未採 |
| 只重產宣告 | 仍把本機研究模組打進 Colab，不能解決邊界；把它加 REASONS 會錯誤宣稱舊訓練依賴新評估，未採。重產宣告是選定清單機制後的必要步驟 |

沿用 p4_bundle／p4_source_compat、既有來源相容測試、tasks GENERATED 宣告與既有 docs／reports／knowledge 容器；不建立新 implementation、worktree、實驗／memory tree 或交付。collector 不讀本機專用 bytes；AST 檢查拒絕包內來源靜態及字面值動態 import 該模組；allowed_path 在 write_archive／extract／verify 拒絕手動注入。這不是任意 Python 動態程式之形式化安全證明；動態構造新載入行為必須另審，Colab 亦不得自行把本機 checkout 加入 sys.path。

宣告由現有 `p4_source_compat.declaration('4e56e9d', current_sources(), previous_declaration['matrix'])` 工具重算，再 JSON serialize（未手填逐檔 hash／diff）。from 不變、to.commit 是 generation-time HEAD `8dc5cc9`，230 個 to sources hashes 綁 working bytes。matrix identity 由已版本化舊宣告保留 2183 bytes／`bc6401095260e7d4153060e3106fff8dbae88cfb6df3f349b808b7dd762c7d9f`，本輪沒有讀正式矩陣或重新認證資料 bytes；deliver 仍須以實際 manifest 重算。六個實際差異仍為 p4_bundle、p4_colab、p4_eval、p4_source_compat、seq_runner、colab-p4.md；本輪只改前述 bundle／source_compat 兩個來源，理由增加「本機專用來源排除」且採用端限定此分類只可用在這兩檔。先前進度／日誌／W≠60／成熟標籤／排練閘門差異完整保留，不冒稱純文件或本輪新改。排除的 P4b 自基準不存在，故不是偽造一筆來源 deletion；宣告另明列精確排除清單。

36 個完成 Q1／Q2 W60 必須全部 resume_fit 原地驗證、保留 bytes／不重訓的採用政策與 64-fit queue SHA `4dc90ec0ada6c6d34b43c429d359424c502634e502c6856c6774bf138de7ba23` 不變。既有回歸繼續驗跨版 GRU／crossstock CPU 權重／預測／RNG／epoch resume、36→首個 Q3→64 fits、原 notebook 保底及 missing/corrupt/wrong-run 拒絕。p4b_blend.py、rolling_seq_scores.py、portfolio_eval.py、P4a contract、runtime／lock 與起始 HEAD 逐 byte 一致；沒有真實重訓或改 P3 規則。

新增負向：清單加入 p4_colab／seq_runner／p4_eval／p4_bundle 必須拒絕；非清單新檔不得未審查進宣告；caller 漏掉 seq_runner 的 to inventory 拒絕；缺失／擴大宣告清單、重新注入本機來源及把排除分類誤套 seq_runner 均拒絕。測試以隔離合成 checkout 兩次完整 pack／declaration／extract／verify，只改 p4b_blend bytes（第二版含無效 Python／NUL），比對所有成員含宣告與 bundle-manifest 的 bytes／SHA 完全相同；隔離 `python -I` 僅加 extracted source 路徑，確認 find_spec 無該模組。三項 in-memory mutation 分別移除 collector filter、archive boundary、policy pin，由 assertion／DID NOT RAISE 殺死，不以 import 或環境錯誤計數。

本輪最終驗證與 identities 見下方補錄。初始 RED 為 `1 failed, 32 deselected in 1.73s`，確實在來源集合仍含 p4b_blend 的 assertion 失敗。


### 最終驗證補錄（執行跨日至 2026-11-25）

開發 focused 實際 `47 passed, 1 skipped in 266.54s`（較早測試集合；不冒充最後版本）。自查發現合成 checkout fixture 原假設 `.git` 為檔案，改用 `git rev-parse --absolute-git-dir`，可承載一般 checkout 與 worktree；新增字面值動態 import 兩種形式、排除分類範圍及實際新增 `p4b_other_local.py` 仍拒絕的案例。第一次尚未完成的全套被主動中止（exit130），log 保留為 `bundle-exclude-full-suite-interrupted.log`，不算 completed run 或 PASS。

修改後 focused 命令：

```bash
PYTHONPATH=V6 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/frank/.venvs/marketmamba-gpu/bin/python -m pytest -q -s V6/research_lab/tests/test_p4_source_compat.py -k 'excluded_content or local_only_in_memory or manual_local_only or colab_dependency'
```

實際輸出（ZIP container SHA 受打包時間 metadata 影響，沒有宣稱兩個 ZIP container hash 相同；逐成員 bytes／SHA 才是本驗收）：

```text
excluded-content isolation: 236 bundle members byte/hash identical; isolated Colab import absent
assertion-killed: collection_filter
assertion-killed: archive_boundary
assertion-killed: policy_pin
12 passed, 42 deselected in 20.41s
```

最後全套沿用原命令，不刪測試、不放寬斷言：

```bash
PYTHONPATH=V6 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/frank/.venvs/marketmamba-gpu/bin/python -m pytest -q V6/research_lab/tests
```

七份保護檔逐 byte 核對 HEAD：P4b adapter、P3 rolling_seq_scores／portfolio_eval／規格、P4a contract／runtime／lock；其中 P4b adapter SHA 仍 `233f08fb7462bf3c5375772b34fa0f7f20bc85a4c44edb99a93f1424bdd5c399`。相對上一份版本化 GENERATED 宣告的 to source 集合差為空，只有 p4_bundle／p4_source_compat 兩個 hashes 變動；相對 4e56e9d 的六檔差異仍完整揭露。宣告當前 identity 為 189414 bytes／SHA `3843dffe9e6a6703520f29ce9aa76ebdac67370fbfb4899313970b19509b4d6f`。

人工 knowledge-impact gate：來源邊界／候選工程 gate／GENERATED 宣告／整合 blocker與next action=YES，故同一 patch 更新 Current State、Authority Map、Training 與 Worktree Register；既有研究語義／正式權威／active外部runtime／deprecation=NO。工具 impact PASS／DEFAULT 不取代此人工判斷。治理 unittest `23 tests OK`；六份文件本機 Markdown references 682 checked／0 missing。code-review-and-quality 五軸自查：來源集合與負向 gate、單一 collector／無新增實作樹、標準庫 AST 不執行研究程式、封裝多層拒絕與無訓練熱路徑影響；不是獨立複審，不能自行宣告 Human acceptance。

Runtime：WSL-native 指派 worktree，Python3.12.3／torch2.10.0+cu128／numpy2.5.4／pandas3.1.1／pytest9.2.0，使用既有 marketmamba-gpu venv 但 CUDA_VISIBLE_DEVICES=''、OMP／MKL=1，執行全限 CPU。MARKETMAMBA_DATA_ROOT unset，Data realpath 是本 worktree 普通 Data 目錄；未讀改正式 Data。沒有 GPU／Drive／Colab／網路或資料取得／production／scheduler／wakeup／recurring job／額外agents／commit／push／merge／deploy。

Remaining／風險：僅 synthetic CPU 工程與歷史來源 objects；不重新驗現場36 artifacts、CUDA官方mixer、Drive持久化或真實資料接續。P4b 全量 price-volume snapshot／全六折組合及研究判讀、獨立複審、Human acceptance 仍待；工程全套通過也不解除這些 STOP。不把原1153/28失敗紀錄或歷史研究成果重標為新證據。最後保留本任務10份 unstaged、0 staged／0 untracked source（logs為ignored），遵no-commit無clean完成宣告；safe-removal須先保留patch／必要證據並另行審查。

全套完成 exit0，完整實際輸出（不含前次中止執行）：

```text
........................................................................ [  5%]
.......................................s...s............................ [ 11%]
........................................................................ [ 17%]
........................................................................ [ 23%]
........................................................................ [ 29%]
........................................................................ [ 35%]
........s............................................................... [ 41%]
.................................s...................................... [ 47%]
........................................................................ [ 53%]
........................................................................ [ 59%]
........................................................................ [ 65%]
........................................................................ [ 71%]
........................................................................ [ 77%]
..........s............................................................. [ 83%]
........................................................................ [ 89%]
........................................................................ [ 95%]
................................s.s........................              [100%]
=============================== warnings summary ===============================
research_lab/tests/test_gpu_env.py::test_verify_requires_cuda_or_raises
research_lab/tests/test_gpu_env.py::test_verify_requires_cuda_or_raises
  /home/frank/.venvs/marketmamba-gpu/lib/python3.12/site-packages/quack/copy_utils.py:15: DeprecationWarning: cutlass.utils.LayoutEnum is deprecated; use cutlass.tensor_utils.LayoutEnum instead. It will be removed in a future release.
    from cutlass.utils import LayoutEnum, block_copy

research_lab/tests/test_gpu_env.py::test_verify_requires_cuda_or_raises
  /home/frank/.venvs/marketmamba-gpu/lib/python3.12/site-packages/quack/copy_utils.py:562: DeprecationWarning: cutlass.utils.LayoutEnum is deprecated; use cutlass.tensor_utils.LayoutEnum instead. It will be removed in a future release.
    layout_c: cutlass.utils.LayoutEnum,

research_lab/tests/test_p4.py::test_zip_inventory_and_payload_gates[duplicate]
  /usr/lib/python3.12/zipfile/__init__.py:1620: UserWarning: Duplicate name: 'folds/fold_2021/X.npy'
    return self._open_to_write(zinfo, force_zip64=force_zip64)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
1204 passed, 7 skipped, 4 warnings in 608.24s (0:10:08)
```

共1211項＝1204 passed＋7 skipped，0 failed。skip為三項套件已存在的缺件訊息測試、三項可選nbformat缺件、以及一項CUDA smoke；原先來源相容交付在產生宣告即失敗，本輪走過既有交付內容／notebook compile斷言後才因nbformat缺件而skip，因此相對原6個skip多1。未新增skip、刪測試或放寬斷言。4 warnings為既有cutlass deprecation與故意重複ZIP member的負向fixture。

最終 source／tests／保護檔與主要logs identity manifest：`.artifacts/p4b/engineering/bundle-exclude-identity.json`，48572 bytes／SHA `b8e470293220544ff346350ab7f5df5323c6215c18b8875e225cfc159b894d33`，277個source／tests／contract／文件identity及7份HEAD byte-identical保護檔；保留matrix未重驗／CPU合成／not accepted限制。完整全套log SHA `3f0efd6b71a196428642aea2c06de1b25a9a7c0698c4a69bbd857eaa35357b0b`。
文件結果補錄後另跑既有三項相關契約測試：`3 passed, 110 deselected in 1.66s`；log `.artifacts/p4b/engineering/bundle-exclude-docs-final.log`，SHA `f6261386a782c4390785970f2d2fba0090b99a00af83b3813422700d86024cbe`。此補充log產生於上列identity manifest之後，故直接於此記錄，不偽稱已收進該manifest。
最終impact（全部10個changed paths）exit0／WARNING，原因為canonical-knowledge及worktree-responsibility要求人工review；上文source/report子集PASS是較早工具輸出。人工已依gate更新四份knowledge。最終 `knowledge_health.py --ci --as-of 2026-11-25` exit0，Project Map／Agent Guides／Domains／Runtime Freshness／Generated Index PASS；Governance／Authority Drift／Legacy Docs WARNING，Overall WARNING，無structural FAIL。外部runtime observed_at未變。

本輪changed paths（10）：

- `V6/research_lab/p4_bundle.py`
- `V6/research_lab/p4_source_compat.py`
- `V6/research_lab/tests/test_p4_source_compat.py`
- `tasks/p4a-source-compat-20e6b54.json`
- `docs/operations/colab-human-requirements.md`
- `reports/v7_p4b_engineering.md`
- `knowledge/00_Project_Map/Current_State.md`
- `knowledge/00_Project_Map/Authority_Map.md`
- `knowledge/00_Project_Map/Worktree_Register.md`
- `knowledge/01_Domains/Training.md`

Completed：bounded bundle來源邊界修補／完整CPU回歸；Changed：上列10paths；Decisions：精確本機專用契約清單，替代方案如表；Evidence：1204 passed／7 skipped／0 failed、236成員hash隔離、3 mutation killed。Remaining：現場／獨立複審／Human acceptance／全量P4b；Blockers：本候選CPU整合無，既有現場／研究STOP仍在；Next recommended action：獨立複審patch與宣告，再由Human決定。Project memory updated：YES；Authority map updated：YES（candidate邊界，不升格authority）；Git status：10 unstaged、0 staged／untracked source，ignored artifacts保留；Commit：N/A（禁止）；Push：N/A（亦無merge／deploy）；Worktree：既有builder_bundle_exclude detached8dc5cc9；Runtime/environment：上列WSL／Python3.12.3／CPU，Data與正式模型未動。完成不是Human acceptance。
