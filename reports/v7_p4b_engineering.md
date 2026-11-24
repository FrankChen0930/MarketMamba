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
