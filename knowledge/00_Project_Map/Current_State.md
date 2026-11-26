## 第三輪 F1–F4 修補驗證候選 — 2026-11-26

既有 `job-run-mwxhve2s-34eacf/builder_fwd_fix_r3`，detached base `5b9f352dd81875261d67bc3a7283b10c1560c140`，起始clean，Human no-commit，**SYNTHETIC_CPU_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED**。[WP5回應](../../reports/forward_wp5_20261126.md)記錄：事故回傳／丟棄回傳及誤導摘要不能繞過 journal 開盲 gate、兩程序 nonce／PID／開始時間 receipt 與四輸出 binding、spec＋sessions＋external-head 唯一開盲 identity；[WP2回應](../../reports/forward_wp2_20261126.md)記錄未來 features／labels finite↔NaN 的210組逐日 oracle 及 missingness mutant KILLED。測試等待30→120秒與 stderr 回報僅改善測試診斷，不放寬產品 deadline。

RED 9 failed；集中GREEN 14 passed；最後標準全部 forward 測試 **428 passed／172 NaN聚合 warnings／386.13秒／exit0**，無basetemp。GENERATED logs／來源、規格、review hashes與環境見本worktree `.artifacts/forward/fix-r3-20261126-5b9f352/manifest.json`。本節更新下方上一輪「尚未收斂」的驗證狀態；不重寫其歷史證據。第三輪審查WP1／WP3／WP4 GO保留，WP2／WP5本輪候選待獨立複審，整包NO_GO／Human acceptance待辦；seed17真實bitwise、WP6、合格日曆、lifecycle可知時間、lagus provenance仍STOP。規格／正式資料／production authority未變，無外部runtime observation更新；next為獨立複審本輪四項及Human驗收。

## F1–F6 有界複審修補候選 — 2026-11-26

終局時間界線：完整 focused 已啟動，程式碼凍結後重跑仍有1個失敗且尚未收斂；依收集順序疑似WP3 SIGKILL測試，未取得traceback前不定因。集中關鍵驗證亦未結束。不得將本輪升格為全GREEN或Human acceptance；NO_GO保留，next為收取完整輸出、診斷失敗後再複審。


本 worktree `builder_fwd_fix_r2`、detached HEAD `0518d05`，Human no-commit 有界核准。CPU_ENGINEERING_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED。[WP5回應](../../reports/forward_wp5_20261126.md)記錄事故鏈不可逆拒絕、calendar/session-domain 唯一開盲 registry、獨立程序 request/origin/output 綁定與發布檔案 allowlist；[WP2回應](../../reports/forward_wp2_20261126.md)記錄連續14個eval sessions三seed逐日oracle與四mutants；[WP3回應](../../reports/forward_wp3_20261126.md)記錄SHA relocation，原歷史manifest不變；[WP4回應](../../reports/forward_wp4_20261126.md)記錄fixture邊界與清理錯誤回報。實作與測試候選不授正式前瞻、production或OS隔離。WP1–WP5整包NO_GO、seed17 bitwise待Human與WP6待Human端點保留。無外部runtime觀測更新；next為獨立複審及Human acceptance。

## P2 WP5 非績效驗收候選 — 2026-11-26

`job-run-mwxe944p-5885dc/builder_fwd_wp5`、既有detached base `e60001207346573e26f03154cd502cada7ca2bd7`，起始clean、Human有界no-commit核准。**SYNTHETIC_CPU_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED；完整前瞻仍STOP。** [腳本](../../tools/forward_wp5.py)／[測試](../../tests/test_forward_wp5.py)／[靜態Human清單](../../docs/operations/forward-retraining-checklist.md)沿WP1 calendar／18:00、WP2 C0+、WP3 immutable reference與WP4存檔／pointer／淨化摘要，驗固定source pin／snapshot／origin／seeds、R→R+1→R+2私有event鏈、截止／初始不啟動、事故／白名單／合成手動一次開盲。兩獨立新程序重做合成48／185矩陣、三seedkeys／scores／ranks及WP4 native roundtrip，全bytes比較；不計績效、不讀blind、不切正式pointer。私有重訓STOP不控制公開來源日鏈節奏；同權限hash鏈不是OS隔離或防竄改。

[報告](../../reports/forward_wp5_20261126.md)記終局98 focused（標準PYTHONPATH、無basetemp）／195相關regression、兩source mutations KILLED；CLI44.42秒／RSS444,472 KiB／exit0，69份source pins及261終局artifacts，R+2成功／超時events與R到R+3來源鏈。所有逐列／模型／logs只在本worktree`.artifacts/forward/`。首輪deadline mutant被另一late guard遮蔽，改oracle為R+3精確18:00；來源投影先於private reference的新RED亦保留，修正後重新驗證，不放寬實作。候選source freeze不是Human簽署，合成252開盲audit不是WP6端點或正式結果釋出。真實P1 seed17 bitwise、合格calendar／PIT／lifecycle、WP3 provenance／daily source與WP4環境lock、20-session穩定鏈／WP6仍STOP／待辦。Next為獨立review及Human acceptance，未授T0／T1啟動。無網路／資料取得／GPU／agents／正式Data／V6/models／TEJ SHA256SUMS／P4d／第三步／production／scheduler操作，無commit／push／merge／deploy／wakeup或外部observed_at更新。

## 前瞻 WP3 同版本 mask／完整 reference 候選 — 2026-11-26

`job-run-mwxdbyz7-28c523/builder_fwd_mask@3f94ae2`，no-commit **CPU_ENGINEERING_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED**。[報告](../../reports/forward_wp3_mask_20261126.md)記錄2,244份prepared cache與22份universe完整hash符合原E5，呼叫原builder在補零前重建mask；重建X完整SHA `9778392d4c337d1f9cb5768002340a0d1aec13f7689b2f601b77bd2329b55363`、cmp exit0／0格差異。mask有2,453,171缺觀測格；F-layout序列化SHA精確符合原manifest的004771dd…，未用舊mask／全1／zero-filled X推回。

真實7,423,927×185 immutable工程reference候選已封版並完整重驗，identity `e50d9db919f148f59685af598b345a40ab679e87434e3fa718016f126dc7098e`，私有`mask-rebuild-20261126/full-reference`；原C0+ helpers完整axis逐欄memmap，256真實列全部185欄重呼原函式bytes相同。reference395.472748秒／RSS10,667,880 KiB／exit0，198相關＋3串流tests通過。既有P1 scores僅opaque bytes；沒有fit／績效／labels或每日append驗收。

原lagus腳本搜尋未恢復，僅roadmap配方重現後精確核X，mask仍為provenance-limited候選；原首輪layout／tree-root摘要誤判與中止的列批次partial完整保留。原WP3缺件報告是歷史觀測，本候選不覆寫舊artifact或解除整包NO_GO、WP1來源／calendar／lifecycle、WP2 seed17等STOP。Next為獨立review候選來源鏈及Human acceptance，再驗每日來源／延伸，不啟用日鏈／排程，無外部observed_at更新。

## 前瞻 F1／F4 審查修補候選 — 2026-11-26

`job-run-mwxcypdo-30712f/builder_fwd_fix12`，既有detached base `d8d76c7cd4f1c8533c3c7c16fc661efb03d76c88`，Human核准有界no-commit，**CPU_ENGINEERING_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED**。回應主checkout私有 `fwd-wp1-4-review.md` 的F1／F4；[WP2測試](../../tests/test_forward_wp2.py)改在origin與+1／+3／+8 sessions逐次改未來labels／features，另按+5／+10收益端點改未成熟labels；只改列日期>t不足以抓直接使用當列eval label。三seed共36 cases核全部≤t scores／ordered keys及實際LightGBM fit target／features與歷史predict features雜湊。審查eval-label及未來feature source mutants在四個時點共8／8被攔，正式oracle對兩mutants各exit1，原實作GREEN；未發現現行計算洩漏，未改fit/predict邏輯。[WP1測試](../../tests/test_forward_wp1.py)鎖定helper與WP2 train≤origin−21、origin−20排除；兩模組僅註解統一。

標準PYTHONPATH=V6:V6/research_lab/tests:.、無--basetemp的focused61 passed／12.81秒，rolling split／runner29 passed／4.29秒；本輪未跑全套或真資料fit。來源PIT／日曆／lifecycle、WP2 seed17 bitwise待Human、WP3同版本mask／reference與WP4環境還原lock仍STOP，**WP1–WP4整包NO_GO**不解除；修補待獨立複審與Human acceptance。詳[WP1回應](../../reports/forward_wp1_20261126.md#f1f4-獨立審查回應--2026-11-26)與[WP2回應](../../reports/forward_wp2_20261126.md#f1f4-獨立審查回應--2026-11-26)。私有GENERATED evidence只在本worktree`.artifacts/forward/fix12-20261126/`，未改規格／acceptance門檻／machine或production authority；無外部observed_at更新，無網路／GPU／agents／正式Data／V6/models／P4d／第三步／排程／blind操作，無commit／push／merge／deploy。

## P2 WP4 模型存檔／合成封盲介面候選 — 2026-11-26

`job-run-mwxb97t0-5e7fef/builder_fwd_wp4`，既有detached base `561392a95c9232c946097630ac51d9e87724b5d6`，起始clean、Human有界no-commit核准。**SYNTHETIC_CPU_ONLY / CPU_ENGINEERING_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED**。[存檔工具](../../tools/forward_wp4.py)沿用Step3 fsync／hash／resume primitives，三seed17／29／43先local封存、durable copy逐檔重驗與native load/predict bytes核對後才atomic切整體current pointer；corrupt／missing pointer STOP，orphan不升格、不自動fallback。每fit開始前fsync完整套件／LightGBM／native binaries／執行緒／CPU flags／affinity／layout與request，模型版本再保存環境；跨matrix切版須明示舊context且保留舊bytes。

[淨化介面](../../tools/forward_wp4_summary.py)只有SYNTHETIC_SOURCE／合成clock，source projection先於private fit決定；Python/native logs與事故私有，公開閉合schema／來源全宇宙計數／私有256-bit nonce commitment／hash chain。沒有讀真正blind、真資料或部署Human runner，固定次日12:00僅clock fixture，非OS隔離、排程或operational證據。終局focused83 passed／105.89秒；相關WP1–WP3／Step3合成回歸114 passed／24.85秒，2／2 source mutants killed，6類實際SIGKILL／新程序恢復；合成CLI7.42秒／RSS426,876 KiB／exit0。終局57個合成檔案／before-fit環境／三seed各37筆預測bytes與public/private chain重驗通過，identity詳[報告](../../reports/forward_wp4_20261126.md)，不混用中間版本證據。

WP1 calendar／PIT／lifecycle、WP2 seed17 bitwise、WP3真實mask／provenance STOP全部保留；完整環境紀錄不證實或解決WP2根因，未保證跨環境重訓相等。中間測試曾混用已載入舊fixture與磁碟新碼而186 pass／1 fail，原log保留；最終固定code新程序結果分開記錄。Next：獨立review／Human acceptance與既定WP5非績效驗收，未授前瞻啟動。只有forward namespace的GENERATED合成產物；無網路／資料取得／GPU／agents／正式Data／V6/models／TEJ SHA256SUMS／P4d／第三步／production／scheduler寫入、無commit／push／merge／deploy／wakeup或外部observed_at更新。

## P2 WP3 矩陣延伸工程候選 — 2026-11-26

`job-run-mwxakzhz-f1f879/builder_fwd_wp3`，既有detached base `2df4504c9aa6e7a3cb1c43f13fa71621c2da6923`，起始clean、Human有界no-commit核准。**CPU_ENGINEERING_ONLY / RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED；WP3整體驗收STOP**。[工具](../../tools/forward_wp3.py)、[合成測試](../../tests/test_forward_wp3.py)、[報告](../../reports/forward_wp3_20261126.md)沿用tools／tests／reports與私有`.artifacts/forward/wp3-20261126/`。已prepared的48欄feature-only切片先建完整reference、再追加獨立版本；原C0+完整prefix重建185欄，舊keys／X／mask／dtype／splits／eligibility與opaque既有scores bytes不變才可封版。固定calendar／stock axis；labels另版且有maturity gate，feature API不讀labels；未提供raw資料到prepared48欄的每日來源程序或前瞻准入。

最終focused／相關regression152 passed／46.96秒、2／2 source mutations killed，含未來X／universe／poisoned labels、真實SIGKILL與partial拒絕。唯一真實掃描4檔、7,423,927×48，6.0373秒／程序8.31秒／RSS1,696,492 KiB，**exit2／STOP_INCOMPLETE_REFERENCE**：prepared缺原始masks.npy；實測X SHA9778392d…符合外層P4 dataset manifest，與內層沿用舊manifest的67358805…不同。未代造mask、未改來源／manifest；真實185欄／scores／全量延伸成本尚未驗。Next：獨立review並定位同版本mask／vintage chain；WP1來源STOP與WP2 seed17裁定保留。完成不是Human acceptance。無網路／GPU／agents／blind／正式Data／V6/models／TEJ SHA256SUMS／P4d／第三步／production／scheduler寫入、無commit／push／merge／deploy或外部observed_at更新。初輪matrix regression曾以TemporaryDirectory寫合成/tmp，最終命令已將TMPDIR／basetemp限定forward；偏離與全部限制見報告。

## 同業代理工作包3資料層候選 — 2026-11-26

`job-run-mwx9y2gu-4a162b/builder_peer_proxies`，既有detached base `d8362c055e68609aa4f5cf327cc5888f9d08699a`，起始clean，Human核准整合規劃§C／D／E工作包3，no-commit。**PARTIAL_RESEARCH_REPLAY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED**。[工具](../../tools/peer_proxies.py)、[合成測試](../../tests/test_peer_proxies.py)、[報告](../../reports/peer_proxies_data_20261126.md)與[彙總JSON](../../reports/peer_proxies_data_20261126.json)只交同業落後價差／營收先公告資料層，逐列及manifest只在本worktree `.artifacts/peer/`，不串接P4d、第三步或正式矩陣。沿用tools／tests／reports與既有project map；既有流程沒有本包無標籤同業日表，因此新增單一工具，不另建頂層實驗樹。

主分類固定TEJ子產業：較細，2958檔中1934檔無變更紀錄（TSE為1622），按結構穩定性選，未用報酬選型。raw變更值方向與known_at、下市終日語義仍UNKNOWN；有產業／市場變更者保留未知，不能推定生效後值或倒填現況。完整歷史成員／strict PIT皆STOP，報告區分可重建子集、未知分母及現況差異上下界。代理至少3家、排除本股；1a用t−1／t−2 raw close價差，市場等權為固定對照；1b只看上個完整月、發布日嚴格之後session收盤、兩期首次單月值都可用才計YoY，目標可用即回缺，跨2013口徑NaN。

真資料3392 sessions／5,805,094候選日股鍵，1a有值912,976、1b有值33,974；focused33 passed／1.64秒，3程式mutation killed、2未來資料不變性測試。6輸入／18產物／6實作SHA與全部唯一鍵重驗；三session獨立數值重算1271／1271代理列與5141市場對照列通過。knowledge health為WARNING、無structural FAIL；詳報告。作者自查非獨立驗收，完成不等於Human acceptance。本輪回歸誤選既有tej_diag整份合成測試，曾執行合成IC／未來報酬測試，已記偏離並停止該測試；未讀真實標籤、未來報酬或blind，未計真資料IC。無網路、資料取得、GPU、agents、正式Data／V6/models／原SHA256SUMS／production／scheduler／帳本／P4d／第三步bundle寫入，無commit／push／merge／deploy或外部observed_at更新。Next：獨立review、補變更欄位方向及可知時間，再另案決定歷史／前瞻准入。

## 季報版本工作包1工程候選 — 2026-11-26

`job-run-mwx9hm8k-43c245/builder_quarterly_versions`，detached base `9613273bf1bc2f902e1d856f60aefcfaf55e302e`，起始clean，Human已核准整合規劃§A／E工作包1，no-commit。**DESCRIPTIVE_SOURCE_VERSION_AUDIT / RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED**。[工具](../../tools/tej_quarterly_versions.py)、[測試](../../tests/test_tej_quarterly_versions.py)、[報告](../../reports/tej_quarterly_versions_20261126.md)及[彙總JSON](../../reports/tej_quarterly_versions_20261126.json)沿用tools／tests／reports；逐列只在本worktree `.artifacts/tej/quarterly/release-20261126/`。未讀價格／報酬／標籤／blind，無網路、資料取得、GPU或額外代理。

自結63,266列／2,581公司；3,898重複列全部是1,949組Y/N雙範圍，1,524組同日、425組異日，同範圍重鍵0，不能當修訂歷史。金額更接近YTD累計，僅為單位×1000假說下的數值線索；自結Q/YTD／單位／幣別仍待原件。2020Q4覆蓋1,930公司（2019Q4為488），現況市場／產業代理顯示跨類增加，但強制規則或供應政策成因UNKNOWN。60,641同範圍日期配對中，自結早於正式45,069、同日15,382、晚於190，中位提前3日；不證日期對應數值版本。

唯讀4-run帳本chain／base/index驗證，實際40業務鍵／40事件轉移從base驗previous_json通過，留61個版本；僅2026-11-25本機觀測可回放，不倒推首次公司公告。21 EPS REVISED為5名稱變更、6浮點表示微差、10實質數值變化；正式16僅符new、4皆符、1皆不符，10實質變化皆符new，仍不證公司重編。19 FIRST_SEEN均非本次四科目。正式115,876列首次／修訂版本不可判定100%；98,823日期／結構候選不等於PIT，strict歷史合格0。下一步獨立review與工作包2原件核版，不讀報酬；未變更原SHA256SUMS、正式Data、V6/models、帳本、P4d、第三步bundle或production／scheduler authority，未commit／push／merge／deploy，未更新外部observed_at。最終focused與治理結果以報告為準；工程完成不等於Human acceptance。

## P2 前瞻 WP2 任意 origin 工程候選 — 2026-11-26

指派 `run-mwx8ka1a-f6fcaa/builder_fwd_wp2`、detached base `850ea71c441b0a7288fbabf2cfafed04899125b7`，Human bounded no-commit 核准，**CPU_ENGINEERING_ONLY / RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED**。[adapter](../../tools/forward_wp2.py)／[focused fixtures](../../tests/test_forward_wp2.py)沿用 P1 C0+ feature／full-cross-section rank／三 seed 抽樣；任意 origin 採 WP1 Calendar 算 origin−21 並**含端點**，等價 P2 `< origin−20`。初輪 WP1 排他 docstring 歧義已由本次 F4 修補統一含端點，依 P1 executable／P2 原文處理。歷史重現僅授權舊 fold_2026；observed-union calendar 明標 reproduction-only，不當正式前瞻 calendar。

[報告](../../reports/forward_wp2_20261126.md)與本 worktree `.artifacts/forward/wp2-20261126/`保存三 seed scores／模型文字／全量及抽樣 keys／replay SHA。三seed完成456.1562秒／exit2：17有50／302,228筆score bytes不同、29／43全相等；全部鍵／seed ranks／三seed rank average相等，**bitwise 驗收仍STOP，不接受近似相等**。9 input／5 reference／21 output hashes重驗、三seed keys replay通過；1／4執行緒及C／F推論均重現candidate，訓練端根因未證實，完整診斷與regression終局見報告。最終focused100 passed／48.57秒；隔離全套1429 passed／1 failed／11 skipped／914.42秒，唯一失敗是既有P4d測試限/tmp與本輪forward basetemp不符（P4d原碼未改），不改P4d或gate，forward目錄bind至/tmp後該項1 passed／18.56秒，確認harness路徑原因；不宣稱完整全套重跑PASS。來源日曆／PIT／publication／vintage／lifecycle gates 不變，strict前瞻仍STOP；WP3–WP6、獨立審查／Human acceptance 待辦。未改規格、WP1、P4d、正式Data、V6/models、第三步bundle、production／scheduler；無網路／GPU／agents／commit／push／merge／deploy／wakeup。無外部 observed_at 更新。

## 共用前瞻 WP1 工程候選 — 2026-11-26

`job-run-mwx7iuz6-61038d/builder_fwd_wp1`，detached HEAD/base `08e69fe24486e3e42cce033b04c289e62984c668`，初始 clean，Human 有界 no-commit 核准，**CPU_ENGINEERING_ONLY / RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED**。[報告](../../reports/forward_wp1_20261126.md)、[模組](../../tools/forward_wp1.py)及[測試](../../tests/test_forward_wp1.py)提供 P2／Step3 共用 session 數學、18:00／A+2、snapshot lifecycle 與 readonly ledger as-of adapter；兩案輸出／模型／日鏈未啟動。focused 79 passed／27.74秒，3 mutations killed；隔離全套600秒逾時exit124／NOT_COMPLETED、約65%而無終局總計（全套載入補充函式前來源，最終版另由focused覆蓋）；獨立驗收／Human acceptance 尚待。

唯讀 W34 官方範圍2024-01-01～2026-10-22共679 sessions，資格只 DEVELOPMENT_ONLY，不升格 forward admission；當前T0及2027／2028日曆缺失，**STOP**。TEJ補充380,514列／2,958檔、跨月靜態衝突0；研究矩陣1,952檔／7,423,927列代碼全覆蓋，但snapshot-derived history的known_at、變更值方向、事件終日語義仍UNKNOWN，strict PIT合格0。4-run帳本chain／blobs唯讀驗證，INITIAL_STOCK不當首次公告。逐列產物／manifest只在本worktree `.artifacts/forward/wp1-20261126-release/`；未讀正式Data／分數／標籤／blind，未變更規格、原SHA256SUMS、P4d、runtime／production／scheduler authority。無外部observed_at更新。下一步先獨立review，Human／Claude補合格日曆、lifecycle語義與可知時間，不用工程PASS解除strict STOP。

## Step3 延後評估驅動程式候選 — 2026-11-26

`job-run-mwx2xzc6-056b2e/builder_step3_eval@10684c754cbe10fb7da2476c9b425775481473ef`，no-commit bounded patch，**CPU_ENGINEERING_ONLY / RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED**。Human 本輪明示已接受歷史來源 gate，要求等 72 fits 全部完成後再由 Claude 評估；本 worker 未讀取實際 fits／Human 決定 artifact、未查程序、未做真資料評估，不更新 runtime observation，也不將使用者敘述當成本輪驗收。

新增 [延後入口](../../tools/run_step3_evaluation.py) 沿用 `step3_evaluation.evaluate`，先重驗 bundle／admission／Human 決定與 72 COMPLETE manifest 的完整 request、逐檔 SHA、全部鍵／樣本／凍結標籤，再計兩段配對 IC。矩陣 session 鍵驗證後映射完整凍結日曆，保留缺日位置，封版上限為矩陣與日曆共有末日。檔名避開 `tools/step3*.py`，不改既有 bundle code identity；凍結規格 SHA 不變。選用既有 tools／tests／reports 與私有 `.artifacts/step3/evaluation/`，無新頂層目錄或替代評估語義。

[公開報告模板](../../reports/step3_results_20261126.md) 及同名 JSON 已於 2026-11-26 執行完成（**COMPLETE_HISTORICAL_ONLY**，commit `ca7d25e`）：B−A 每日 h20 IC 增量 2015–2020 +0.005769 [0.003228, 0.008537]、2021–2026 +0.004953 [0.000634, 0.009025]；獨立重算 GO（6 位小數吻合）；2024／2025 年差為負，增益非逐年穩定。執行時須明示 `--consume-placeholder`，只替換完全相符的待執行模板，既有完成輸出或私有 namespace 一律 STOP。私有保存逐日 IC、輸入 SHA、72 job manifests 及輸出 SHA；公開只列彙總。證據等級固定「已曝光、量級與工程驗證、非確認」，不解除 strict 前瞻 STOP、不 supersede 歷史結果／來源／P2／P4d／production authority。合成測試與執行交接見 [工程報告](../../reports/step3_engineering_20261126.md#step3-延後評估驅動程式--2026-11-26)；獨立 review、真資料執行與 Human acceptance 均待辦。

## Step3c 審查修補候選 — 2026-11-26（目前狀態）

**本輪實際完整72-job bundle未完成／不可准入。** 準備及未核驗快取已停止（exit130），精確inventory在本worktree `.artifacts/step3/preparation-status-step3c.json`；partial不當complete。來源待Human決定與完整性STOP分開；本輪不啟動正式72 fits／評估。 最新focused49 passed／9.35秒；全套timeout1500在30分鐘界線中止exit130／約66%／NOT_COMPLETED，無終局總計，不宣稱全套通過。

`job-run-mwwgd60e-d894d4/builder_step3c`，既有 detached `31f89ed95feb182ccdc4669cc4c09f5766e1cd08`，起始 clean，Human no-commit bounded approval；**HUMAN_DECISION_REQUIRED / CPU_ENGINEERING_ONLY / RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED**。本段取代下方Step3b目前歷史PASS主張；舊數值只保留為歷史工程觀測，不撤改歷史artifact。[Step3c報告](../../reports/step3_engineering_20261126.md)附Human兩個決定選項；是否接受歷史來源gate由Human決定，本輪沒有接受紀錄、fits或IC／績效。

[prepare](../../tools/step3_prepare.py)改為完整72-job（2015–2026／17,29,43／A,B）身分；只重用原warmup遮罩，每年度依凍結日曆重算purge。admission強制完整job／target／fit／eval／標籤鍵與全部來源manifest身分，runner待決來源無Human紀錄即拒絕，明示旗標只處理此待決，不解除其他STOP／strict前瞻STOP。resume補prediction必備；評估入口增加凍結labels值／鍵／session逐列核對及infinity拒絕。私有產物只在本worktree `.artifacts/step3/`，實際完成度及最新測試以報告補記為準，code presence不代表完整bundle通過。

唯讀跨來源：380,514列，配對291,847，TEJ×1000對FinMind NTD相對差<0.1%有291,360（99.833132%），>1%有446（0.152820%）；未配對／非有限88,667。配對一致才標NTD_BY_CROSS_SOURCE_AGREEMENT，其他UNIT_UNVERIFIED／NaN。不宣稱原文件授權幣別或首次版本；公司更正與最早「相關資訊」日期對應該數值知悉時間仍未知，合併可比性假設待Human。原規格SHA `9cd7cb8681e3d881193efa3c6fa3c72bd74fba84a96f6a16e6da3a85398eaf34` 未改，無machine／production／P2／P4d authority supersession、無外部observed_at更新。

## Step3b 歷史准入裁定接續 — 2026-11-26

`job-run-mwwebldn-201d27/builder_step3b`，既有 detached HEAD/base `b8c4c8f574ea280057bb87a0a852f8ca0b830ef4`，起始 clean，Human 明確 no-commit bounded approval；**CPU_ENGINEERING_ONLY / RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED**。[工程報告](../../reports/step3_engineering_20261126.md)記錄協調者 Claude A–D 裁定（待獨立審查）：TEJ原始文件或逐期公告擇一，限2013/01起有發布日之首次單月值、NTD千元轉元，跨2012比較NaN且禁用匯出去年／累計欄；raw觀測bytes配既定SHA凍結；完整bundle不符STOP。strict前瞻官方日曆／lifecycle／公司行動STOP保留，歷史只限完整價格研究股票，不升格為可成交／含息／全市場證據。

[串接入口](../../tools/step3_prepare.py)沿用既有tools與私有 `.artifacts/step3/`，將矩陣X／48欄順序／日曆／價格／成員／h20／五欄／共用抽樣／程式／版本證據凍結；[runner](../../tools/step3_runner.py)每次逐檔重驗及核對job陣列。僅本次2026 A/B seed17資源量測授權，不執行或查看任何IC／績效。最終 `resource-release-20261126/bundle.json`（45 identities，SHA `5c734cf4997b5d9cc7b7ba6a6c6436290eacb7ac69955d19d2cdc27fa931c824`）重驗後歷史admission **PASS**、strict前瞻**STOP**；準備165.05秒／RSS3,635,872 KiB／exit0，無OOM。全套1383 passed／11 skipped／5 warnings／862.94秒／exit0；最後內部效能修補另focused33 passed／3.86秒及真實來源列一致性驗證，未宣稱全套程序重載最後函式。2026 A/B seed17均COMPLETE／exit0（各500,000 fit、302,228推論、200棵）：A fit34.8334s／infer0.9111s／RSS1,331,400 KiB，B fit36.0945s／infer0.9898s／RSS1,365,460 KiB；沒有OOM、IC／績效或其他研究fits。最後重驗與限制以報告Step3b節為準；下方Step3a STOP／fits=0保留為當時歷史觀測，不再代表本輪授權邊界。無正式Data／V6/models／P4d／P2／production／scheduler／網路／GPU／額外agents／commit／push／merge／deploy，未刷新外部observed_at；frozen規格SHA不變，沒有machine authority或歷史結果supersession。

## Step3 月營收 h20 工程候選 — 2026-11-26

`job-run-mwwctc1a-f7f0c1/builder_step3a`，既有 detached HEAD/base `224a81439838fb3b569291baaa4f50399249b71a`，起始 clean、依 Human no-commit 指示保留 patch；**CPU_ENGINEERING_ONLY / RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED**。凍結[規格](../../tasks/spec-step3-monthly-revenue-h20-20261126.md) SHA `9cd7cb8681e3d881193efa3c6fa3c72bd74fba84a96f6a16e6da3a85398eaf34` 相符且未修改。

[工程報告](../../reports/step3_engineering_20261126.md)是本輪檢查與未完成項目導航；[准入／特徵](../../tools/step3.py)、[選樣稽核](../../tools/step3_audit.py)、[runner](../../tools/step3_runner.py)、[延後評估模組](../../tools/step3_evaluation.py)沿用既有 tools 邊界，不修改 P4d／P2／P4a 來源宣告。TEJ 原始 HTML 條文與 raw／PIT SHA 已核，發布日意義、單月值不隨次年更新、2013/01 口徑轉換及一月累計起點均找到原文；逐期首次版對應、幣別／合併可比範圍及合格日曆／lifecycle 仍未齊，**准入 STOP，不授予歷史擬合或 strict PIT**。runner 明確鎖住真資料擬合，資源量測 A/B fits=0；評估模組未執行，沒有本案研究績效或分數比較。

選樣按年對回 5,334,161 個成員鍵，W60 4,900,707，沿用 2b 價格遮罩缺價 256,550；停牌／下市／公司行動原因未知，不從缺價推定。逐列產物、擷取原文、版本及 source/input/output hashes 僅在本 worktree gitignored `.artifacts/step3/`，必須保存後另案審查／退役。測試終局與限制以報告為準；builder 自查不是獨立驗收。Next：獨立審查、補原件與可比範圍／日曆／生命週期、凍結完整 identity bundle；另案才可解除執行鎖與量測2026最大折。無 blind／網路／GPU／正式 Data 寫入／production／scheduler／排程／agents／commit／push／merge／deploy；未刷新外部 runtime observed_at。

## TEJ 第二步 2b 描述診斷候選 — 2026-11-25

最終診斷430.75秒exit0；月營收等權h20早期0.0263／後期0.0152，季報0.0347／0.0263；四條延遲曲線未觸描述性驟降警旗，不證明PIT。h20有效期實際到2026-08-21。描述性建議值得規劃固定月營收h20第三步預先登記，未啟動實驗。最新focused26 passed／兩項新mutation killed；較早collection全套1347 passed／11 skipped／5 warnings／898.37秒exit0，修正後未再完整全套，限制見報告。五份正式parquet SHA均相符，私有manifest SHA `ec10a1d6781db6fa4e3f165d8e2301d86d1dce963598f186522fe74bffd65b13`；partial-before-coverage-fix不是完成證據。

`job-run-mwwag5jb-fb775c/builder_tej_2b`，既有 detached base `099e17c6ee67ad5dfedc42a2b3045fa91557e26c`、起始 clean；本輪 no-commit patch，**DESCRIPTIVE_ONLY / RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED**。[程式](../../tools/tej_diag.py)與[合成測試](../../tests/test_tej_diag.py)沿用 tools／tests；[報告](../../reports/tej_diag_20261125.md)與[JSON](../../reports/tej_diag_20261125.json)是彙總，所有逐列特徵／分數只在本 worktree `.artifacts/tej/diag/`。來源唯讀主 checkout TEJ 雙表及事前 frozen raw-price snapshot，逐份 SHA、source／config／support／輸出身分見 JSON；C0+ 逐日 date／stock 成員鍵、2021 後歷史分數僅描述，不用 X/y、不重訓。

20 日主標籤在診斷前已鎖定，5／60 日僅描述；發布 D 後第一個 session 收盤才可形成訊號，age=0 不混入 1–5 桶。完整四控制 PIT 殘差 IC **STOP：無合格 PIT log 市值**，僅另報三價量控制及 C0+ 殘差對照。月營收範圍欄全缺、累計比較值版本未核；IFRS 排除長尾／非曆年／未知起迄／非 Q／現況金融與分類未知，值版本（含 EPS）仍 UNKNOWN。raw 價無股利、日曆為 observed union、研究成員不是 tradability／lifecycle authority；不升格 source admission 或嚴格 PIT。最終數值、第三步描述性判斷、驗證與未完成範圍以報告為準，builder 自查不等於獨立審查或 Human acceptance。

不讀 blind、不改正式 Data／V6/models／實際帳本／production／scheduler／P4d，不用網路／GPU／額外 agents，不 commit／push／merge／deploy；不更新外部 observed_at。私有產物與 patch 必須先保存、驗 hash／references，才可另案審查 worktree 退役。

## TEJ 第二步 2a 資料層候選 — 2026-11-25

`job-run-mww8tui1-604e40/builder_tej_2a`，detached base `aafd0d9c41ac2a42a7277936353a101f93aab95c`、起始 clean；Human 核准有界第二步，依 no-commit 保留 patch，**RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED**。[loader](../../tools/tej_pit.py)與[合成測試](../../tests/test_tej_pit.py)只做 TEJ 資料層，沿用既有tools離線資料工具邊界，避免納入P4a訓練打包；不改既有gate。[報告](../../reports/tej_pit_audit_20261125.md)／[JSON](../../reports/tej_pit_audit_20261125.json)記 source／input／輸出 hash 與實際統計；raw zip 和所有逐列衍生資料（含代碼日期示例）僅在 gitignored `.artifacts/tej/`，不入 PUBLIC git。

最終 `.artifacts/tej/pit/` 生成月營收380,514列、IFRS115,876列；manifest SHA `f3532527b51ed05ad2b0360aad0c14de0531d18f230cf00c7ef456d79fa254f1`。缺發布日12,982／4,237列不倒填，所有 strict_pit_eligible=False。IFRS以名目年月當期末的14列負延遲（min −82日）均為非曆年期間，按實際財報迄日改算為0列；>365日3,298列全是曆年期間，成因／版本仍UNKNOWN，不升格PIT。月營收配對291,847列、>1%差446；Revenue／EPS差異不識別首次或重編值。C0+覆蓋只讀獨立date／stock身分陣列，以月內研究成員聯集為分母，非lifecycle authority；2026/10分母UNKNOWN。

合成focused14 passed（0.94秒）、兩項in-memory mutation killed；P4a source-compat位置回歸1 passed（11.35秒）。隔離後全套因本輪30分鐘時間界線中止（exit130），collection1351項但無終局總計，NOT_COMPLETED／不宣稱全套通過；須續段完整重跑報告所列隔離命令。重要限制：早期未隔離全套測試已執行到既有外部真實fold讀取區段，不能宣稱本輪全程未讀真實標籤；發現後中止，最終以bwrap遮蔽外部資料後重跑，不改既有測試或skip條件。未把該值用於本案稽核或研究判定；偏差詳報告，待Human／獨立review判斷。無blind、實際帳本寫入、正式Data寫入、模型權重、網路、GPU、production、scheduler、P4d、agents、commit／push／merge；不更新外部observed_at。Next：核授權原件與IFRS版本語義、長尾／差異、精確春節順延及獨立驗收；不直接加回特徵或改authority。

補充唯讀21筆EPS REVISED：16筆TEJ僅符新值、4筆新舊皆符、1筆皆不符；全部科目名稱同時改變，不能推定公司重編或整表版本。帳本SHA前後一致，未寫帳本。

## PIT 帳本複審修補候選 — 2026-11-25

`job-run-mww7h6b0-bba9f3/builder_pit_fix2`、detached base `04e724dd09ef0d44974dfae9e235e777937e6784`，起始 clean，no-commit patch；**RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED**。[工具](../../tools/pit_ledger.py)精確比對完整 sqlite_master 名稱／定義集合（含自動索引），拒絕 sqliteXdrop 等額外物件。舊 schema v2 在續寫前稽核不可變索引業務鍵，None／NaN／Unicode 空白均 STOP；合成百萬鍵估計全量約38–39秒，因此成功續寫才同交易升至 schema v3、在封存 run 記索引 SHA 稽核標記，後續仍逐次驗 blob SHA，避免重掃。失敗 rollback 保留 v2；schema v1／v3 缺標記 STOP。

focused56 passed；指定全套1333 passed／4 skipped／5 warnings／0 failed（749.61秒、exit0）；量測見[報告](../../reports/pit_arrival_audit_20261125.md#複審-findings-修補--2026-11-25)。symlink 檢查後至寫入間競態未修，依 Human 單機意外損壞／程式錯誤威脅模型接受為已知限制，不宣稱安全寫入邊界。每日排程 NO_GO 保留；無正式 Data／實際帳本／GPU／網路／agents／P4d／production／排程／commit／push／merge操作。Next：獨立複審與 Human 另案驗收；報告第4步含主 checkout 的精確 validate／replay／NO_CHANGE runs+1 events+0 命令。本輪未執行該實際帳本程序，完成不是接受。

## PIT 帳本獨立審查修補候選 — 2026-11-25

`job-run-mww6g0bv-340962/builder_pit_fix1`、detached base `47bf4659e1143269b622bfd80ebed687f3538973`，起始clean、依no-commit保留patch。**RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED；每日排程 NO_GO 待複審／現場驗收**。[工具](../../tools/pit_ledger.py)在標準化後拒絕缺失／NaN／空白業務鍵，驗必要schema及四個append-only trigger的正規化SQL SHA；寫入限明確帳本namespace，拒絕root／祖先／既有後代symlink與Data／來源重疊。parquet索引暫存留帳本內，SQLite temp_store=MEMORY；真實全量新RAM峰值未量測。

RED21 failed／18 passed，最終focused43 passed／17.47秒、兩項新mutation killed。tmp_path真實SQLite／journal寫入中SIGKILL與RLIMIT_FSIZE拒寫後，重開validate及四表重播皆為原完整1 run；重試CHANGED、再跑NO_CHANGE，不重複events。僅本機合成檔案工程證據，不是實體裝置斷電／Windows／publication PIT驗收；原實際帳本與正式Data完全未寫入。指定全套1320 passed／4 skipped／5 warnings／0 failed（709.05秒、exit0）；完整輸出見[報告](../../reports/pit_arrival_audit_20261125.md#獨立審查-required-修補--2026-11-25)。無網路／GPU／agents／抓取或P4d／production／排程／commit／push／merge變更，不刷新外部observed_at、不改machine authority。Next：獨立複審，Human另案依報告先停用註冊／手動action驗證後才決定啟用；完成不等於接受。

## 最小 PIT 帳本容量修補候選 — 2026-11-25

Human 核准[計畫 §J](../../tasks/plan-fundamentals-supplychain-20261125.md)第一步及本輪 REQUIRED；`job-run-mww5b81x-8af923/builder_pit_step1b`，detached base `fb32e4c1a99fafb05abf80fd7b041b08859f5f24`，由前輪複製9檔接續，no-commit patch，**RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED**。[工具](../../tools/pit_ledger.py)改為不可變 SHA-256 來源 parquet＋zstd 每鍵列hash索引；SQLite只存後續diff與run chain，來源SHA不變記NO_CHANGE、不解碼來源列。完整性／重播與實際容量／測試以[報告](../../reports/pit_arrival_audit_20261125.md)及[JSON](../../reports/pit_arrival_audit_20261125.json)為準。

真實初始化13,760,451列／556,400,149 bytes、614.602秒；NO_CHANGE1.748秒、增11,913 bytes、events仍0。focused15 passed／2 mutations killed；指定全套1292 passed／4 skipped／5 warnings／0 failed、944.62秒、exit0，完整輸出見報告。新版持久帳本 `/home/frank/projects/MarketMamba/.artifacts/pit-ledger-v2/`，EXTERNAL_LARGE_DATA，獨立初始觀測、不移植舊時間；舊worktree v1約9.8GB原樣保留由Claude處理，禁止隨worktree清理丟失。INITIAL_STOCK不是首次公告、不倒填歷史PIT。create_time中位+11／大量+42日很可能為供應商建檔／回補，不能作歷史遲交上限；TEJ發布日等其他來源待Human確認授權，法律遲報比例null。

不讀blind／報酬、不改正式Data／抓取／P4d／production／scheduler；排程只文件指令，指向新持久路徑、抓取成功完成後獨立執行、失敗不回傳upstream。未連網／GPU／agents／commit／push／merge。未刷新外部runtime observed_at，不升格strict PIT／admission／canonical artifact。Next：Claude獨立複審與Human接受；完成不是Human acceptance，現場接線與備份另案。

## P4d 排練 heartbeat 與本機資源量測修補候選 — 2026-11-25

`job-run-mwvqcr5w-e515cb/builder_p4d_fix1`，既有 detached HEAD/base `d36e709a21088ce836c6d84e2964518c018cc12e`、起始 clean；依核准範圍保留 no-commit patch。**UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED / CPU_ENGINEERING_ONLY**。獨立審查原 NO_GO 的 REQUIRED 為排練未傳 Progress；本候選以 `tools.p4d_bundle.launch` 在 binding／排練前啟動同一發布器，穿過預處理、六條排練 fit、GRU 全列診斷／推論、C0+ 特徵與擬合至正式模式。每約60秒更新 `progress/formal/PROGRESS.txt`／`progress.json`，明標 REHEARSAL 與 stage，寫入失敗繼續；不改 P4a 共用來源、正式帳本／checkpoint／研究契約或成功 gate。

本機 `python -m tools.p4d_train --measure-c0plus --origin 2026 --data-dir <prepared>` 僅量測完整2026 outer fit 的185欄特徵＋CPU LightGBM200輪，不抽樣、不推論、不選研究結果；略過C0+不用的rank-gauss。新程序輸出 `.artifacts/p4d/measure-<uuid>/measurement.json`，記錄 process-lifetime 峰值RSS、耗時、矩陣bytes及source／contract／input／split／環境身分。**本輪未執行真實完整量測**；36列合成邊界替身只證明入口／輸出schema可用。

[工程報告](../../reports/v7_p4d_engineering.md#排練-heartbeat-required-與最大-fit-量測入口--2026-11-25)列RED→GREEN、focused31 passed、兩項in-memory mutations、完整回歸1275 passed／4 skipped／5 warnings／0 failed（653.96s、exit0）及限制。排練／正式共用publisher與寫入失敗以CPU替身驗證；真實2026峰值RSS、T4／Colab／Drive持久性與長操作heartbeat、獨立複審及Human acceptance仍待。原NO_GO不因builder完成解除，不更新外部observed_at／production authority；Next：獨立複審，再由Claude另案在本機新程序量測2026最大fit。

## P4d stage1b 接續工程候選 — 2026-11-25

`job-run-mwvoxf7h-970cae/builder_p4d_stage1b`，指派 worktree／detached HEAD `b1871a9`（前輪WIP），起始clean，本輪no-commit patch；**STAGE1_FROZEN / E2 RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED**。指定初始全套1265 passed／4 skipped／0 failed；修後完整回歸1269 passed／4 skipped／5 warnings、617.86s、exit0，P4a來源相容未壞；保留tools精確P4d邊界，不搬research_lab、不改舊policy／宣告。補空排練證據STOP、綁identity的持久化排練產物／成功標記與重連重驗，下載包含帳本／進度／排練及bundle manifest；研究語義／threshold不變。

prepared完整anchor／2021 split核對後，只跑65股／650-fit列的CPU支援替身六fit與描述IC；五項具名in-memory mutation全killed、notebook五個code cells compile與解包import／下載格驗證通過。最終回歸、source／產物hash與限制見[工程報告](../../reports/v7_p4d_engineering.md#第一段接續-stage1b--2026-11-25)。CPU不是現場gate，CUDA顯存／耗時、全列C0+RAM、全量推論、Drive持久性、排練長等待heartbeat、獨立審查與Human acceptance仍待；無真研究fit或外部runtime變更。Next：獨立session審查本patch，再由Human決定另案現場排練／交付；本輪不正式交付、不commit／push／merge。

## P4d 第一段分段凍結與CPU工程候選 — 2026-11-25

`job-run-mwvnttp4-a9ebe0/builder_p4d_stage1`，既有detached base `35cd9b88ddec3bce3849ba0363c0514393d16d47`；**STAGE1_FROZEN / E2 RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED**。Human對話核准只涵蓋baseline、GRU B（固定八seed前1/2/4/8）與A六格；原規則SHA `c36aed96787e2b7e69fc92cd736ef3692c5542f1aecf350c4e0047470790fe54`，原全文bytes保留且只附紀錄；其他段落仍DRAFT。下方歷史「P4d仍DRAFT」不代表本次第一段仍未核准。

[工程報告](../../reports/v7_p4d_engineering.md)、[獨立契約](../../tasks/p4d-stage1-contract-20261125.json)、[prepared切分盤點](../../tasks/p4d-stage1-prepared-splits-20261125.json)與[新入口](../../tools/p4d.py)：140 fits＝126 GRU＋14 C0+，新佇列SHA `32506a22ecc1154096ca5c616ef0f33dc565d1f14a45a412cea5c8f620d8892f`，帳本 `budget_p4d_stage1`。2021前anchor126sessions、嚴格20-session不等式、共同finite labels／W60、完整交易年等值池、macro36…47與support/cache隔離；僅anchor baseline選更新、outer六折fresh固定更新。P4a程式／來源相容宣告／64-fit budget及歷史結果不改。使用既有tools容器，避免research_lab全掃描把新來源納入舊P4a；本機評估器不入包。

唯讀prepared七個origins切分全通過：anchor fit3,935,478列／3,528sessions；val185,595列／126sessions（2020-06-04～2020-12-03）。只執行CPU合成邊界／單步路徑與metadata盤點，未訓練研究模型，沒有新績效。T4約19.01h為baseline等效估計，容量、完整train IC／全列C0+ RAM、現場T4／Drive、price snapshot與獨立審查仍未驗。Holm2含未核准Mamba2，第一段不縮成Holm1、不宣告完整核心家族勝出。

Next：獨立審查候選patch；通過後由Human自行產生正式交付，現場排練失敗即STOP。依no-commit保留指定dirty/untracked；無GPU／Colab／Drive／blind／正式Data／models／production／scheduler／額外agents／commit／push／merge。完整實測輸出與限制以工程報告為準，不更新外部runtime observed_at。

## P4b 對話凍結與 CPU 實作候選 — 2026-11-24

## P4b 本機專用來源邊界修補候選 — 2026-11-24

`job-run-mwuv96vj-55da06/builder_bundle_exclude`，既有 detached HEAD/base `8dc5cc961f520272cd94281b8e104347cdecd034`、起始 clean；**UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED / CPU_ENGINEERING_ONLY**。本輪沿用 [bundle](../../V6/research_lab/p4_bundle.py)／[相容工具](../../V6/research_lab/p4_source_compat.py)／[測試](../../V6/research_lab/tests/test_p4_source_compat.py)，用精確路徑＋固定 policy SHA 的本機專用契約清單排除 `p4b_blend.py`；collector 共用、宣告明列／採用端精確驗證，archive／extract／verify 拒絕注入，靜態／字面值動態 imports 拒絕。必要來源清單擴大、漏報新來源或缺少必要 to source 均 fail closed；不是放寬差異 gate。

[工具重產宣告](../../tasks/p4a-source-compat-20e6b54.json) from=4e56e9d、to.commit=8dc5cc9（generation-time HEAD；hashes 綁 working bytes），230 個 to sources；六個歷史差異完整列出，bundle／source_compat 兩檔增加「本機專用來源排除」，不把這次改動冒稱純文件。matrix identity 沿用版本化宣告，本輪未重新讀取正式矩陣認證。36-fit 全數 resume_fit／不重訓、64-fit queue SHA `4dc90ec0ada6c6d34b43c429d359424c502634e502c6856c6774bf138de7ba23`、舊 notebook 保底、P3 凍結檔與 P4b 行為維持；合成證據不等於現場採用驗收。

原 28 failed log 保留；本輪最終實測與 SHA 見 [P4b 工程報告](../../reports/v7_p4b_engineering.md)「本機評估模組與 Colab bundle 來源邊界修補」及後續補錄，未以早期 focused 結果宣稱全套通過。工程檢查完成仍不解除既有研究／現場 NO_GO 或 Human acceptance；全量 P4b 評估／price-volume snapshot／獨立複審仍待。無 GPU／資料取得／正式 Data／V6/models／Drive／Colab／production／scheduler／排程／額外agents／commit／push／merge／deploy／wakeup；外部 runtime observed_at 未更新。Next：獨立複審 patch、生成宣告及證據，再由 Human 決定接受及另案交付。


Human對話已核准P4b採預設值，原規格SHA `8d4f194730c2a147b42afe69750f722fccab1d0111e721d2a1f279df369f53c2`；不是數位簽章，僅P4b FROZEN，前瞻／P4d仍DRAFT。`job-run-mwuunk1u-38d24c/builder_p4b`、detached base bc549b8、起始clean；**CPU_ENGINEERING_ONLY / E2 RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED**。

建立前盤點後沿用research_lab／tests／reports，新增 [P4b adapter](../../V6/research_lab/p4b_blend.py)、[測試](../../V6/research_lab/tests/test_p4b.py)與[工程報告](../../reports/v7_p4b_engineering.md)，不改原64-fit gate、rolling_seq_scores或portfolio_eval。固定36-fit子集唯讀resume_fit／BEST/hash/ledger/source-upgrade驗證，三個固定混合先seed rank→臂rank→blend、B2外層rank；D由凍結P1 date/split hashes產生，完整NaN及95%／≥50 gate。統計／候選及持股診斷實作仍待全量真實驗收。

實際本機排練36 fits＋18 P1原seed檔全驗、ledger769 events；僅2021混合378,547列／244日期、四臂976個每日gate，保留率最小1.0、共同股票數最小1493。最終排練manifest SHA `78090080bcbc3c4d4a2e5564a130fe3ee0a44310658ebdb52fbda4e5825122f4`，輸出在指派worktree `.artifacts/p4b/rehearsal-2021-final/`，未跑真實組合／統計。先前29項focused及9項mutation通過，完整回歸與identity詳報告。已曝光、C0+三seed調參與神經兩seed固定配方不對稱；raw/PIT/universe/下市限制未解除，無研究成功／可交易宣稱。

Knowledge-impact：P4b凍結狀態／候選入口／blocker及next decision=YES，已同步Research／Authority Map／Worktree Register；production、scheduler、active外部runtime、既有machine thresholds／64-fit authority、deprecation=NO。未更新外部runtime observed_at。Next：獨立複審、保存候選patch及產物，先凍結price/market_value/volume snapshot再由Human另行執行全量評估與判讀；本次未提供價格snapshot、未全六折真實評估、未Human acceptance。no-commit故保留dirty，非clean完成宣告。

最終補錄：adapter `233f08fb7462bf3c5375772b34fa0f7f20bc85a4c44edb99a93f1424bdd5c399`；工程identity `21b33e7a4a7bca49ca375e07bba8f570f7f71d822dfbf819e4956a85c5c931d3`；最終2021排練manifest `0dcd0477c5318edfb764bf646585d08fd132d5f3616231393e045cc54ace35e4`（`.artifacts/p4b/rehearsal-2021-verified/`），8份分數hash與前次相同。focused **30 passed／9 mutants killed**；全套 **1153 passed／28 failed／6 skipped**（384.86s），28項均P4a source-compat gate拒絕未獨立審查的新增P4b模組，已單測重現。**整合驗收STOP**；不放寬gate或偽稱全套通過。需獨立審查數值中立／來源打包範圍，及price-volume snapshot／Human另行全量評估；未變production／64-fit authority。

最終 CPU 回歸（跨日至2026-11-25）：**1204 passed／7 skipped／4 warnings／0 failed，608.24s，exit0**，1211項完整collection；focused最終12 passed／42 deselected，3項assertion-killed mutations、236成員bytes/hash隔離與實際未宣告新增檔拒絕通過。原來源inventory整合阻擋在本候選CPU工程層閉合；原歷史失敗log保留，獨立複審／現場NO_GO／Human acceptance不因此解除。skip明細及完整輸出見工程報告，未新增skip或刪測試。

## 三份DRAFT複審修補候選 — 2026-11-24

`job-run-mwuu88xi-da927f/spec_fixer2`，既有detached HEAD/base `abc12f50a8246bbece3f187636f697c9a536a235`，起始clean；本次僅文件patch，**DRAFT / E2 RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED**。前瞻P2將公開全宇宙觀測比例與逐必要欄適用集合100% gate分開，合法停牌不冒充當日OHLCV、停牌／估值證據對持有及未持有皆100%；補成對fixture與固定來源錯誤處置，修正入口為`portfolio.run_backtest`。P4d模組表B同步為訓練隨機性與集成，保留batch／更新長度選擇混淆。未實作adapter／摘要runner或訓練，不宣稱修補已通過獨立複審或Human acceptance。

P4a既有失敗判定為**文件段落結構錯置**：Current State及Authority Map的第13次修補標題之後插入P2段落，原`job-run-mwtn0pir-48e95f`正文被歸入P2。只把原正文移回原P4a標題，不補造64-fit驗收／現場狀態，測試不改。已重現原focused結果1 failed／2 passed／110 deselected；搬回後3 passed／110 deselected。

建立前盤點：tasks三份規格已有各自Human決定清單，Current State可承載跨規格的一次凍結導航，故沿用此處彙整表，不新增freeze-checklist檔案、目錄或memory樹。表格是待Human簽署的決策索引，完整語義仍以各DRAFT正文為準，不能提升為machine authority。既定決定只核對，不重新當未決問題；任一替代須在新結果前完整改稿／重算影響，不並跑挑優。

驗證紀錄（本次CPU自查，非獨立複審）：`PYTHONPATH=V6 CUDA_VISIBLE_DEVICES='' /home/frank/.venvs/marketmamba-gpu/bin/python -m pytest -q V6/research_lab/tests/test_p4.py -k 'knowledge_preserves_candidate_authority_status or draft_documents or operations_require'`為3 passed／110 deselected；`python3 -m unittest discover -s tools/project_knowledge -p 'test_*.py'`為23 tests OK。實際import `research_lab.portfolio.run_backtest`成功，簽名(prices,predictions,config=Config(...))；全文無錯誤backtest入口，P4d「初始化方差」剩2處均明示不可歸因。P4a原正文與HEAD逐字相同，僅移回正確標題。合成算術：公開99/100=99%但必要99/99=100%為PASS；公開98/100=98%、必要98/99=98.9898989899%為FAIL；成對公開投影相同是文件預期，未執行尚不存在的摘要runner。7份Markdown本機引用688個缺失0、41個表格欄數一致、DRAFT標示3/3；`git diff --check`通過。impact為WARNING，已人工核對Operations／GOVERNANCE，無操作子系統／治理規則變更；health `--ci --as-of 2026-11-24` exit0／WARNING（Governance／Authority Drift／Legacy Docs），無structural FAIL。環境WSL CPU，系統python3及既有venv均3.12.3，MARKETMAMBA_DATA_ROOT unset，Data realpath為本worktree普通目錄；不讀其內容、不更新runtime observed_at。7份指定文件unstaged、無staged/untracked；依Human no-commit保留patch，非clean-worktree宣告。

## 三份DRAFT Human凍結決定彙整

**前瞻：仍待Human凍結；P4d：第一段STAGE1_FROZEN，其餘DRAFT（詳本表分段列）；P4b：FROZEN（2026-11-24 對話核准採預設值，無數位簽章）。不是額外執行／費用授權。** 請逐列記「採預設／採替代＋完整修訂／不執行」，並綁定最終文件及輸入hash、批准者、UTC／台北時間；不得用本表勾選代替正文所需證據。三案可分別凍結，但跨案衝突須在各自第一筆新結果之前處理。

| 案／決定 | 建議凍結內容（正文位置） | 重大替代與代價／尚待證據 |
|---|---|---|
| 全案身分與邊界 | 三份DRAFT完整版本／hash、source commit＋dirty bytes、環境、輸入manifest、獨立審查與後續執行分案；P4b/P4d全為已曝光描述性 | 不執行任一案；文件GO／凍結不等於Human acceptance，不改P4a／production authority |
| 前瞻端點F1 | [前瞻P2](../../tasks/spec-forward-p2-primary-20261124.md)「252日估值與收益契約」A：s0現金、s1 open首建含費、s1…s252 close估值、無終止交易，策略／基準同時鐘及原成本不對稱 | B末日open清倉／C末日close清倉須先完整改收益公式、終止費用、停牌及事件規則與fixture，不能由adapter現場決定 |
| 前瞻公開schema F2 | `forward-public-summary-v1`封閉欄位、source slots／枚舉、全宇宙expected／valid觀測比例、逐欄R_c,d必要gate100%、合法停牌適用表、S/M成對fixture；次日12:00固定發布、256-bit私有nonce及hash chain | 全部私有至期末可減公開資訊風險但失去每日公開連續性監督；無nonce digest不可採。來源分類不得依持股、公開比例99%不得單獨判失效 |
| 前瞻來源／事故 | 合格calendar、來源與欄位必要性、publication／as-of／provenance、lifecycle／停牌／下市／估值、預測資格及keys100%、0漏日、固定錯誤表／SOURCE多碼 | 證據不足即不啟動／失效；不能放寬門檻、假OHLCV、依持股移分母。正式來源及事件證據尚未驗收 |
| 前瞻兩階段／重訓N | 已決定嚴格T0僅試運行、年報／Q2兩錨點、T1正式起算；待凍結N_A=10／N_H=8、A+2-session資料緩衝、18:00、R+1/R+2同快照重試與超時STOP | 8/8、10/10或更久依非績效來源延遲／CPU實測選；1–2太緊。公司期限例外不自動漂移整體時鐘，未知不得啟動 |
| 前瞻正式日曆／截止 | 合格2027／2028calendar及假日／延期例外，實際T0/T1、R、origin−21、252收益終點、末日+11安全成熟及每版訊號／再平衡暴露；至少一次成功更新及更新後既定再平衡 | 日期未驗維持UNKNOWN；20-session連續流程、T1前20／5-session截止、跨Q2未啟動STOP均待簽署。依工程gate延後T1需重訓，開始後不重設／補日 |
| 前瞻揭曉／隔離 | 階段0到正式唯一端點才描述性揭曉；Human為保管／重訓owner，凍結head外部保管、提醒管道、淨化與事故／手動開盲演練 | T1規格簽署且已啟動後揭曉階段0，或永不揭曉；均不得績效調規。可另選OS使用者／ACL，現有同權限紀律不能稱OS隔離 |
| 前瞻統計／成本／影子臂 | 已決定252單端點、H-F1／Holm1、雙邊p、raw主要與比例成本基準；C0+三seed17/29/43、185欄固定；固定／神經影子臂預設不收集；WP1～WP6先驗收 | total_return、單邊p、新影子臂只能在結果前另補完整契約／成本及身分，不中途替換；CPU/RAM/disk預算仍待實測及另案授權 |
| P4b凍結身分 | Human 於 2026-11-24 以對話核准採預設值凍結 P4b；凍結身分＝本規格檔案 SHA-256 8d4f194730c2a147b42afe69750f722fccab1d0111e721d2a1f279df369f53c2＋ 所有輸入與程式 identity（實作完成後補入）；此凍結僅限 P4b，不涵蓋前瞻與 P4d；不是執行授權之外的任何授權（Human 已同意以本機 CPU 實作與描述性評估） | 無數位簽章；確切時間 UNKNOWN；工程身分見 [P4b報告](../../reports/v7_p4b_engineering.md)，不得改凍結規則 |
| P4b公式（FROZEN／對話核准） | [P4b](../../tasks/spec-p4b-blend-20261124.md)「凍結參數」三個0.5/0.5，先seed rank→臂rank→blend，B2保留R_all外層rank，不搜尋權重 | 無外層rank僅能結果前改為單一版本；其他權重需新設計，不兩版擇優 |
| P4b工程身分補錄 | adapter SHA `064e8b8404bc0aefbb60598471c58ac78a71e952be203c5dff8126dfd10b5823`；269檔code／契約identity manifest SHA `3a9b6d8bd48d9218927d53d2d82aed6c507b792af71a4d1774c5e4554f16c276`；實讀輸入及2021排練manifest SHA `78090080bcbc3c4d4a2e5564a130fe3ee0a44310658ebdb52fbda4e5825122f4` | 僅CPU_ENGINEERING_ONLY；price/market_value/volume snapshot未提供／未驗證，不能全量評估；無Human acceptance |
| P4b支援／輸入 | 36-fit固定subset獨立准入且不降64-fit gate；P1原seed scores、共同S先裁後rank；事前D＋日期聯集、每日95%與≥50、NaN不刪組合列；固定snapshot／volume與完整Config identity | 只看交集／日期聯集均不足；缺fit／日／折即STOP或不具提名資格。本機匯出、逐fit lineage與來源驗證待另案，不讀Drive |
| P4b統計／候選／限制 | Holm3只2021–2023，block20／1000／seed17雙邊；全六折揭露；兩折CI正／三折mean正／近期方向及完整診斷，優先序B1→B2→B3；E2及不對稱限制 | 無候選就停止；多臂前瞻需新家族／alpha，不加入已開始P2。CPU subset loader／評估資源與授權另案，不重訓 |
| P4d分段凍結（STAGE1_FROZEN） | Human 2026-11-24對話核准預設值，無數位簽章、確切時刻UNKNOWN；原全文規則SHA `c36aed96787e2b7e69fc92cd736ef3692c5542f1aecf350c4e0047470790fe54`；只含baseline、新C0+參考、GRU B前1/2/4/8與GRU A六格；原規則bytes不改，只附凍結紀錄 | Mamba2全部、超六格容量、D、N12核心搜尋仍DRAFT待第一段結果後另決；工程完成非Human acceptance，無GPU／費用／正式交付授權 |
| P4d macro schema凍結名單 | [P4d](../../tasks/spec-p4d-equal-tuning-20261125.md)「預處理隔離契約」神經歸零36…47：TWII_Return、SPX_Return、VIX、TNX、Gold_Return、Oil_Return、USD_TWD、Futures_OI_Foreign、Options_PC_Ratio、Fear_Greed、Business_Signal、FED_Rate；綁48欄order SHA及cache／fit support identity | 替代僅用合格tuning fit完整截面估mask後全案固定；不可全矩陣抽樣／按fold或fraction重估。future-mutation／cache hit/rebuild／新程序驗收尚未實作；C0+不刪macro |
| P4d核心搜尋／選擇（DRAFT，未核准） | §2–5三臂C0+/GRU/Mamba2各N12×seed17/29、2021前126-session全列validation／purge／maturity、先生成random表RNG20261125、IC5/BEST及tie規則、固定更新六折fresh fit、§3完整搜尋空間與新dropout／loss接線 | N6省費／N24較廣；等牆鐘另案且硬體不等價；S-Mamba預設不納，若納須同預算Holm3與重算；不Optuna，不按outer／pilot績效改N |
| P4d統計／完整性 | GRU−C0+及Mamba−C0+，centered雙邊公式、Holm2／seed20261125、2021–2023 pool，逐日同列、共同成熟IC、組合保留NaN、六折完整揭露 | 單邊需另案公式與驗收，不能除二；所有歷史結果均描述性，少fit不可稱equal_budget_complete |
| P4d A／B／D開關與配方 | §6各模組可獨立開關；預設P4a固定配方、baseline選BEST更新數後各格固定更新，七origins共用合格baseline去重，本預設不重用P4a fits | 調參後配方混合搜尋／容量效應須先改設計，不兩版擇優；B只描述訓練隨機性與集成，保留batch／更新長度選擇混淆 |
| P4d A容量／B種子／D資料 | A六格d32/64/128/256×2層＋d64×1/4層，headdim8/state16/expand2；B固定17/29/43/59/71/89/101/113之前1/2/4/8；D可關閉或25/50/100%最近完整年等值全列 | A12格可估交互但336fits且昂貴角落未測；改heads／state／expand會加混淆；D不是隨機抽樣。B無挑最佳seed，baseline新seed更新次數交替對應須一併凍結 |
| P4d預算／順序／停續 | §4／6先baseline、GRU B/A→Mamba B→按排練資源決定原案Mamba A／D；pilot前4候選計入固定N；全模組＋N12共380神經fits約115.74 T4h基準等效，173.6–312.7 CU暫估非上限 | Human須定CPU/RAM/disk及分階段費用範圍、當次CU率與最大shape排練條件；超資源停並報partial，不按勝負擴格。CUDA／Colab／Drive現場及大容量倍率未知，現在不執行 |

Knowledge-impact：DRAFT coverage判定／next decision細化與文件結構修復=YES，已同步Research與Authority Map；已生效machine authority、subsystem、active runtime、既有correctness thresholds、canonical artifact新增／supersession、deprecation=NO。現有worktree責任記Worktree Register；無外部runtime觀測，不更新observed_at。Next：獨立複審此patch，再由Human依本表凍結；正式calendar／來源／事件、工程實作、未來mutation／推導測試與現場驗收仍為blockers。no-commit保留指定文件unstaged，不能宣稱clean-worktree。

## 三份DRAFT審查修訂候選 — 2026-11-24

`job-run-mwutomlv-e3974c/spec_fixer`，既有detached HEAD/base `3ed6876`，起始clean、6份Markdown未提交patch；**DRAFT / E2 RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED**。[前瞻P2](../../tasks/spec-forward-p2-primary-20261124.md)補F1完整252收益契約：建議T1純現金→次日open首建含費、s1…s252 close估值、無終止清倉費、停牌／未知下市fail closed及同基準時鐘；與舊engine末entry open清倉不同，WP6只驗已選語義。F2封閉公開來源／全宇宙摘要schema、分母、持股診斷禁公開、私有256-bit nonce承諾、固定錯誤處置及與公開市場資訊結合推導驗收。[P4b](../../tasks/spec-p4b-blend-20261124.md)補B1事前日期域與聯集核對、整日／整折消失不得被交集隱藏。[P4d](../../tasks/spec-p4d-equal-tuning-20261125.md)補D1明確36…47欄macro schema／fit support cache與未來特徵mutation；D2雙邊centered公式＋Holm2；D3改稱訓練隨機性與集成的描述性反應，anchor選更新數有選擇偏向。

唯讀CPU實證：prepared及pilot09六折共12組，X實算SHA `9778392d4c337d1f9cb5768002340a0d1aec13f7689b2f601b77bd2329b55363`；整矩陣60日、train60日、2021前60日、train逐日及2021前逐日的mask全同36…47。P1／P3六折X／split identity均吻合，P4a限本機prepared；**已完成結果不受此問題影響（僅限該mask及所列identity）**，不升格P3完整性／P4a逐fit驗收或PIT。輸入hash、方法與逐折數值在P4d附註，不依賴臨時輸出。

Next：獨立再審後Human選定端點A（收盤估值；替代開盤／收盤清倉）、摘要／nonce／處置表及macro schema（替代僅fit估計後凍結），並決定其餘既有DRAFT選項；文件修訂不解除原NO_GO或代替Human凍結。正式calendar／來源／事件、實作、future-mutation／推導測試及現場驗收仍blockers。只文件與指定唯讀實證，無正式Data／blind／models／GPU／Colab／Drive／production／scheduler／排程／額外agents／commit／push／merge；不更新外部runtime observed_at，不改machine authority或歷史結果。遵照no-commit保留6份unstaged，非clean-worktree完成宣告。

本候選實際驗證：knowledge unittest 23 tests OK；shared significance 8 passed；既有DRAFT exposure文件測試1 passed／112 deselected。合成符號反轉雙邊p均0.3806193806193806、Holm2手算[.01,.04]→[.02,.04]；無假日s0／s1／s20／s21／s240／s241／s251／s252／s253／s262／s263邊界重算符合規格，非正式交易所日曆。6份Markdown共659本機引用無缺失、36表格欄數一致；git diff --check通過。impact WARNING指向GOVERNANCE，已人工檢查、無治理規則變更；health --ci --as-of 2026-11-24 exit0／WARNING（Governance／Authority Drift／Legacy Docs），無structural FAIL。未執行尚不存在的adapter、未來mutation／摘要推導或正式資料測試，作者自查非獨立複審；no-commit故保留指定6份unstaged文件。

## P4d 對等調參／三軸曲線草案 — 2026-11-24

`job-run-mwut6qrj-7d9826/spec_writer_p4d_modules`，沿用指派detached HEAD/base `ffe98b4`、初始clean，只交三文件patch、不commit。[P4d規格](../../tasks/spec-p4d-equal-tuning-20261125.md)仍 **DRAFT／E2 RESEARCH_ONLY／UNMERGED／NOT_DEPLOYED／NOT_ACCEPTED**，全部歷史資料已曝光、僅描述性，無執行／費用authority。核心維持三臂各N12×2seed、2021前126-session全列驗證、random／IC5與固定更新六折重訓、Holm2；不以outer選超參。Human同意加入容量A與種子B設計，三模組預設固定P4a配方：A六格（寬度32/64/128/256×2層＋d64×1/4層），headdim8／state16／expand2固定；B固定序列17/29/43/59/71/89/101/113之前1/2/4/8排名平均；D可選25/50/100%最近完整年等值。baseline更新數在2021前固定，各格不再早停；不同配方核心結果不得默認共用。

P4a原始ledger證據已由Human的 `p4a-results (3).zip` 提供派生摘要；本輪實讀主checkout `.artifacts/p4/p4a-fit-durations-min.json`（64 fit記錄／12群組），W60 GRU均值9.05／max10.5、Mamba2均值27.5／max29.0min；W120／250 Mamba均值38.4／61.8、max40.9／62.8min。未重新開ZIP驗hash-chain或逐fit環境，不能稱正式結果驗收；P4a逐折早停／80k val cap與P4d切分／更新不同，**本預設不重用P4a兩seed fits**。三模組7 origins共享baseline去重共308神經fits＝93.81 T4h基準等效；加N12核心為380 fits／115.74h，觀測max情境125.08h。資料D增量17.06h，A六格增量42.64h均需明列；容量倍率未知，這不是總工時上界。T4估計1.5–2.5 CU/h，全案基準至max情境173.6–312.7 CU，需當次面板觀測替換；不核准費用。P1累積調參總次數仍UNKNOWN。

Next：Human凍結六／12格、三模組開關、固定配方／種子及分階段資源；另案實作／獨立審查，先baseline及GRU B/A，再Mamba B，依排練資源決定大容量停／續；d256×4只屬12格替代且先排練，不能結果後擴格。全列CPU/RAM、實際split／hash、最大容量時間／顯存、原始逐fit身分與當次CU率仍待驗證，未自行啟動pilot。Knowledge-impact為候選設計與blocker／next decision更新；Research同步、Authority Map已檢查但不變，無machine authority supersession或runtime observed_at更新。工作樹保留三文件dirty符合no-commit指示；無程式／GPU／Data／blind／Colab／Drive／排程／額外agents操作，完成不等於Human acceptance。

## 專案洩漏稽核／Colab 排練清單候選 — 2026-11-24

`job-run-mwuchypx-211f4f/skill_writer`，既有 detached HEAD/base `8eff9fef3386d5b852b79f5c66d09a0893891122`；UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED，Human 核准僅文件／skill、no-commit。新增 [ml-leakage-audit](../../.claude/skills/ml-leakage-audit/SKILL.md) 與 [colab-preflight](../../.claude/skills/colab-preflight/SKILL.md)，AGENTS.md 加 Codex 共用清單來源指標；本文不授予研究、執行或 production authority。

建立前盤點：既有 advisor／agents 涵蓋角色隔離，ruflo-project 涵蓋 worker／artifact 交接；quant_rigor_checklist 是通用複習，Training domain／P3／P4a reports 保存契約與事故，colab-human-requirements 是 Human 偏好單一來源。缺口是任務觸發的短清單與逐項可追溯檢查，故僅沿用 `.claude/skills/` 增兩個入口，不建立新流程、記憶或報告樹。選擇連結既有證據而非複製報告／Human 需求；不塞入 advisor 或 AGENTS 長清單，避免角色協定與跨 CLI 規則分叉。歷史事故、已知限制與未驗證候選分開描述。

Knowledge-impact：清單導航／同份 canonical 清單來源新增=YES；研究語義、primary authority、subsystem、active runtime、correctness thresholds、deprecation 與研究 blocker=NO。已人工檢查 GOVERNANCE，不改治理規則或 machine authority。既有文件測試發現 HEAD 的 Current_State／Authority_Map 首個 P4a 標題空段落，`test_p4_knowledge_preserves_candidate_authority_status` 原已失敗；保留原文件結構、不順修無關段落。獨立審查待後續 session，作者自查不等於 Human acceptance；沒有 GPU／網路／Colab／Drive、資料取得、正式 Data／V6/models、production／scheduler／通知／wakeup／額外 agents／commit／push／merge／deploy，也未更新 runtime observed_at。

Evidence：兩份 skill 為72／61行，quick_validate 均通過；本機引用核對13個連結、36個測試名稱通過；`python3 -m unittest discover -s tools/project_knowledge -p 'test_*.py'` 為23 tests OK。CPU／`PYTHONPATH=V6` 的 `test_p4.py -k 'draft_documents or operations_require or knowledge_preserves'` 為2 passed／1 failed／110 deselected，失敗即上述 HEAD 既有空標題（首次未設 PYTHONPATH 的 collection error 不算通過）。impact checker WARNING，已人工核對 Operations／GOVERNANCE，無操作子系統或治理語義需改；health `--ci --as-of 2026-11-24` exit0／WARNING，含 authority repin、stale runtime 與 legacy docs，無 structural FAIL。skill SHA-256：ml=`7c9001821293558331efbab82850c0c7dfee149fdaab4569d238af59656adc1f`、colab=`cea48735bcc3a4e755df0d497ed025274040aaf9217afa4a3c2d5220ce81ccbb`。Next：獨立 session 核對清單與原始證據並由 Human 驗收；既有空標題文件測試失敗另案處理。工作樹身分與保留條件見 [Worktree Register](Worktree_Register.md)。

## P4a 第 13 次複審 REQUIRED＋兩項 MINOR 修補候選 — 2026-11-24

`job-run-mwtn0pir-48e95f`／`builder_p4a_gate3`，沿用既有 detached worktree，base/HEAD `40d621cc9ac586e45fff1e095f59465b388094a1`；UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED，依明確授權保留未提交 patch。第 13 次獨立複審為 NO_GO；本輪修補 Q5 巢狀 schema、超大數值／遞迴損壞 marker，以及 metadata 符號連結。完成不解除 NO_GO，不等同 Human acceptance。

[runner](../../V6/research_lab/p4_colab.py) 的階段結果、發布 metadata、Q5 診斷與成功 marker 共用單一封閉遞迴 schema；必要／允許欄位、精確型別、數值有限性／值域、ISO 日期、凍結 rule、R=10 與 permutation/date 支援均 fail closed。`excluded_dates` 依 producer 定義為日期字串清單（可空）；布林只在契約的 bool 欄位合法。Q5 producer 原先傳 session index，現在傳同列 `dates.npy` 的日期，合成證據確認分組與數值診斷不變。marker 的 OverflowError／TypeError／RecursionError 與既有解析錯誤視為無效，顯示期望／實際／下一步後移除並重排練。完成、發布、result JSON 與 marker 拒絕符號連結；產物 bytes、請求、來源與正式完成驗證沿用原 gate。

[工具重產宣告](../../tasks/p4a-source-compat-20e6b54.json) 的 `to.commit` 為本輪可得 HEAD `40d621c`，230 個來源 hashes 綁定 working bytes；後續 commit 可不同，deliver 必須以最終 bytes 重算，不可直接採用舊宣告。36 個 Q1／Q2 W60 完成 fit 全部 resume_fit 驗證／保留 bytes／不重訓的條件與 64-fit SHA `4dc90ec0ada6c6d34b43c429d359424c502634e502c6856c6774bf138de7ba23` 不變；研究／環境／預算契約、P3 adapter／portfolio engine 及正式 Data／模型未動。AST pins 僅限定候選範圍，非獨立審查證明。

Evidence：本機 CPU 合成與真子程序發布替身；RED→GREEN、逐欄位型別矩陣、三項新 in-memory mutation、實際完整測試與 artifact identity 見[工程報告](../../reports/v7_p4a_colab_engineering.md)。CUDA／官方 mixer／真實 fold／W250 峰值／Colab／Drive FUSE／現場 36 artifacts 與正式接續仍 UNVERIFIED。Next：獨立複審 patch、宣告與測試，通過後由 Human 決定現場排練及 acceptance。無 GPU／網路／Colab／資料取得／production／scheduler／排程／additional agents／wakeup／recurring jobs／commit／push／merge／deploy，未更新 runtime observed_at。

## P2 前瞻／P4b 描述性規格草案 — 2026-11-24

`job-run-mwuc1teo-b3dba8/spec_writer_fwd_p4b`，既有 detached worktree/base `b7e4f50d4f061cce32f35f466de7ab992ffc35ed`；僅文件 patch，UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED，依指示不 commit。[P2 前瞻草案](../../tasks/spec-forward-p2-primary-20261124.md)與 [P4b 混合草案](../../tasks/spec-p4b-blend-20261124.md)均 **DRAFT 待 Human 凍結**，沒有新研究結果或執行授權。沿用 tasks／research_lab，歷史規格不能承接新的研究身分，未建立新目錄、程式或輸出樹。

唯讀本機 artifacts 發現 P1/P3 分數到 2026-09-11，但 P2 收益評估到 **2026-11-19**；本機價格到 2026-11-23。Human 於2026-11-24已選前瞻T0嚴格晚於已曝光／現有價格末日、Human凍結及前置驗收，不追認11/20；主假設為三種子C0+加半年重訓policy，依下方最新兩階段修訂從T1起252-session整段端點、單一假設／Holm m=1。固定模型僅替代／描述性，前後段不新增正式假設。矩陣延伸功能、舊區間逐位元一致、模型持久化、PIT／lifecycle／coverage 與封盲仍為 Blockers。P4b 固定三個等權混合與 Holm3、沿共同列 NaN 語義；需要另案 36-fit subset loader，不降低既有 64-fit gate。36/36 完成是 Human 報告，不是本輪現場驗證。

2026-11-24最新兩階段文件修訂：`job-run-mwuoasa7-93afef/spec_writer_phases`／detached base `0db633bcedc97484bff6c316d7cc9d19334137d2`，初始clean，4份文件未提交patch；DRAFT / E2 RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED。Human新決定取代Q1／Q3：嚴格T0（候選2026-11-25之後）僅試運行整日流程，績效封盲且永不調規；年報後重訓T1才正式起算252收益sessions、唯一H-F1／Holm m=1／雙邊保守p。一般年度3/31＋Q2 8/14作整體更新時鐘，75日、金融半年8/31、順延／延期／非曆年等按公司期別核實，不能當可得日。DRAFT提案年報N_A=10／Q2 N_H=8＋必要資料實際公開／入管線2-session緩衝，保留18:00與最多2-session重試／超時STOP；N是作者提案、非已凍結Human數值。試運行預設直到正式端點才描述性揭曉，另列T1簽署且啟動後揭曉／永不揭曉替代風險；正式規則／日曆必在任何階段0績效曝光前簽署。

無假日週日曆示例為T1=2027-04-14、Q2更新8/25、252末日2028-03-31、次年年報更新4/14；正式交易所calendar未驗，真實端點會受休市影響，第二次更新可能入窗，正式次數UNKNOWN，不按績效選日。工程按兩次容量準備，逐版計算訊號與20-session再平衡暴露。WP1～WP6拆T0前最低安全日鏈與T1前完整驗收，建議T1前20／5 sessions截止；20日穩定流程及跨Q2仍未啟動就STOP是待凍結提案。試運行不穩只按無績效gate延後T1，重新as-of重訓、不回填；正式開始後不得重設T1／延長252日。所有工程仍另案批准／獨立驗收，文件完成不等於Human acceptance。

本輪source核對：V6 builder revenue月末+11日、financials預設+45日，只有fundamentals_v2顯式啟用才年度+90日；非逐公司實際公告／revision PIT。**P1的48欄manifest已移除財報／營收等11欄，C0+185欄也未加回**，不能宣稱報後重訓會直接納入季報。嚴格source／PIT gate不因固定lag、季報期限或本文件放行。

Human已維持252日單端點／雙邊保守p、神經與固定影子臂預設不收集、P1／分段描述性、raw主要／基準比例成本。Human兼任封盲保管與重訓執行owner，指定 `D:\Desktop\work\ProjectForMe\MarketMamba-blind\`（WSL `/mnt/d/Desktop/work/ProjectForMe/MarketMamba-blind/`）；Claude／Codex不得讀寫，worker prompt明載，只看append-only hash chain／行數／coverage／錯誤計數／缺日，不能還原績效。Claude本機拒絕設定為Human報告、未查驗；Codex另程序同WSL權限，這是防疏忽紀律，不是OS隔離或防竄改。一次性開盲由Human在端點成熟及完整性核對後手動觸發，Claude開盲後才取結果；偷看即確認性降級。隔離與健康檢查目前只有規格，沒有已部署保護的主張。

Next：Human分階段凍結N_A／N_H、T0／T1 calendar／source gates及全文，另案批准並驗收附錄WP1～WP6（日期／as-of→重現→矩陣延伸→模型存檔／隔離→每次驗收→252日adapter）。CPU時間僅估計，峰值RAM／IO未知，前置驗收未完成保持STOP。P4b文件／固定歷史C0+分數／三混合Holm3／36-fit subset／64-fit gate不變。沒有新結果、程式／資料／GPU／Colab／Drive／production／scheduler／排程／額外agents或commit／push／merge；未讀封盲區、未更新observed_at。完成不等於Human acceptance。

本輪文件驗證：`python3 -m unittest discover -s tools/project_knowledge -p 'test_*.py'` 23 tests OK；規格10個本機連結缺失0、8個Markdown表格欄數一致、48欄／11個排除欄與manifest相符、6項無假日日曆算術assertions通過。P4b SHA-256維持 `da508479fb417d0a0b65ed84d7ca8ca6ef2548832b78ce176af13fbc1621e3ca`。impact WARNING（GOVERNANCE已人工檢查，無治理規則／machine authority改動）；health `--ci --as-of 2026-11-24` exit0／WARNING（Governance、Authority Drift、Legacy Docs，無structural FAIL，與修改knowledge前分類相同）。`git diff --check`通過；作者自查非獨立審查，正式日曆／實際流程／資源仍未驗證。初始無dirty/untracked/staged，交付保留上述4份未stage文件，依Human本任務禁止commit而非clean-worktree；沒有其他檔案修改。





## P4a 第 12 次複審兩項 REQUIRED 修補候選 — 2026-11-24

`job-run-mwtlq2cv-a82151`／`builder_p4a_gate2`，沿用既有 detached worktree，base/HEAD `99b4e0e3547868847262aa305cd5fe7100c3c9f5`；UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED，依授權保留未提交 patch。第 12 次複審為 NO_GO：排練只憑返回碼放行、有效 JSON 的成功標記可被清空／竄改仍略過。本候選收緊 [runner](../../V6/research_lab/p4_colab.py) 的工程 gate；不解除原 NO_GO，不取代獨立複審與 Human acceptance。

閘門逐階段讀獨立結果 JSON，核對 Q3 W120 mamba2／Q3 W250 mamba2 各 1 fit、Q5 2 fits 與 validation／evaluation 完成，核對 source／prepared data／reference／environment 與本次 run command、nonce；每個 fit 契約綁定請求，並由 gate 自行重驗既有 completed-output validator 的 scores／reference／checkpoint／complete identity 與 bytes。空輸出、吞例外後正常退出、錯誤 selection／identity 或缺產物一律 STOP，不寫成功 marker、不啟動正式 runner。成功 marker 要求完整 schema、恰好三階段 identity／耗時／REHEARSAL evidence／fit 數及內容 digest；缺欄位、型別錯誤、未知欄位、空 stages 或不符即重排練。digest 只防誤改，非惡意防護；重連快取不依賴 `/tmp` 產物仍存在，marker 不是正式完成或研究 authority。

[工具重產宣告](../../tasks/p4a-source-compat-20e6b54.json) 的 `to.commit` 從舊 `bf0114d` 更新為本任務 HEAD `99b4e0e`，sources hashes 綁定 dirty working bytes；不是已提交或已接受的認證。36 個 Q1／Q2 W60 completed fits 全部 resume_fit 驗證的採用條件、64-fit SHA `4dc90ec0ada6c6d34b43c429d359424c502634e502c6856c6774bf138de7ba23`、研究／環境／預算契約及 P3 adapter／portfolio engine bytes 均維持。AST pins 為候選範圍限制，仍待獨立複審。

只有本機 CPU 合成工程證據與真子程序替身；CUDA／官方 mixer／W250 峰值／Colab／Drive FUSE／現場 36 artifacts 仍 UNVERIFIED，未更新 runtime observed_at。沒有 GPU／網路／Colab／資料取得／production／scheduler／排程／額外代理／commit／push／merge／部署。實際測試、變異、限制及交接見 [工程報告](../../reports/v7_p4a_colab_engineering.md)。Next：獨立複審 patch、宣告與測試，再由 Human 決定接受及現場驗證；本次完成不等於 Human acceptance。

## P4a 正式前自動排練閘門候選 — 2026-11-24

`job-run-mwtk8k4r-a80ee8`／`builder_p4a_gate`，沿用既有 detached worktree，base/HEAD `bf0114dad9dc2d9bef7502c21d6fd36737eb0458`；UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED，依 no-commit 指示保留 patch。獨立審查 NO_GO 的唯一 REQUIRED 為 notebook 缺正式前自動排練；[產生器](../../V6/research_lab/p4_bundle.py) 現在於正式 runner 同一格前段依序跑 Q3 W120 mamba2、Q3 W250 mamba2、Q5（兩 prerequisite fits），各 1 step，真實資料／官方 mixer 由現場執行；本輪只有 CPU 合成工程證據。

FORMAL／resume 預設啟用、SPEED_TEST 預設關閉，第 0 格可明設 `REHEARSE_FIRST`。任一步失敗／中斷不啟動正式 runner；輸出失敗階段、`/tmp`、期望／實際／下一步。全部成功才寫 progress 的 `rehearsal_passed.json`；同一 run command／已驗 bundle source／prepared matrix＋rankgauss／資料 realpath／實際 environment identity 相同才略過，標記不是研究或完成 fit authority。[runner](../../V6/research_lab/p4_colab.py) 新增 W≠60 的 sessions／eligible dates 前置檢查，W60 不走新路徑；排練成功輸出 CUDA peak allocated bytes。

[來源相容宣告](../../tasks/p4a-source-compat-20e6b54.json) 由工具重產，誠實分類自動閘門與長窗前置檢查；固定 queue SHA `4dc90ec0ada6c6d34b43c429d359424c502634e502c6856c6774bf138de7ba23`、36 個 Q1／Q2 completed fits 全部 resume_fit 驗證採用條件與研究／環境／預算契約不變。沒有改 P3 adapter／portfolio engine 或正式 Data／模型。完整實際測試輸出、位元比對、三項 pytest exit=1 的 in-memory 變異與限制見[工程報告](../../reports/v7_p4a_colab_engineering.md)。

Next：獨立複審此 patch，再由 Human 驗證真實 CUDA／官方 mixer／Colab／Drive 與正式接續；本輪完成不等於 Human acceptance，不解除 NO_GO。W250 windows 張量／GPU 峰值、首次全量推論耗時、Drive 真實 36 artifacts 均 UNVERIFIED；無 GPU／網路／Colab／production／scheduler／additional agents／commit／push／merge／deploy，無 runtime observed_at 更新。

## P4a Q3 事故修補／上線前排練候選 — 2026-11-23

`job-run-mwt8u5js-7f212c`／`fixer_p4a_q3`，detached base `685f6ece1faa7ed39ff7ebff7056b2c4ad7eb4b2`；UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED，依 no-commit 指示保留 patch。Human 報告原 02_formal@4e56e9d 已完成 36 個 Q1／Q2，Q3 W120 因尾端 NaN label 中斷；本輪未觀測 Drive 或驗證現場 36 份產物。修正列身分 NaN 等價／±inf 拒絕，Q5 IC 僅用共用成熟標籤支援；新增同一訓練／發布／輸出驗證路徑的 `--rehearse`，產物僅 `/tmp`、明標不是研究結果。

[Training note](../01_Domains/Training.md)／[工程報告](../../reports/v7_p4a_colab_engineering.md)記錄實際 CPU 回歸、跨版 bytes／RNG、72 次 W60 呼叫路徑、36 完成後首筆 Q3 接續、四項 in-memory 變異與限制。[工具重產的 GENERATED 宣告](../../tasks/p4a-source-compat-20e6b54.json)綁定本工作樹 bytes，新增採用前原地 resume_fit 驗證 36 份完成 fit；只擴充本事故的相容分類，不改研究／queue／matrix／環境／預算 authority。

Next：獨立審查後，Human 以真實 fold/reference 在本機 RTX 3060 或 Colab 排練 Q1、Q3 W120＋W250、Q4、Q5，才決定新版正式接續。現場 CUDA／官方 mixer／T4／Drive FUSE／Human acceptance 仍 UNVERIFIED；無 GPU、網路、Colab、正式 Data／模型、production、scheduler、commit／push／merge 或額外 agents；runtime observed_at 未變。

## P4a 來源相容升級候選 — 2026-11-22

`job-run-mwrxutx3-d28fa4` builder，沿用既有 detached worktree `builder_p4a_upgrade`，base/HEAD `20e6b5408d9f9b13d56290a72a2dc313c35d1b94`；UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED，依 no-commit 指示保留 patch。依 [Human 需求第 8 節](../../docs/operations/colab-human-requirements.md)新增 [來源驗證模組](../../V6/research_lab/p4_source_compat.py)，沿用 [runner](../../V6/research_lab/p4_colab.py)／[交付工具](../../V6/research_lab/p4_bundle.py)，沒有新 worktree／實驗樹。現有 runner 沒有跨來源證據層，因此在相同 research_lab 新增單一 helper；不複製訓練／完成驗證實作。

目前只認證 `4e56e9d` → 本工作樹逐檔 bytes。`deliver --resume-from-commit 4e56e9d` 由 git objects 重算全部舊來源與逐檔 diff／分類，綁定未變的契約／64-fit queue／DATASET_MANIFEST／環境／record_only 預算，宣告隨 ZIP 打包；[本輪 GENERATED 宣告](../../tasks/p4a-source-compat-20e6b54.json)供審查，不是新研究 authority。以舊 run 身分綁定的原 ZIP manifest 補足舊 run_identity 缺少的來源清單，完整核對 from／to、差異集合及非來源輸入後才採用舊 run_hash。run 目錄與帳本追加 SOURCE_UPGRADE，歷史 complete／pointer／checkpoint／run_identity 不改寫。缺證據、差異或多 lineage 一律 STOP；無舊 run 才新建，明設 allow-new-run 才允許新批次，既有帳本／STOP gate 不略過。

本節取代先前「任何換版只能重新綁身分開新 run」的操作限制：執行中的 kernel 仍不可換碼；斷線後才用 `03_formal_resume` 新 notebook，同一 EXPERIMENT_DIR／budget，保留原 02_formal ZIP 及原 notebook 保底。只限數值中立的進度／日誌／speed-test 帳本分離／文件；每個新來源基準先補跨版測試與獨立審查。研究、production、環境 authority 與 runtime observed_at 均未改動；本機唯讀原 ZIP 的 229 個來源與 git objects 完全一致，不能推定 Drive 現況。

最終完整 CPU 回歸 **774 passed／1 skipped／5 warnings（382.55s）**，4 項 in-memory mutation 皆攔截。CPU 跨版 GRU／crossstock（mixer 替身）權重／預測／RNG、舊 epoch pointer 接續、舊完成 fit 驗證及 12 完成＋52 剩餘的 runner／ledger 整合證據見 [工程報告](../../reports/v7_p4a_colab_engineering.md)。Colab／T4／Drive FUSE 現場 UNVERIFIED；無 GPU／網路／資料取得／production／scheduler／通知／額外代理。Next：獨立審查 patch、宣告與測試，接受後重新 deliver；現場測試與 Human acceptance 仍未完成，不解除原 NO_GO。

## P4a 進度兩項 REQUIRED 修補候選 — 2026-11-22

`job-run-mwrpdfba-d4ad9d` builder，既有 detached worktree `fixer_p4a_progress`，base/HEAD `5156087da0e2aaba5d0cf703db3cdd67200cfbe9`；UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED，依 no-commit 授權保留 patch。本節補正前節進度語義，沿用 [runner](../../V6/research_lab/p4_colab.py) 與 [進度測試](../../V6/research_lab/tests/test_p4_progress.py)：驗證 checkpoint 後設定本次 step 起點，至少 10 新 steps／1 秒才估 fit ETA；3900 → 4000／5000、10 秒的接續案例為 100 秒。佇列目前 fit 優先用本次 step ETA，後續 fit 用本次完成 fit 固定實測耗時均值（至少 1 秒），cached skips 不作樣本；無可靠量測即 null／預估中，Q5 仍額外未知。

略過已驗證 fit 也更新編號、Q、arm／year／seed／window／lr，清除前筆訓練欄位，進度行與 TXT／JSON 標示「已完成（略過）」並保留至正常退出；全部快取與 mixed skip/train 均逐 fit 驗證。中途接續恢復已提交 epoch／step，最後 fit 完成時兩個 ETA=0；不更改 checkpoint／完成 gate、64-fit 順序、模型／RNG、凍結研究與環境契約、帳本或 production authority。

CPU 合成完整測試 **745 passed／1 skipped／5 warnings（273.22s）**，四項 in-memory 變異皆被新測試攔截；notebook nbformat 與 8 個程式格 compile 通過。工具 knowledge-impact=PASS／DEFAULT，人工 gate 判定進度語義與 next decision 有影響，因此同步更新本節與 Training note。沒有 runtime observed_at 更新；Colab／T4／Drive FUSE 仍 UNVERIFIED。獨立複審仍待進行，不以 builder 完成取代原 NO_GO 或 Human acceptance。Next：獨立複審此 patch；接受後另案重新產生綁定新 source identity 的交付，現場驗證另行授權。實際輸出、artifact hashes 與限制見[工程報告](../../reports/v7_p4a_colab_engineering.md)。

## P4a 進度與 speed-test 帳本候選 — 2026-11-22

`job-run-mwrof3ay-8d0c43` builder，既有 detached HEAD/base `f6a173eab03332bc3c50d78885a1d865366d2518`；UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED，依 no-commit 授權保留 patch。依[Human 持續需求](../../docs/operations/colab-human-requirements.md)沿用 [notebook 產生器](../../V6/research_lab/p4_bundle.py)、[P4 runner](../../V6/research_lab/p4_colab.py)與 `seq_runner.train_one`：串流／flush、每 50 steps／epoch、60 秒時間心跳，Drive `progress/formal/` 與 `progress/speedtest/` 的 TXT／JSON、可行動錯誤及離線摘要格。快照是觀測，不是 checkpoint／完成／研究 authority；FORMAL 才顯示驗證 IC。ETA 未知即預估中，佇列以本次完成 fit 的實測耗時外推，不假設 Q5 耗時。

本節取代舊候選「speed 與 formal 共用 budget」的操作描述：speed 自動使用 `budget_speedtest/`，FORMAL 保持 `budget/`；各自跨交付延續，既有 chain／identity／缺帳 gate 不變，不遷移或重置歷史帳本。正在跑的原始 02_formal 不可中途換版；後續交付須重新綁定 source identity。FROZEN 研究契約、OBSERVED runtime bytes／observed_at、P3 adapter 與 portfolio engine 不變，無 production／scheduler／正式 Data／模型操作。只有 CPU 合成工程證據（權重／預測 bytes 與 torch RNG 一致、真子程序串流、帳本先 speed 後 FORMAL、in-memory 變異），Colab／T4／Drive FUSE 仍 UNVERIFIED；不等同 Human acceptance。Next：獨立審查 patch，接受後另案重新 deliver 與現場驗證。實際輸出與限制見[工程報告](../../reports/v7_p4a_colab_engineering.md)。

## P4a 三項 REQUIRED 修補候選 — 2026-11-22

`job-run-mwric0jj-32d745`，既有 detached HEAD/base `d82d0209cba6686c10fc1a01c78172446a6796fa`；UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED，依 no-commit 授權保留 patch。修補範圍限獨立複審的三項 REQUIRED：[evaluator](../../V6/research_lab/p4_eval.py) 對 Family A/B 列集合不一致明確拒絕並報差異，只有 C 取共同交集且先限制每個 seed 再重算 percentile rank／平均；保留交集與排除比例。[completed-fit validator](../../V6/research_lab/p4_colab.py) 綁定 frozen queue／每 fit 的 window、lr 有效配方，驗證連續 history、重算 BEST／bad、8 epochs 或有效早停、metrics／pointer／預測 fit 設定。發布與續跑共用驗證；舊缺 recipe／history 證據的完成標記 fail closed，不遷移或重標歷史 artifacts。

機器研究契約與 OBSERVED runtime bytes 不變，未更新 observed_at；P3 adapter／portfolio engine 原樣，無 production／scheduler／正式 Data／模型／GPU／網路／Colab變更。只有 CPU 合成工程證據；Colab／T4／Drive FUSE 現場仍 UNVERIFIED，工程完成不等於 Human acceptance。實際驗證與限制見[工程報告](../../reports/v7_p4a_colab_engineering.md)。Next：獨立複審本 patch，再決定接受與重新產生綁定新 source identity 的交付；本輪不執行現場作業。

## P4a 完整固定佇列與 Drive 資料候選 — 2026-11-22

`job-run-mwrgy396-cf1213` builder，既有 detached HEAD/base `f333421fafdcb02f716b16376ff65717021edfa4`，UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED；依 Human no-commit 指示保留 patch。此次核准的 [FROZEN 規格](../../tasks/spec-p4a-colab-20261121.md)／[機器契約](../../tasks/p4a-colab-contract-20261121.json) 取代下列舊 P4a DRAFT、25% reserve、wall/CU caps、completed-fit-only、矩陣隨 ZIP 交付的目前操作描述；歷史紀錄保留，P3 source/artifacts/14h 契約完全不重標。

沿用 [bundle](../../V6/research_lab/p4_bundle.py)／[executor](../../V6/research_lab/p4_colab.py)／`seq_runner.train_one`；資料 manifest 綁定 matrix_v1 的指定 X identity，Drive 只上傳一次，複製到SSD後才mmap，rank-gauss共享快取回存Drive；ZIP只帶source/契約/環境/manifest/C0+鍵檔。Q1/Q2主格36、Q3長窗24、Q4學習率4，共64 fits，再Q5排列診斷。epoch檢查點恢復model/optimizer/scheduler/BEST/history/RNG；latest/previous/best先持久驗hash再pointer，orphan不自動升格。已完成fit驗證後跳過；P4只記錄耗時與GPU telemetry，>=85°C STOP仍鎖定，P3預算不變。

新增 [P4 evaluator](../../V6/research_lab/p4_eval.py) 因P3 evaluator不可改，重用P2公開函式。Family A/B/C Holm m=3/2/4；長窗採全比較系統/W60/C0+顯式交集並報排除比例。P3已曝光，mamba2/crossstock是新環境replication；GRU與長窗為新假設但資料仍曝光。CI/Holm門檻通過只屬歷史比較，唯一獨立確認仍是未曝光P2前瞻資料；C0+曾調整、神經臂未調參偏向C0+，P4d本機CPU等預算調參另案OPEN。

本機只有CPU合成證據：epoch中斷/接續（含dropout）權重及預測位元一致、64fit/Q5替身佇列與重跑skip、安全交付/manifest/共同列/統計測試。完整驗證數字與限制見 [工程報告](../../reports/v7_p4a_colab_engineering.md)。無正式Data／V6/models／GPU／網路／Colab／scheduler／production改動，沒有P4結果；現場Drive FUSE持久性、T4長窗/OOM、wheel/編譯、Python3.13整批與下載均UNVERIFIED。Next：獨立審查候選，Human完成一次資料上傳與真實交付後另行現場驗證；工程完成不等於Human acceptance。現場操作見[指南](../../docs/operations/colab-p4.md)。

## P4a Drive 分次交付路徑候選 — 2026-11-22

`job-run-mwqscvk1-a3db0f` builder，base/HEAD `df78d6cb3bb706f6f4d58b94310c3d352aab8385`；UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED，依本輪 no-commit 授權保留 patch。沿用 [p4_bundle](../../V6/research_lab/p4_bundle.py) 的 deliver／notebook；新增 `--run-name`（01_speedtest）／`--drive-root`（MyDrive/MarketMamba）與兩行 README.txt，不建立平行流程或新 worktree。

第0格預填 `RUN_DIR=.../MarketMamba/p4a/01_speedtest`、`EXPERIMENT_DIR=RUN_DIR.parent`；runner／摘要／匯出都使用 `EXPERIMENT_DIR/budget`，speed 與 formal 跨交付共用同一本帳。ZIP 依 RUN_DIR → EXPERIMENT_DIR/inputs → /content 搜尋；殘留 /content SHA 不符明確 WARNING 並忽略。notebook 綁定實際 ZIP bytes/SHA，解壓前另核對 manifest `total_bytes`（含 aliases 邏輯大小）及實體成員大小，不混比壓縮與還原大小；不符 STOP 並提示上傳不完整。WHEEL_DIR 預設 MarketMamba/wheels，空／不存在才回退 Mamba_Wheels_T4_new，均無 wheel 才編譯寫回新目錄；部分 wheel／不相容仍 STOP。

SPEED_TEST 上限改1.0小時／15 CU；FORMAL 無預設，須 Human 明確填入。其餘0.5小時每fit/stage、10 CU/hour、25% reserve 保留；全新三臂 speed 計畫至少需2.5小時／25 CU加已花時間，因此小上限下計畫檢查會STOP，需Human依實際保守估計另行決策，不能放寬gate。既有舊根帳本不自動搬移／重置；若已有消耗，須保留原 EXPERIMENT_DIR 或先獨立盤點遷移。DRAFT 正式 gate、研究／production authority、runtime observed_at 與 lock 未變。

本輪僅本機 CPU 合成工程證據；沒有上傳 Drive、網路、GPU、Colab、資料取得或研究結果，現場行為仍 UNVERIFIED。此節取代前節的6小時／100 CU、/content優先與舊預設wheel路徑操作描述，保留歷史驗證紀錄。既有 fold resolved-root 邊界與壓縮比兩項 OPEN 限制未處理。Next：獨立審查 patch 與預算設定，核准真實輸入後重新 deliver；現場 Colab 驗證及 Human acceptance 另行處理。詳見 [指南](../../docs/operations/colab-p4.md)、[工程報告](../../reports/v7_p4a_colab_engineering.md)。

## P4a 兩檔 Colab 交付候選 — 2026-11-22

`job-run-mwqr9yiu-8c464e` builder，既有 detached HEAD/base `74795a4a56f25585000d26d281e305381de6230a`；UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED，明確 no-commit 授權保留 patch。[產生器](../../V6/research_lab/p4_bundle.py) 的 `deliver` 一次輸出 `p4a-bundle.zip` 與 `p4a-colab.ipynb`，沿用既有 pack／executor／研究契約，沒有平行流程或新 worktree。ZIP 以內容 identity 去重跨 fold 的 X.npy，manifest 完整邏輯檔案清單加 aliases；安全解壓還原各 fold 的一般檔案並逐份驗 size/SHA。fold 輸入 symlink 只在打包端解析，解壓仍禁止 links，tar 相容與白名單／資源／cache gate 維持。

Notebook第0格集中預算與路徑，預設SPEED_TEST；找 /content 再找Drive、標準庫取內含bootstrap後完整驗證，不需手貼SHA或另傳bootstrap。先自檢環境，已符合契約則略過安裝；cp313 Drive wheel路徑 `MyDrive/Mamba_Wheels_T4_new`，cp312不符仍STOP。兩wheel均缺才使用官方sdist編譯備援（mamba C++20、MAX_JOBS=4）及Mamba2小型前後向冒煙。CUDA gate、DRAFT正式拒絕、既有resume／預算帳本與speed輸出隔離不變。

[Colab runtime契約](../../environments/p4a-colab-runtime.json)已是Human提供的OBSERVED Python3.13.16／torch2.14.1+cu130／CUDA13.0／T4／mamba2.3.2.post1／causal1.7.0；本節取代舊候選「等待填TBD版本」的next action，但不修改原觀測時間，不把Human smoke升格為專案正式GPU證據。本輪只有CPU合成測試／nbformat解析／三項in-memory反證，無網路、GPU、資料取得、正式Data、模型、production、scheduler或研究結果。Colab現場ZIP解壓、編譯儲存格、Drive掛載／持久寫入／下載仍 **UNVERIFIED**。Next：獨立審查patch；另行授權真實bundle打包及Colab現場驗證；FORMAL仍需凍結研究契約。細節與限制見[指南](../../docs/operations/colab-p4.md)、[工程報告](../../reports/v7_p4a_colab_engineering.md)。

## P4a Colab 操作修訂候選 — 2026-11-21

`job-run-mwqi30jf-aa0d7f` builder，既有 detached HEAD/base `36aa27da6e26c1877332c9cdea927e6409c3d105`，UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED，依本輪明確 no-commit 授權留 patch。無 GPU／網路／Colab／正式 Data／模型／production／scheduler 變更，未 commit/push/merge。

Human 已核准 archive／bootstrap 外部 SHA 不再必填：extract 自動印出 archive SHA，提供時嚴格核對；bundle 內逐檔 size/SHA、資源上限、symlink 防護與白名單仍強制。這取代本頁舊輪次「外部 SHA 必填」操作描述，不重標歷史證據。Drive `--wheel-dir` 模式只裝兩個 wheel（no-deps／no-index／find-links），記錄檔名／cp與平台標籤／SHA，標籤不相容或缺失／多版本 STOP；不改 torch 或其他套件。

新增獨立 [Colab runtime 契約](../../environments/p4a-colab-runtime.json)，初始 TBD／UNOBSERVED，待 Human pip list 與 runtime 自檢填實際觀測版本，不猜 torch／CUDA 組合。本機 [P3 參考 lock](../../environments/p4a-colab.requirements.lock.txt) bytes 不改。speed-test 保存逐項 environment_mismatch 與「結果不可與正式環境比較」，正式模式要求 OBSERVED 契約完全相符，CUDA 不可用仍一律 STOP；DRAFT 研究 gate 不變。P4a 三臂須在同一 Colab 環境 identity；與本機 P3 的比較需揭露環境差異。程式／測試／[指南](../../docs/operations/colab-p4.md)／[報告](../../reports/v7_p4a_colab_engineering.md)是工程候選，沒有新研究結果或 Colab acceptance。Next：獨立審查 patch；Human 提供 pip list／runtime 自檢，完成契約觀測後重新 pack，仍須研究凍結與現場 GPU／Drive 驗證。

## P3 神經模型組合評估結果（探索性）— 2026-11-21

## P4a Colab 工程候選 — 2026-11-21

`job-run-mwqd0vuy-7d9826` builder，既有 detached worktree base `e4f21c3747124f4c6cc46a873eae891697910067`；UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED，依限定授權僅離線程式／CPU合成測試，原候選與第一輪修補已分別提交於 `8cb61c4`、`05cb840`；第二次修補已提交為 `a76e98d`（此後提交狀態以 git log 為準，不在此逐次更新）；未push/merge、無GPU／Colab／網路／正式資料或排程變更。[P4草案](../../tasks/spec-p4a-colab-20261121.md) 與 [機器候選契約](../../tasks/p4a-colab-contract-20261121.json) 均DRAFT；正式入口拒絕DRAFT，沒有研究結果。Human提供筆電熱降頻與Colab授權背景，不是本輪runtime observation。

沿用research_lab訓練器、P3 split／row checks，新增 [bundle](../../V6/research_lab/p4_bundle.py)、[Colab executor](../../V6/research_lab/p4_colab.py)、兩層width64 GRU及可選checkpoint observer。[操作指南](../../docs/operations/colab-p4.md)說明safe-extract、逐檔SHA、exact lock預編譯wheel、rankgauss重建與tag驗證、三臂2epoch×100步speed-test、Drive fit-level續跑／BEST權重及至少25%餘裕雙上限帳本。CU依Human保守費率估算；85°C直接STOP、不等待，未知中斷區間保守扣帳並鎖STOP。partial checkpoints不自動resume或升格best。P3凍結契約、帳本、歷史結果不修改；新增source會改變現checkout的provenance，既有P3 evidence仍須使用原pinned source，不能重標。

第一輪 `job-run-mwqe7o12-d33ef8` fixer 以 `8cb61c4` 為 base 進行審查修補，已由後續流程保存為 `05cb840`，仍 UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED。修補 speed 全輸出隔離、解壓資源 gate、executable-cache 排除與 pre-import 清理、completed-fit 必要 schema。規格保留 OPEN 研究決策，已曝光 2021–2025 只作描述性；GPU／ABI／Drive 持久性仍 UNVERIFIED，CPU 工程完成不授予 Colab 或研究接受。

第二輪 `job-run-mwqew35j-886363` 在既有 detached HEAD `05cb840` 修補 raw PAX／GNU header 資源預檢、每次與累計解壓上限、Linux 全路徑 dir_fd／O_NOFOLLOW／O_EXCL 寫入及 bundle 路徑白名單。外部 SHA 必須由 Human 在可信環境記錄並在 Colab 重核；SHA 失守則 bundle 不可信。此輪依 no-commit 指示保留 patch，UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED。最終全套 CPU：**646 passed／1 skipped／4 warnings（161.32s）**；四項記憶體變異皆被殺死。詳見[工程報告](../../reports/v7_p4a_colab_engineering.md)。Next：獨立複審本 patch；Human／Claude 凍結仍 OPEN 的研究決策，後續另行現場驗證 wheel／kernel／Drive。CPU 測試不授予 Colab／正式研究 acceptance。


E2 RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED。24 次官方 Mamba2／S-Mamba 風格擬合完成 21 次（14 小時預算用盡，帳本 STOP；缺 2026 折 mamba2 s29、crossstock s17／s29）。確認性 2021–2023 三折神經分數完整，但契約要求 24 次與完整 manifest，故只能做探索性評估：H-P3a（Mamba2 vs C0+）日均 −0.0154% CI [−0.0414%, +0.0084%]、H-P3b（S-Mamba 風格）−0.0240% CI [−0.0595%, +0.0046%]，Holm p .298，兩者「未證實」且方向偏負；十組 IC 配對差神經全部低於 C0+。結果與後續計畫：[reports/v7_p3_portfolio_evaluation.md](../../reports/v7_p3_portfolio_evaluation.md)。評估輸出與腳本在 ignored `.artifacts/p3/explore-21of24/`（SHA-256 `c607522c…4860`）。P3 正式記為未證實／21 of 24。未決：是否核准 P4a（同配方 GRU 對照＋對等調參＋保存檢查點）與 P4b（神經／C0+ 混合，描述性）。教訓：預算須含餘裕、訓練中不得提交 Git（source_commit 在 run identity）、停機監看須用 Python 本體 PID。

# MarketMamba Current State


## P3 修訂3 預算錨點候選 — 2026-11-20

`job-run-mwoibr4r-44988a` fixer，detached base `39320a04512c446fb75aa993690a7496e434e57b`；UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED。依[P3 修訂3](../../tasks/spec-mamba-portfolio-p3-20261120.md)，[adapter](../../V6/research_lab/rolling_seq_scores.py) 的全域 ledger 每次事件追加後原子寫入獨立 `budget-anchor.json`（sequence／head_sha256／spent／observed_at）；啟動完全核對錨點與 sidecar 前綴，拒絕 empty／valid_prefix／換 out-dir 的預算重置。fit sidecar 在 END 扣帳後發布，manifest 同記三元組。缺 ledger 或 anchor 而存在 fit／sidecar／run_identity 時拒絕，須 `--rebuild-budget --reason TEXT`；REBUILT 保留原始 state identities，以所有 sidecar 最大 spent 及存活可驗帳本／anchor 的最大值扣帳，再鎖 STOP 待顯式檢查／clear-stop。未知 legacy spent 不自動當0，舊 fit 不因重建升格。詳見[工程報告](../../reports/v7_p3_portfolio_engineering.md)。

本輪只有合成 CPU／破壞性回歸與 in-memory 反證；防疏失，不防刻意同時刪 ledger、anchor、全部 sidecar；UNRESOLVED / GPU_SMOKE_NOT_RUN，無正式研究結果。GPU、正式資料、模型、production authority／scheduler 不變。依 no-commit 指示保留 dirty candidate；Next：獨立審查 diff 與實際 logs，Human acceptance 仍未完成。

## P3 序列分數／組合工程候選 — 2026-11-20

`job-run-mwnv3018-83e5c9` fixer，detached base `a91f38e86dab0d332fc63c6dc8a6ab7361a19b99`；UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED，依指示不 commit/push/merge/GPU/新增 agents。[P3 修訂2](../../tasks/spec-mamba-portfolio-p3-20261120.md) 的[adapter](../../V6/research_lab/rolling_seq_scores.py) 新增 CUDA-only `--gpu-smoke`，由執行器量測官方 temporal／cross mixer 前反向與逐檔環境／source hashes；smoke 不自行審查。獨立 review job 完成後才用 `--record-review` 產生另一份 exclusive artifact，記錄 reviewed_by／standing_authorization，不接受 v1／human_reviewed 布林。[evaluator](../../V6/research_lab/portfolio_eval.py) 現在重新驗 evidence path/SHA、review、source/spec/environment、cfg/mixer、全域帳本與完整24fit sidecar/run_hash，不再信任手寫 RESOLVED。

全域 `.artifacts/p3/budget-global.jsonl` 含 hash chain，flock 鎖定 artifact 目錄 inode，所有 namespace／smoke 共用14h；fit/run_identity 與帳本不符即拒絕。STOP 事件及檔案先 fsync 再 END；刪檔不能清除 latch，必須 `--clear-stop <name> --reason <text>` 留 CLEARED，不重置 spent。硬中斷仍保守扣帳至重啟。`collect()` 是需 active ledger attempt 的內部入口。凍結補 dropout/window/normalize/train_probe_sessions，官方其餘預設從 signature 讀出；寫死常數由 source hash 重驗。P1/P2 default、purge、CI/Holm 與研究門檻不改。

**UNRESOLVED / GPU_SMOKE_NOT_RUN**：本輪只有 CPU／合成 E2E、反證與唯讀六折 metadata 測試，無官方 CUDA 量測／24 fits／研究結論；真實 GPU 環境未由本輪驗證。完整 RED→GREEN、實際全套數字、解鎖與 STOP 程序見[工程報告](../../reports/v7_p3_portfolio_engineering.md)。既有 per-namespace 歷史不得當成新全域帳本或重置預算；需要另行盤點遷移。Next：獨立審查本 diff／logs；本 builder 的 No GPU 範圍不執行後續 GPU 工作。正式 Data、V6/models、production/scheduler、研究 evidence class 與 Human acceptance 不變。

## P2 low-turnover result — 2026-11-20

Pre-registered P2 ([spec](../../tasks/spec-lowturnover-p2-20261119.md) with amendment 1, [report](../../reports/v7_p2_portfolio_evaluation.md)), source commit `fcfc0e6`, E2 RESEARCH_ONLY / NOT_DEPLOYED. Hold buffer plus 20-session rebalance cut annual turnover from about 15x to about 5x and cost drag from about 13% to about 4.5% for both C0+ and LightGBM (descriptive, six folds). Confirmatory tests on clean folds 2021-2023 remain unconfirmed: H-P2a (net excess vs proportional-cost benchmark) and H-P2b (vs P1 primary) have 95% CIs that cross zero. The P2 primary design was informed by P1 descriptive results on the same folds, so it is not an independent test. Minimum fee (20 TWD per order) costs about 0.6 points a year at 1M TWD capital and vanishes at 10M. Next: an unexposed-forward confirmatory test of the fixed P2 primary cell, adj_close/delisting evidence, and Mamba2/S-Mamba under the same portfolio protocol.

## P2 amendment 1 bounded repair candidate — 2026-11-19

本輪 `job-run-mwnpp2vy-e42792` builder，沿用 detached base `1920fde0abfead53e84a2a9354a695e59a03764c` worker；UNMERGED / NOT_DEPLOYED，Human 禁止 commit/push/merge/新增 agents。依 [P2 登記修訂 1](../../tasks/spec-lowturnover-p2-20261119.md) 修正既有 [engine](../../V6/research_lab/portfolio.py)、[evaluator](../../V6/research_lab/portfolio_eval.py) 與 [P2 tests](../../V6/research_lab/tests/test_p2.py)：P2 主要基準不收 min_fee，只收比例成本，另存也收最低費用的 descriptive sensitivity；策略成本不變。OFAT＋三指定組合去重為 14 格／fold×system（a/c 與 OFAT 重複），六折兩系統 168 格，另 12 個同分數 P1 references，共 180 cells。改 top_k 維持 buffer=2×top_k。凍結加選股數超過 top_k 時該次不新增／不強制賣出可成交持股、保留原股數；terminal liquidation 不變。P2 所有格明確啟用 preserve_frozen_overflow（含buffer=top_k），P1預設關閉，保留原浮點算式。子集 --fold／--system 的 grid_complete 必為 false，不進完整確認性。P2 schema／run_hash 與 P1 不同，舊 P1 parts 無法續跑；位元一致僅指預設路徑逐日策略與基準報酬。

指定全套 **474 passed／1 skipped／4 warnings（90.94s）**；P2 focused 28 passed／6.97s，初始 RED 8 failed／16 passed，補強相等buffer RED 1 failed後GREEN；額外 benchmark／Holm／golden mutations 均被拒絕後還原 GREEN。pre-P2 ace2eef 原始引擎獨立產生 29 日合成 golden；實際 2023 P1 reference 與舊 part 231 日／3,696 bytes 位元一致。2023/C0+ CPU 主要格冒煙 **109.417992s**（含啟動 110.895806s），2 COMPUTED cells／0 ERROR，221 主格報酬日；策略區間淨報酬 25.522813%，比例成本基準 33.948999%，日均超額 -0.000326306791；也收最低費用基準為 13.862428%，僅敏感度。H-P2a/b 都 INCOMPLETE_ENGINEERING_ONLY／未證實（缺 2021/2022、primary-only/subset）。修訂揭示原基準最低費用抬高淨超額，不把敏感度升格。

實測 P2 cell 9.182894s、P1 reference 11.158765s、其餘 90.554148s；完整 180 cells 線性外推單程序 29.45 分鐘，12 個 fold×system job／3 程序理想估計 15.35 分鐘。實測峰值 RSS 2.629967 GiB／程序，加20%裕度3.155960 GiB，三程序加2 GiB系統保留約11.47 GiB；六核不代表12GB可跑六程序。完整網格／多程序未執行，其他格、年份、I/O與累積RSS未知，不作時程上界或承諾。子集各自run_hash，無跨hash確認性彙整器。

GENERATED 本機 evidence `.artifacts/p2/job-run-mwnpp2vy-e42792/`；report SHA-256 `77acb7d28f6337345c0552d1ef08a8cec261e3db2d5aead1d711212679ee6394`，identities/logs/估時見 [工程報告](../../reports/v7_p2_portfolio_engineering.md) 與其 verification-final.json。沿用唯讀 full P1 scores，不重訓；Python3.12.3 pilot09 venv／CPU單執行緒／MARKETMAMBA_DATA_ROOT unset，data realpath `/mnt/d/Desktop/work/ProjectForMe/MarketMamba/Data/processed_v6`；未碰formal Data／V6/models、GPU、acquisition、production、scheduler、wakeups、recurring jobs。起始clean；成果依no-commit指示留核准範圍unstaged dirty，不宣稱clean。Knowledge-impact：基準／網格／frozen gate語義改變，已更新Current State、Authority Map、Portfolio／Research notes及既有 [Worktree Register](Worktree_Register.md)。knowledge_health.py --ci exit0／WARNING，無structural FAIL；syntax／artifact hash／link reference／git diff --check通過。下一步為主控獨立審查diff/artifacts並決定完整執行；工程完成不等於Human acceptance，E2 RESEARCH_ONLY，既有P1與修訂前歷史artifacts不修改、不升格。

## P1 portfolio-level evaluation result — 2026-11-19

Pre-registered P1 ([spec](../../tasks/spec-portfolio-eval-20261119.md), [report](../../reports/v7_p1_portfolio_evaluation.md)), source commit `ace2eef`, E2 RESEARCH_ONLY / NOT_DEPLOYED. Clean folds 2021-2023, primary cell (top 50, 10-session rebalance, 10bp slippage, raw prices): C0+ and LightGBM net excess over the same-universe equal-weight benchmark are both unconfirmed (95% CI crosses 0); C0+ minus LightGBM is also unconfirmed. Net cost drag (about 8-18 points a year at 14-18x annual turnover) is the main go-live risk; slower rebalancing lowered C0+ drag descriptively (needs new pre-registration). No adj_close, unverified delisting coverage, non-PIT market value. Next: pre-register a low-turnover study (20-session rebalance, hold buffer) and compare Mamba2/S-Mamba under the same portfolio protocol. Output 1.35 GB stays under `.artifacts/p1/` (not in Git).

## P1 bounded evaluator follow-up — 2026-11-19

本輪 `job-run-mwmgvttf-837c02`，detached base `d2390b56d633627d942e6f4969771b1382744ba9`，既有 worktree `.worktrees/ruflo-workers/run-mwmgvttf-837c02/fixer`；UNMERGED / NOT_DEPLOYED，Human 明確禁止 commit/push/merge。只修改 [evaluator](../../V6/research_lab/portfolio_eval.py)、既有 [P1 tests](../../V6/research_lab/tests/test_p1.py) 與本節／Authority Map；根目錄 `tests/test_p1.py` 不存在，未建立重複測試樹。重用 frozen [P1 spec](../../tasks/spec-portfolio-eval-20261119.md)，不改 fitting、成本或容量算式。

缺價診斷以 finite positive close 為有效價，零量不等於缺價。依 fold 範圍分列 `pre_fold_terminated`、`in_fold_terminated`（末次有效價在 fold 內，之後連續至少 10 個觀測 session 無價）、`transient_gap`（恢復／不足 10 日／無已知歷史有效價，後者仍 UNKNOWN）。穩定無缺價代碼不進異常清單；只對 in-fold 類別附持有／封鎖次數與價值，舊 `count/codes` alias 也只代表 in-fold。保留 window 前最後有效價 metadata，包括 fold 中已完全無列的舊代碼；這不是補價或正式下市證明。Calendar 仍為 observed union，非合格 exchange calendar。容量欄位與 caveats 明示 `approximation`、NAV=1 名目金額未隨 NAV drift 調整。

`volume_units` 保存 2330、OTC 大型參考股 8069、小型參考股 1236 的實際樣本／年度量級檢查，以及 2,490 股全宇宙 5,378 個觀測日的 raw close×volume 分位數。三參考股有效 rows 為 5,377／5,308／5,370；原始量中位數 37,103,997／7,978,000／45,000。全宇宙每日成交值 p05/p50/p95 為 46,786,720,736.51552／102,778,450,957.32242／622,192,728,217.9954 TWD（shares 假設）；實測 checks 一致支持 shares×1。缺探針、UNKNOWN 或跨參考／年度量級不一致都 fail closed。探索時 6488 的 2014–2016 初期量級使大型股成交值探針出現 lots/UNKNOWN，不能據此斷言單位混雜；最終固定使用較長歷史的 8069 與相同 volume 門檻，未放寬 gate。量級支持不是每列來源、全資料同質性或 PIT 證明，固定參考股也屬 hindsight 診斷選取。

Part 同時驗 run identity、payload SHA-256、fold/system envelope 與每列 fold/system；改檔名搬移其他 fold/system 會拒絕。預設仍保留 ERROR，caveats 說明其可能暫時性。明確 `--retry-errors` 只重算 ERROR parts 中的 ERROR cells，保留成功 cell、原始 part 與原報告，追加 hash-linked `-retry-000001.json` attempts／報告。完整驗證的連續 attempt 鏈擁有最新恢復狀態，缺 base、缺中間 attempt 或 hash 損毀都停止；不升格正式結果。成對描述性差異分 era 列出 paired folds、requested folds 與 missing primary folds/systems，缺任一主格即 `INCOMPLETE_PAIRED_SUPPORT`。

最終指定全套測試 **446 passed／1 skipped／4 warnings（146.57s）**；warnings 均為既有 stale-audit mutation 的 pandas API deprecation。逐項 RED→GREEN：缺價 6 failed→7 passed、容量 1→2、多檔 volume 7→10、part fold/system 2→2、missing paired fold 1→2、ERROR retry 1→1。另 7 個 contract/seed/matrix mismatch、retry chain corruption/missing/orphan、publish 後實際 SIGKILL/restart 測試通過。metadata 隔離新增 RED 1 failed，與缺價／retry 回歸 GREEN 13 passed；歷史 map 僅進診斷，不傳播至 engine pandas 操作。

CPU 2023/seed17/train500 冒煙只訓練一次：每系統 390,606 eval rows、合計 781,212 rank rows，export 111.36s。初次 evaluator 過慢，exit 130 中止，未發布 part／report；隔離診斷 attrs 後，重用原 immutable scores，不重訓，evaluation 145.54s 完成 2 parts／8 COMPUTED cells／0 ERROR。兩系統主格 2023-01-03～2023-12-20，共 231 returns；pre-fold 301／in-fold 16／transient 345，16 個 in-fold 代碼持有／封鎖次數與價值皆 0。清單包含全 raw universe／歷史代碼，不限 scored candidates，不代表正式下市宇宙。2021–2023 paired 只有 2023，明列 2021／2022 缺兩系統，仍 `INCOMPLETE_PAIRED_SUPPORT`，20 bootstrap samples 僅工程 smoke。

Ignored `.artifacts/p1/job-run-mwmgvttf-837c02-2023-s17-r500/portfolio.json` SHA-256 `03b9269d2e9fd0677baf3c1a4bf900b0d14f348ef892f151c6653af01580c8b1`；同 namespace `verification.json` 保存 source/input/output identity、實際 RED/GREEN/full-suite logs 與 interrupted attempt。2 part payload hashes、fold/system、score manifest outputs、輸入未變與當前 implementation hashes 已重新核對。Python 3.12.3 pilot09 venv、CPU（smoke CUDA_VISIBLE_DEVICES 為空）；MARKETMAMBA_DATA_ROOT unset；本 worktree Data 為普通目錄，實際唯讀價格仍從 `/home/frank/projects/MarketMamba/Data/processed_v6` 解析至 `/mnt/d/Desktop/work/ProjectForMe/MarketMamba/Data/processed_v6`，價格 SHA-256 `5df865ee9d57acf4d77de4e716ec3e2dd94dc8afae1869ec8adfdcb0bd10fc14`。無 GPU、資料 acquisition／修改、正式權重、production／scheduler 變動；未 commit/push/merge。

Knowledge-impact 檢查因 canonical map 變更提示 REVIEW_REQUIRED／WARNING；已檢查 GOVERNANCE，artifact 分類／流程不變，毋須修改治理規則。Current State／Authority Map 已更新；knowledge health 為 WARNING、無 structural FAIL（既有 authority repin candidates／legacy drift 保留）。工作樹只留上述 4 個核准檔案的未提交修改，起始無其他 dirty/untracked/staged 狀態。Blockers：本輪工程工作無；Remaining／Next recommended action：獨立檢查此 diff 與 verification evidence，再由 Human 決定 acceptance；完整 registered study 尚未執行。E2 RESEARCH_ONLY，不是 Human acceptance。


## P1 review repairs — 2026-11-19

Bounded builder repair at detached `abf0368d709b83bc916144292da8b5930cc0ff26`, existing `.worktrees/ruflo-workers/run-mwmgdq5i-f34fa7/fixer`; UNMERGED / NOT_DEPLOYED, no commit authorized. [Evaluator](../../V6/research_lab/portfolio_eval.py), [engine diagnostics](../../V6/research_lab/portfolio.py) and [tests](../../V6/research_lab/tests/test_p1.py) now persist immutable fold×system parts with input/implementation/parameter identity and payload hash; resume skips matching parts, refuses mismatch, and preserves cell ERROR type/message including nonfinite results. Final aggregation alone can promote candidate_confirmatory cells after the full clean-fold gate. Paired C0+ minus LightGBM inference separates 2021–2023 and 2024–2026.

Reports identify actual evaluation dates/return counts, early valid-price termination codes and last dates, held/blocked mark counts and normalized values. Termination cause remains UNKNOWN, not verified delisting. Caveats explicitly retain tail truncation, net-to-net benchmark costs and bootstrap blocks crossing fold boundaries. Actual read-only prices_raw unit evidence: 2330 5,377 rows, raw Volume median 37,103,997, latest sample 45,159,352 on 2026-11-18; magnitude and fetcher shares contract support shares (multiplier 1), not row-level source verification. Lot input is converted to shares before both liquidity gates and capacity; UNKNOWN units stop evaluation. Local smoke namespace `.artifacts/p1/review-2023-s17-r500/` reuses the existing P1 artifact contract without overwriting older evidence. Full registered study and Human acceptance remain outstanding; E2 RESEARCH_ONLY engineering evidence only.

Repair verification: focused review GREEN 10 passed; original behavior RED 9 failed, with direct old-evaluator exception/NaN RED 2 failed and stratum-mask/lot-conversion mutations rejected. Requested full suite: 419 passed, 1 skipped, 4 warnings / 70.11s. 2023/seed17/train500 smoke exported 390,606 rows per system (781,212 ranks); after SIGKILL (-9) with one durable part, resume skipped exactly one and computed exactly one new part, preserving original bytes/hash/mtime, then produced 8 cells/0 ERROR. Main cells span 2023-01-03–2023-12-20 / 231 returns; 19 early-termination codes each, zero held/blocked exposure for those codes in this smoke. Report SHA-256 `4930224ad55424f528e3f7ceea2b33d2ba933acd2e5b035ec6c49035d76f4eb7`; local `verification.json` pins resume and report evidence. Python 3.12.3 pilot09 venv, CPU; MARKETMAMBA_DATA_ROOT unset; prices realpath `/mnt/d/Desktop/work/ProjectForMe/MarketMamba/Data/processed_v6/prices_raw.parquet` (read-only), SHA-256 `5df865ee9d57acf4d77de4e716ec3e2dd94dc8afae1869ec8adfdcb0bd10fc14`. No GPU, data acquisition, protected weights, scheduler or production change.


## P1 portfolio evaluation builder candidate — 2026-11-19

Bounded Human-approved builder task at detached base `a0e1f00b70057b7e3fd17a8e4a5fb1f65c3acea1`, worktree `.worktrees/ruflo-workers/run-mwmfoyei-243923/builder`: [registered P1 spec](../../tasks/spec-portfolio-eval-20261119.md), [score exporter](../../V6/research_lab/rolling_scores.py), [portfolio evaluator](../../V6/research_lab/portfolio_eval.py) and [regressions](../../V6/research_lab/tests/test_p1.py) are uncommitted candidates, UNMERGED / NOT_DEPLOYED. Existing `c0plus.run`, rolling split/purge, 60-session eligibility and `portfolio.run_backtest` are reused; `rolling_runner` outputs are unchanged. Optional portfolio universe eligibility changes selection/benchmark membership only, preserving sellability of existing holdings. New thin CLIs are needed because existing IC-only rolling output discards per-row scores and the existing backtest CLI does not orchestrate the frozen grid/strata/capacity study.

Default dry-run, explicit `--run`, fresh ignored `.artifacts/p1/` namespaces only. Export requires existing read-only `stock_idx.npy`, uses unchanged fitting settings and per-date seed rank averaging. Portfolio grid has 432 parameter/basis cells per system/fold; primary and three lagged market-cap strata are separately identified. `--primary-only` is incomplete engineering smoke evidence. Confirmation requires clean folds 2021–2023, all three registered seeds, registered training cap and full grid; other folds are descriptive. No new evidence admission or trading readiness. Input/source hashes and environment accompany artifacts; absent adjustment data makes total_return cells unavailable with explicit primary raw fallback. Market-cap shares remain NON_PIT despite lagging; delisted coverage and exchange-limit flags remain unverified. Full training/full grid and independent Human acceptance remain outstanding. No data, protected models, GPU, scheduler or production changes; no commit/push/merge authorized in this task.

Builder verification (not Human acceptance): requested research suite **372 passed, 1 skipped / 71.24s**; four in-memory causal/accounting mutations each produced exit 1 and one failing regression. Local ignored evidence is `.artifacts/p1/smoke-2023-s17-r2000/verification.json`; its final report is `portfolio-scored-universe.json`. The earlier portfolio diagnostic predates scored-universe benchmark alignment and is retained unchanged. Smoke: 2023 / seed 17 / train 2,000, 390,606 evaluation rows per system, 781,212 rank rows, 91.75s export and 65.23s final evaluation. Main net annualized C0+ 11.15% / LightGBM 10.87%; scored-universe benchmark period return 31.24%; neither incomplete smoke confirms benefit.

Prices: 0 missing predicted keys, 39 missing/nonpositive opens, 15,634 absent code-date grid cells; 31,254 price rows lack prior-session cap. Coverage reasons/delisted support remain unverified. Observed-session terminal semantics end this main smoke on 2023-12-20 (231 returns), while scores extend through 2023-12-29 (239 sessions); no tail extension. Evidence/source hashes retain the evaluation-only alignment change; fitting inputs/exporter are unchanged. Full study remains outstanding.

## Research code import from pilot09 — 2026-11-19

Imported at file level (histories are unrelated) from pilot09 `rd/hybrid` commit `65d0b68da0c8b443dfb0a43b0220ef51614921ac` (also preserved on origin as `archive/pilot09-rd-hybrid`): `V6/research_lab/` (study code, rolling folds, C0+, Mamba2/S-Mamba-style models, repair and audit tools), `V6/experimental/v7_evidence_tier*` and `v7_portfolio_metrics*`, two root tests, and the roadmap/handoff/design notes under `tasks/`. NOT imported: pilot09 `V6/marketmamba` (this repo's fetch-hardening lineage is newer), `.agents`, `.claude*`, `V6/results` and app/tool changes. Verification here: 406 passed, 1 skipped, 8 subtests (pilot09 venv, `PYTHONPATH=V6:V6/research_lab/tests:.`). Large data (12 GB `.local-data`: E5 matrix, rolling folds, backups) stays in pilot09 as EXTERNAL_LARGE_DATA; any use must record path and hash. Prior findings (evidence class E2 RESEARCH_ONLY, simulated labels, survivorship bias): C0+ beats LightGBM on clean folds 2021-2023 (+.0207) but the gain decays (2026 about -.005); Mamba2 and S-Mamba-style models are statistically indistinguishable from LightGBM and show no gain from cross-stock mixing. Status: UNMERGED_TO_PRODUCTION / NOT_DEPLOYED; code presence is not production authority.

## Human policy update: V7 direction, version control, model split — 2026-11-19

Human authorized commit and push of branches and `main` to `origin` (no force-push/history rewrite; deploy, scheduler changes and paid Colab/GPU still need separate approval). V6 is legacy: no dual V6/V7 maintenance; the new system ships as V7. V6 daily fetch remains the data supply until a V7 data pipeline replaces it. Codex is the default executor for routine work; Claude is for planning, independent review and cross-checking ([AGENTS.md](../../AGENTS.md)). Local CPU/RTX 3060 first, Colab only for runs that cannot finish locally. Short-term goal: bring a pure-quant V7 model online, chosen by evidence (Mamba2 + S-Mamba is a hypothesis, not a result). Long-term: fundamentals, news, supply-chain relations. Prior research repo `/home/frank/projects/ruflo-evaluation/pilot09/marketmamba` (`rd/hybrid`, unrelated history) is read-only reference; its results are not current evidence here until imported with identity. Local evidence class: POLICY / DEVELOPMENT_ENGINEERING_ONLY.


## Ruflo development integration — 2026-11-19

Human approved Claude-primary background work, persistent memory and Claude/Codex workers. Branch `development/ruflo-integration`, base `f501a7eb28fab26c86f1bc776ad80a0ff13cfe81`: UNMERGED / NOT_DEPLOYED. [Project contract](../../tools/project_agents/integration-contract.json) and [operator guide](../../docs/operations/ruflo.md) define explicit submitted development jobs, pinned Ruflo/CLI 3.77.1 and official Codex adapter 3.0.8. Reuses official DualModeOrchestrator; project adapter owns process/exit/artifact verification, not a second MAS. One active job, at most3 workers/1writer; no periodic schedule or automatic retry. Memory is non-authoritative, explicitly shared at Git common root `.artifacts/ruflo/memory.db`; formal data/models/research/production boundaries unchanged. Local evidence is DEVELOPMENT_ENGINEERING_ONLY. Bounded acceptance: 21 offline agent tests +23 governance tests; real Claude entry submitted job run-mwmdmets-74c208, Codex fixture worker and independent Claude reviewer completed; 27-byte snapshot/hash checked. [Acceptance manifest](../../reports/ruflo_integration_acceptance.json) records source/outputs/limits. MCP↔CLI durable SQLite readback PASS; AgentDB bridge disabled after observed missing native binding/non-durable write. Local primary checkout uses development/ruflo-primary; main/production not merged/deployed.


## V6 fetch date binding and internal gaps — 2026-11-18

`fix/macro-sync-fetch-hardening`, WSL primary checkout, base `bc88a181a15ad20e41daf29ec68a623d99ccf5d8`:
UNMERGED / NOT_DEPLOYED. [Fetcher](../../V6/marketmamba/data/fetcher.py) now
rejects absent, invalid, mismatched or conflicting response date/title evidence
for BWIBBU_ALL, TWT96U and TWSE/TPEx daily institutional endpoints.
Margin/daytrade/generic/market-value catch-up reuses `_trading_calendar()`
(prices_raw distinct dates). The follow-up to `721eb84` bounds the scan start to
`max(first stored date, today - lookback_days)` (default 30 calendar days;
non-positive lookback rejected), retaining the latest-10 missing-session cap.
A missing weekday today is always a candidate even without today's price calendar;
None/empty responses write no rows. Empty margin/daytrade sources bootstrap only
today; generic/market-value missing-file behavior remains unchanged.
Market value skips candidate gaps older than 10 calendar days with a warning
pointing to `V6/scripts/backfill_stale_202607.py` and its non-PIT share limitation.
It reports only dates actually written; missing prices cannot create market value.
This supersedes `721eb84`'s unbounded historical scan and calendar-only today
selection, not its response-date gates. Whole missing sessions only: incomplete
stock coverage, contaminated existing values and price/institutional historical
catch-up remain outside this patch.

[Offline regressions and in-memory mutation runner](../../V6/scripts/test_fetcher_date_gaps.py):
16 gap/date tests + 8 official parsers + 3 raw append + 3 fetch-only = 30 pass
(original 23 retained, two expectations updated for the requested new semantics).
New regressions against `721eb84`: 17 assertion failures, 0 runtime errors across
16 methods. 13/13 in-memory mutants killed by assertions, 0 runtime errors.
Fetcher SHA-256: `0074a54c31b031accf9d4f3542d9ccebe6ef487ec45d7659ad6d3b7204c0b0ec`.
Evidence class: SYNTHETIC_OFFLINE_ENGINEERING, not actual source validation or PIT
admission. All data writes are temporary and HTTP mocked; no formal data or
.env/credentials accessed. Python 3.12.3 environment:
`/home/frank/.venvs/marketmamba-fetch`, pandas 3.0.2, pyarrow 24.0.0;
MARKETMAMBA_DATA_ROOT unset; formal symlink resolves to
`/mnt/d/Desktop/work/ProjectForMe/MarketMamba/Data/processed_v6` (not written).
Current MOPS shares remain non-PIT even within the 10-calendar-day limit; capital
changes can bias recent historical values. No existing historical rows repaired.
Next: review branch; any formal-data remediation/deployment needs separate Human
authorization. No production authority, scheduler, model, push or merge change.

## Run status count patch adoption — 2026-11-13

Human-authorized independent adoption on `feat/run-status-count-validation`,
worktree `.worktrees/run-status-count-validation`, base
`8b0786018d763942d0ea68eb9ce537cb33f8b1fd`; LOCAL_MAIN_INTEGRATED / NOT_DEPLOYED.
Human-authorized fast-forward integrated `e06fb256211f21612a25af4b10f709a756540d32` into local main.
The feature worktree is retained, clean at that commit; retired N/A.
The original MAS delivery patch (2,542 bytes, SHA-256
`d8c25509dab82400f212e803a8c44a92d5b5d854bf926c8b41f0e60f898d20ad`)
changes only `runStatusSummary.mjs` and its test: non-negative integer counts
retain localized formatting; invalid counts display an em dash without coercion.
Four existing Node test files: baseline 25/25; patch regression RED 5 pass/2 fail;
complete patch GREEN 28/28, no skipped tests, using `--test-isolation=none`.
JSON fixture `research/v7/portfolio-history-artifact.json` is readable (7 events,
SHA-256 `d7fd650132e1261a7fb5275cf44c7481b9f86f226253f35f8dd49003953035d6`).
Original MAS goal `19459dd37dab44748c88a69b469eccad` evidence/state
remain untouched; historical Git Delivery remains incomplete. No push/deploy.
Local integration is complete; no further action is authorized by this adoption.

> Status: `CANONICAL_CURRENT_PROJECT_MAP`

## Window37 W34 Read-only Model Integration — VERIFIED CANDIDATE

Main72d35e7/W34 source82cd49c diverged atc57ac3d. Isolated `integration/v7-window37-w34-models` ports only readonly loader and original scores function, plus byte-identical model/recipe/feature/replay evidence. No new Fit/GET, no old training/fetch entrypoints, W35 Recovery unchanged. W34 original161files+restore verified; necessary JSON model/scaler/recipe/serving-schema pins retained. [Integration](../../research/v7/window37-integration.json), [adoption](../../research/v7/window37-model-adoption.json). Independent Gate GO Critical0 Required0；2models tests＋15W35 regression PASS，512原分數exact Replay；Human已授權必要驗證後main ff-only，實際整合identity由W37 closure補記。
W36 remains separate protected branch b2bc041; no Task/runtime/artifact/budget mutation. Its limited tasks are independently active under HumanW36 only, not a Shadow authorization. W34 models OFFLINE_VALIDATED, no Production/StrictPIT admission. BACKUP_PENDING deferred by Human; no independent backup claim. W37 manual Shadow implementation only after integration, no schedule or waiting; current Nov2 night is not Preopen.

## Window35 Five-session Collector Closure / Recovery v0.4 — COMPLETE / STOP

W35 branch `research/v7-window35-recovery-v04`，base main `c57ac3d7822f229ae08849e56581ebb3c969441c`，UNMERGED / NOT_DEPLOYED。W19cf1與main分岔於87695，無合併/Runtime移植；main既有verified_record/QC/snapshot/restore可驗證既有scheduled records。W34僅exactGit82cd固定64ID只讀，無模型載入/修改/整合。
五日5/5runs、10/10sources、原10GET證據確認12031rows/11814basicPASS/217missing/Fatal0。完整Raw TWSE72 vs69exceptions、TPEx145 vs100，補48明細；121個market/security缺1/2/3/4/5天69/23/17/9/3。217全部OUT_OF_W34_TWSE64_UNIVERSE；cause仍UNKNOWN，已確認合法無價/停牌0、SourceFailure0，不因持續性或商品樣式推定。64股每day64有效OHLCV/兩Featureinput，共320，0RankingScores。
Task唯讀觀測2026-11-02T21:27:33+08：Ready/enabled/last19:00:01result0/NextRun=null（不據此稱停用），原End23:59:59觀測時未到，EXPIRING_PROTECTED_EVIDENCE。W19Task/政策/10GETbudget/135原檔hash均保留未改；不延長或重啟。
最小v7_data_recovery.py僅offline分類/決策/immutable claims/outcomes；resource/session整份CSV計數非per-stock，1content與候選2network retries分開，offline physical參考cap4，正式營運Retry參數未凍結。來源停止跨Session保留，wrongSession/sameContent/stillMissing停止，Backfill不冒充Forward。
主要Review原NO_GO/Required1保留；R1 failure-parent retry分叉已單輪修復與[Independent Closure](../../research/v7/window35-review-closure.json) GO Critical0／Required0。15tests PASS；同event冪等、每失敗parent一次、下一retry只能跟新failure。原replay保留，postrepair獨立[Replay](../../research/v7/window35-replay-closure.json) 重現12031/217與Synthetic修復/limit-stop，0GET/Fit/GPU/Taskchanges。
10個W19只讀replay、41offline檔9409116bytes/份restore，135原W19files未變；BACKUP_PENDING，同機非獨立備份。[Evidence](../../research/v7/window35-evidence.json)、[完整Missing](../../research/v7/window35-missing-distribution.json)、[Report](../../reports/v7_window35_collector_recovery.md)。
[Activation Candidate](../../research/v7/window35-activation-candidate.json) 僅PROPOSED：Normal19／Conditional07、Interactive限制/原W19期滿隔離/新限期Task/保存備份監測。真正live Normal/Recovery dispatcher仍未移植接合，不能直接啟用；下一Human最少批准W35整合、有界live adapter/驗收及限期操作設定、calendar/retry/update-evidence與獨立備份位置。
恢復：W35 worktree → Current_State → 唯一Backlog → window35-evidence.json → closure/report。無剩餘本輪READY，STOP；StrictSTOP/FormalAdmissionNOT_APPROVED/W34OFFLINE_PREPARATION_ONLY/W12SAFETY_HALTED。不得自整合W35/新GET/長期Task/Shadow/下一窗口。

## Window33 Recent Data & Initial Shadow Model — DATA PRESERVED / FIT BLOCKED / STOP

W32 fixed `8fa41869a2583f0b5db698db83eed98611f24a7e` 已由main67e2d25本機ff-only整合；Review GO/11Synthetic原檔及restore/13tests/治理核對。W33 branch `research/v7-window33-recent-shadow`，base同固定SHA，worktree `.worktrees/v7-window33-recent-shadow`；UNMERGED / NOT_DEPLOYED。GET前91d7cca固定W29同64檔/日期/guard。
64 FinMind price GET全200、0metadata/retry/token/fallback，7604573rawbytes；最小間隔5.001249s、最高known prior hour63<100，額度CLOSED/剩6不重用。43437rows，2024-01-02～2026-10-22、679OBSERVED sessions（242/243/194）；58zero、19union-relative missing，不補值或推定停牌。
[Data](../../research/v7/window33-data-evidence.json)：43319數值Feature候選，5d42720／10d42173數值Label候選；近期Calendar沒有合格官方/例外休市/時鐘證據，W07回CALENDAR_QUALIFICATION_UNKNOWN。全部新mask fitFalse；未因Vintage未知全面拒絕資料，也不把價格union升成qualified。8422於2025-11-17疑似單位不連續，事件原因UNKNOWN，沿用既有flags不修價。
[Compatibility](../../research/v7/window33-compatibility.json)：W16/W18同Oct22原始CSV；64檔OHLCV精確相等，官方current＋明示FinMind prior的兩Feature hybrid全相等；非連續兩官方Snapshots獨立parity。Feature order/shape2保持。
實際Training N0／Ridge0／LGBM0／Offline Scores NOT_RUN_DEPENDENCY_BLOCKED；沒有新模型、Prediction或RankIC。Prediction Replay N/A，不能冒充完整Shadow準備成功。W30Recipe與W27 safety不變；2024–2026 EXPOSED。
264files58910660bytes/份獨立程序Raw/Derived restore，68old rawpins未變；CPU28.817576＋60reserve<1800，artifact+restore117822087<1GiB。Private `.artifacts/window33-recent-shadow/`／`.artifacts/window33-restore/`，BACKUP_PENDING同機非獨立備份。
8targetedtests PASS；獨立Review GO（資料保存與阻塞收尾）Critical0／Required0，0輪review rework；見window33-review.json，不代表訓練Gate已通過。[Evidence](../../research/v7/window33-evidence.json)、[Report](../../reports/v7_window33_recent_shadow.md)。初期interpreter/import/probe shape失敗trace與CPU計帳保留，無額外GET或輸出覆寫。
下一Human：有界補足近期合格calendar/休市證據，或批准另具身份的觀測Calendar代理語義；不得只改qualification字串。解除Gate後另授權固定4fits及Offline驗收；真正PREOPEN另需模型/source/time/missing/recovery及操作授權。本輪STOP，不自動續期Collector或開下一窗口。
恢復：W33 worktree → Current_State → 唯一Backlog → window33-evidence.json；不得重跑已CLOSED source。StrictSTOP/FormalAdmissionNOT_APPROVED/W19PROTECTED無任何讀寫/W12SAFETY_HALTED/E1BLOCKED/H2NOT_APPROVED，無Scheduler/GPU/Push/Deploy操作。

## Window32 Expanding Closure & Prospective Shadow — PREPARATION COMPLETE / STOP

W31 fixed `67e2d25a710273646d41c0ad861b4a993593ba4d` 已由main b64ac35本機ff-only整合；72原檔/restore、24views/24model identities、maturity及10tests核對。Refit−Train−Validation差額5d300/305/310、10d597/610/620均為TRAIN邊界排除後已成熟的舊labels，非Forward。未重Fit/重算舊RankIC，W31歷史Warning保留。
W32 branch `research/v7-window32-shadow-preparation`，base同固定SHA；worktree `.worktrees/v7-window32-shadow-preparation`，LOCAL_INTEGRATED_FIXED8fa4186 / NOT_DEPLOYED。
[Shadow Contract](../../research/v7/window32-shadow-contract.json) 為PROPOSED；正常目標07–08Asia/Taipei，持久發布嚴格早於合格Entry開盤才PREOPEN，晚到MISSED_PREOPEN/RESEARCH_ONLY。未凍結精確source freeze/wait/missing政策或啟用排程。
Synthetic-only薄入口重用W07時鐘/日曆、W09 strict JSON/lock/safe path及既有atomic-no-clobber；無真實模型載入、Fit或Inference CLI。Feature-only keys先保存，5d/10d成熟後才Join，Backfill不冒充Forward。13targetedtests PASS；11個Synthetic檔16123bytes/份同機restore；BACKUP_PENDING。
[Readiness](../../research/v7/window32-readiness.json)：已核immutable訓練資料僅2021–2023，不能宣稱2026FreshRefit就緒。W31兩Feature schema可重用，Daily CSV尚需units/security/market/feature parity及合格Universe/Calendar/events；V6的56/59維Inference不可直接沿用。
W19僅讀固定Git policy物件作參考，沒有Task/worktree/artifact/runtime存取或續期；五次單日snapshot不能取代多年訓練資料，實際Task狀態未觀測。W06/W12/W19均未整合。
[Evidence](../../research/v7/window32-evidence.json)、[Report](../../reports/v7_window32_shadow_preparation.md)、window32-review.json GO Critical0／Required0，0輪review rework。下一Human：初始資料/事件/Label准入與限額FreshFit、Shadow時間/missing/evaluation契約、模型pins、起訖/環境/儲存備份/監測及操作授權。無剩餘已授權READY；STOP，不產生真實排名或開下一窗口。
恢復：W32 worktree → Current_State → 唯一Backlog → window32-evidence.json；只讀檢視，不啟用任何Collector或Shadow。StrictSTOP/FormalAdmissionNOT_APPROVED/W19PROTECTED/W12SAFETY_HALTED/E1BLOCKED/H2NOT_APPROVED；InferenceRecoveryImplementation仍DEFERRED。

## Window31 First Expanding CPU Development — COMPLETE / STOP

W30 fixed `b64ac353833a9248121f12ebe61e1333decb2dbf` 已由main e83dfe6本機ff-only整合；38pins/restore及6mock-only tests通過。970 daily＝940 defined＋30 None；2910 paired＝2820 defined＋90 None；原模型351/379 split nodes、預測非退化，未重Fit。見 [Integration](../../research/v7/window31-integration.json)。
W31 branch `research/v7-window31-expanding`，base同固定SHA，worktree `.worktrees/v7-window31-expanding`；LOCAL_INTEGRATED_FIXED67e2d25 / NOT_DEPLOYED。首次Fit前 `ddcf5d835b43578ed9fc5567ee677d88e928e203` 固定 [Contract](../../research/v7/window31-contract.json)／[Prepared](../../research/v7/window31-prepared.json)。
三Origin2023-01-03／04-06／07-03，Forward54／59／63sessions。5d TRAIN14113/17559/21321，REFIT29229/32574/36276；10d TRAIN13711/17143/20890，REFIT28743/32046/35715。持有窗跨界/未成熟Label排除，額外purge/embargo0不是E530sessions。每Origin Validation Fit與Fresh Refit不同run/scaler/state；Validation診斷後refit，全部Forward評分延至三Origin完成，無Forward-feedback tuning。
Ridge12fits＋LGBM12fits已達本輪上限；L0±固定。453192完整Feature-only Predictions先保存後Label Join；7264daily/10896paired，含Validation明確None尾端不當0。Forward352個session×horizon全部可算，Daily N59–64，min40未降。
Forward Ridge5d O1/O2/O3 Mean .028606/.035161/.024427；10d −.012394/.008584/−.002463。LGBM5d .023592/.058135/.065496、10d −.021556/.026531/.000766。全部為EXPOSED DEVELOPMENT，不選winner/不宣稱獨立統計或收益。
10純數值/mock tests PASS；獨立程序Replay453192pred/7264daily/10896paired完全吻合，0追加Fit。72files91000410bytes/份同機Restore，310 prior W29/W30pins未變；CPU87.690630＋60reserve<1800，artifact+restore182002029<512MiB；BACKUP_PENDING。
初始導航authority_class缺漏使healthFAIL；prefit shell未set-e，未攔住O1。已修導航，保留原FAIL與修後WARNING證據，數值code/contract未變；不可宣稱所有preflight從未失敗。獨立Review GO Critical0／Required0，0輪review rework；見window31-review.json。無額外API/GPU/W19或V6操作。
[Run](../../research/v7/window31-run.json)、[Results](../../research/v7/window31-results.json)、[Evidence](../../research/v7/window31-evidence.json)、[Report](../../reports/v7_window31_expanding_cpu.md)。W31當輪main保持b64ac35、成果未合併；W32已獲Human授權ff-only整合至67e2d25，無其他未授權整合。
下一Human：正式Dataset/Universe與UNKNOWN事件政策、Decision-time/Label/maturity/purge、selection/refit/evaluation/statistics、Exposure/prospective評估、運算/獨立備份。無剩餘已授權READY；不得重跑已用完額度的O1/O2/O3。
恢復只讀入口：上述worktree → Current_State → 唯一Backlog → window31-evidence.json；W29/W30 raw/models/restore、W11環境依賴保留。StrictSTOP/FormalAdmissionNOT_APPROVED/W19PROTECTED/W12SAFETY_HALTED/E1BLOCKED/H2NOT_APPROVED。STOP，不開下一窗口。

## Window30 Fixed Real-data LightGBM — COMPLETE / STOP

W29 fixed `e83dfe68e4bc73b59cb6390ec130ba967138e7f1` 已由main1d9a2a7本機ff-only整合；340pins/restore、4tests、ReviewGO核對。
W30 branch `research/v7-window30-real-lightgbm`，base同固定SHA，worktree `.worktrees/v7-window30-real-lightgbm`；LOCAL_INTEGRATED_FIXEDb64ac35 / NOT_DEPLOYED。
任何Fit前60e7ca7固定唯一Recipe/版本/native-lib SHA/feature order/seed/TRAIN view/planned ModelSpec IDs；weights SHA僅Fit後產生。
W11環境LightGBM4.6.0/NumPy2.0.2/SciPy1.16.3/sklearn1.6.1不變；L2/64trees/7leaves/depth3/lr.05/minleaf100/L2=1/maxbin63/seed0/thread1，無eval_set/earlystop/search。
W29 Cohort/Data/兩Features/Labels/Fold/support完全沿用；TRAIN14113/13711，最新exit2021-12-30；identity preprocessing，bin mapper只學TRAIN。
2個fresh LGBM fits各horizon一次，皆64trees，fit約.02777/.02791秒，modeltext52369/54849bytes；Ridge/API/GPU0。
60726完整Feature-only Predictions先immutable保存後Label Join；同W29 support/AVERAGE/min40，970groups940defined N58–64，30tail/maturity保留拒絕。
LGBM Mean/Median：2022/5d .018814/.021627、2022/10d −.012617/−.016793、2023/5d .021182/.026036、2023/10d −.013374/−.013470。
5d正但四組Mean均低於既有Ridge，10d兩年負；沒有改Recipe重Fit、選winner或批准Mamba2。2022/2023 EXPOSED，UNKNOWN actions/asof/vintage與Cohort bias仍保留。
6 mock-only tests PASS；獨立Replay60726pred/970daily/2910paired PASS，0Fit；1主要Review GO Critical0 Required0／0 review rework。
Reviewer不能只由stdout獨立重證shell timeout wrapper；已保存實際tool command/session/start/complete reference，未重跑Fit；OS CPU guard及實測fit保留。
11files34874071bytes/份同機restore、W29原299files hash未變；artifact+restore69749181<256MiB，CPU30.723458＋60reserve<1200；BACKUP_PENDING。
[Recipe](../../research/v7/window30-recipe.json)、[Prepared](../../research/v7/window30-prepared.json)、[Run](../../research/v7/window30-run.json)、
[Results](../../research/v7/window30-results.json)、[Evidence](../../research/v7/window30-evidence.json)、[Report](../../reports/v7_window30_fixed_lightgbm.md)、
[唯一Backlog](../../research/v7/research-program.md#window30-fixed-real-data-lightgbm-baseline)。
下一Human：Expanding origins/model-ready cutoffs/train/validation/forward/maturity/purge/selection/refit/evaluation與data/proxy/universe/events/exposure/CPU/storage上限。
固定CPU E2E基礎已驗，但W29 masks為固定split，各新Origin需重新核對maturity/purge，不能直接宣稱Expanding實驗已批准或認證；無剩餘已授權READY，STOP。
StrictSTOP/FormalAdmissionNOT_APPROVED/W19PROTECTED/W12SAFETY_HALTED/E1BLOCKED/H2NOT_APPROVED；不mergeW30/push/deploy/nextwindow。

## Window29 Expanded Cross-sectional Development — COMPLETE / STOP

W28 fixed `1d9a2a7a1ee57c455ba58250f63681efbbdd9050` 已由main39e2fed本機ff-only整合；35pins/restore、6tests、ReviewGO核對。
W29 branch `research/v7-window29-expanded-development`，base同固定SHA；worktree `.worktrees/v7-window29-expanded-development`；LOCAL_INTEGRATED_FIXEDe83dfe6 / NOT_DEPLOYED。
GET前746ca04固定68新Cohort（W24重疊0）：60pre2021 current issuers＋4periodlisting＋4delisted start/typeUNKNOWN；64檔進Prediction，仍HINDSIGHT/存活者偏差，非HistoricalUniverseVerified。
68真實FinMind price GET／metadata0／8387840rawbytes，全部HTTP200；單並發、實際起始間隔min5.001297秒，known-local100/hour（外部匿名用量UNKNOWN），unused預算CLOSED。
一次本機mutable counter修復：seq1保留不重送、seq2由Trace證明尚未GET；E1atomic_json只修counter，失敗CPU保守扣600秒未reset；不清除providerSTOP、不擴預算。
最小feature-only介面分離原W24未來Label gate；只讀當前/前一session和當下membership，Predictions先保存後Label Join；未來Label刪除/eligibility變動不能改PredictionUniverse。
48061raw rows／44899feature-only rows；940核心groups N58–64，min40未降；30tail/maturity groups/model N0保留拒絕證據，尾端Predictions仍保存。
原兩Features，L0正／反（反向post-W25 control）、Ridgealpha1兩次freshfit、TRAIN-only scaling，TRAIN N14113/13711。LGBM於GET前略過既有synthetic≤64/toy guard，沒有新Recipe或繞過。
5d Ridge Mean/Median：2022 .037333/.033014、2023 .047138/.050515，正向延續；10d2022 .012073/−.004079、2023 −.008580/−.002417，一致正向未延續。
不能宣稱跨Cohort泛化/優勢/正式統計成立：來源/Universe/support/protocol不同，2022/2023皆EXPOSED；UNKNOWN actions/suspension/asof/vintage保留。
182178predictions，3模型相同可評估Support；18targetedtests，獨立Replay Raw-derived feature/masks＋2910daily/2910paired PASS，0額外Fit。獨立Review GO，Critical0／Required0，1主要輪／0 review rework。
299files／74612661bytes同機restore；總artifact+restore149226490bytes<1GiB，CPU計帳647.740401秒（含failed600）＋60reserve<3600；BACKUP_PENDING。
[Plan](../../research/v7/window29-plan.json)、[Cohort](../../research/v7/window29-cohort.json)、[Coverage](../../research/v7/window29-coverage.json)、
[Run](../../research/v7/window29-run.json)、[Results](../../research/v7/window29-results.json)、[Evidence](../../research/v7/window29-evidence.json)、[Report](../../reports/v7_window29_expanded_development.md)、
[唯一Backlog](../../research/v7/research-program.md#window29-expanded-cross-sectional-ranking-development)。
下一Human：補強ordinary/historical membership與事件未知政策，事先固定新資料/新時間或prospective的5d primary/10dsecondary＋L0±/Ridge研究與精確資料/CPU/storage/backup上限；不因本結果調參或選勝者。
本輪無剩餘已授權READY，STOP；StrictSTOP/FormalAdmissionNOT_APPROVED/W12SAFETY_HALTED/E1BLOCKED/H2NOT_APPROVED/W19完全隔離。

## Window28 Clean Training Refit & Temporal Causality — COMPLETE / STOP

W27 fixed `39e2fedeb550a08d00f4abddc9a6caa7b1d570f8` 已本機 ff-only 整合（main前367005d）；35pins＋原計算source/restore、14tests、ReviewGO核對。
W28 branch `research/v7-window28-clean-refit`，base同固定SHA，worktree `.worktrees/v7-window28-clean-refit`；LOCAL_INTEGRATED_FIXED1d9a2a7 / NOT_DEPLOYED。
Plan/code/prefit先commit `ec0b072` 再執行；原W24兩Features／Ridgealpha1／W27Policy與Mask不變，2021各horizon只fresh fit一次，總2fits，0resume/search。
5d TRAIN 7338→7164、10d 7188→6841；mean/std僅在clean TRAIN重估並逐值驗證。2022/2023完全不參與Fit/Scaling。
同一W27 Eval support、940日期×horizon、N>=20未降：Ridge舊→新 2022/5d .033873→.034682、2022/10d .045471→.044561、
2023/5d .080008→.080407、2023/10d .053166→.054248；2022/2023皆EXPOSED Development，非模型選擇或獨立效能證明。
18個future有效OHLCV數值擾動案例，Features/固定模型Predictions exact-equal、當日positive control有效，Raw不改。
只驗數值因果性：既有feature wrapper也有未來Label資格gate，不宣稱row eligibility為as-of線上接口，亦非歷史publication/vintage認證。
原July1之2321不在W24資料；任何Fit前依原Mask第一共同合格日固定2022-07-18，不改Mask或樣本池。
6targetedtests；獨立Replay1880daily/940paired/29143newRidge scores/18cases PASS，Replay0Fit。獨立Review GO，Critical0／Required0，1輪主要審查／0 review rework。
227 prior private source identities核對；W28 11files／28170524bytes同機restore，BACKUP_PENDING，原W24/W27 models/predictions/metrics均保留。
[Plan](../../research/v7/window28-plan.json)、[Run](../../research/v7/window28-run.json)、[Results](../../research/v7/window28-results.json)、
[Causality](../../research/v7/window28-causality.json)、[Evidence](../../research/v7/window28-evidence.json)、[Report](../../reports/v7_window28_clean_refit.md)、
[唯一Backlog](../../research/v7/research-program.md#window28-clean-training-refit-and-temporal-causality)。
UNKNOWN actions/suspension/asof/vintage、hindsight/universe/已曝光仍保留；不宣稱完整污染已去除或投資收益。
下一Human：限定新資料/universe/事件未知政策及Label用途，凍結Exposure-aware Study/Fold/Recipe/evaluation與資料/CPU/儲存/備份上限；無剩餘已授權READY，STOP。
0 API/GPU/Colab/Bootstrap/LOSO/V6/W19操作。StrictSTOP／FormalAdmissionNOT_APPROVED／W12SAFETY_HALTED／E1BLOCKED／H2NOT_APPROVED。

## Window27 Pragmatic Label Safety & Frozen-Prediction Sensitivity — COMPLETE / STOP

W26 fixed `367005dfaba0988fca5e294d9b3e281408bbfce8` 已本機 ff-only 整合（main 前27055f7）；38 pins／3 tests／獨立 Closure GO。
W27 branch `research/v7-window27-label-safety`，base367005d，worktree `.worktrees/v7-window27-label-safety`；LOCAL_INTEGRATED_FIXED39e2fed / NOT_DEPLOYED。
Policy `W27_RAW_PRICE_DIAGNOSTIC_DEVELOPMENT_ONLY_v1`／Predicate／Mask 先在 `e77f32d` commit 固定，再重評既有 W24 預測。
原44266→新43148樣本；排除1118（Training521僅診斷，evaluation597），disjoint primary：confirmed event3／missing33／zero1082。
UNKNOWN event/suspension/asof 對所有樣本保留，不升格 CLEAN；新 Mask 不修改原 W24 mask，沒有 refit 或消除原模型 Training 風險。
四組原940日期×horizon仍全可計算；N>=20未降，Ridge均值：2022/5d .037016→.033873、2022/10d .043930→.045471、
2023/5d .080310→.080008、2023/10d .054056→.053166。這是 post-hoc development sensitivity，非正式模型選擇／獨立驗證。
2023/5d排除68筆持有路徑零價，沒有本輪已確認減資跨界；不得用三筆10d事件解釋5d。無 event-only 追加評分。
14 targeted tests PASS；獨立 Replay1880 daily／940 paired，原44266 Mask 由 raw 重建吻合；W24/W25/W26 218 prior byte pins 未變。
一次 import namespace 修正在任何 metric 前；Replay int-key JSON 比較問題以 canonical hash 修復，原計算source留存、calculate AST相同，metrics不覆寫。
獨立 Review GO，Critical0／Required0，1輪主要審查／0輪 review rework；新成果留候選。0 API／fit/refit／new prediction／Bootstrap／LOSO／GPU／V6或W19修改。
Private `.artifacts/window27-label-safety/` 與同機 `.artifacts/window27-restore/`；BACKUP_PENDING，不宣稱獨立備份。
[Policy](../../research/v7/window27-policy.json)、[Mask](../../research/v7/window27-mask-manifest.json)、[Results](../../research/v7/window27-results.json)、
[Report](../../reports/v7_window27_label_safety.md)、[Evidence](../../research/v7/window27-evidence.json)、[唯一Backlog](../../research/v7/research-program.md#window27-pragmatic-label-safety-and-frozen-prediction-sensitivity)。
下一 Human：更大研究 Universe/事件覆蓋/未知政策/Label路線、Fold/maturity/selection/refit/evaluation/exposure及精確資料/CPU/保存上限；無剩餘已授權 READY，STOP。
StrictSTOP／FormalAdmissionNOT_APPROVED／W12SAFETY_HALTED／E1BLOCKED／H2NOT_APPROVED；W19完全隔離。

## Window26 Label Integrity & Next Study — COMPLETE / STOP

W25 固定候選 `27055f7ec34cd4aa6d47f67bd89e5b6616244bc8` 已由 main `90b1327` 本機 ff-only 整合；
4 focused tests、12 private artifacts／21 deliverable pins 與 Review GO 已核對。W26 base 等於該固定 SHA。
W26 branch `research/v7-window26-label-contract`，worktree `.worktrees/v7-window26-label-contract`；LOCAL_INTEGRATED_FIXED367005d / NOT_DEPLOYED。
2321 已確認 68% 虧損減資；2023-02-17 市場參考價單位變更與 2022-12-08 會計減資基準日分離。
三筆既有 10d Label 算式／市場索引吻合，但跨股數單位變更，不可作未調整持有期報酬。
六個缺行情日期是市場交易日，非市場休市；原始官方停牌公告尚未核實，精確停牌區間保留 UNKNOWN。
其他公司行動覆蓋／historical availability／vintage 未認證；不重寫 W24／W25 歷史結果，不重新評分。
兩路 Label Policy、新 Study、L0 正反向對照與所有日期／閾值／資源額度均為 PROPOSED，待 Human 決定。
曝光帳本核對 E5／E1 machine contracts；2022／2023 已曝光，未列期間為 EXPOSURE_UNKNOWN，不自動成為獨立測試。
FinMind 非價格 1/6、行情 0；官方文件 loads/attempts 4/4；全部預算 CLOSED。0 fit/refit/new prediction/RankIC/GPU。
W26 private root `.artifacts/window26-label-integrity/`；15 files／4,785,564 bytes 獨立同機 Restore PASS；
W24 191＋W25 12 files 原 Hash 保持。BACKUP_PENDING，不宣稱正式備份或完整重現。
[Report](../../reports/v7_window26_label_integrity.md)、[Evidence](../../research/v7/window26-evidence.json)、
[事件矩陣](../../research/v7/window26-event-matrix.json)、[Study Candidate](../../research/v7/window26-next-study-candidate.json)、
[Exposure Ledger](../../research/v7/window26-exposure-ledger.json)、[唯一 Backlog](../../research/v7/research-program.md#window26-label-integrity-and-next-study-contract)。
Review／tests 狀態以 window26-evidence.json／window26-review.json 為準；本窗口無剩餘已授權工程，STOP。
下一 Human：Label 路線及未知／停牌／除息／下市政策、資料與 Universe 資格、Development-only 或 prospective、
Fold／Selection／Refit／評估／資源上限；本次未批准其執行。Strict STOP／E1 BLOCKED／H2 NOT APPROVED／W12 SAFETY_HALTED，W19 不操作。

## Window25 Ranking Robustness & Leakage Audit — complete / STOP

W24fixed `90b1327560f717727ad3fd53dfe86cb421c70f5b` 已LOCAL_INTEGRATED；main由87af76b ff-only，
26tests／191sourcefiles+33deliverablepins／GO與rawrestore驗證。W25 `.worktrees/v7-window25-robustness-audit`，
branch `research/v7-window25-robustness-audit`，base90b1327；LOCAL_INTEGRATED_BY_W26_FIXED27055f7/NOT_DEPLOYED。
單一read-onlyanalysis，0GET/fit/refit/newpredictions/GPU；既有W24policy/predictions/masks/data不改。
44266sourceformulas/timing檢查0確認跨期數值錯誤；TRAINhash/mean/std吻合，historicalavailability/maturity/vintage仍UNKNOWN。
四組RidgeMean/Median正，positive約59–62%；LOSO33×4=132cases/31020paired-days不翻負，非排除群組或TRAIN影響。
固定10/20circularblocks敏感度：2022relativecomparison脆弱/部分bounds跨0；2023direction一致但EXPOSED。
公司行動未知；2321約+224%10dproxy跨6缺行情session，來源/停牌/公司行動原因UNKNOWN，無發明過濾門檻或改寫結果。
分類ENGINEERING_VALID_WITH_EXPLORATORY_SIGNAL僅條件rawproxy；2022comparisonSIGNAL_FRAGILE，actions/asofINSUFFICIENT_EVIDENCE；非Humancorrectness/admissiongate。
Root.artifacts/window25-robustness-audit保存LOSO/casetrace/source；12files獨立同機restorePASS，BACKUP_PENDING。
[Report](../../reports/v7_window25_robustness_audit.md)、[matrix](../../research/v7/window25-leakage-matrix.json)、
[evidence](../../research/v7/window25-evidence.json)、[唯一Backlog](../../research/v7/research-program.md#window25-ranking-signal-robustness-and-leakage-audit)。
10focused/26mainpost/4independent新testsPASS；ReviewGO0CriticalRequired，0rework；16CBBbounds/264LOSOdailycase獨立核對。
下一Human：公司行動/停牌/adjustment與2321有界補證、universe/PIT或proxy政策、matchedbaseline方向與新未曝光資料/正式統計設計。
本輪無剩餘READY；STOP。StrictSTOP/FormalAdmissionNOT_APPROVED/E1BLOCKED/H2NOT_APPROVED/W12HALTED，W19完全不操作。

## Window24 Expanded Retrospective Pilot — complete / STOP

W23fixed `87af76b038fcbbda9e1e4b3a4349e693a3b1ba48` 已LOCAL_INTEGRATED；main由7027f03 ff-only，
26tests／55priorlocalfiles+25deliverables／GO與rawrestore重驗。W24 `.worktrees/v7-window24-expanded-pilot`，
branch `research/v7-window24-expanded-pilot`，base87af76b；LOCAL_INTEGRATED_BY_W25_FIXED90b1327/NOT_DEPLOYED。
下載前固定36檔HINDSIGHT_SELECTED cohort（currentcapital非historicalmarketcap）；31新FinMindGET+5舊raw，
0metadata/retry/redirect/token/fallback，unused14CLOSED。24511rows，QC24160VALID/346zero/5UNKNOWN；不fill或調整價混用。
TWSE2021–2023既有729calendar；52488三分mask，engineering46291／本輪exploratoryrawproxy44266／strict0。
33current/new股票入比較；3下市startUNKNOWN僅保留診斷，不能宣稱已消除存活者偏誤。
固定L0／Ridge各horizon一次（2realRidgefits），Train2021/Val2022/exposed2023；0search/selection/refit。
940core日期×horizon全部N27–33，預先固定min20未降；1880daily/940paired與scores獨立ReplayPASS。
公司行動/publication/vintage/可交易資格仍UNKNOWN，所有Label為獨立rawdiagnosticproxy，非收益或模型優劣證據。
本輪LIMITED_EXPLORATORY授權已行使；FormalResearchAdmission仍NOT_APPROVED／StrictSTOP/E1BLOCKED/H2NOT_APPROVED/W12HALTED。
Root.artifacts/window24-expanded-pilot/private原始與完整predictions，193files+5舊raw獨立RestorePASS，BACKUP_PENDING。
[Report](../../reports/v7_window24_expanded_pilot.md)、[evidence](../../research/v7/window24-evidence.json)、
[唯一Backlog](../../research/v7/research-program.md#window24-expanded-retrospective-ranking-pilot)。
50focused/26mainpost/44independenttestsPASS；ReviewGO0CriticalRequired，0rework；191pins/restore驗證。
下一Human：正式raw/adjusted/actions政策與historicaluniverse/calendar/Label/eval/compute批准；本輪無剩餘READY，STOP。
W19工作樹/task/artifacts/budget/runtime不操作不干擾，不新增runtime觀測；不mergeW06/W12/W19或W24，不push/deploy。

## Window23 Historical Calendar & CPU Smoke — complete / STOP

W22 fixed `7027f03ebdb7b7b3bd90bc2bf5ce97a4c267c470` 已LOCAL_INTEGRATED，main由428a5fa ff-only；
23tests／78localfiles+23deliverablehashes／既有GO重驗。W23 `.worktrees/v7-window23-calendar-smoke`，
branch `research/v7-window23-calendar-smoke`，base7027f03；LOCAL_INTEGRATED_BY_W24_FIXED87af76b/NOT_DEPLOYED。
官方documentloads10已CLOSED（8directGET+2webopen），FinMind0。TWSE2019/20tableempty，2021/22schedule244/246，
2023schedule加官方0803臨時休市239，與observedreference相同；不提升strictPIT／完整TPEx日曆資格。
十檔日期membership僅listinginterval/knownboundary；8272不套用2024TPEx身份，停牌/可交易資格UNKNOWN保留。
W22原19728strictmask不改，Stricteligible0；A195價格幾何失敗／B7009獨立工程診斷／C12524其他阻塞。
Human本輪LIMITED_ENGINEERING_SMOKE_ONLY：五檔TWSE，Train範圍masked到2021、Validation2022、Forward-style2023；
固定L0／兩horizons各一次Ridge，0search/selection/refit；1880dailyrows，1864defined、16minN不足跳過，非模型性能結論。
公式/feature/fold/policy於fit前封存；rawreturn非可執行收益，actions/publication/vintage仍UNKNOWN，不是研究admission。
唯一rework修復JSONmap排序造成replaylist順序不同；原bundle/models/metrics及run-code留存，無重新fit，獨立ReplayPASS。
Root.artifacts/window23-calendar-smoke及同機window23-restore保存資料，56files獨立RestorePASS，BACKUP_PENDING。
[Report](../../reports/v7_window23_calendar_smoke.md)、[evidence](../../research/v7/window23-evidence.json)、
[唯一Backlog](../../research/v7/research-program.md#window23-historical-calendar-and-cpu-smoke)。
正式study仍PROPOSED/DataAdmissionNOT_APPROVED；StrictSTOP/E1BLOCKED/H2NOT_APPROVED/W12HALTED；W19完全不操作。
58focused/23mainpost/26independenttestsPASS；獨立GO0CriticalRequired，唯一replay修復無refit。
下一Human集中決定正式historicalcalendar/universe/action/label/availability/evaluation/compute；無本輪剩餘工程READY，STOP。

## Window22 Continuous Historical Panel — complete / STOP

W21兩Required獨立複核CLOSED，15testsPASS/Critical0Required0；133localfiles驗Hash。
main由6a34520 ff-only至 `428a5fa5cb7b771db39a053f88c7046501399f7e`；沒有提前main文件commit。
W22 `.worktrees/v7-window22-historical-panel` / `research/v7-window22-historical-panel`，base428a5fa；LOCAL_INTEGRATED_BY_W23_FIXED7027f03/NOT_DEPLOYED。
10FinMindpriceGET、0metadata/diagnostic，均HTTP200，unused20CLOSED；10檔9864rows/1762931rawbytes，2019–2023。
6檔同1216reference日期（不是合格officialcalendar），9725structuralVALID/131UNVERIFIED/8zero，不fill或算return/RankIC。
官方18419/25036/11543CSV各1GET；TPExdelisted403立即停，共4resourceattempts；cataloguedocs另3GET。
6515TWSE上市20210120、8272TPEx上櫃20241023、2499終止上市20201110有officialCSV證據；逐日suspension/membership未知。
實際5d/10d eligible0：calendar/actions/availability/labelmaturity仍UNKNOWN，historicalstudyPROPOSED，不作admission。
Root.artifacts/window22-historical-panel與同機restore保存raw/panel/mask；15產物獨立程序restorePASS，BACKUP_PENDING。
[Report](../../reports/v7_window22_historical_panel.md)、[coverage](../../research/v7/window22-panel-summary.json)、
[唯一Backlog](../../research/v7/research-program.md#window22-continuous-historical-panel)。
23focusedtestsPASS；一次主要獨立ReviewGO/Critical0Required0，0rework；15份restore皆PASS。
下一步只建議精準calendar/lifecycle補證與Human統一data/label/eval/CPU授權；本輪完成後STOP。
W19工作樹/排程/Artifacts/budget不讀寫不觸發不整合；W12SAFETY_HALTED/StrictSTOP/E1BLOCKED/H2NOT_APPROVED不變。

## Window21 FinMind Historical Proof — complete / STOP

W20fixed `6a34520d8f9c1166172af3503cad531e23f5ecec` 已LOCAL_INTEGRATED，main由87695e4 ff-only；
11deliverable／8HTML+manifest／既有ReviewGO驗證。W21 research/v7-window21-finmind-proof，base同上，
`.worktrees/v7-window21-finmind-proof`，COMPLETE/STOP/LOCAL_INTEGRATED_BY_W22/NOT_DEPLOYED；固定428a5fa。
18anonymousHTTP200＝14price+3metadata+1repeat；4名額未用且CLOSED，無Token/retry/redirect/fallback。
6檔116price／48dates／2010-01-04..2025-08-01；100structuralVALID、3NO_PUBLISHED_PRICE（含有量零價）、
2documented-date市場例外、11UNVERIFIED；零ffill／PriceAdj混用／研究准入。2499有2010price及provider下市身份，近/後窗口空不推論漏資料。
Snapshot收到838854bytes，去重後836995rawbytes。root.artifacts/window21-finmind-proof獨立於worktree，
18份初次／修後獨立程序Restore與reviewerRestorePASS；原records/code保留。Backup仍同機BACKUP_PENDING。
VendorCalendar含future及已知20261023休市日期，禁止當actualcalendar；StockInfoupdated不等於active/listing，Universe/lifecycle仍BLOCKED。
來源僅CONDITIONAL_RETROSPECTIVE_CANDIDATE，多年份小窗口≠全市場連續性／StrictPIT／Executable證據。
W21原一次Review 2Required及單輪修復證據保留；W22已針對性獨立複核CLOSED/GO，見Window22integration記錄。
[Proof](../../reports/v7_window21_finmind_proof.md)、[evidence](../../research/v7/window21-evidence.json)、
[唯一Backlog](../../research/v7/research-program.md#window21-finmind-historical-data-proof)。
下一Human：獨立少特徵retrospective dataset身份、合格calendar/lifecycle、missing/actions/Label/maturity與eval/CPUcompute批准；
本輪無剩餘READY。W19工作樹/排程/artifact/budget完全不操作、不整合；無新runtime觀測。
StrictSTOP/E1BLOCKED/H2NOT_APPROVED/HistoricalPROPOSED/FormalDataNOT_APPROVED/W12SAFETY_HALTED。

## Window20 Historical Source Decision Sprint — complete / STOP

Branch `research/v7-window20-historical-sources`，base/main `87695e4d5dc6e047f5ab39ca3cd8294b971bbc3a`；
`.worktrees/v7-window20-historical-sources`，UNMERGED/NOT_DEPLOYED。僅最多三類來源的兩單元文件調查，沒有行情API或工程實作。
FinMind TaiwanStockPrice為有條件主要候選；富邦HistoricalCandles為帳號/使用權BLOCKED的備用；
W14官方歷史route使用資格未釐清，未套用W15CSV授權。多年份日期皆DOCUMENTED宣稱，不是假定Coverage。
非商業/顯名/禁止raw再散布與匿名300/hr文件已核對；永久保存條件UNKNOWN，0價/興櫃open/調整價/下市生命週期須驗證。
建議下一次無Token資格Pilot至多22requests（18price+3metadata+1重取）、6檔角色×3小窗口，PROPOSED未批准。
不是5d/10d Label／Dataset／研究admission，沒有L0/Ridge/LightGBM訓練或RankIC。
W19固定cf1a84f的限期Task授權獨立保留；本輪不整合/改動/停用/觸發，無新增runtime觀測；以其固定來源及timestamped記錄為準。
[Decision brief](../../reports/v7_window20_source_decision.md)、[matrix](../../research/v7/window20-source-matrix.json)、
[evidence](../../research/v7/window20-evidence.json)、[唯一Backlog](../../research/v7/research-program.md#window20-historical-data-source-decision-sprint)。
一次獨立審查GO／Critical0／Required0，8官方HTML924253bytes／manifest及分類預算驗證PASS；無實作或套件重跑。
下一Human只需決定是否批准有限無帳號非商業來源資格驗證及保存範圍；富邦另需帳號/行情權限，付費並非首輪必要。
完成後STOP，沒有本輪剩餘READY；StrictSTOP/E1BLOCKED/H2NOT_APPROVED/HistoricalPROPOSED/FormalDataNOT_APPROVED/GPU_SAFETY_HALTED。

## Window18 手動 Forward Collector Pilot — complete / STOP

W17 fixed `96e8c0c4f2f7f78c508048e31f0f163d6c707128` 已 LOCAL_INTEGRATED/NOT_DEPLOYED，
main 由000b0b9 ff-only至固定候選；38treesclean／19deliverablehashes／9CPUtests確認；W06/W12未整合。
W18 `development/v7-window18-manual-collector`，base同上，`.worktrees/v7-window18-manual-collector`，
WINDOW_COMPLETE/STOP/UNMERGED/NOT_DEPLOYED；finalHEAD為含本收尾commit。僅本輪manualpilot批准，daily仍NOT_APPROVED。
TWSE/TPEx exactlicensedCSV各1GET HTTP200，0retry/redirect/fallback；均2026-10-22：1383/1017rows，
137956/141480bytes，raw合計279436。原record/QC/code保留；TPEx初判27INVALID，修後離線27MISSING，TWSE15MISSING；不補零。
本機根目錄`.artifacts/window18-manual-pilot`保存raw+manifest+provenance，獨立程序初次／修後restore均Hash+rowsPASS。
同機restore不是獨立backup：BACKUP_PENDING。無Content-Length證據明確UNKNOWN，不宣稱來源宣告長度驗證。
Capture MANUAL_RESOURCE_CAPTURE_TIME_UNQUALIFIED，publicationUNKNOWN，非VERIFIED_FORWARD_VINTAGE／ResearchDataAdmission。
W17calendar僅PROVISIONAL TWSEboundedmatch；TPEx日曆未知，不以牆上日期判stale，不將商品全列入股票Universe。
新增manualCLI/CSVQC，共重用v1snapshot/W09safe+lock/W07clock/W17calendar；W15adapter仍Synthetic-only，沒有第二套平台。
25targetedtestsPASS（含原W17 9）、integration9PASS；一次review Critical0/Required3→唯一修復輪全部以RED/GREEN＋
獨立程序實際restore自行驗收。沒有第二次獨立GO，不把首次NO-GO改寫。W14/16/17原bytes與W12HALTED不變。
[Brief](../../reports/v7_window18_manual_pilot.md)、[evidence](../../research/v7/window18-evidence.json)、
[唯一Backlog](../../research/v7/research-program.md#window18-limited-real-forward-collector-pilot)。
下次Human：daily runtime/scheduler、19:00候選與更新/日曆/QC、有限retry、獨立備份、監測通知與backfill/停止撤銷。
本輪GET額度已用完；無剩餘授權執行。StrictSTOP/E1BLOCKED/H2NOT_APPROVED/HistoricalPROPOSED/GPU_SAFETY_HALTED，
FairEval/PreNextOpen PROPOSED／InferenceRecovery DEFERRED。不得自動重跑/排程/訓練或開下一窗口。

## Window17 Forward Vintage 操作準備 — complete / STOP

W16 固定 `000b0b9c40448b11a41d05396dffc6e0f3ea1235` 已 LOCAL_INTEGRATED／NOT_DEPLOYED；
main 由 `27dcb36` ff-only 至固定候選，原有 W16 UNMERGED 記錄僅保留歷史意義。W06／W12 未整合。
W17 `research/v7-window17-forward-readiness`，base 同上，`.worktrees/v7-window17-forward-readiness`；
WINDOW_COMPLETE／STOP／UNMERGED／NOT_DEPLOYED，最終 HEAD 為含本收尾的 local commit。
W16 CSV 279436 bytes＋2 manifests＋request ledger 已獨立程序驗證並保留至根目錄
`.artifacts/window16-durable-evidence/`，原工作副本及 Restore 不動；同機同檔案系統，BACKUP_PENDING。
獨立Review一次GO／Critical0／Required0／0rework。32 次必要定點驗收（整合3＋最終29）、6日離線面板；新9測試僅工程／Synthetic，不是歷史 PIT 准入。
TWSE 官方10/23休市標題／ID／發布時刻及10/26年度補假確認；PDF正文未取得。
TPEx直接公告仍缺，不能以TWSE推定；10/27預定交易、非已實際交易。日曆修訂須有鏈與asof身份。
兩資源每1日更新不等於自動存取配額／SLA；定期服務條件與正式啟用仍 BLOCKED。
新元件限離線calendar／storage狀態guard，重用W07/14/15及Snapshot；沒有網路Collector或Calendar Platform。
[Activation brief](../../reports/v7_window17_activation_brief.md)、[evidence](../../research/v7/window17-evidence.json)、
[唯一Backlog](../../research/v7/research-program.md#window17-forward-vintage-operational-readiness)。
下一Human：先確認兩Resource定期服務條件及各市場有效日曆，再批准runtime／時間頻率／有限重試／
QC缺漏／儲存與獨立備份／監測通知／backfill及停止恢復撤銷。沒有剩餘本輪READY或啟用權限。
StrictSTOP／E1BLOCKED／H2NOT_APPROVED／HistoricalPROPOSED／CollectorNOT_APPROVED／GPU_SAFETY_HALTED；
FairEval／PreNextOpen PROPOSED、InferenceRecovery DEFERRED。不啟動下一窗口。

## Window16 Real Resource Proof — complete / STOP

W15fixed27dcb36b5431b70e8aa831edb6cb9234cb3a40b3 LOCAL_INTEGRATED/NOT_DEPLOYED,
ff-only from4da1bcf;25targetedCPUtestsPASS,source/review/hash/36cleantrees verified.
W14 raw/W15 evidencebytes remain originaltrees;not moved to main. Mainheld27dcb36;
W06/W12/othercandidates notmerged. OlderW15UNMERGED text historical for integration only.
W16 research/v7-window16-resource-proof,base27dcb36,.worktrees/v7-window16-resource-proof,
WINDOW_COMPLETE/STOP/UNMERGED/NOT_DEPLOYED;2units only,no newvalidator/platform. Two exactlicensedCSV GETs,
1each/0retry/redirect/fallback. BothHTTP200 UTF8,only2026-10-22/1session:
TWSE1383rows/1383securities/137956bytes,1368basicQCpass+15missing;
TPEx1017/1017/141480bytes,990basicQCpass+27missing. Nozero-fill orcalendar/universe/
pricecurrency/adjustment/tradabilityadmission. HistoricalTrainingCoverageBLOCKED;
no hiddenhistorysearch. Forwardrawresource CANDIDATE_ONLY, activationNOT_APPROVED.
IndependentPythonrestore2records/279436bytes/hash+rowsPASS;3focusedSnapshot testsPASS.
Raw+restore both localuntrackedcopies in W16tree,not certifiedbackup;tree removalrisk.
[Brief](../../reports/v7_window16_resource_proof.md),[evidence](../../research/v7/window16-evidence.json),
[sole queue](../../research/v7/research-program.md#window16-real-resource-coverage-and-capture-proof).
SingleindependentreviewGO0Critical/Required,0repair;28targetedtests, no fullsuite repeat.
Fallbackpolicy PROPOSED,whitelistempty,no provider switch/collector/fit/RankIC/GPU.
Old120files+4Frozenhashes unchanged. StrictSTOP/E1BLOCKED/H2NOT_APPROVED/
ExploratoryPROPOSED/CollectorNOT_APPROVED/GPU_SAFETY_HALTED. No nextwindow.

## Window15 Licensed Foundations / Dual-track — complete / STOP

W14fixed4da1bcfa81c8ff786de82d29ec461c0851d411d5 LOCAL_INTEGRATED/NOT_DEPLOYED,
ff-only fromc78098f;pre/post150CPUtestsPASS,independent0Critical/Required.
W14original337221rawbytes/4snapshotfiles remain in W14tree, NOT moved to main.
Mainheld4da1bcf; W06/W12/othercandidates notmerged. OlderW14UNMERGED prose historical.

W15 research/v7-window15-dual-track,base4da1bcf,.worktrees/v7-window15-dual-track,
WINDOW_COMPLETE/STOP/UNMERGED/NOT_DEPLOYED. Humanconfirmed dualtrackdirection only.
U1 COMPLETE_REVIEWED:govdatasets11549/11371 exactCSV scopedOGDL1.0+attribution confirmed;
periodicservice/rate/activation conditional; oldhistoricalroutes permissionUNCLEAR.
3officialmetadata/licenseHTML1547650bytes locallyretained/independentverified;0market
payloadrequests. Licensing never implieshistoricalPIT/dataset/researchadmission.
U2 COMPLETE_REVIEWED:independentexploratoryStudy/Dataset/Feature drafts PROPOSED;
7synthetictests. Lag/dates/parameters/universe/calendar/labels/bytes/compute unapproved,
DATA_READY=false; oldsyntheticadapters require separatelyreviewed realdata scope.
U3 COMPLETE_REVIEWED:SYNTHETIC-onlythinarchive adapter reusesSnapshot/W14/W09;
18tests,2boundedrepairs,0remainingCritical/Required. Idempotentraw/newrevision,
currentQC/late vsfirstcapture separated; missingintent/binding/snapshotNOT_RESTORABLE,
no hiddenretryrepair. Initialdryrun+oldsource retained, finalofflinevalidation separate.
175jointCPUtestsPASS; allthree reviews0Critical/Required, U1/U2rework0. No livecollector/
fits/realRankIC/GPU. W11/W12/W14old92files+4Frozenhashes unchanged; no cleanup.
[Brief](../../reports/v7_window15_dual_track_brief.md),
[evidence](../../research/v7/window15-evidence.json),
[sole queue](../../research/v7/research-program.md#window15-licensed-dual-track-foundations).
NextHuman:boundedlicenseddata/coverage/restore qualification, newstudy/eval/admission
policy, separateTrackBoperation/storage/frequency/QC/notice authorization. No READY
execution remains in this completedgrant. FinalHEAD containingclosurecommit.
StrictSTOP/E1BLOCKED/H2NOT_APPROVED/FairEvalPROPOSED/PreNextOpenPROPOSED/
GPU_SAFETY_HALTED/InferenceRecoveryDEFERRED remain. No nextwindow activation.

## Window14 Evidence-first Data Pilot — complete / STOP

W13fixedc78098f3fdddb74e461bc54d700bde5baac0fcc1 LOCAL_INTEGRATED/NOT_DEPLOYED,
ff-only from8db5b911;pre/post140CPUtestsPASS,independentGO0Critical/Required;
12artifact/24source/88protectedhashes verified,34trees preflightclean. Mainheldc78098f;
W06/W12/othercandidates notmerged. OlderW13 UNMERGED text historical for integration only.
W14 research/v7-window14-data-admission,basec78098f,.worktrees/v7-window14-data-admission,
WINDOW_COMPLETE/STOP/UNMERGED/NOT_DEPLOYED;2boundedunits,each1reviewrepair.
U1COMPLETE_REVIEWED:2dailyofficialrequests TWSE20220901/TPEx20220902,337221bytes,
4selectedstocks,currentstructuralQC. Both proposedRETROSPECTIVE_ONLY;0asof/0proxy
qualified. Historicalpublication/revision/calendar/universe/labels and rights remain
unresolved. Contentclass != usagepermission != researchadmission;no moredatarequests.
U2COMPLETE_REVIEWED:PROPOSED forwardcontract+readonlyprototype reusesSnapshot/W07/W09;
10synthetictests,150jointCPUtestsPASS;both0Critical/Required. No collector/scheduler/
newplatform. PROTOTYPE_TIMELY neverauthenticates callerclock/pins oradmitshistoricaldata.
[Brief](../../reports/v7_window14_decision_brief.md),
[evidence](../../research/v7/window14-evidence.json),
[sole queue](../../research/v7/research-program.md#window14-evidence-first-admission-pilot).
NextHuman:permitdata route/rights, qualify asof/vintage or distinctlag-proxy/forward
proposal, then calendar/universe/labels/restore+research/eval/CPUcompute. No trueCPUstudy
ready or remainingauthorizedexecution. Raw4snapshotfiles retained locally;hash-only
restore impossible. ProtectedW11 83bytesfiles/W12 5diagnostics and4Frozenhashes unchanged.
StrictSTOP/E1matrixBLOCKED/H2NOT_APPROVED/FairEvalPROPOSED/PreNextOpenPROPOSED/
GPU_SAFETY_HALTED/RecoveryDEFERRED. FinalHEAD containingclosurecommit;no newwindow.

## Window13 Thermal / Admission Preparation — complete / STOP

Branch research/v7-window13-admission-preparation,base/main8db5b9111b98d4965a9f56f10a51c5157284b900,
.worktrees/v7-window13-admission-preparation,WINDOW_COMPLETE/STOP/UNMERGED/NOT_DEPLOYED.
Main unchanged; W11 localintegration verified (olderUNMERGED text historical).
W06 fixed96c60564/ProtocolAudit8601c737/Recoverye181f366/W12 fixed16e444f9 only
UNMERGED_REFERENCE_ONLY; no merge or authority promotion.

U1 COMPLETE_REVIEWED,sourcef26dbf5: HUMAN_CONFIRMED_DESIGN_DIRECTION/GPU_RETEST_DEFERRED;
RTX3060 local-first candidate,WindowsHost target87C reference/futurestop87C;
83C warningPROPOSED, not NVIDIA longtraining certification. OldW12 80C/HALTED/
32.106942s unchanged. GPUkernel/F-B/resume/AMP/fullmodelshape NOT_VERIFIED.
U2 COMPLETE_REVIEWED: [Human Decision Brief](../../reports/v7_window13_decision_brief.md),
DT1–4/EV1–6 options PROPOSED only. W05 audit reused0VERIFIED/21CONDITIONAL/27UNKNOWN;
2TPEx oddlot archives176096bytes currentlyverified, no historicalavailability/QC/
calendar/universe/labels/full48panel proven. No realRankIC or dataadmission.
ENGINEERING_READY scopedW07–11 tools; POLICY_REQUIRED/DATA_BLOCKED/RUNTIME_BLOCKED
separate. 140mainCPUtests+13fixedW06prototypePASS; bothreviewCritical0/Required0,
0rework. ProtectedW11 83files+W12 5diagnostics unchanged; no newsource/framework.
[Decision](../02_History/Decisions/20261024-window13-thermal-admission-preparation.json),
[evidence](../../research/v7/window13-evidence.json),
[sole queue](../../research/v7/research-program.md#window13-thermal-and-admission-preparation).
NextHuman: freeze timing/missing/source/evidenceclass, identify qualified bytes or
approve futurevintage collection, then evaluation/compute contract. No READY execution
in this completedgrant; new boundedauthorization required. Do notclearHALTED/retestGPU.
StrictSTOP/E1matrixBLOCKED/H2NOT_APPROVED/FairEvalPROPOSED/PreNextOpenPROPOSED/
GPU_SAFETY_HALTED/RecoveryDEFERRED unchanged. FinalHEAD containingU2closurecommit.

## Window11 L2 & CPU Resume — complete / STOP

Window10 fixed `f25181637e40138c84dc3722292647d4325882fc` LOCAL_INTEGRATED/NOT_DEPLOYED
ff-only from `91fa82c2ac5e3b8f22109c55a73fa4e540a916b9`;pre/post167testsPASS,
40hashes/independent26 GO,0Critical/Required. Main heldf251816; no main closure commit.
Older Window10 UNMERGED/STOP entries historical, superseded for integration only.

Window11 `development/v7-window11-l2-cpu-resume`,basef251816,
`.worktrees/v7-window11-l2-cpu-resume`, WINDOW_COMPLETE/STOP/UNMERGED/NOT_DEPLOYED.
U1 actualLightGBM4.6.0 fit/refit/forward/common-support/independentReplay PASS;
5officialwheels pinned in dedicated overlay, baseCPU Torch/NumPy/SciPy unchanged.
U2 reuses E1 actual train loop/store/adapter, safe weights_only=True probe codec.
Three new-process exits0/75/0,step4(epoch1batch1) afterwarmup ->12steps/4epochs;
23nontelemetryfields exactlyequal forLATEST/PREVIOUS/BEST. CPU_STEP_BOUNDARY_RESUMABLE
and CPU_MID_EPOCH_RESUMABLE only deterministic daily cursor/0workers/FP32/1CPUthread.
No separateepochinterrupt/GPU/AMP/shuffleworker/generalExactResume/full reproduction.
211jointtestsPASS. U1 environment setupcorrection1,reviewrework0;U2rework2,0remaining
Critical/Required. Source U1 f00bebab33cdeaa75b8401c6a6988dfaa3d3e0e7;U2 e28ac62ed95007e246185c5c2a134d17a5d0e08b.
Final HEAD is containing artifact/closure commit; no further engineering/Window12.

CPUbytes83files/~1.79MB retained `.artifacts/window11-cpu-resume-proof` under this
worktree; exactinventory in CPUevidence. Hash-only Git metadata cannot restore lost
checkpoints. Preserve this tree/basevenv/overlay/cache until separately reviewed.
Current runtime-bound L2 source snapshot uses U2commit and fresh run identity;
initialsourcebundle/report retained unchanged, requires original source to replay.
[CPU proof](../../research/v7/window11-cpu-resume-evidence.json),
[current L2](../../research/v7/window11-l2-runtime-evidence.json),
[evidence](../../research/v7/window11-evidence.json),
[sole Backlog](../../research/v7/research-program.md#window11-l2-and-cpu-resume).
StrictSTOP/E1matrixBLOCKED/H2NOT_APPROVED/FairEvaluationPROPOSED/PreNextOpenPROPOSED/
InferenceRecoveryDEFERRED persist. No realresearch/GPU/production/admission/othermerge.
Resume namedtree HEAD/clean/maps/Charter/evidence; new bounded authorization required.

## Window10 Classical Synthetic Foundation — complete / STOP

Window09 fixed `91fa82c2ac5e3b8f22109c55a73fa4e540a916b9` LOCAL_INTEGRATED /
NOT_DEPLOYED, ff-only from `bdea30d877fa2c5d02308f343fa5591a35616aec`.
Pre/post141CPUtests PASS;39hashes/independent40tests GO,0Critical/Required;30trees
clean. Main held91fa82c, all new closure/development documents stay isolated.
Older Window09 UNMERGED/STOP entries superseded for integration only.

Window10 branch `development/v7-window10-classical-e2e`, base91fa82c, worktree
`.worktrees/v7-window10-classical-e2e`. UNMERGED/NOT_DEPLOYED; two units only,
<=2rework/unit. Human now permits tiny CPU synthetic fits, not real-data research.
U1 COMPLETE_REVIEWED:13tests/independent13PASS,1rework (full selection run hash and
null containers). L0 explicit toy factor; L1 NumPy train-only scaled Ridge; L2
adapter code present, dependency BLOCKED (lightgbm absent; no install performed).
U2 COMPLETE_REVIEWED:13tests/independent53 including40oldW09testsPASS,1rework
for fit/prediction/config lineage; validation+toyselection and forward pair replay
verified in independent CPU process. Final167joint testsPASS. U1 source 4ec22a4330ed6db7df0b3dafb4106cea820516e2;
U2+closure final HEAD is this containing commit. WINDOW_COMPLETE/STOP.
W09 v1 no-fit profile unchanged; v2 honest tiny-classical-fit profile added. No new
split/evaluator/platform. METADATA/EVALUATION replay proven; TRAINING_RESUMABLE/
FULL_REPRODUCTION_READY unproven. L2 actual fit remains dependency BLOCKED.

StrictSTOP/E1matrixBLOCKED/H2NOT_APPROVED/FairEvaluationPROPOSED/PreNextOpenPROPOSED/
InferenceRecoveryDEFERRED. No Frozen E5/E1 or production mutation; no other branch
integration, GPU/Colab/search/real research. [Evidence](../../research/v7/window10-evidence.json),
[sole backlog](../../research/v7/research-program.md#window10-classical-synthetic-foundation).
Resume named branch HEAD/clean/maps/Charter/evidence/queue; final candidate is this
containing U2/closure commit. No READY unit remains in this grant; no Window11.

## Window09 Durable Artifacts & Offline Replay — complete / STOP

Window08 fixed `bdea30d877fa2c5d02308f343fa5591a35616aec` LOCAL_INTEGRATED /
NOT_DEPLOYED, ff-only from `1631020bce8223ac4337c7e97473a2a3d9a555f0`.
Pre/post101CPUtests PASS;19hashes/24cases reverified; independent34tests GO,
Critical0/Required0. Main remains bdea30d; no extra main closure commit.
Older Window08 UNMERGED/STOP entries are historical, superseded for this integration only.

Window09 isolated `development/v7-window09-artifact-replay`, base bdea30d, path
`.worktrees/v7-window09-artifact-replay`. UNMERGED / NOT_DEPLOYED. Human authorizes
only two CPU/SYNTHETIC units, <=2evidence-based rework/unit, then STOP.
U1 COMPLETE_REVIEWED:20focusedPASS;2bounded rework rounds corrected traversal,
metadata symlinks/JSON overflow, publication-null/sources containers. Immutable bundle profiles, required
artifacts, staging+diagnostics, fsync/flock/rename, idempotence and verified external
bytes references implemented. U2 COMPLETE_REVIEWED:20testsPASS,1rework for nested
containers and actual maturity recheck; independent final40testsPASS,0Critical/Required.
Final141joint tests;2schemas/3outputs validated. METADATA_VERIFIED and
EVALUATION_REPLAYABLE proven; TRAINING_RESUMABLE/FULL_REPRODUCTION_READY not proven.
WINDOW_COMPLETE/STOP. U1 source 79d6c556d1bb5648a40a02c3b6f0e628c6512b4a; final U1 container repair
and U2 source/evidence are in this containing closure commit.
No tracking platform, retention/GC policy or training/research admission added.

StrictSTOP/E1matrixBLOCKED/H2NOT_APPROVED/FairEvaluationPROPOSED/PreNextOpenPROPOSED/
InferenceRecoveryDEFERRED. No W06/recovery/audit integration or original matrix search.
[Evidence](../../research/v7/window09-evidence.json),
[backlog](../../research/v7/research-program.md#window09-durable-artifacts-and-offline-replay).
Resume named worktree cleanHEAD/maps/Charter/evidence/sole queue; final HEAD is
containing U2/closure commit. No remaining work in this grant; no Window10 grant.

## Window08 Common Ranking Evaluation — complete / STOP

Window07 fixed `1631020bce8223ac4337c7e97473a2a3d9a555f0` LOCAL_INTEGRATED /
NOT_DEPLOYED by ff-only from `465dca229fc1452aaf1b46572b06886a7d8a2c8f`.
Pre/post67tests PASS;10artifact/frozen hashes and16cases verified; independent24tests,
Critical0/Required0 GO. Main remains exactly1631020; no main closure-doc commit.
Window07 STOP/UNMERGED prose below is historical, superseded only for this integration.

Window08 branch `development/v7-window08-ranking-evaluation`, base1631020, worktree
`.worktrees/v7-window08-ranking-evaluation`. UNMERGED / NOT_DEPLOYED. Human grant
permits only common evaluator and paired comparison, synthetic CPU fixtures, max2
rework/unit. U1 COMPLETE_REVIEWED:826b8ddf798e752584d180181d24cc1db967e019;19tests/independent19PASS,1rework.
U2 COMPLETE_REVIEWED in containing closure commit;15tests,0rework. Both Critical0/
Required0;101joint CPUtests,24explicitcases,2schemas/3outputsPASS. WINDOW_COMPLETE/STOP.
Existing E5 average-tie RankIC, canonical_hash and W07 calendar/fold/role guards reused.
Policy remains named SYNTHETIC_NOT_ADMITTED; minN/weighting explicit, no statistics
or acceptance policy frozen. Metadata-first label gating, raw references and daily
rejection ledger are engineering evidence, not real historical PIT certification.

No W06/recovery/audit branch integrated. Strict Phase0 STOP; E1 Frozen Matrix BLOCKED;
H2 NOT_APPROVED; Fair Evaluation and Pre-next-open PROPOSED; Inference Recovery DEFERRED.
[Evidence](../../research/v7/window08-evidence.json),
[unique backlog](../../research/v7/research-program.md#window08-common-ranking-evaluation).
Final HEAD is this containing U2/closure commit. Resume by HEAD/clean, this entry,
Authority Map, Charter, evidence and backlog; a new bounded grant is required.
No automatic Window09, real-data evaluation, training or main merge of Window08.

## Window07 temporal/isolation foundation — completed / STOP

Human approved two CPU-only synthetic engineering units, no formal study config.
Branch `development/v7-window07-temporal-isolation`, worktree `.worktrees/v7-window07-temporal-isolation`,
base/main `465dca229fc1452aaf1b46572b06886a7d8a2c8f`. UNMERGED / NOT_DEPLOYED.
U1 commit5d736e9:13focused/independent13PASS,1owner-binding repair.
U2 complete:11focused/independent11PASS,
1source-hash repair. Final67related tests/16expectedcasesPASS;0remainingCritical/Required.
WINDOW_COMPLETE / STOP. No further work under this grant. Containing Unit2/closure
commit is final checkpoint. Resume via cleanHEAD here/maps/Charter/uniquequeue;
new bounded authorization required. CommonEvaluator synthetic foundation can be
next scope, not approved policy/data/Window08. [Evidence](../../research/v7/window07-evidence.json),
[decision](../02_History/Decisions/20261023-window07-temporal-isolation.json),
[unique backlog](../../research/v7/research-program.md#window07-temporal-fold--research-isolation).
No fixed validation/forward months/origins/30purge or metrics adopted; unknown policy/
data failclosed. No realPIT certification, no formalFeatureLabel/E5E1/admission change.
Window06/audit/recovery branches only pinned references, not merged or promoted.
StrictSTOP/E1BLOCKED/H2unapproved/FairEvaluationPROPOSED/PreNextOpenPROPOSED/
InferenceRecoveryDEFERRED persist. This grant does not authorize a Window08.

## Window06 — Window05 integrated; decision-time preparation authorized

Fixed `65462ec185783eedb2ba81b66cfa04783f86c6e2` integrated ff-only from main
`2ee4ca83e709af64be4dd63032ba4fb47903d769`; pre/post87 tests PASS,13source/output hashes
verified, independent13tests/0Critical/Required. LOCAL_INTEGRATED / NOT_DEPLOYED.
This containing main commit is integration closure documentation only.
[Evidence](../../research/v7/window06-integration.json).

Same Human grant authorizes isolated `research/v7-window06-decision-time` at
`.worktrees/v7-window06-decision-time` from this closure: VERSIONED PROPOSAL,
<=3source families/~10sessions diagnostic panel and synthetic eligibility tests.
New pre-next-open proposal does not replace frozen E5/E1. No formal cutoff/time,
feature/label/admission or execution policy selected. StrictSTOP/E1BLOCKED/H2unapproved/
FairEvaluationPROPOSED remain. No F2 repeat search/training/production changes.
Read local Current State, Authority Map, Charter and unique Backlog on that branch
for subsequent progress; all pending documents stay isolated. Older STOP entries
are historical for their windows; no indefinite authorization.

## Window05 Feature Integrity and Data Research — complete / STOP

Human authorized F0/F1/F2 only: read-only source audit, synthetic candidate formulas,
and bounded monthly-revenue PIT feasibility. Branch `research/v7-window05-feature-integrity`,
worktree `.worktrees/v7-window05-feature-integrity`, base/main `2ee4ca83e709af64be4dd63032ba4fb47903d769`.
UNMERGED / NOT_DEPLOYED. Source-derived macro Date joins do not prove t-close
availability; missing source values may become defaults. F0 identified material
unquantified risks; no real factor/performance analysis may depend on these
unverified payloads. Historical conclusions are preserved; no result reclassification.

F0 source audit complete:48ordered, VERIFIED0/CONDITIONAL21/UNKNOWN27/INVALID0,
all empirical quality UNKNOWN; F0 commit9fbf1bd,4tests/independent4 PASS.
F1 completed PROPOSED shortlist/formulas,9tests/independent9 PASS,1timezone rework;
source commit1a6a452, real diagnostics BLOCKED. F2 complete NOT_CURRENTLY_FEASIBLE
for broad historical PIT recovery;4historical metadata blobs independently verified,
0newqualified rows,0rework. All independent reviews0Critical/Required;87related tests
PASS. Window complete: STOP / UNMERGED / NOT_DEPLOYED, no automatic continuation.

The containing F2/closure commit on this branch is the final resume checkpoint.
Verify HEAD/clean in the named worktree, then read this section, Authority Map,
`knowledge/GOVERNANCE.md` and the local unique Backlog's Window05 section.
[Backlog](../../research/v7/research-program.md#window05-feature-integrity-and-cost-effective-data-research),
[evidence](../../research/v7/window05-evidence.json). Main stays2ee4ca8; pending
integration documents stay here. No formal manifest/cutoff/label/admission changed.
Strict Phase0 STOP; E1 matrix BLOCKED; H2 NOT_APPROVED; Fair Evaluation and all
new feature/timing/forward-vintage candidates PROPOSED. Human timing/data identity/
policy choices precede real-data diagnostics; a new bounded grant is required.
Earlier window execution pointers below are historical; this entry owns latest state.

## Window04 W04B-LOG integration closure — STOP

Human-approved fixed `97fb4c9d669f3aa237ef5545a0ef322a5cc2d6a9` integrated by local ff-only
from `31f7accc475b5b735684d5a06725a237d85d5693`. Engineering source `27b9b41b01a44d3459ab66c6a0ea83395fc2d30a`;
no new merge commit. This containing main commit adds closure documentation only.
LOCAL_DEVELOPMENT_INTEGRATED / NOT_DEPLOYED / WINDOW_COMPLETE / STOP.
Pre/post274 tests each PASS; independent runtime6 PASS,0 Critical/Required.
Both source hashes match;23 existing worktrees clean before integration and preserved.
[Closure evidence](../../research/v7/window04-log-integration-closure.json).

Current continuation entry is WSL main: verify HEAD/clean, read this section,
Authority Map, `knowledge/GOVERNANCE.md` and the local unique
`research/v7/research-program.md`. Older branch-local/unmerged pointers below are
historical, superseded only for W04B-LOG local integration. No new work authorized.
No additional high-value READY item was established by the prior bounded discovery;
future concrete CPU defects can be considered under a new bounded grant.
Strict Phase0 STOP; E1 matrix BLOCKED; H2 NOT_APPROVED; research options PROPOSED.

## Window04 Stage B — completed / STOP

Branch `development/v7-window04-cpu-continuation`, base/main `31f7accc475b5b735684d5a06725a237d85d5693`,
worktree `.worktrees/v7-window04-cpu-continuation`. UNMERGED / NOT_DEPLOYED.
WINDOW_COMPLETE / STOP: W04B-LOG source `27b9b41b01a44d3459ab66c6a0ea83395fc2d30a`; final274 tests PASS.
No additional high-value independent READY unit established. A new bounded grant
is required for further development; this grant is exhausted. Resume by verifying
HEAD/clean here, reading this section, Charter and the unique backlog; do not
continue automatically. The containing docs-only closure commit is final checkpoint.
W04B-LOG fixes admitted E1 runtime reader-error propagation: log write/decode
failure reaches caller with original cause/path and stops a still-running child;
a failure observed after child exit also cannot return success. CPU fixtures only,
no Colab/Drive certification or training readiness claim. Focused6 tests PASS,
independent6 PASS,0 Critical/Required,0 review rework rounds (one initial fix).
[Unique backlog](../../research/v7/research-program.md#window04-stage-b--cpu-continuation)
records evaluated alternatives; [evidence](../../research/v7/window04-cpu-continuation-evidence.json).
Strict Phase0 STOP, E1 matrix BLOCKED, H2 NOT_APPROVED and proposals PROPOSED.
This section owns branch-local progress; earlier integration/continuation pointers
below are historical. Stage B source/closure stays here, main remains unchanged.

## Window04 integrated; bounded CPU continuation authorized

Human-approved fixed `98f270dd63f05831b2a42695d8005638fda2bbf5` was ff-only integrated
from main `c733a53f81ff7bd572739b5d75f8baabf1189976`. LOCAL_DEVELOPMENT_INTEGRATED /
NOT_DEPLOYED. Pre/post272 tests each PASS; independent metadata5 PASS,0 Critical/Required.
This containing main commit is integration closure documentation only.
[Evidence](../../research/v7/window04-main-integration.json).

The same Human grant authorizes Stage B, at most four valuable CPU engineering
units with at most two evidenced rework rounds each. Use isolated
`.worktrees/v7-window04-cpu-continuation`, branch `development/v7-window04-cpu-continuation`,
created from this closure; its `research/v7/research-program.md` owns subsequent
selection and completion state. All pending development/closure stays on that branch.
No Stage B merge/push/deploy/training. Strict Phase0 STOP, E1 matrix BLOCKED,
H2 NOT_APPROVED; comparison proposals remain PROPOSED. Earlier awaiting-approval /
no-StageB text below is historical and superseded solely by this explicit grant.

## Window04 Git reconciliation — current candidate

Human authorized isolated version reconciliation only. Branch
`integration/v7-window04-reconciliation`, worktree `.worktrees/v7-window04-reconciliation`,
base/main `c733a53f81ff7bd572739b5d75f8baabf1189976`; second parent fixed
`489a127cf951f93f7bfc873e9333f05113f5b8e9`. Both histories are retained without rewrite.
Original STOP audit `3330427a1d774d30e65216ddab89d668d97c187c` remains separate evidence,
not a third merge parent. This section supersedes older continuation pointers below;
they describe prior checkpoints, not the current working location.

Status: GO_CANDIDATE_ONLY / AWAIT_HUMAN / UNMERGED_IN_MAIN / NOT_DEPLOYED.
Main is unchanged. No Stage B work or research planning is authorized here.
Merge commit `225efa9bb13b67bd5258faba79a47666c8906148` retains exact two parents above.
272 joint tests PASS; independent metadata5 PASS,0 Critical/Required; no unresolved
conflicts. [Integration evidence](../../research/v7/window04-reconciliation-evidence.json). The containing docs-only
closure commit is the final candidate SHA; main and original candidate are ancestors.
Resume by verifying branch/HEAD/clean here, reading this section and the evidence,
then await Human approval of that fixed SHA. Do not start Stage B.
Strict Phase 0 STOP; E1 Frozen Matrix BLOCKED; H2 NOT APPROVED; comparison options
remain PROPOSED. Read the local unique `research/v7/research-program.md` first
for reconciliation checkpoint; original preparation artifacts are unchanged.

## Fair evaluation preparation complete — fixed continuation pointer

Window02/03 source is integrated in local main at approvedeba6b649; this main
commit adds navigation only. New Window04 work remains UNMERGED/NOT_DEPLOYED on
`development/v7-fair-evaluation-decision@489a127cf951f93f7bfc873e9333f05113f5b8e9`, base`f31fcbba4d82d260915d7b640d80abc9c9ba8fc1`.
Worktree `/home/frank/projects/MarketMamba/.worktrees/v7-fair-evaluation-decision`.
WINDOW_COMPLETE / STOP. Readiness tool2a9de32 and decision package489a127 completed;
64 related non-training tests PASS,15 source hashes/cost arithmetic verified,
independent reviews0 Critical/Required. New comparison options remain PROPOSED;
no dates/thresholds/training/H2 approved. Original matrix BLOCKED, Strict Phase0 STOP.

Latest unique backlog is that pinned commit's `research/v7/research-program.md`
Window04 section; report `reports/v7_fair_evaluation_decision_package.md`, evidence
`research/v7/fair-evaluation-preparation-evidence.json`. New audit is not merged
into main. To resume, cd named worktree, verify HEAD/clean, read Current State,
Authority Map and that queue. No full historical rediscovery or repeated matrix
search. New bounded execution needs Human grant; no background continuation.
This pointer supersedes earlier Window04 authorized/READY text below.

## Fair evaluation preparation — completed checkpoint

Isolated `development/v7-fair-evaluation-decision` from integrated main closuref31fcbb.
UNMERGED/NOT_DEPLOYED for new work only. Existing-contract read-only E1 metadata
audit completed,5tests/independent5 PASS,0 Required; always execution_ready=false.
Decision package completed:15 source hashes, exposure ledger, fit/cost formulas and
Colab stop conditions independently verified;0 Critical/Required.64 related tests
PASS. WINDOW_COMPLETE/STOP, all new options PROPOSED. New execution requires
Human bounded grant. Resume at this branch via main pinned closure pointer.
[Unique backlog](../../research/v7/research-program.md#fair-evaluation-preparation-window-04),
[evidence](../../research/v7/fair-evaluation-preparation-evidence.json).

## Window02/03 integrated; fair evaluation preparation authorized

LOCAL_DEVELOPMENT_INTEGRATED / NOT_DEPLOYED. Human approved fixedeba6b649;
main ff-only9956446→eba6b649, pre/post267 tests each PASS, independent0 Critical/
Required.19 worktrees clean before merge;23 source/2 parser/1 report/7 result hashes
verified, protected contracts/data/models/lock unchanged. Fixed candidates retained.
[Integration evidence](../../research/v7/window0203-integration-20261021.json),
[Human scope](../02_History/Decisions/20261021-engineering-integration-fair-preparation.json).
This supersedes UNMERGED/no-main-merge text below only for admitted Window02/03.
No deployment, policy or research classification change. StrictSTOP/E1BLOCKED/H2
unapproved persist. A separate bounded CPU preparation window is authorized;
its current queue is [the unique program](../../research/v7/research-program.md).

## Data reliability / evaluation preparation window03 — 2026-10-21

WINDOW_COMPLETE / STOP / UNMERGED / NOT_DEPLOYED. Current Human grant superseded the
Window02 STOP solely for this separate window. Main remains9956446; candidate
`development/v7-data-evaluation-preparation` joins main and fixed Window02f34f9f3
in isolation. Five navigation conflicts retain Window02 content, with this new
checkpoint taking precedence; main's previous pointer remains in parent history.
[Decision](../02_History/Decisions/20261021-data-evaluation-window03.json),
[unique backlog](../../research/v7/research-program.md#data-evaluation-window-03),
[evidence](../../research/v7/data-evaluation-window03-evidence.json).
Stage1 isolated integration GO at9bb4247 (259 tests, independent0 Required).
Parser complete: TWSE response date binding and TPEx unique header mapping,
8 tests + independent8 PASS; MOPS source/policy unchanged. No provider certification.
Fairness/readiness completed:12 source identities,7 result hashes,2seed identity
chains; independent0 Required. New designs PROPOSED; NOT_RUN_READY without matrix.
Runtime correction: candidate Data/contains3 inherited tracked historical files,
unchanged and parquet contents unread; no processed_v6 mount. Do not reuse older
Data-absent assumption. Final267 tests PASS, health WARNING/no FAIL. Main stays9956446; next session starts
at `.worktrees/v7-data-evaluation-preparation`, reads this queue/evidence and waits
for a new bounded grant. No main merge authorized. Strict Phase0 STOP/E1 matrix BLOCKED unchanged.

## Observation Metadata

| Field | Value |
|---|---|
| `observed_at` | `2026-09-22T13:25:21+08:00` for this runtime reconciliation; individual observations retain their own timestamps |
| `owner` | Human + project agents |
| Repository context | WSL development `main@3b0df0f` at observation; Windows operational `main@8b82a68`; research authority remains branch-qualified |
| State updated | `2026-10-21` engineering/decision update; Scheduler observation remains `2026-09-22T13:21:35+08:00`, not refreshed |
| Source basis | Reviewed K0/contracts plus [new Scheduler observation](../runtime-observations/scheduler-20260922T132135+0800.json) and [operational path observation](../runtime-observations/operational-path-20260922T132521+0800.json) |
| Human review | Prior research/Phase A scope accepted `2026-09-20`; the new 2026-09-22 runtime observations are agent-observed and have not received separate human review |
| Runtime freshness | External scheduler/deployment observations expire after 7 days and become `STALE_OBSERVATION`; they are retained until re-observed |
| Repository freshness | Contracts do not expire by time; commit identity or explicit supersession changes them |

Semantic change owners must update this file in the same change. K0 remains discovery evidence. `CANONICAL_CURRENT_PROJECT_MAP` means current accepted navigation/state authority, subject to semantic updates and explicit supersession; it does not mean immutable or permanently final.

## Cross-domain window02 final checkpoint — STOP

Isolated `development/v7-cross-domain-window-02` from main closurec1179fe,
UNMERGED/NOT_DEPLOYED. W02-DATA/PREP/EVAL/MODEL completed; this bounded Human grant is exhausted;
[unique backlog](../../research/v7/research-program.md#cross-domain-window-02)
owns acceptance and state. No production parser edits or training permitted.
W02-DATA completed7-case offline diagnostic with independent review0 Required;
TWSE/TPEx source fixes and MOPS policy are now explicit Human decision items.
W02-PREP completed frozen contract/matrix metadata guards;34 tests and independent14
PASS. Metadata-only, no training readiness/admission; original matrix remains BLOCKED.
W02-EVAL rejects missing seeds/nonfinite gate values;6 tests + independent6 PASS.
Acceptance thresholds unchanged; historical evidence not recomputed.
W02-MODEL completed source-level selection/evaluation reuse audit;11 pinned sources
independently verified. New research choices remain PROPOSED/Human-only.
Final259 tests PASS, four independent reviews0 Critical/Required. No additional
high-value READY item demonstrated; see queue for P1/P2 decisions and recovery.
[Evidence](../../research/v7/cross-domain-window02-evidence.json).

## Charter integrated and window 02 authorized — 2026-10-21

LOCAL_DEVELOPMENT_INTEGRATED / NOT_DEPLOYED. Human approved exact
`f7af348710467ac381b2a84e8d192670ced89932`; WSL main ff-only from
`6d74d7384caa5fa90b1b050ce507bfb8282f19df` to that SHA. This containing
closure commit is docs/evidence only. [Integration evidence](../../research/v7/charter-integration-20261021.json)
records pre/post254 tests each (PIT49, non-training E1 31, portfolio67, ops81,
fetch3+governance23), 6 source/4 authority hashes,75 protected blobs unchanged,
17 clean worktrees and independent0 Critical/Required. No tiny training or GPU probe run.
Window01's UNMERGED wording below is historical and superseded only for integration.

[New Human decision](../02_History/Decisions/20261021-charter-integration-window02.json)
authorizes one next isolated CPU window, maximum3–5 meaningful units. Current
selection/checkpoint remains the [unique program](../../research/v7/research-program.md).
Discover Data, Model Research and Experiment Preparation before optional system
work; do not change V6 production. StrictSTOP/E1BLOCKED/Human research gates persist.
The fixed candidate branch remains unchanged. No push/deploy/Windows/MAS activity.

## Autonomous charter first window — 2026-10-21

**WINDOW_COMPLETE / STOP_AFTER_FIRST_WINDOW / UNMERGED / NOT_DEPLOYED.**
Human's [Charter](../GOVERNANCE.md#autonomous-development-charter--2026-10-21)
is persisted in existing governance, and the
[single backlog/checkpoint](../../research/v7/research-program.md#autonomous-charter-window-01)
records the complete first window. Branch `development/v7-autonomous-charter`
starts at unchanged main `6d74d7384caa5fa90b1b050ce507bfb8282f19df`.
Five units: Charter `be7998d`, E1 split completeness `3f07f80`, research decision
preparation `5847afc`, checkpoint fallback `2c60415`, snapshot pre-publication
validation in the containing closure commit. No further work starts in this window.

[Evidence](../../research/v7/autonomous-window-01-evidence.json): E1 33 + PIT49 +
governance23 =105 PASS; independent reviews0 Critical/Required after checkpoint's
second/final repair; snapshot one round; four research sources verified. Source
and output hashes/JSON/syntax/diff checks pass; knowledge health WARNING, no FAIL.
CPU synthetic engineering and PROPOSED research preparation only; no historical
checkpoint payload/matrix/data reads, external network, new training, GPU or MAS.
Existing E1 unittest includes a one-step synthetic CPU smoke. Loader remains
trusted-local and unrestricted: no historical-pickle safety certification.

This supersedes older no-next-program or in-progress statements solely for this
completed window. H0 remains locally integrated; Strict Phase0 STOP, E1 Frozen
Matrix BLOCKED, H2 and production/research gates unchanged. Another session reads
Charter + this backlog + fixed branch evidence, verifies Git/runtime, and waits
for a new bounded window grant before executing. No merge/push/deploy authorized.
No confirmed unexecuted READY item remains in this window; do not infer global
absence of defects or repeat E1 searches without new evidence.

## Portfolio History H0 completed — 2026-10-21

**LOCAL_DEVELOPMENT_INTEGRATED / NOT_DEPLOYED.** Human accepted the limited
fixture Product Slice and [approved exact4a66928](../02_History/Decisions/20261021-portfolio-final-h0-approval.json).
WSL main fast-forwarded from `83a6a2e6f4a56db318ee07e7e737cc01de3ce48e` to
`4a66928aa4c9bd8a1ddf99a5b42866e9b22efb74`; subsequent containing commit is
**documentation/evidence only**, not a different product candidate. Pre/post CPU,
API/HTTP, frontend and real Chrome font/selection/error checks passed.
[Integration report](../../reports/v7_portfolio_h0_integration_closure.md),
[evidence](../../research/v7/portfolio-h0-integration-closure.json).
CPU01/02, PORT01–04, UX01–04, final UX and font correction are in local main.
This supersedes historical UNMERGED/H0-pending/UX-review-pending prose below for
these units only. Original reports, fixed branches and all16 worktrees are retained;
audit7ecb920 remains separate historical failed-preflight evidence, not merged.
No Required/Critical remains within this tested slice. Main is a valid source
baseline for future separately authorized CPU work; no new Program starts now.
Strict Phase0 STOP, E1 Frozen Matrix BLOCKED, H2 unapproved, research correctness
unapproved. No push/deploy/Windows operational change/training. STOP after closure.
The existing UI's static UNMERGED badge is historical candidate wording, not a live
Git integration indicator; canonical status above owns integration authority.

## Final H0 browser blocker resolved — 2026-10-21

Human [authorized only the late-font visibility repair](../02_History/Decisions/20261021-portfolio-font-blocker-authorization.json).
New branch `fix/v7-timeline-font-visibility` starts from exact `6c894794` product
candidate; `audit/v7-h0-preflight@7ecb920` is evidence only, NOT an ancestor.
True-font RED reproduced desktop3.5px/mobile2.5px clipping. Existing ResizeObserver
now observes all event rows as well as the fixed-height container; GREEN confirms
font-driven layout correction without page movement, focus loss or data change.
[Report](../../reports/v7_portfolio_font_visibility.md),
[evidence index](../../research/v7/portfolio-font-visibility-evidence.json).
Original Required finding RESOLVED within tested fixture scope; bounded preflight
engineering GO, pending Human approval of the new fixed SHA. Final candidate is the
containing commit, not the audit commit. Main remains `83a6a2e`; no merge occurred.
Human limited UX product acceptance remains in force; earlier pending-UX-review
language below is historical. Old H0 authorization does not extend to this new SHA.
UNMERGED / NOT_DEPLOYED; H2 unapproved, Strict Phase0 STOP, E1 Frozen Matrix BLOCKED.
STOP here: no further UX/research work authorized. Native Goal creation was rejected
because the previous H0 goal remains paused; no automatic continuation claim.

## Project Summary

**Final UX closure (2026-10-21):** Human accepted the dual-mode direction and
[authorized only two finishing changes](../02_History/Decisions/20261021-portfolio-ux-closure-decision.json).
On `development/v7-portfolio-ux`, baseline `1553a59`, timeline selection stays visible
inside its container, with explicit selected text and no page anchoring jumps;
event explanations use observable exact-state differences and conservative Chinese.
Both tasks COMPLETED_VERIFIED / UNMERGED / NOT_DEPLOYED. Final identity is the
containing closure commit (see `git log -1`); [report](../../reports/v7_portfolio_ux_closure.md)
and [evidence index](../../research/v7/portfolio-ux-closure-evidence.json).
Desktop/mobile general seven events, desktop advanced exact values and mobile advanced
basic smoke verified. No additional mobile advanced visual work. Stop for Human final
review; H0/H2 remain unapproved, Strict Phase 0 STOP, E1 BLOCKED. No further UX READY work.
Earlier UX/PORT evidence below is retained as historical acceptance, not rewritten.

**Completed UX refinement (2026-10-21):** Human approved UX-01–04 default dashboard /
advanced research view on isolated `development/v7-portfolio-ux` from fixed `5a2bc6f`.
[Decision](../02_History/Decisions/20261021-portfolio-ux-human-decision.json),
[skills/design/acceptance](../../reports/v7_portfolio_ux_acceptance.md).
UX-01–04 COMPLETED_VERIFIED within fixture UI scope: desktop1440/mobile390
Chrome155, both modes/all7events/exact values/keyboard PASS. Implementation
`764dfbf`; final evidence in containing UX04 commit. Stop for Human UX Product
Slice Review. Focused lint PASS; full frontend baseline58errors/3warnings unchanged.
[UX evidence](../../research/v7/portfolio-ux-evidence.json).
UNMERGED / NOT_DEPLOYED; existing PORT completion remains preserved.
No H0/H2, accounting, data or production authority change. Strict STOP persists.


**Strategic recovery / integration candidate (2026-10-21):** this Human-authorized
planning window is complete; H1 was subsequently approved for the fixture slice. `main@83a6a2e`
remains unchanged; candidate `integration/v7-cpu-roadmap` starts at fixed
`4c388de`. Five CPU fixes already form continuous ancestry; no cherry-picks,
source changes or main merge. Joint 60 portfolio/health/audit, 47 PIT, 81 ops,
3 fetch-only, 23 governance and 6 historical-output fixtures PASS. Independent
review: GO for local development candidate only, no Critical/Required.
[Integration/objective/status evidence](../../reports/v7_strategic_roadmap_recovery.md)
and [hash/test index](../../research/v7/cpu-integration-candidate-20261021.json).

The **single active cross-domain development backlog/program** is the dated
section of [existing research program](../../research/v7/research-program.md#current-development-program--2026-10-21).
Prior CPU task files remain completed-window evidence. Product intent includes
trustworthy data, relative rankings, weight-based simulation and performance;
current engine is complete only as a deterministic component. PORT-01–04 now have explicit
[Human H1 fixture-only engineering approval](../02_History/Decisions/20261021-portfolio-history-h1.json).
Completed isolated branch `development/v7-portfolio-history` from `bebf19e`:
PORT-01 COMPLETED_VERIFIED (7 focused / 67 combined, independent review closed
one compatibility repair); PORT-02 COMPLETED_VERIFIED (4 new + 4 existing API
tests, independent live startup review); PORT-03 COMPLETED_VERIFIED (8 focused +
8 existing JS cases, build and real desktop/mobile/keyboard acceptance); PORT-04
COMPLETED_VERIFIED (actual HTTP/restart, browser corruption/recovery, 47 PIT,81 ops,
3 fetch-only,23 governance). All four units independently reviewed; stop for Human
Product Slice Review. No additional H1 implementation remains READY.
[Evidence/checkpoint](../../research/v7/portfolio-history-evidence.json).
UNMERGED / NOT_DEPLOYED; H0/H2 absent. Strict Phase 0 STOP and E1 matrix BLOCKED
persist. MAS Core Live unverified; no production/runtime authority changed.
Native Goal is V7-bound; automatic future continuation remains unverified.


**CPU-only window 02 (2026-10-21, WINDOW_COMPLETE):** reused window 01 lineage
from `7eb9cd9` on `development/v7-cpu-window-01`; main stays `83a6a2e`.
[Current backlog/checkpoint](../../tasks/v7-cpu-development-window-02.md) and
[evidence index](../../research/v7/cpu-window-02-evidence.json).
U1 `735ae6f` rejects normalized quote collisions before persistence; U2 (commit
containing this closure) rejects duplicate JSON object keys at every depth.
Both COMPLETED_VERIFIED: final 60 portfolio/process/health/audit, 47 PIT,
23 governance PASS; independent review no Critical/Required, one fix round each.
No remaining confirmed READY item in inspected backlog; stop after two units.
Goal API created a correctly bound V7 objective after observing no goal; future
automatic continuation unverified. No MAS/global settings touched. This supersedes
window 01 next-action status only. UNMERGED / NOT_DEPLOYED, no research approval;
strict Phase 0 STOP and E1 matrix BLOCKED remain unchanged. Extra Human decisions 0.

**CPU-only window 01 (2026-10-21):** Human authorized bounded autonomous
engineering without training or research changes. Isolated
`development/v7-cpu-window-01`, base `11ae71d`; main remains `83a6a2e`.
[Ranked backlog and acceptance](../../tasks/v7-cpu-development-window-01.md)
select three existing portfolio scheduling/journal recovery contract units.
All three complete: early-signal state preservation, pre-write semantic
validation and shared-lock reading. Final bounded suites: 53
portfolio/process/health/audit, 47 PIT, 23 governance PASS; independent review
has no Required/Critical findings. See the
[evidence index](../../research/v7/cpu-window-01-evidence.json).
Historical next READY at window 01 closure: normalized quote collisions;
now fixed by window 02 above. Window 01 stopped after three units.
E1 matrix remains BLOCKED; strict Phase 0 STOP and Phase 1–5 gates unchanged.
This branch is UNMERGED / NOT_DEPLOYED; no live MAS Core capability is assumed.

**E1 recovery checkpoint (2026-10-21, branch-local):**
`recovery/e1-frozen-evidence` verified six checkpoint byte hashes, two prediction
hashes, seven result/pointer/evaluation files and seven fixed diagnostic files.
Historical matrix `a590b280...947f62` was not found in the explicit local search;
same-name fixtures rejected, remote Drive not inspected.
**STOP_REPLAY_MATRIX_IDENTITY_UNAVAILABLE**. No checkpoint deserialization,
matrix generation, inference or training. See
[inventory/report](../../reports/v7_e1_evidence_recovery.md).
Human next step: locate the original matrix/manifest or exact archive path.
Do not repeat completed Phase A work or infer permanent matrix loss.

**Current integration closure (2026-10-21):** Human authorized fixed `89f9496`;
WSL development main fast-forwarded from `50c5b78` to that exact candidate.
Pre/post checks each passed 81 Operations, 47 PIT/Snapshot, 3 fetch-only,
38 Portfolio and 23 governance tests. See
[closure](../../reports/production_phase_a_integration_closure.md).
Phase A is `IN_DEVELOPMENT_MAIN / NOT_DEPLOYED`; old pending-merge language below
is historical. Remote tracking was not fetched; Windows operations are untouched.
E1 bounded evidence recovery is now authorized in isolation, not new training.

**2026-10-21 current Phase A decision:** Human explicitly approved Task 4 limited
engineering correctness at `1e119ef`, after revalidation of its index, 12 raw
SHA-256 records and 36 generation digests. Decision provenance:
[Human record](../02_History/Decisions/20261021-task4-human-approval.json).
This supersedes older pending-Task-4 observations below, not research or
production gates. New isolated `integration/phase-a-task5` starts at `63a040d`
and cherry-picks the 17 Phase A commits `9efed7b..1e119ef` with `-x` provenance.
Integrated Operations files match `1e119ef`; snapshot files match `63a040d`.
Joint regression: 75 ops, 47 PIT, 3 fetch-only, 18 governance PASS.
Task 5 limited offline engineering correctness is now **HUMAN_APPROVED** at
fixed `1c5250e67461456b7ca0b361f8da3d9cb3f22514`, after artifact/digest verification.
The [Task 5 Human decision](../02_History/Decisions/20261021-task5-human-approval.json)
retains the explicit scope and exclusions. Original acceptance/report artifacts
remain unchanged historical pre-decision evidence.
Final Operations 81, governance 19, non-training portfolio regression 38 PASS;
joint PIT 47 and fetch-only 3 PASS. See the
[Task 5 report](../../reports/production_phase_a_shadow_ledger.md) and
[evidence index](../../research/production-system/phase-a/task5-evidence-index.json).
Ten fixed replays, real-process restart/concurrency and measured overhead are
offline engineering evidence, not production certification. Final engineering
review recommends **GO for development candidate integration**; this is an Agent
assessment, not final Phase A Human acceptance or permission to merge.
See [final review and guarded integration procedure](../../reports/production_phase_a_integration_review.md).
The decision-validator change passes 23 governance tests (including rejection
of mutated, missing or corrupt Task 5 approval evidence).
The earlier review had no merge authority; the subsequent explicit integration
decision above authorized only local development main. No push/deployment/training.

**2026-10-20 bounded engineering recovery (branch-local):** base development
`main@50c5b785e8c373913c3301a056497eb955f36681`; candidate
`fix/v7-snapshot-immutability`, `UNMERGED / NOT_DEPLOYED`. One engineering unit
fixes immutable V7 source snapshot publication; it does not promote research
evidence or close any correctness gate. See the checkpoint below.

**Historical window 01 objectives (superseded by strategic checkpoint above):** CPU window 01 complete; next bounded engineering candidate
is journal quote-collision refusal. Separately await original E1 frozen
matrix/manifest location.
Bounded local inventory is complete; replay is stopped until identity and safe
inspection gates can be met. No new research hypothesis or training is authorized.

MarketMamba contains separate production-data, legacy model runtime, V7 correctness, research-training, historical-simulation, portfolio, delivery, and operations-control-plane subsystems. Production acquisition uses mutable provider data; V7 research authority uses versioned PIT/provenance evidence. The current corrected research baseline is `E5-PIT-Clean-v1`, with 48 features, no graph and no industry neutralization. Historical simulation is `PASS`, while strict trading correctness remains `STOP`. The observed production scheduler runs fetch-only; full V6.2 inference/portfolio/publication is `DISABLED`. E1 is a completed research experiment with `PASS_REFRESH_HYPOTHESIS`, not trading evidence. Phase A operations work is `IN_DEVELOPMENT_MAIN` and `NOT_DEPLOYED`; historical pending-integration paragraphs below are retained observations.

The `environments/v7-colab-2026.04/` identity files are preserved in development main. Its exact lock matches the E1 branch copy; the E1 frozen contract and run manifests still own experiment semantics and result provenance. The dated local verification record is not proof of a current Colab GPU runtime.

## Knowledge Layers

Canonical development/research checkout is WSL-native `/home/frank/projects/MarketMamba` in the distro named `Ubuntu-24.04` (`3b0df0f` at observation). Windows Scheduler invokes the separate D: checkout `D:\Desktop\work\ProjectForMe\MarketMamba` (`8b82a68`, clean) through the distro named `Ubuntu`, not the development checkout. Both distros report Ubuntu 24.04, but they are distinct registrations. The batch activates `mamba_env` with `/home/frank/miniconda3/etc/profile.d/conda.sh` and uses Python 3.11.15 at `/home/frank/miniconda3/envs/mamba_env/bin/python`. Windows operational `Data/processed_v6` is the D: physical directory; the development checkout symlink resolves there. The read-only interactive query shell in `Ubuntu` had `MARKETMAMBA_DATA_ROOT` unset; the scheduled process environment was not captured; the Windows checkout V6 config falls back to its own `Data/`. Colab remains separate. No path or runtime was changed.

- **Layer 1 — Current State:** this short file and its JSON projection hold active objectives, blockers, next actions, current runtime and branch/worktree context. [Authority Map](Authority_Map.md) determines which evidence wins; [Worktree Register](Worktree_Register.md) records checkout responsibility.
- **Layer 2 — Project Knowledge:** [Domain Notes](../01_Domains/), [GOVERNANCE.md](../GOVERNANCE.md), and stable architecture/decision contracts describe boundaries and durable rules.
- **Layer 3 — History / Evidence:** [History](../02_History/), research contracts/manifests/results, dated runtime observations, reports, Obsidian and Git history preserve completed or superseded work. Machine contracts, hashes, source and executable tests bind technical claims; historical prose does not override them.
- `AGENTS.md` and `CLAUDE.md` are the agent operating contract outside the three memory layers.

## Knowledge Governance State

| Topic | State | Authority / notes |
|---|---|---|
| Domain layer | `COMPLETE_CURRENT_SCOPE` | Twelve reviewed domain notes; machine contracts still bind technical truth. |
| Governance | `ACTIVE` | [`knowledge/GOVERNANCE.md`](../GOVERNANCE.md); impact, supersession, runtime-freshness and history protocols are defined. |
| Unified health | `AVAILABLE` | `tools/project_knowledge/knowledge_health.py`; PASS/WARNING are non-blocking, structural FAIL is blocking. |
| Runtime observation store | `ACTIVE_WITH_STALE_SEMANTICS` | `knowledge/runtime-observations/`; timestamps change only after an actual observation is recorded. |
| Generated navigation | `DETERMINISTIC` | `generate_indexes.py`; declared generated files are not hand-edited. |

## Production State

| Topic | State | Authority / observation |
|---|---|---|
| Active scheduler path | `ACTIVE`, observed 2026-09-22 | `MarketMamba_DataFetch` (Ready/enabled) → `wscript.exe` → `run_hidden.vbs` → D: `v62_daily.bat --fetch-only` → `wsl -d Ubuntu` → `mamba_env` → D: checkout `V6/run_v62_daily.py`. [Scheduler observation](../runtime-observations/scheduler-20260922T132135+0800.json). Last run 2026-09-21 22:30 +08:00, Task result 0; log exit 0 at 22:39:52. |
| Fetch-only behavior | `ACTIVE` | [`V6/scripts/v62_daily.bat`](../../V6/scripts/v62_daily.bat) and [`V6/run_v62_daily.py`](../../V6/run_v62_daily.py) stop before matrix, inference, portfolio and publication. |
| Operational data freshness | `10/10 MATCH_REFERENCE_DATE` | Ten `DAILY_SOURCES` have Parquet metadata max date 2026-09-21, matching `prices_raw` and last fetch-only log. This is date completeness, not data-quality certification; log recorded a stock-universe delta warning. [Operational observation](../runtime-observations/operational-path-20260922T132521+0800.json). |
| Full V6.2 schedule | `DISABLED` | `MarketMamba_V62` is disabled; last run 2026-08-25 22:15, last result 255. Code remains manually runnable. |
| Result publication | `DISABLED_AS_SCHEDULED`; manual state `UNKNOWN` | Full pipeline contains publication logic, but no active scheduled publication was observed. |
| API deployment | `UNKNOWN` | Source exists; live deployment/traffic was not verified. |
| Frontend deployment | `UNKNOWN` | Source exists; live deployment/traffic was not verified. |

Runtime observations are not permanent facts. The new Scheduler observation expires after `2026-09-29T13:21:35+08:00`; the prior 2026-09-20 observation remains preserved. A successful fetch-only run does not prove the next run, deployment, trading, or complete data validity.

## Data / Correctness State

| Topic | State | Authority |
|---|---|---|
| Reusable PIT/provenance source | `IN_MAIN_RESEARCH_ONLY` | Six fail-closed primitives and their focused tests were adopted from `fix/v7-historical-pit-reconstruction@e3bce91`; source and policy bytes match branch blobs, with one test's extra EOF blank line normalized. Blob map: `research/v7/source-adoption-register-20260922.json`. Frozen experiment evidence/readiness remains branch-qualified. |
| Historical-simulation readiness | `PASS` | `fix/v7-historical-pit-reconstruction@e3bce91:research/v7/ohlcv-provenance-remediation-v1/readiness.json` |
| Strict Phase 0 | `STOP` | Same readiness evidence and corrected Phase 0 audit; it is intentionally separate from historical-simulation readiness. |
| Verified executable labels | `5d=0`, `10d=0` | Corrected E5/E1 machine contracts. |
| Corrected feature contract | `48 FEATURES`, ordered and manifest-pinned | `fix/v7-historical-pit-reconstruction@e3bce91:research/v7/corrected-baseline-v1/feature-manifest.json` |
| Graph | `DISABLED`, zero edges, fusion branch absent | Completed corrected E5 contract. |
| Industry neutralization | `DISABLED` | Corrected baseline and E1 contracts. |
| Historical labels | `RESEARCH_ONLY`, evidence class `HISTORICAL_SIMULATION_PROXY`, policy `P0` | Never call these `VERIFIED_EXECUTABLE` or tradability evidence. |

## Research State

| Topic | State | Authority / notes |
|---|---|---|
| Corrected baseline | `CURRENT_REFERENCE`: `E5-PIT-Clean-v1` | Corrected baseline manifests and completed incumbent contract. |
| Corrected E5 | `COMPLETED_PARTIAL_ENSEMBLE` | Seeds 17 and 29 completed. Architecture authority is `d_model=64`, `d_state=32`; the old 32/8 draft is superseded. |
| Formal seeds | `17`, `29` | Used by diagnostic conclusions. |
| Seed 43 | `INCOMPLETE_EXCLUDED_FROM_FORMAL_CONCLUSIONS` | Pointer metadata referenced `best_epoch=8`, but pointer-named best/latest/previous generations are missing. The surviving orphan checkpoint is epoch 15, step 39,915, `terminal=false`, contract/matrix matched, and is convergence evidence only. It is not a formal seed and must not be promoted or directly resumed. |
| Extended diagnostic | `STOP_FOR_TRADING_CONTINUE_RESEARCH` | Broad 2026 degradation and near-redundant heads; authority is diagnostic v2. |
| E1 rolling-origin refresh | `COMPLETED_RESEARCH_EXPERIMENT`; `PASS_REFRESH_HYPOTHESIS` | Formal seeds 17/29; `eec63c5:research/v7/e1-rolling-origin-refresh-v1/results/result-manifest.json`. The specification retains its pre-run wording as history. Historical-simulation proxy only; strict Phase 0 `STOP`. |
| E1 post-run diagnostic | `MONTH_SENSITIVITY_FOUND`; `EPOCH_DEGRADATION_INCONCLUSIVE`; `REGIME_DEPENDENCE_INCONCLUSIVE` | `ef020e5:research/v7/e1-postrun-diagnostic-v1/diagnostic-summary.json`. Logged validation declines after epoch 1; final-support replay and causal regime attribution remain unavailable. |

## Production-System State

| Topic | State | Authority / notes |
|---|---|---|
| Architecture audit | `COMPLETE_CURRENT_REFERENCE` | `docs/production-system-architecture-audit@9efed7b:reports/marketmamba_production_system_audit.md` and its production-system roadmap. |
| Phase A implementation | `IN_DEVELOPMENT_MAIN`, `NOT_DEPLOYED` | Fixed `89f9496` integrated locally; original source and candidate branches retained. |
| Phase A task state | Tasks 1–3 `COMPLETED_REVIEWED`; Task 4 limited approval at `1e119ef`; Task 5 limited offline approval at `1c5250e` | Limited local engineering integration completed; production readiness remains outside scope. |
| Current blocker | `NONE_FOR_LOCAL_PHASE_A_INTEGRATION` | Completed local integration; production and research gates remain unchanged. |
| Production success authority | Legacy fetch result remains authoritative | Shadow-ledger failures must not change fetch-only success semantics. |

## Delivery/UI State

- Tracked FastAPI and React legacy delivery surfaces exist; live deployment and deployed revision remain `UNKNOWN` because public endpoint queries were inaccessible and provider dashboards were not accessed.
- V7 status API/UI source is `COMMITTED_NOT_DEPLOYED` in the WSL development main checkout. The public API projects only the summary schema/state/decision/data ID/time/counts; original health artifacts remain local research evidence. This source commit does not change Windows operational checkout or live deployment authority.
- V7 status design is read-only and fail-closed, but live behavior is `UNKNOWN`.

## Active Blockers

1. `STOP`: strict Phase 0 has no verified executable 5d/10d labels.
2. `NOT_DEPLOYED`: Phase A local integration complete, no production promotion.
3. `UNKNOWN`: E1 final-support epoch dynamics and causal regime dependence require the frozen matrix and ex-ante market-state evidence; completed result is admitted.
4. `UNKNOWN`: live API/frontend deployment and publication state.

## Current Decisions / Next Actions

The first Autonomous Charter window is complete. Use the single backlog linked
above; all five selected units are committed/reviewed. No automatic continuation.
Next Human decisions: separate exact-candidate integration, future bounded CPU
window, original E1 matrix location if available, and research admission/design/
compute. Strict gates remain unchanged; no need to repeat completed H0 or E1 search.

## Active worktree boundary

Observed 2026-10-21: main plus16 linked worktrees (17 total after this window's
creation). Only `development/v7-autonomous-charter` changed; main remains6d74d738.
All other checkouts were clean at inventory and preserved. The new tree retains
the closed window for review; no tree removal authorized. See Worktree Register.
No Windows runtime observation was refreshed.

## Repository Rationalization Inventory

Phase 3A read-only inventory completed on 2026-09-22; report: `D:\Desktop\work\project-audit\market-mamba\repository-rationalization\MANIFEST.md`. It classified the three tracked dirty files, 69 collapsed untracked status entries (73 files), eight worktrees, local branches, large artifacts and duplicate candidates. That phase made no source/data/worktree/branch cleanup, Windows checkout sync, push or deployment. Its evidence does not alter research, scheduler or deployment authority.

Phase 3B-1 source canonicalization completed on 2026-09-22 in WSL development main; `D:\Desktop\work\project-audit\market-mamba\phase3b-source-cleanup\MANIFEST.md` is its external audit artifact. The V7 status UI/API is `COMMITTED_NOT_DEPLOYED`, E1-matching environment identity is in Git, early V7 source has a historical SHA register, and local research/task notes are preserved and explicitly dated. `research/v7/current-state.md` defers to this canonical state file. The development main source working tree was clean at completion. Corrected E5/E1/recovery/postrun and Phase A implementation/result authority remain branch-qualified for Phase 3B-2 selective review. No worktree, branch, large artifact or Windows operational checkout was changed, and nothing was pushed or deployed.

## Authority Links

- [Authority Map](Authority_Map.md)
- [Domain knowledge index](../README.md)
- [K0 discovery report](../../reports/marketmamba_k0_project_knowledge_discovery.md)
- [K0 machine authority map](../../research/project-knowledge/k0/authority-map.json)
- [K0 boundary map](../../research/project-knowledge/k0/boundary-map.json)
- Corrected/PIT authorities are branch-qualified in [Authority Map](Authority_Map.md).

## Historical bounded recovery checkpoint — 2026-10-20

The observation-time pending approvals and suggested next steps in this section
are superseded by the current 2026-10-21 decision summary above. They are not
instructions to repeat completed work.

### Snapshot Reader contract continuation (Unit A)

Base `8309ee366aea4c002f9a79d0d97d214b29578fcf`, same isolated branch.
The v1 reader now rejects traversal, absolute/noncanonical raw paths, symlink
files/directories, non-regular files, duplicate keys, malformed/missing/wrong-type
fields, digest/request/publication identity mismatch and incomplete publication.
Directory-relative no-follow reads are anchored at the caller-selected snapshot
root; root ancestors must remain trusted/stable. Integrity is not authenticity,
HTTP success, PIT eligibility or Human correctness approval.

Two repair rounds: initial reader validation (22 failing assertions before fix);
independent review then reproduced nested JSON float overflow (`1e999`), closed
with finite-number parsing. Valid v1 GET/POST, empty payload, naive/offset/UTC
writer timestamps, HTTP error observations and finite floats stay compatible.
Final focused snapshot tests: 17/17; six PIT suites: 47/47 (zero skips);
governance tests: 15/15. Independent reviewer reran 17/17 and approved the
second-round reader changes with no remaining required finding.
No research data was read or modified; all malformed manifests are offline
fixtures. Strict Phase 0 STOP and Task 4 approval pending remain unchanged.
Reader implementation committed as `0a7928061755855faf9aeb55c1702e26a5a62a34`.
The next authorized unit (Task 4 evidence recovery) completed below, on its
separate operations lineage; no operations implementation was imported here.

### Phase A Task 4 evidence recovery (Unit B)

Isolated `fix/phase-a-task4-evidence@46df7b3402f85877a5fc55600d53b15a6f3a7849`
derives directly from original `f5b0ff3df7645098cc01a9eeb36f5df22c4a3e41`.
The original checkout remains unchanged. Evidence index at that commit:
`research/production-system/phase-a/task4-evidence-recovery.json`.
It records specification/report hashes, exact test names, commands and limits.

Recovered a deterministic defect: one artifact referenced by two stages was
counted once but its completed flag and bytes were summed twice. The fix filters
artifact identities by attempt membership before aggregation. Public LedgerStore
fixtures reproduce the baseline failure (`complete_count` 2 instead of 1);
the candidate also verifies equal-sized distinct artifacts, latest empty retry
isolation and unchanged logical ledger contents.

Two bounded rounds: query fix after genuine RED; review-driven test strengthening
from connection-local change count to complete before/after ledger dump.
Final offline ops 70/70 and mocked fetch-only regression 3/3 PASS. Independent
reviewer ran 70+3 in round 1 and reviewed round 2 without rerunning tests; no
Required/Critical finding. These are engineering evidence, not Human acceptance.

At the initial `46df7b3` checkpoint, Task 4 correctness approval remained
**NOT_ESTABLISHED**. The sequential rebuild
test does not prove concurrent committed-only isolation: uncommitted-writer and
mid-read-commit tests are still missing. Sanitizer regression corpus passes but
requires final Human review. Task 5's process crash/restart/replay, >=10 outcome
comparisons, formal artifacts and overhead benchmark remain **NOT_STARTED**.
Do not proceed directly as though Task 4 were approved. Next bounded proposal:
close the two snapshot-isolation evidence gaps, present Task 4 for Human review,
then separately authorize offline Task 5. No research rules/data changed;
strict Phase 0 **STOP** and all research authority remain unchanged.

### Historical Task 4 isolation checkpoint — prior to 2026-10-21 Human approval

`fix/phase-a-task4-evidence@1e119ef855f8fd319f774162beca0bd729d4b921`
adds tests and evidence only, directly after `46df7b3`. The two isolation gaps
above are now closed for the tested SQLite WAL/synthetic-ledger contract:
real spawned writer/reader, uncommitted commit/rollback/SIGKILL cases, and four
deterministic reader boundaries repeated twice. No sleep-based ordering.
Whole projection identity/counts and digest are checked; an explicit no-BEGIN
negative control detects a mixed version. No production correctness bug found
or production code changed. No production repair rounds were needed.

Final Operations 75/75; mocked fetch-only 3/3; separate snapshot branch PIT
47/47 and governance 15/15 PASS. Independent reviewer ran the five isolation
tests (12 scenarios), no Required/Critical finding. Optional cleanup-return
assertion improvement remains deferred; successful child exits/liveness were
directly checked. No cross-branch integration was performed or claimed.

Human review package: Task 4 branch
`research/production-system/phase-a/task4-evidence-recovery.json`,
with 12 raw records and hashes in `isolation-evidence-20261020/`.
Evidence is sufficient to present the bounded Task 4 engineering correctness
decision; **NOT_APPROVED** until Human decides. Tests do not establish research
PIT/label truth, arbitrary-secret absence, power-loss durability or production
readiness. Task 5 remains **NOT_STARTED** and requires a separate authorization:
fixed >=10 observed-vs-shadow replay cases, crash/restart/concurrency, formal
artifacts/report and overhead evidence. Strict Phase 0 remains **STOP**.

- **Completed:** reproduced silent manifest replacement for the same request and
  retrieval timestamp, including parser-only conflicts; identical retries also
  replaced inode/mtime. Replaced shared temporary-name/overwrite publication with
  unique staged files and atomic no-clobber publication. Identical bytes reuse
  the committed file; conflicts, corrupt raw content and manifest symlinks fail
  closed without replacing the original. Valid raw blobs created before a
  rejected manifest remain preserved and do not imply a completed observation.
- **Evidence:** snapshot suite 10/10; all six adopted PIT primitive suites
  40/40, no skips, CPU/local synthetic fixtures, ResourceWarning promoted to
  error. Real competing subprocesses yield one committed manifest; controlled
  SIGKILL before/after manifest publication yields safe retry and valid hashes.
  This is engineering/fault-injection evidence, not market-data or trading
  correctness. Independent review found no required code changes.
  Governance regression 15/15; syntax/JSON and `git diff --check` passed.
  Knowledge-impact review completed; unified knowledge health exits 0 with
  pre-existing Governance/Authority Drift/Legacy Docs WARNING, no structural
  FAIL. Operations/GOVERNANCE rules reviewed and unchanged: no control-plane,
  authority or lifecycle policy change. Full project/model-training tests were
  not run; verification is bounded to the six PIT primitives and governance.
- **Runtime:** Python 3.12.3; related pandas tests use the existing
  `.worktrees/v7-e1-colab-recovery/.venv/bin/python` with GPU hidden. No installs,
  model training or data writes. `MARKETMAMBA_DATA_ROOT` was unset; canonical
  development `Data/processed_v6` resolves to
  `/mnt/d/Desktop/work/ProjectForMe/MarketMamba/Data/processed_v6`; that data was
  not loaded or mounted into the new worktree. Only temporary fixture data used.
- **Limits:** trusted snapshot directories and hard-link/fsync-capable filesystem
  required; unsupported storage fails, with no overwrite fallback. No certification
  of Windows/Drive semantics, adversarial directory mutation or host power-loss
  durability (new-directory parents are not fsynced). Killed-writer temporary
  files stay non-authoritative and are not auto-cleaned.
- **Remaining gates:** strict Phase 0 STOP, verified labels 5d/10d = 0; Phase A
  Task 4 approval not established / Task 5 not started; E1 matrix/replay and
  causal-regime evidence limitations unchanged. No merge, deployment or approval.
- **Resume:** inspect this branch and commit/tests before using the candidate;
  the main source-adoption register remains historical byte-identity evidence.
  Existing memory-hardening branch `91db7d1` remains unmerged; no new hooks or
  memory system were installed. Next bounded candidate: validate source snapshot
  reader path/schema rejection with offline fixtures, without admitting new data.

Reproduce the bounded regression from the candidate worktree:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' /home/frank/projects/MarketMamba/.worktrees/v7-e1-colab-recovery/.venv/bin/python -B -W error::ResourceWarning -m unittest V6.experimental.v7_source_snapshot_test V6.experimental.v7_temporal_contract_test V6.experimental.v7_executable_labels_test V6.experimental.v7_ohlcv_provenance_test V6.experimental.v7_historical_security_master_test V6.experimental.v7_historical_label_materialize_test -v
```

## Knowledge-impact Rule

Update canonical knowledge in the same change for: semantic changes, authority changes, subsystem-boundary changes, active runtime-state changes, correctness-gate changes, new canonical artifacts, deprecations, or major blocker/next-decision changes.

No knowledge update is normally required for formatting, comments, pure refactors, tests-only changes, or internal renames that do not change public semantics or authority.

## Phase 3B-2 selective source integration (2026-09-22)

Development main now contains the reusable corrected PIT primitives, corrected E5 research core, formal E1 research core and frozen contract copies, plus the E1 telemetry correction and reusable Colab log wrapper. The [source adoption register](../../research/v7/source-adoption-register-20260922.json) records source commits, blobs, destinations and test evidence. This is research source authority in main, **not** result, portfolio, operational, Colab run or deployment promotion. The old 32/8 baseline draft remains branch-qualified; the accepted E5/E1 architecture is 64/32. E1 historical-simulation `PASS_REFRESH_HYPOTHESIS` and strict Phase 0 `STOP` remain unchanged. E1 postrun diagnostics remain active; Phase A Task 4 approval and Task 5 acceptance remain open. Eight worktrees were inspected and retained; no branch/worktree/artifact was removed. Windows operational checkout, Scheduler and deployment were not changed; no push.

## Phase 3B-3A governance checkout retirement (2026-09-22)

The four authorized K0/K1/K2 governance checkouts were removed using normal Git worktree removal after clean-state, ignored-file, branch-history and path-dependency checks. Their branch refs and commits remain as `HISTORICAL_LINEAGE / NO_ACTIVE_CHECKOUT`; the four E1/Phase A worktrees remain present. Details and local bytecode disposition are in [Worktree Register](Worktree_Register.md). No research artifact, branch, Windows operational checkout, deployment or scheduler was changed; no push.

## Phase 3C storage audit (2026-09-22)

Read-only reachability/rebuildability audit completed; [external manifest](/mnt/d/Desktop/work/project-audit/market-mamba/storage-audit/MANIFEST.md). No storage item was deleted. D: scheduled fetch-only inputs/outputs, historical baseline caches, dated ZIPs, E1 formal bundle/results, V7 provenance/label snapshots and active recovery `.venv` remain protected. A clean offline frontend install/build demonstrated that local `node_modules` and `dist` are future low-risk generated cleanup candidates (~187 MB), subject to separate cleanup authorization. Large-cache/archive deletion has no approved exact survivor or rebuild path. `DATA_QUALITY_DIAGNOSTIC` remains separate.

## Repository/governance rationalization closure (2026-09-22)

The **repository/governance rationalization program** is complete: canonical governance and authority navigation are established, runtime boundaries were reconciled, development main source is clean, reusable PIT/E5/E1 research source was selectively integrated, historical source was preserved, the four K governance checkouts were retired with branch history retained, and storage received a retention decision without deletion. The E5/E1 refresh, E1 recovery, E1 postrun, and Phase A worktrees remain intentionally present. Development main is ready for remote publication; the finalization audit records the actual Git push and remote-HEAD verification separately.

This closure does **not** complete V7 research, strict Phase 0 (`STOP`), E1 postrun diagnostics, Phase A acceptance, `DATA_QUALITY_DIAGNOSTIC`, Windows operational promotion, backend/frontend deployment verification or promotion, or large-storage cleanup. The Windows operational checkout and Scheduler remain separate and unchanged. Research result authority and branch-local operations authority remain as stated above and in the Authority Map.
