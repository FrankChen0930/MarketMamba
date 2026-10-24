# Window13 — Human Decision Brief

**DECISION_PREPARATION_ONLY / NOT_ADMITTED / UNMERGED / NOT_DEPLOYED。**
目的：讓下一次只需決定資料範圍與研究參數；不重開完整 Roadmap、Feature Audit 或 GPU Audit。
Main/base `8db5b911`；W06 `96c60564`、Protocol Audit `8601c737`、Recovery `e181f366`、
W12 `16e444f9` 都是固定來源引用，未合併、未升格為 main 或正式新研究契約。
來源完整SHA／路徑／hash見 [驗證紀錄](../research/v7/window13-verification.json)。

## 1. 已確認與仍待決定

Human-confirmed direction：cross-sectional ranking、Expanding 初始歷史方向、fresh-state
Refit、Local-first Mamba-2 工程候選；偏好下一合格交易日開盤前、07–08 Asia/Taipei 預測。
新研究的5d主要／10d次要、Mean Daily RankIC5 Selection、約12月Validation／一季Forward／
3 Origins、Fixed Recipe／Refit Budget 是待凍結候選，不能複用 E1 正式參數冒充新批准。
Inference Recovery 僅設計封存、實作 DEFERRED。以下所有「建議」均 PROPOSED。

Thermal 已封存 [Decision Record](../knowledge/02_History/Decisions/20261024-window13-thermal-admission-preparation.json)：
87°C target reference／未來受控測試stop方向，以Windows Host nvidia-smi為量測來源；
83°C初步預警仍是候選。CPU過熱、監測失效或硬體異常可提前停止，不等Epoch。
**不是NVIDIA長時間安全認證，也不是GPU重測授權。** 原W12 80°C gate／HALTED budget
及32.106942s記帳未動；Kernel、finite FP32 F/B/step、GPU Resume、AMP、代表全模型
shape／VRAM／throughput 全部未驗證。正式長訓前重新確認散熱/RAM/VRAM/接電/整機穩定性。

## 2. Data/PIT：目前可以信任到哪裡

| 現有證據 | 可支持 | 不能支持／本輪結果 |
|---|---|---|
| W05完整48-feature audit，與65462ec原檔byte相同，順序與frozen manifest相同 | 0 VERIFIED／21 CONDITIONAL／27 UNKNOWN／0 INVALID；公式/來源風險盤點 | 非row-level歷史PIT比例；無新raw48panel、missingness/correlation或績效統計 |
| W06指定2份TPEx 2022-09-01 raw，2026-09-17取得 | 本輪復驗manifest與176,096 raw bytes／SHA，當前snapshot完整；來源段為零股交易與零股彙總 | 不是普通整股OHLCV或48-feature panel；歷史發布/availability/QC、完整修訂鏈、universe/calendar及labels未知／缺失 |
| W06三來源族群/10個fixture sessions/13cases | aware timestamps、DST、missing≠0、late/revision fail-closed的工程行為 | sessions/latencies多為synthetic；一般市場收盤時刻≠provider當時發布或本地取得證明 |
| W07–W11 source/contracts/tests、bundle/checkpoint保存 | 已有時間/role隔離、common-support evaluator、artifact/replay工具及受限CPUresume證據 | 不自動認證historical48、calendar、labels、GPU或FULL_REPRODUCTION_READY；hash≠可還原bytes |

在**僅本次指定樣本**內，找到0個可組成新研究全48-reference的合格historical panel。
不是宣稱所有分支/資料來源都沒有資料。本輪沒有搜尋其他歷史位置、讀Windows operational
payload、下載新資料、重建E1 matrix或製造真實RankIC；不足即停止這條實證分支。
W11 83檔/1,788,382bytes與W12 5份runtime diagnostics均按既有inventory核實保留。

合格資料還需逐sample的 observation／exact-version publication／source availability／
local receipt／QC PASS／revision pin、交易所calendar與歷史universe/security lifecycle，
以及現有label定義的entry/exit、maturity、價格/label來源資格；可恢復bytes+manifest身份。
現在驗hash通過只證明保存的bytes，不能回填當時QC、發布時間或完整原始filing版本。
W05 SPX/QQQ映射、日期join、missing→0、macro vintage問題仍是來源風險，不改歷史結果。
月營收沿用W05 NOT_CURRENTLY_FEASIBLE（已稽核2002–2026/2334issuer/455040rows，
歷史publication 0 verified）；不重新搜尋，也不把Forward vintages當歷史恢復。

## 3. 最小 Decision-Time 決策（請一次明確選擇；全部尚未批准）

| ID | Human需凍結 | 建議與取捨 | 未知／所需證據 |
|---|---|---|---|
| DT1 | 正式採pre-next-open？固定freeze或bounded-wait？實際freeze、latest deadline、推論/export buffer | 延續pre-next-open方向；第一份可比較研究優先考慮fixed freeze，資訊預算易重現；bounded-wait可提高必要來源到齊率，但每日資訊集合不同且易selection bias | 不在此替Human填07:00或08:00硬截止。freeze≤decision<next qualified entry open，季度refit須在model-ready origin前另行完成 |
| DT2 | Required/Optional source及missing時skip-session/skip-stock/顯式missing表示；coverage分母 | 優先任何實際使用欄位都需clock/QC/vintage證據；缺失保持unknown，不當0。相同被拒session/stock需保留在coverage | 所有48維若維持原recipe，每個used dependency都必要；把institutional/macro等改optional需另批feature/mask/recipe，不能偷偷drop或fill原manifest |
| DT3 | Late/stale/revised eligibility | Late：freeze前未合格者不進舊snapshot；Missing：skip或另批missing-aware recipe；Stale：reject或明訂各來源age上限；Revised：原pin不覆寫，新版本只進未來新snapshot | 需分來源freshness/age、最大wait、outage條件與版本規則；不假定ffill或晚到修正可補回已產生prediction |
| DT4 | 歷史可用性證據最低標準、可接受research evidence class、資料保存位置/還原方式 | 先資格小型具來源panel；不足則另批Forward Vintage Collection累積。以raw/version/receipt/QC/日曆/coverage保存，避免只留hash | Forward collection不能補證歷史publication或未曝光Final期間；也不能先收後聲稱回溯PIT。Source許可/保存期限/容量仍待決定 |

Required候選分類：訊號/entry calendar、security/universe、被recipe使用的price及derived
完整依賴；訓練/評估再加合格label entry/exit/provenance/maturity。Institutional、margin、
foreign holdings、US/global/macro **只是未來可選feature-family候選**：若保留48-reference，
不能視為可任意缺失。月營收不在正式48內且歷史PIT blocked，不列本輪必要來源。

Fixed-freeze不能只寫某日日期；須保留預定/實際freeze與receipt/QC，晚到不變更既有輸出。
Bounded-wait須先定required-set、上限與timeout後skip規則，不能看prediction/labels後決定等多久。
美股觀測日可不同於台股signal日：W06時間換算支持此可能性，**每個provider仍需實際發布/
取得/QC先於freeze**。SPX、TNX、FX不能一律套NYSE 16:00，日期shift也不能修復缺失vintage。

## 4. Evaluation：執行前還缺的最小凍結項

| ID | 尚未凍結的項目／可行選項 | 現有可重用／真正阻塞 |
|---|---|---|
| EV1 | 合格歷史起點、實際train/validation/forward ranges、3個origin是否採用、model-ready deadline/activation | W07 session-calendar fold guard；日期與calendar/data admission屬POLICY_DECISION_REQUIRED+DATA_EVIDENCE_BLOCKED，不能從12mo/3mo自動產生批准日期 |
| EV2 | 保持既有5/10 entry/exit定義下，joint maturity或按head eligibility/masking；required heads；purge資訊重疊與額外embargo | W07逐交易session/label interval/maturity；不得用panel列數或直接沿舊30-session。任何新mask政策需批准，無新label定義 |
| EV3 | RankIC tie、minimum valid N、missing/constant/insufficient處理、按日aggregation權重與common-support denominators | W08可配置具名policy、daily ledger與paired delta；synthetic average-tie/minN2/equal-day不是正式預設，不可計算值不是0 |
| EV4 | Selection僅validation；metric及secondary用法、固定candidate/seed/search budget、失敗run記錄；fresh refit用epoch或optimizer-step換算、固定refit budget | W07 isolation/exposure + W10/11 synthetic adapters/E2E。Refit新Run ID/全新model-optimizer-scheduler-RNG，不可用併入refit的validation再earlystop；不得與resume混同 |
| EV5 | 統計uncertainty/block length/CI/multiplicity/acceptance與真正independent final支持集、exposure ledger | 已曝光E5/E1期間/已被用來挑方案的forward不再untouched；3origins不保證獨立或顯著。沒有新final identity時標未完成，不選一段已看過結果的日期代替 |
| EV6 | 具體L0/Ridge/L2/Mamba recipe及資料表示差異、資源/保存/compute授權 | Classical tiny synthetic已驗，非real-data fit授權；CPU真研究仍須另批。GPU安全halt另線blocked；不必等GPU才規劃資料/CPU研究准入 |

共同support需相同fold/session/security/horizon/label contract和exact-version identity；
逐日成對算IC及delta，保存排除原因與coverage，不能各算各的mean後稱公平改善。
若模型輸入是tabular48與60-session序列，必須披露資訊集合差異，不能將結果全歸架構。
研究label若採明確已批准的historical-simulation proxy仍須獨立研究准入及限制揭露，
不等於VERIFIED_EXECUTABLE，亦不必先宣稱已解除Strict Phase0才能討論研究資料資格。

## 5. 工程、政策、資料、runtime 分開判斷

| 分類 | 現狀 | 不自動推得 |
|---|---|---|
| ENGINEERING_READY（限定scope） | W07 Fold/成熟/role guards；W08 common RankIC/paired support；W09 immutable bundle/replay；W10/11 L0/Ridge/L2 tiny synthetic E2E、受限CPUresume既有證據 | 不是全新historical loader、真研究配置或production readiness。此次只復驗W07–09/Snapshot/治理140tests、W06原branch13tests；不重跑模型fit/resume |
| POLICY_DECISION_REQUIRED | DT1–4、EV1–6、正式data/evaluation/admission與compute授權 | Human-confirmed方向不能填入尚未批准的時刻/日期/N/threshold/seeds |
| DATA_EVIDENCE_BLOCKED | 全48歷史clock/vintage/QC、qualified calendar/universe、label/support/restore身份未在指定樣本滿足；E1 matrix仍缺 | 不以hash、synthetic或今日歷史下載證成當時可得；不再無線索搜尋E1 |
| RUNTIME_BLOCKED | GPU SAFETY_HALTED，hostRAM連續觀測/散熱、官方kernel完成與GPUresume/AMP/代表截面均未認證 | 新87C設計方向不解除舊HALTED、不代表Local/Colab優劣或正式Training Ready |

建議下輪只解除三個實際依賴（均需新有界授權）：
1. **先DT1–4：** Human選freeze/缺漏/source/evidence-class；才可對指定小panel做qualification。
2. **資料最小證據：** Human指定可讀immutable bytes與合格calendar/universe/label lineage；
   若無歷史clock，改審Forward Vintage Collection的保存規格與唯讀來源範圍，而非重建歷史。
3. **再EV1–6：** 同一次凍結日期/support/maturity/purge、selection/refit/statistics/final與CPU
   baseline compute/storage邊界；有qualified資料才另批第一個小型CPU研究。GPU retest獨立延後。

## 6. 驗證、持久化與停止

驗證以原source/已有工具為主，無新Evaluator/Trainer/Data Platform/Registry或thermal framework。
W06原branch測試13 PASS；main共140 CPU tests PASS。新raw範圍僅2 manifest+2 raw，
網路請求0，realRankIC0，GPU呼叫0。驗證檔記錄24個固定source identities及88個保護檔案。
最後review/rework/commit狀態見 [evidence](../research/v7/window13-evidence.json)（closure生成）。

最小恢復：cd `.worktrees/v7-window13-admission-preparation`，查branch/HEAD/clean；讀本頁、
CurrentState、AuthorityMap、GOVERNANCE、唯一research-program Window13和verification。
本輪完成後STOP；不自行整合main、不啟動下一研究／資料收集／GPU窗口。
