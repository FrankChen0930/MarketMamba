# Step3b 准入與最大折資源量測接續 — 2026-11-26

## Step3c 獨立審查修補 — 2026-11-26（目前狀態）

**HUMAN_DECISION_REQUIRED / CPU_ENGINEERING_ONLY / RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED。** 本節取代下方 Step3b 的「目前歷史 PASS」主張；舊節保留為歷史工程紀錄。Human 本輪只核准有界修補，不是接受來源 gate。凍結規格 SHA `9cd7cb8681e3d881193efa3c6fa3c72bd74fba84a96f6a16e6da3a85398eaf34` 不變。起始 clean、detached `31f89ed95feb182ccdc4669cc4c09f5766e1cd08`；不 commit／push／merge，不讀 blind，不執行 fits／真實評估或查看 IC／績效。

### 來源證據與限制

TEJ HTML 只能支持原件中的特定更新規則、2013/01 轉換及「營收相關資訊最早揭露」描述，沒有 NTD 幣別條款，也不能證明每個匯出值都是首次版本。本輪不再聲稱文件授權 NTD 映射或首次版本。版本 0 只表示此匯出單一版本的重播順序，不是首次版本認證；合併優先範圍／跨期可比性仍是待 Human 接受的歷史假設。

唯讀使用既有 TEJ PIT（SHA `987c7c32636fcaebcff66a6ba774ff37f1c8fca8d20f589e794db8e28b45e44b`）及 FinMind revenue_raw（SHA `573118a101fd8f90c8be707451048ee7843d6d46fff9b4754326efd1d301c942`），按 stock_id／營收年月精確配對，自 2013-01-01 至 2026-10-01；拒絕 reference 重複鍵。比較 `abs(TEJ 單月千元×1000 − FinMind NTD)/abs(FinMind NTD)`；兩者零為差異零，reference 零但 TEJ 非零為無限差異。門檻嚴格 `<0.1%`／`>1%`，分母為可配對且兩值有限者。這是數值與單位交叉一致證據，兩來源都可能事後更正，不能證明 PIT／首次公告或可成交性。以事後跨來源一致來決定哪些歷史值記 NaN 本身也是 hindsight coverage／版本選擇限制，不代表當時可以取得該交叉驗證。

全 380,514 列中，配對 291,847；`<0.1%` 291,360（99.833132%）；`>1%` 446（0.152820%）；中間含端點 41；未配對／非有限 88,667。只有配對一致列標 `NTD_BY_CROSS_SOURCE_AGREEMENT`；其餘標 `UNIT_UNVERIFIED`，營收 NaN，跨期比較依既有缺值規則傳遞，兩臂仍保留共同鍵。沒有比較資料時 adapter 全部維持未驗證，不因原來標 TWD 就自動放行。

KY／外國是代碼／名稱含 KY、DR、外國，或代碼以 91 開頭的代理辨識；無標記不等於已認證本國公司。分類沒有上市當時 domicile authority；不能聲稱完全涵蓋外國公司。

| 年度 | 全列 | 配對 | <0.1% | 一致率 | >1% | >1%比例 |
|---|---:|---:|---:|---:|---:|---:|
| 2013 | 24,583 | 18,403 | 18,398 | 99.972831% | 5 | 0.027169% |
| 2014 | 25,315 | 18,960 | 18,958 | 99.989451% | 2 | 0.010549% |
| 2015 | 25,898 | 19,701 | 19,692 | 99.954317% | 5 | 0.025379% |
| 2016 | 26,193 | 20,224 | 20,212 | 99.940665% | 8 | 0.039557% |
| 2017 | 26,470 | 20,669 | 20,662 | 99.966133% | 4 | 0.019353% |
| 2018 | 26,852 | 21,105 | 21,098 | 99.966833% | 6 | 0.028429% |
| 2019 | 26,990 | 21,309 | 21,304 | 99.976536% | 5 | 0.023464% |
| 2020 | 27,319 | 21,559 | 21,536 | 99.893316% | 23 | 0.106684% |
| 2021 | 27,873 | 21,669 | 21,654 | 99.930777% | 14 | 0.064608% |
| 2022 | 28,436 | 21,633 | 21,611 | 99.898304% | 21 | 0.097074% |
| 2023 | 29,083 | 21,675 | 21,563 | 99.483276% | 111 | 0.512111% |
| 2024 | 29,825 | 22,259 | 22,194 | 99.707983% | 63 | 0.283032% |
| 2025 | 30,468 | 22,996 | 22,894 | 99.556445% | 86 | 0.373978% |
| 2026 | 25,209 | 19,685 | 19,584 | 99.486919% | 93 | 0.472441% |

| 年度／辨識層（KY外國／未見外國標記） | 配對 | <0.1% | 一致率 | >1% | >1%比例 |
|---|---:|---:|---:|---:|---:|
| 2013／KY外國 | 713 | 713 | 100.000000% | 0 | 0.000000% |
| 2013／未見外國標記 | 17,690 | 17,685 | 99.971735% | 5 | 0.028265% |
| 2014／KY外國 | 812 | 812 | 100.000000% | 0 | 0.000000% |
| 2014／未見外國標記 | 18,148 | 18,146 | 99.988980% | 2 | 0.011020% |
| 2015／KY外國 | 947 | 947 | 100.000000% | 0 | 0.000000% |
| 2015／未見外國標記 | 18,754 | 18,745 | 99.952010% | 5 | 0.026661% |
| 2016／KY外國 | 1,092 | 1,092 | 100.000000% | 0 | 0.000000% |
| 2016／未見外國標記 | 19,132 | 19,120 | 99.937278% | 8 | 0.041815% |
| 2017／KY外國 | 1,230 | 1,230 | 100.000000% | 0 | 0.000000% |
| 2017／未見外國標記 | 19,439 | 19,432 | 99.963990% | 4 | 0.020577% |
| 2018／KY外國 | 1,318 | 1,318 | 100.000000% | 0 | 0.000000% |
| 2018／未見外國標記 | 19,787 | 19,780 | 99.964623% | 6 | 0.030323% |
| 2019／KY外國 | 1,246 | 1,246 | 100.000000% | 0 | 0.000000% |
| 2019／未見外國標記 | 20,063 | 20,058 | 99.975079% | 5 | 0.024921% |
| 2020／KY外國 | 1,396 | 1,395 | 99.928367% | 1 | 0.071633% |
| 2020／未見外國標記 | 20,163 | 20,141 | 99.890889% | 22 | 0.109111% |
| 2021／KY外國 | 1,383 | 1,382 | 99.927693% | 1 | 0.072307% |
| 2021／未見外國標記 | 20,286 | 20,272 | 99.930987% | 13 | 0.064084% |
| 2022／KY外國 | 1,327 | 1,327 | 100.000000% | 0 | 0.000000% |
| 2022／未見外國標記 | 20,306 | 20,284 | 99.891658% | 21 | 0.103418% |
| 2023／KY外國 | 1,329 | 1,326 | 99.774266% | 3 | 0.225734% |
| 2023／未見外國標記 | 20,346 | 20,237 | 99.464268% | 108 | 0.530817% |
| 2024／KY外國 | 1,360 | 1,355 | 99.632353% | 5 | 0.367647% |
| 2024／未見外國標記 | 20,899 | 20,839 | 99.712905% | 58 | 0.277525% |
| 2025／KY外國 | 1,433 | 1,423 | 99.302163% | 9 | 0.628053% |
| 2025／未見外國標記 | 21,563 | 21,471 | 99.573343% | 77 | 0.357093% |
| 2026／KY外國 | 1,199 | 1,194 | 99.582986% | 5 | 0.417014% |
| 2026／未見外國標記 | 18,486 | 18,390 | 99.480688% | 88 | 0.476036% |

KY／外國可辨識者：19,321 列、配對 16,785、<0.1% 16,760（99.851057%）、>1% 24。未見標記：361,193 列、配對 275,062、<0.1% 274,600（99.832038%）、>1% 422。

差異 >1% 的 446 列按可觀測數值分類：TEJ_BELOW_REFERENCE=328；TEJ_ABOVE_REFERENCE=111；NEGATIVE_VALUE_PRESENT=5；REFERENCE_ZERO_TEJ_NONZERO=2。此分類不推定原因；公司更正、供應商版本與合併口徑原因仍 UNKNOWN，需原公告才能歸因。逐列證據／差異列只在本 worktree `.artifacts/step3/source-evidence-20261126/pairs.parquet`，完整分層彙總與輸入 SHA 在同目錄 summary.json，公開無公司逐列數值。

### 給 Human 的決定

1. **接受受限歷史 gate**：以原始說明＋跨來源一致作為歷史工程證據，接受公司更正版本未知、發布日僅為「相關資訊最早揭露」、合併可比性及來源未配對造成的缺值限制。其後歷史結果必須明列這些限制，不能升格為確認性／strict 前瞻／交易證據。只有 Human 明確決定並另有執行授權後才可啟動正式研究；本輪沒有代寫接受紀錄，也沒有 fits。
2. **維持 STOP**：歷史檢驗不跑，等待具實際收到／驗證時間及版本鏈的前瞻資料。影響是沒有本案歷史增益估計，需等前瞻成熟及其他 strict gate 完整，不能把本輪跨來源一致率當作前瞻合格證明。

runner 預設拒絕 `HUMAN_DECISION_REQUIRED`。選項 1 的紀錄檔是 JSON，需含 `decision_date`（ISO 日期）、非空 `decision_text`（Human 決定文字）及 `decision: "ACCEPT_HISTORICAL_SOURCE_GATE"`；由 Human 提供，透過 `--human-accepted-source-gate <紀錄檔>` 明示，request 封存原文與 SHA。此旗標只處理來源待決，不能解除 STOP、資料身分／程式 mismatch、完整 schema 或 strict 前瞻 STOP；不自動授權研究／排程。JSON 是決定留痕，並非使用者認證或密碼學簽章。

### 本輪完成驗證補記

最終focused命令及全套timeout1500命令在 `.artifacts/step3/commands-step3c.txt`；最新focused實際輸出（exit0）：

```text
.................................................                        [100%]
49 passed in 9.35s
```

自查另增EUR明確外幣反例，先RED `1 failed, 48 deselected in 0.79s`，再把adapter改成原有TWD／NTD／待證標記的allowlist加上跨來源一致，其他明確幣別即使數值碰巧一致也NaN。上列49項包含此增補；全套collection在此既有測例加assert之前，不能把全套單獨當作新增EUR assertion的執行證據，該邊界由最新focused覆蓋。五欄、標籤、split、seeds與門檻沒有修改。

6個Python AST解析通過；spec SHA完整相符；19個本輪local link targets無失敗；`git diff --check`通過。knowledge-impact為WARNING（受影響knowledge已同change更新），knowledge-health --ci exit0／WARNING，無structural FAIL；詳私有delivery logs。自查依code-review-and-quality五軸進行，不是獨立審查，沒有額外agents。完整bundle仍未完成，不能據tests通過升格72 fits readiness或Human acceptance。

全套使用timeout1500，但在Human本輪30分鐘界線前主動中止，實際exit130／**NOT_COMPLETED**；最後進度約66%，沒有終局passed／failed總計，不宣稱完整regression通過。原始輸出 `.artifacts/step3/full-step3c-final.log` 最後為：

```text
........................................................................ [ 61%]
........................................................................ [ 66%]
..........s.
```

較早全套（首次41項focused版）也已中止exit130，不是最終完成證據。全套、focused及所有prepare／cache程序均已結束，沒有留下背景工作。最終source／runtime manifest為 `.artifacts/step3/manifest-step3c.json`；這是patch與partial inventory，**不是完整72-job admission manifest**。

本輪固定交接：Completed＝來源待決三態與跨來源彙總、72-job prepare／schema程式、prediction resume與label identity／infinity guards、Human旗標及RED→GREEN反例；Changed＝四個tools、tests、本報告及六份knowledge（共12 paths）；Decisions＝來源交Human、其他工程缺漏仍拒絕執行；Evidence＝focused49 passed／9.35s、AST6、specSHA、19 links、diff check、knowledge WARNING；Remaining＝完整72-job物化及逐檔核驗、全套regression、獨立重審、Human決定；Blockers＝本輪時間／I/O及來源待決；Next recommended action＝另段工程預算完成bundle／regression後重審，Human決定另行留痕，不自動研究執行；Project memory updated＝是；Authority map updated＝是（只更新導航與候選狀態）；Git status＝12份unstaged修改，無stage、原先clean，依no-commit保留patch；Commit／Push＝無；Worktree＝既有builder_step3c／detached31f89ed；Runtime/environment＝WSL CPU／pilot09 Python3.12.3，MARKETMAMBA_DATA_ROOT unset，Data realpath本worktree普通目錄；無網路／GPU／blind／正式Data改動／V6/models／P2／P4d／production／scheduler／agents／deploy。

### 實際72-job交付限制（NOT_COMPLETE）

完整72-job的prepare／schema程式已修，但本輪**沒有產出可准入的完整72-job bundle**。原始逐job重算準備及私有build-once快取嘗試，都因本輪30分鐘界線與實測本機I/O進度而停止（各exit130），不是完成證據；未放寬任何門檻、未擬合模型。已產生2015 seed17的兩臂fit矩陣及部分共用陣列；未完成全部fold／seed。快取未完成canonical逐值比較，不可使用或宣稱等價。`base185_cache.npy`、`base185_cache.partial-rowmajor.npy`皆是partial，即使檔案header／logical size完整也不當complete。私有driver／未執行finalizer及所有partial保存，不刪除、不覆寫共用來源。

精確inventory／環境／attempt退出狀態在 `.artifacts/step3/preparation-status-step3c.json`。`source-gate-step3c.json` 明確記錄來源HUMAN_DECISION_REQUIRED與殘餘風險；這不是完整bundle admission，缺bundle仍STOP。因此「完整72-job凍結交付」尚未完成，正式72 fits及評估不可開始，待獨立重審及Human來源決定。本輪沒有製造Human接受檔。後續需沿用既有namespace／manifest規則，在另段工程預算完成可重建矩陣與72-job完整身分核驗；不能直接把本輪partial當完整bundle。

### 工程修補與驗證邊界

沿用 tools／tests／reports／knowledge 及唯一私有 `.artifacts/step3/`，沒有新頂層樹。prepare 擴為 12 折×3 種子×2 臂的實際 fit／eval／target／sample／keys／label_keys／eval_labels 身分，每折 eval 兩臂共用，各種子 target／sample 兩臂共用。已有 shared 僅六個後期 fold，因此只沿用 fold2026 splits 的原 warmup（code 0）遮罩，每年度 fit/eval 候選都由凍結完整日曆及原 strict purge 重新算，不沿用2026的 train/eval 切分、不新增變更 shared split。逐 job 陣列 hash 在 fit 前再驗；完整 bundle 要求恰好72 jobs、全部角色 references、來源 identities，並驗 pinned source manifest 的全部 input/output SHA、原件與目前 implementation SHA。

resume 的 COMPLETE 必須含 prediction.npy 及其 manifest identity；缺檔、刪除 manifest 項或 hash 損壞均拒絕。SIGKILL／完整 orphan／seed isolation 既有測例也補足合成預測檔。評估入口逐列核對 label_keys 與六 job 鍵、row_session 與鍵第一欄、標籤值與 frozen_labels（NaN 等價）；±inf 明確 STOP，不當缺值。新增入口拒絕測試將 scoring／bootstrap 設為禁止呼叫，只驗 defensive guards，沒有執行真實評估。

RED：新增審查反例 8 failed／33 deselected；補強測例再重現 2 failed／47 deselected。GREEN 與最終全套結果見本節完成驗證補記。較早41／49項中間 focused 或被停止的全套不當最終證據。首次 bwrap 加 unshare-net 因環境 NETLINK_ROUTE 限制於啟動前 exit1，改沿用既有資料遮蔽；無任何網路呼叫，未申請權限。全套以 bwrap 遮蔽正式 Data、主 artifacts、真實 fold／local-data；只執行既有合成回歸，不讀真實模型分數。完整 timeout 為1500秒。


**CPU_ENGINEERING_ONLY / RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED。本節 supersede 下方 Step3a 的「尚未授權解除歷史硬鎖」現況敘述；歷史觀測與產物不改寫。沒有執行或查看本案 IC／績效。**

既有 assigned worktree `job-run-mwwebldn-201d27/builder_step3b`，detached HEAD/base `b8c4c8f574ea280057bb87a0a852f8ca0b830ef4`，起始 clean。Human 明確核准本次 bounded admission 與2026 A/B seed17工程量測；不 commit／push／merge、GPU、網路、blind、正式 Data、排程、P4d／P2 或額外 agents。凍結規格不修改。

## 協調者裁定（Claude，待獨立審查）

- A：§3「可核 TEJ 原始欄位說明或逐期原公告」為 OR；已 pin 原始 HTML SHA `2e0e9b5e6897e2eaf7af3565a1344f34cb4b475e1e1da786ee1c58a07dc62dfa`。依條文只接納 period≥2013/01 且有發布日的首次單月值，發布 D 後首 session 收盤可用；個股千元 NTD 轉元。2012 及以前不進比較歷史，因此2013 YoY／跨2012 MoM／YTD／high12／surprise缺值預期。不得使用 TEJ 去年單月／累計／去年累計欄；非NTD／幣別異常記NaN及原因。2013年度匯出去年欄的IFRS轉換值不納入。本裁定建立可用範圍，不要求逐期公告俱全。
- B：完整 identity bundle 是工程待辦，凍結矩陣X、48欄順序、日曆、價格、成員、h20、月營收特徵、程式、共用抽樣、版本文件、source commit與逐檔bytes/SHA；runner每次重驗，不符即STOP。公開只報彙總，逐列及完整manifest只在本worktree `.artifacts/step3/`。
- C：raw ZIP 當前觀測 bytes 與已凍結 SHA 一同凍結；不再要求另份事前 size。SHA仍逐一驗證，不接受內容漂移。
- D：strict 前瞻維持STOP（官方日曆、lifecycle與公司行動證據不足），不阻擋完整價格研究股票之歷史嵌套工程。observed calendar、研究成員與 raw h20不升格可成交／全市場／含息證據。

## 實作與驗證邊界

沿用現有 `tools/step3*.py`／tests／本報告及 `.artifacts/step3/`，新增 `tools/step3_prepare.py` 是既有primitives與shared矩陣／price snapshot的串接入口；既有audit僅有布林選樣，不能承載h20／月營收／共用抽樣／凍結bundle，故此單一入口必要，不新增頂層tree或重建shared矩陣。所有共享來源唯讀，不觸模型權重。實際source/runtime與逐檔身分在私有bundle。

`monthly_events`先移除2013以前月份；adapter只讀首次單月原欄，不讀去年／累計值。非NTD使用UNKNOWN currency，使依賴它的比較為NaN；逐事件缺值原因私存。兩臂以同列建構並逐值核對185欄含NaN相等，保留W60全部推論鍵，不因未來label或B缺值刪列；完整訓練日中心平均同秩後以seed17抽500,000列。2026使用shared原splits排除warmup，凍結observed日曆執行21-session purge。

准入不接受單獨PASS字串：驗spec及完整bundle逐檔size/SHA。fit_job另核year、seed、兩臂sample／target／keys／fit／eval陣列digest與bundle相符，推論不載入評估標籤，僅查預測shape與有限性，私存模型／預測及資源telemetry。此次bundle只登記2026 seed17，其他年／seed均不能直接沿用，需要自己的凍結job identity。

RED：新增反例先得到2 failed／29 passed（2012未排除、缺bundle gate），修正後31 passed；追加adapter禁止去年／累計依賴、USD→NaN及fit陣列不符STOP，focused最終33 passed／5.40s。去年度欄mutation實際注入函式並由assertion殺死。全套以bwrap遮蔽所有外部真實fold／正式Data／主artifacts，防止既有real-fold測例讀取真實研究績效；不執行Step3 evaluation模組。最後新增2個tests於全套collection後加入，由focused另覆蓋；全套數量以實際log為準。

## 重播效能診斷與保留partial

第一輪 `resource-20261126/` 在逐月重播耗時，1,000真實來源列probe為4.064秒；profile顯示每鍵小型DataFrame的drop_duplicates、每session重建itertuples及重複metadata檢查為瓶頸，並非OOM。主動中止exit130，保留原h20／日曆／source-check與partial，不稱完成或准入。

只做不改語義的效能修補：整表logical+value去重後檢查logical碰撞、每股一次itertuples搭配有序groupby、每歷史月份快取已知可比identity／有限單月值。相同1,000列全部特徵（含NaN）與attrs（缺值原因、去重鏈）逐值相等，修補後1.146秒；不是完整資料耗時外推證明。現有版本碰撞、同session亂序、晚到舊月、UNKNOWN與缺月反例全部通過，focused33 passed／2.98秒。新版在獨立 `resource-optimized-20261126/` 重新建構並以實際程式SHA凍結，未減樣本或500,000 cap。

全套程序在效能修補前collection／import，實際終局數字見後節；最終效能修補由33項focused及上述真實列一致性驗證補覆蓋，不能宣稱同一全套程序重載了最後版本Step3函式。沒有為求全套通過而改既有skip／gate。

第二輪已完成全量重播，但對齊時發現pandas3會把15MB attrs深拷貝至每個group，單股分組實測1.29645秒，去除工作attrs後0.015562秒、數值表相等。中止第二輪exit130（非OOM），保留完整monthly_events及event_reasons；最終 `resource-release-20261126/` 先獨立持久化全部metadata，再清除工作表attrs，重新重播／對齊。這是metadata儲存位置與內部效能修補，不新增模型欄位、不刪研究樣本；舊partial均非准入artifact。最終focused及實跑終局如下。

## Step3b 支援與原始檔大小（無績效統計）

| 檢查 | 實際數值 |
|---|---:|
| 原矩陣列數 | 7,423,927 |
| 2013起h20有限研究列 | 5,077,611 |
| 2026完整訓練日候選 | 4,755,330 |
| 2026推論候選（W60前） | 324,395 |
| 最晚候選訓練訊號／出口 | 2025-12-01／2025-12-30 |
| 首個評估session | 2026-01-02 |
| 全候選嚴格purge | True（t+21 < origin） |
| 月營收raw ZIP bytes | 12,571,050 |
| IFRS raw ZIP bytes | 7,290,770 |
| basic raw ZIP bytes | 18,697,958 |

來源SHA仍為下方Step3a表中同三份既定SHA，當前bytes一同凍結；沒有取代raw或重命名歷史artifact。切分檢查私存 `resource-20261126/split_support_check.json`，只驗有限性／數量／日期，不讀取標籤績效。

## 五欄覆蓋（原年度成員列，未套W60，非績效）

| 年 | 分母 | YoY | MoM | YTD YoY | high12 | surprise |
|---|---:|---:|---:|---:|---:|---:|
| 2013 | 316,349 | 0 (0.00%) | 263,849 (83.40%) | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| 2014 | 334,125 | 304,041 (91.00%) | 332,808 (99.61%) | 304,328 (91.08%) | 303,450 (90.82%) | 222,338 (66.54%) |
| 2026 | 324,395 | 322,632 (99.46%) | 323,247 (99.65%) | 323,313 (99.67%) | 324,027 (99.89%) | 322,168 (99.31%) |

2013同比類全缺為口徑限制之預期；2014 surprise暖身不足保留NaN。未知／非NTD欄本批0列，adapter合成USD測例仍驗NaN及原因；只使用千元NTD的原始單月欄，不讀匯出去年／累計值。

## 最終凍結准入與驗證

`resource-release-20261126/admission/admission.json`：歷史 **PASS**、stops=[]、文件條款4/4 True；strict前瞻 **STOP**。完整私有bundle `resource-release-20261126/bundle.json`，45個逐檔身分，13,538 bytes，SHA `5c734cf4997b5d9cc7b7ba6a6c6436290eacb7ac69955d19d2cdc27fa931c824`。准入時fits=0是該階段計數，後續量測須以各job COMPLETE manifest為準，不能拿它當最終fit數。最終準備165.05秒、峰值RSS3,635,872 KiB，exit0，無OOM；ulimit -v 9,961,472 KiB（9.5GiB），並非GPU或外部CPU。

全套隔離命令同下方Step3a，加timeout1500與OPENBLAS_NUM_THREADS=1；本輪 **1383 passed／11 skipped／5 warnings／862.94秒／exit0**，完整log `.artifacts/step3/full-step3b.log`。本輪最後focused **33 passed／3.86秒／exit0**，log `focused-step3b-release.log`；全套載入時31個Step3測例，之後新增2例由focused覆蓋。最後內部效能修補全套未重新載入，已誠實列為驗證範圍限制，不以首次全套代替最終來源全套驗收。沒有新增skip或改既有門檻；既有11 skip包括被隔離的外部真fold及optional依賴，均非本案故意略過失敗。

knowledge-impact逐項：Semantics YES（2013可用範圍），Authority NO（production／machine contracts未變），Subsystem boundary NO（沿用tools），Active local runtime YES（本地工程量測，無外部scheduler觀測），Correctness gate YES（bundle gate取代無條件歷史鎖），Canonical artifact supersession NO（新GENERATED bundle），Deprecation NO，Major blocker YES（歷史准入解除，strict保留）。Current State／Authority Map／Worktree Register／Data／Features同步更新；GOVERNANCE與Operations規則人工檢查後無語義變更，不需改穩定治理條文。knowledge-impact最終含新prepare入口、exit0／WARNING為人工review提示，不是結構FAIL。Builder依code-review-and-quality五軸自查：可比性／PIT／purge／共同列、私有輸出與hash、tools boundary、pandas metadata深拷貝及矩陣記憶體；無獨立review，未取代Human acceptance。

## 公開identity彙總（完整45項只存私有bundle）

| 身分 | bytes | SHA-256 |
|---|---:|---|
| matrix | 1425394112 | `9778392d4c337d1f9cb5768002340a0d1aec13f7689b2f601b77bd2329b55363` |
| feature_order | 4328 | `26862fcbd29b1ef485b1aff981a78abf0fdfc55a4d238c721f89ac0b9d7456cd` |
| calendar | 43184 | `fac6872b4598601caf2071b62adf7563ce6040462140b866be445e523b3f94e0` |
| prices | 231509693 | `294338229d04e20a2708d00ef1a3c37a34f67d173c0b21f441adbc1fe48d9f42` |
| membership | 950262784 | `8d5cc83037d2470f051d8ff40e5e7c8edb2d436e0db195dea78df5fca8567c21` |
| labels | 59391544 | `4bd46c50a7ff2c6ab457ac99d8e2251801476f7e4f0468cda0635784c976c549` |
| features | 148478668 | `8b56d339ab4921330980f3cc512ce6e524c30b5a37394c0fd78f848d6765f645` |
| sampling | 4000128 | `1ce3c4c90b03259d6b8a4687a8c3f8fe62204eafeebeb7aab84ae7598e2b1bfb` |
| version_evidence | 103453 | `2e0e9b5e6897e2eaf7af3565a1344f34cb4b475e1e1da786ee1c58a07dc62dfa` |
| splits | 7424055 | `dfd3fa021b5d29447b67c65ecbc602e990cf92778828d151123c714fce57fa6b` |
| keys | 2417952 | `c3f529548b63c875b9c899b29e2993759d03238c435c0e0a387d5a937577303d` |
| implementation | 17194 | `a7a31887fdf4cfd30dc95c3c31c0e43b5f13b38ea798bcc7fdefa6fb85c30458` |

## 2026最大訓練折資源量測（seed17，無績效）

| 臂 | 擬合秒 | 推論秒 | fit+infer秒 | 峰值RSS KiB | fit矩陣bytes | infer矩陣bytes |
|---|---:|---:|---:|---:|---:|---:|
| A | 34.8334 | 0.9111 | 35.7445 | 1,331,400 | 370,000,000 | 223,648,720 |
| B | 36.0945 | 0.9898 | 37.0843 | 1,365,460 | 380,000,000 | 229,693,280 |

兩個job均COMPLETE／exit0，串行CPU四執行緒，各500,000 fit列／302,228 W60推論列；完整200棵、無早停／選種子，ulimit -v 9,961,472 KiB。程序wall（含import、bundle重驗與陣列hash）A45.83秒；B46.52秒（含身分重驗，詳measure-B-step3b.log）。RSS為整個該process高水位，與準備程序3,635,872 KiB分開；最大值均遠低於保護上限，未發生OOM，無需減樣本或轉用GPU／高RAM付費算力。若將來有新fold的資源問題，優先重用已凍結memory-map／逐job載入，保持研究樣本與cap，不自行改研究配方。

最終 `verification-step3b.json` 重驗完整45項bundle、spec SHA與兩個job所有輸出hash，檢查模型各200棵、共同keys／sample／target SHA一致。prediction.npy只由runner檢查shape／有限性後私存及hash，未查看值、未計IC、未呼叫Step3評估器。這只交付兩fit成本，不是72-fit研究或Human acceptance。

最終knowledge-health --ci **exit0／WARNING**：Project Map、Agent Guides、Domains、Runtime Freshness、Generated Index PASS；Governance／Authority Drift／Legacy Docs WARNING，無structural FAIL。6個Python檔syntax compile、6個本輪直接local link targets、untracked prepare whitespace及git diff --check均通過。全套程序、準備程序與A/B量測程序皆已結束，無留下背景工作。最終45項bundle／兩job的模型與predictions hashes、200棵、共同鍵／抽樣／目標身分已重驗，證據在 `verification-step3b.json`；完整本輪source／outputs／environment inventory在 `manifest-step3b.json`，不納入歷史partial作完成證據。

## Step3b 固定交接欄位

- Completed：依協調者A–D建立歷史admission PASS／strict STOP，凍結45項identity，實跑2026 A/B各seed17共2完整fit＋inference，無IC／績效。
- Changed：`tools/step3.py`、`tools/step3_prepare.py`、`tools/step3_runner.py`、`tests/test_step3.py`、`reports/step3_engineering_20261126.md`、`knowledge/00_Project_Map/Current_State.md`、`knowledge/00_Project_Map/Authority_Map.md`、`knowledge/00_Project_Map/Worktree_Register.md`、`knowledge/01_Domains/Data.md`、`knowledge/01_Domains/Features.md`。
- Decisions：協調者Claude裁定待獨立審查；2013前比較NaN，不用匯出去年／累計欄；bundle／job陣列不符STOP，strict前瞻STOP；只凍結此次2026 seed17，不授權其他fits。
- Evidence：bundle SHA `5c734cf4997b5d9cc7b7ba6a6c6436290eacb7ac69955d19d2cdc27fa931c824`；上表兩fit資源；全套1383 passed／11 skipped／5 warnings／862.94秒，最終focused33 passed／3.86秒；metadata逐值一致性、syntax、hash、links、diff及knowledge gates詳本節與私有logs。
- Remaining：獨立審查／Human acceptance；最終效能修補後重新完整全套（現有全套在修補前import，最後兩個新增測例／修補由focused覆蓋）；其他fold／seed的凍結job identities、72-fit與評估數值驗證均未執行，本任務禁止IC／績效。
- Blockers：strict前瞻仍欠合格官方日曆／lifecycle／公司行動與來源管線驗收；本次歷史工程無source／RAM blocker。
- Next recommended action：獨立審查裁定、bundle與兩fit工程證據，重跑最終來源全套；任何後續研究擬合／績效均待另案授權。
- Project memory updated：YES，Current State／Worktree Register／Data／Features，同一no-commit patch。
- Authority map updated：YES，只記branch-local歷史准入與strict STOP，不改production／machine authority。
- Git status：起始clean；最終9個tracked unstaged＋1個untracked（prepare）；staged=0；依Human禁止commit保留patch，不宣稱clean。
- Commit：N/A，明確禁止，HEAD仍 `b8c4c8f574ea280057bb87a0a852f8ca0b830ef4`。
- Push：N/A，禁止push／merge／deploy。
- Worktree：既有 `.worktrees/ruflo-workers/run-mwwebldn-201d27/builder_step3b` detached，created UNKNOWN，本輪接手2026-11-26，不新增／刪除；私有bundle／models／predictions／partial需保存，不可因ignored刪除。
- Runtime/environment：WSL本機CPU，pilot09 Python3.12.3／NumPy2.5.4／pandas3.1.1／LightGBM4.7.0，CUDA_VISIBLE_DEVICES空，MARKETMAMBA_DATA_ROOT unset；Data realpath為本worktree普通目錄未讀寫。輸入只讀canonical artifacts／p4-bundles，逐列只寫本worktree `.artifacts/step3/`；無網路／GPU／blind／正式Data／V6/models／排程／P4d／P2／agents或外部observed_at更新。

下方完整保留Step3a歷史報告；其中STOP／fits=0／未授權解鎖只描述其當時狀態。

---

# Step3 月營收 h20 歷史工程交付 — 2026-11-26

**CPU_ENGINEERING_ONLY / RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED。准入 STOP；真資料 fits=0；Step3 評估模組未執行。完成工程檢查不是 Human acceptance。**

本輪僅做核准的歷史工程。凍結規格 SHA-256 `9cd7cb8681e3d881193efa3c6fa3c72bd74fba84a96f6a16e6da3a85398eaf34` 先驗相符、完成前再驗相符，規格未改。HEAD/base `224a81439838fb3b569291baaa4f50399249b71a`，既有 detached worktree，起始 clean，無 staged／untracked 既有變更；依本輪指示不 commit。沒有新增本案 IC、績效或分數比較；不讀舊模型分數檔或 blind。

## 1. 範圍與狀態

沿用 `tools/`、`tests/`、`reports/`；不在 research_lab 新增會被 P4a collector 自動打包的來源，不改 P4a／P4d 相容宣告，不建立替代實驗樹。Step3 必須有獨立私有 namespace，不能覆寫既有 TEJ 2a／2b 或 P2 的產物；所有本輪逐列產物均留 `.artifacts/step3/`（gitignored）。

| 優先項 | 實際完成與邊界 |
|---|---|
| §3 准入 | raw／PIT 位元檢查、原始文件擷取及年度選樣實跑；整體 STOP，詳下節。 |
| 五欄 | 合成首次單月歷史重播、YTD 重建、發布可用規則、穩定順序、版本衝突、原因 metadata／去重鏈；未產生正式真資料特徵矩陣。 |
| 標籤／切分 | h20 函式、12 年度折、21-session 嚴格切分、完整日截面中心排名後共用抽樣、W60、185／190 欄相等檢查已實作及合成驗證；未串接全量真資料標籤／訓練 plan。 |
| runner | 固定 72-job queue、兩臂三種子固定配方、私有 job manifest、續讀／損壞 STOP／完整孤兒恢復及延後 fold driver。真資料 admission lock 明確保留，不能靠傳入 PASS 字串啟動。 |
| 評估 | `tools/step3_evaluation.py` 已寫、syntax compile；無 CLI、自動執行或 runner import，未呼叫任何評估函式。數值正確性與完整 session 軸驗收留待另案，不能視為通過。 |
| 最大折量測 | **BLOCKED_SOURCE_ADMISSION**；2026 A/B 各 seed17 的擬合＋推論均未跑，fit 秒數／RSS 仍 UNKNOWN。 |

## 2. §3 准入結果

私有 machine artifact：`.artifacts/step3/admission-verified-20261126/admission.json`，記錄16個已驗身分（來源、checksum、子檔與擷取文字），source commit、implementation SHA、runtime。歷史較早 admission namespace 保留，不視為最新實作的唯一身分。

| 項目 | 結果 |
|---|---|
| 月營收 raw | SHA PASS：`8ba52155ed383005a8999b45aff179eecca41c9d9ad6882099d29c03f055eaae`。 |
| 另外兩份凍結 raw | IFRS `f343d8a27ae99a9ac943cedb8d9f93c0c2491bedf089e0f746a3f3273f708341`；basic `87c9aa30c829690ad88ecbdd7d0d5732b43d7f7881206784dca3901fccf295f7`，PASS。 |
| PIT manifest | SHA PASS：`f3532527b51ed05ad2b0360aad0c14de0531d18f230cf00c7ef456d79fa254f1`。 |
| PIT 子檔 | manifest 全部8個輸出逐檔重驗大小／SHA，PASS；包括月營收 `987c7c32636fcaebcff66a6ba774ff37f1c8fca8d20f589e794db8e28b45e44b`。 |
| raw ZIP 大小 | 記錄觀測 bytes；缺獨立 frozen raw-size 期望值，嚴格 size admission 記 STOP，未用檔案存在冒充驗證。 |
| 原始 TEJ 欄位文件 | 同目錄 SHA256SUMS 核對兩份 HTML；月營收原件另外 pin `2e0e9b5e6897e2eaf7af3565a1344f34cb4b475e1e1da786ee1c58a07dc62dfa`。 |
| 版本與口徑 | **STOP**：尚未建立每個輸入期／異常鍵與首次公告原件的對應及可用範圍；歷史合併／幣別證據不完整。 |
| 完整 matrix identity bundle | **STOP**：尚未凍結矩陣 X、48欄順序、日曆／價格／成員／h20標籤／版本證據的完整聯合 identity；只核身分陣列與既有 snapshot 不算全套。 |
| strict 前瞻 | **STOP**：官方日曆、逐日 lifecycle、停復牌／下市及公司行動證據未齊；不由 observed union／研究成員推定。 |

原件條文摘要（不公開授權原文）：發布日是三類資訊來源中最早揭露營收相關資訊的日期；該發布日欄自2013開始提供。單月營收不因次年比較欄更新而同步更新。2013/01 改以合併為主，與先前母公司口徑不一致，跨轉換月比較受限制。非曆年公司累計統一自一月起。這些條文明確改善文件證據，但沒有證明每筆匯出值就是當時首次版，也沒有補齊本批全缺的合併旗標或歷史幣別。

另對已驗SHA的月營收表只讀版本／口徑metadata：380,514列的合併旗標全缺，380,514列幣別皆仍是待文件確認的TWD假設，版本欄皆明記尚未獨立驗證；有可用發布日期367,532列。發布日的「最早相關資訊」也尚未逐鍵證明就是該單月數值的首次知悉日，不能僅憑非更新條文通過版本gate。彙總證據在私有 `version_scope_aggregate.json`，沒有讀取或比較模型分數。

私有擷取文字34,489 bytes，SHA `0928594fcfbe9b4791601d5fc78d7d1321df5478cddb954db98f55d7273907fa`；位於同 namespace `monthly_field_doc_extracted.txt`。公開僅放上述摘要與雜湊。缺件不補、不放寬門檻；`require_admission` 目前直接 STOP，因此 runner 不是可直接執行的已驗收研究入口，後續解鎖需另案審查及授權。

## 3. 不計算報酬的選樣對帳

核 snapshot 價格、snapshot manifest、shared 的 dates／stock_ids／manifest SHA 後，只建立價格有效性與 session 窗口布林遮罩，**沒有計算任何真資料 h20 數值標籤**。沿用2b `validate_prices`，21個 session（t…t+20）皆有效才記價格支援。時間成熟表示 snapshot session 軸已越過出口，與價格齊備是兩個旗標。本批所有矩陣訊號都已時間成熟，仍有缺價標籤，不將兩者混稱。

完整 observed calendar 為2005-01-03…2026-11-24，5,382 sessions；矩陣研究訊號為2013-01-02…2026-09-11，這是預排訊號末日，**不是最後成熟有效研究日**。尚未生成共同成熟評估鍵，故實際最後可評估日 UNKNOWN，未硬套2b的末日。矩陣日期不在價格聯集的列數0；沒有官方 calendar 差異逐日證據，strict gate 仍 STOP。

| 年 | 成員鍵 | W60應推論鍵 | 時間成熟 | 缺價／品質不合格 |
|---|---:|---:|---:|---:|
| 2013 | 316349 | 293463 | 316349 | 11276 |
| 2014 | 334125 | 313240 | 334125 | 10634 |
| 2015 | 339828 | 312782 | 339828 | 16442 |
| 2016 | 352331 | 313060 | 352331 | 21641 |
| 2017 | 367351 | 332634 | 367351 | 18596 |
| 2018 | 380497 | 336574 | 380497 | 27709 |
| 2019 | 385151 | 337437 | 385151 | 24057 |
| 2020 | 398332 | 360897 | 398332 | 18857 |
| 2021 | 406509 | 378547 | 406509 | 14316 |
| 2022 | 418635 | 385419 | 418635 | 17680 |
| 2023 | 417365 | 390606 | 417365 | 11658 |
| 2024 | 437791 | 418138 | 437791 | 8092 |
| 2025 | 455502 | 425682 | 455502 | 14645 |
| 2026 | 324395 | 302228 | 324395 | 40947 |

合計成員5,334,161，W60 4,900,707（只看 t 以前及當日矩陣存在性），時間成熟5,334,161，缺價／品質不合格256,550，因此未套 W60 的完整價格支援5,077,611，與規格對帳完全一致。W60 且完整價格支援4,813,396。應推論鍵在特徵缺營收或未來缺標籤時仍保留，不將價格成熟遮罩用於推論資格。

互斥主要原因順序為：W60不足 → 時間未成熟 → 價格無效且原因未知 → 完整價格研究支援；每年整數合計精確等於成員分母，另附比例。重疊旗標另列 W60、時間成熟及缺價。停牌、下市、公司行動均為 null／UNKNOWN，不填0、不根據缺價猜原因；當時可知流動性採前一 session 的20-session原價成交值平均之當日截面三分位，另保留unknown。逐年／流動性／lifecycle UNKNOWN 分層見私有 `selection_summary.json`；因無合格 lifecycle，無法做「已知下市／非下市」分層，這項未完成。

第一版錯把2013年前價格送入2b的品質驗證，造成256,551個排除鍵。根因唯讀對照找到2筆原始跳價旗標差異（2013與2023各一筆），其中1筆影響本案對帳；精確逐列證據只留 `.artifacts/step3/validation_boundary_difference_private.parquet`。新增反例後，只把驗證起點改為2b的2013-01-01；完整日曆仍保留作W60。原namespace保留，不重新命名或改寫；修正版精確回到256,550。不是放寬 gate 或新增特例。

最終私有 namespace 為 `.artifacts/step3/selection-release-20261126/`，含 `selection_flags.parquet`、summary、source/input/output hashes及環境。最終稽核耗時41.13秒、峰值RSS3,524,220 KiB（約3.36 GiB）；虛擬位址上限以 `ulimit -v 7500000` 保護（7,680,000,000 bytes）。這是選樣稽核，不是最大折 LightGBM 的成本。

## 4. 工程驗證

RED：首次 import 失敗；分階段新增 runner／凍結日曆origin／價格驗證起點／missing-reason-lineage 反例均先失敗。最新 focused **28 passed**，3.81秒，exit0，詳 `.artifacts/step3/focused-final.log`。未知／NaN 的布林證據直接STOP，避免truthiness放行。兩項 mutation 使用實際函式注入並由 assertion 殺死：發布日 inclusive search、h20出口偏移一 session；另有未來特徵／W60支援改動不影響過去的反例。

另外以 `git show ace2eef:<path>` 比對現有檔案 bytes：`c0plus.py` SHA `9b4b20b5dbc838f9e4c610b52ea3f4db03e05d277393c4205f29260c01c1201f`、`matrix_baselines.py` SHA `708801fc2e8e37607c8130f9174dece02ea625399b09855f609d5fbaf2cdee01` 均與 P1 anchor 完全一致。這只認證重用來源，未宣稱全量矩陣特徵相等。

測例涵蓋發布當日不可用、同session亂序、晚到舊月、版本碰撞、同值去重留鏈、單月缺漏、跨年一月、未知可比性、非正分母、下市型缺出口／尾端未成熟、完整截面平均同秩中心化後無放回排序抽樣、缺session W60、年度origin缺首日仍依凍結日曆、185欄逐值／NaN／鍵一致、seed／matrix mismatch、損壞子檔、完整孤兒恢復、SIGKILL重啟。續跑測例使用合成模型 bytes／鍵，不擬合任何研究模型、不產生新預測比較。

第一輪隔離全套：**1366 passed、11 skipped、5 warnings，759.13秒，exit0**，log在 `.artifacts/step3/full.log`。最終版本重跑於680秒程序時限停止，**exit124／NOT_COMPLETED**，最後進度約82%，沒有終局總計，log在 `.artifacts/step3/full-final.log`；不宣稱最終版本完整全套通過。程序已結束，沒有留下背景測試。測試開始時為14個Step3測例的初版；後續新增／變更由最新28個focused覆蓋。完整命令未改既有測試／skip邏輯；用bwrap遮蔽外部真資料，避免既有real-fold測試讀取研究標籤。不能把合成CPU測例當成真資料研究驗收；評估模組未動態執行，其驗證只到compile／runner隔離。

```bash
bwrap --ro-bind / / --bind "$PWD" "$PWD" --bind /tmp /tmp \
  --dev-bind /dev /dev --proc /proc \
  --tmpfs /home/frank/projects/ruflo-evaluation/pilot09/marketmamba/.local-data \
  --tmpfs /home/frank/projects/p4-bundles \
  --tmpfs /mnt/d/Desktop/work/ProjectForMe/MarketMamba/Data \
  --tmpfs /home/frank/projects/MarketMamba/.artifacts \
  env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=V6:V6/research_lab/tests:. \
  /home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python -m pytest V6/research_lab/tests tests -q -p no:cacheprovider
```

syntax compile五個新Python檔通過；完成前重驗spec SHA、16個admission身分、4個implementation SHA、5個selection輸入及1個selection輸出，全相符。knowledge impact PASS；knowledge health exit0／WARNING，Project Map、Agent Guides、Domains、Runtime Freshness、Generated Index PASS；Governance、Authority Drift、Legacy Docs WARNING，無structural FAIL。`git diff --check`與所有新增檔案 whitespace檢查通過；12個新增本機link targets可解析。完整私有inventory／最終source及output SHA記於 `.artifacts/step3/manifest.json`（封存時產生）。Builder按code-review-and-quality自查，不是獨立審查，沒有派agents。

## 5. 未完成及資源方案

最大2026折 A/B 各seed17完整擬合＋推論未執行（0 fits），故耗時及峰值 UNKNOWN，不把上述稽核RSS當作模型RSS。版本 gate STOP 優先於資源量測。特徵建構只針對P1共用抽樣列呼叫既有 `c0plus.build_features`，避免為6百多萬列先分配185欄；保持完整日截面排名後抽樣、研究樣本不變。這只是候選實作，尚無真資料峰值證據。

後續順序：獨立review、補逐期原件／可比範圍及lifecycle／calendar，凍結matrix＋h20＋特徵／版本證據聯合manifest，再另案解鎖真資料特徵與2026兩fit工程量測。若本機11GiB下仍OOM，保存失敗與確切矩陣bytes／RSS，再請Human另案核准高RAM CPU與授權資料處理位置；不減研究樣本、不更改500,000 cap、不佔P4d或改用GPU。未發生本案LightGBM OOM，不能宣稱已證實需40GiB。

最終版本全套未完成；尚未完成完整私有特徵／標籤／抽樣鍵manifest、全量逐鍵相等驗證、真實fit/inference、72-fit研究、獨立驗收，以及評估數值測試。評估器與runner目前是待獨立檢查的工程候選，沒有任何模型結果或確認性結論。新增能力不變更production、P2封盲、P4d、正式Data、排程或成功門檻。

## 6. 固定交接欄位

- Completed：凍結SHA、raw/PIT/document檢查、年度選樣實跑；工程primitives與合成測試；准入誠實STOP。
- Changed：`tools/step3.py`、`tools/step3_audit.py`、`tools/step3_runner.py`、`tools/step3_evaluation.py`、`tests/test_step3.py`、本報告、Current_State／Authority_Map／Worktree_Register、Data／Features domain notes。
- Decisions：保持版本／來源STOP及執行鎖；只公開彙總；不執行Step3評估。
- Evidence：私有admission-verified與selection-release manifest／檢查log；首輪全套1366 passed／11 skipped，最終版本全套680秒逾時／exit124／NOT_COMPLETED。
- Remaining：本節及§5所列全量串接、模型量測、評估驗證、獨立審查。
- Blockers：逐期版本／口徑證據、官方日曆／lifecycle、完整凍結identity bundle。
- Next recommended action：獨立review本patch並補來源證據，另案才可解鎖資源量測。
- Project memory updated：YES，Current_State、Worktree_Register、Data、Features；同一no-commit change。
- Authority map updated：YES，只記候選與STOP，不改machine authority或升格結果。
- Git status：起始clean；終局保留指定unstaged／untracked no-commit patch，不宣稱worktree clean。
- Commit：N/A，明確禁止。
- Push：N/A，明確禁止；無merge／deploy。
- Worktree：`.worktrees/ruflo-workers/run-mwwctc1a-f7f0c1/builder_step3a`，既有detached，不新增／移除。
- Runtime/environment：WSL CPU、指定pilot09 Python3.12.3，MARKETMAMBA_DATA_ROOT unset，Data realpath為本worktree普通目錄；來源是canonical checkout私有TEJ及事前price snapshot，shared身分唯讀。無正式Data／V6/models寫入、網路、GPU、blind、排程、agents或外部runtime observed_at更新。
