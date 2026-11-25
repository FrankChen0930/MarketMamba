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
