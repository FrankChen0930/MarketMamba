# P4a：上傳資料夾一次＋每次兩檔

> Human 的持續有效需求（交付形式、Drive 結構、進度顯示等）見 [colab-human-requirements.md](colab-human-requirements.md)；本檔為 P4a 的逐步操作細節。

**UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED；Colab 現場行為 UNVERIFIED。** 凍結內容、完整網格與外推耗時見[規格](../../tasks/spec-p4a-colab-20261121.md)。無研究結果，不是交易就緒。操作只對自己的可信交付；內部 hash 不能證明來源。

## 一次準備資料

Human 已用 `cp -L` 備妥 Windows `D:\Desktop\work\ProjectForMe\MarketMamba-drive-upload\data\matrix_v1`。請在可存取該目錄且安裝專案依賴的 Python 執行（WSL 請使用對應 `/mnt/d/...` 路徑）：

```bash
PYTHONPATH=V6 python -m research_lab.p4_bundle dataset-manifest /path/to/data/matrix_v1
```

CLI 嚴格檢查 shared 八檔、folds2021–2026各 splits.npy，禁止 symlink，逐檔 size/SHA，並核對指定 X identity。把整個 matrix_v1（含 DATASET_MANIFEST.json）用 Drive 網頁上傳到 `MyDrive/MarketMamba/data/matrix_v1`，只需一次。`rankgauss_cache` 由 notebook 建立，不需手動準備。

## 每次產生兩檔

```bash
PYTHONPATH=V6 python -m research_lab.p4_bundle deliver \
  --folds-root /path/to/data/matrix_v1 \
  --p1 /path/to/verified/p1-scores \
  --output-dir /path/to/fresh-delivery --run-name 02_formal
```

`--folds-root` 在此指定 matrix_v1 根目錄，沿用既有 CLI 命名。輸出 `p4a-bundle.zip`、`p4a-colab.ipynb`；README.txt 是本機路徑提示，不需上傳。ZIP 無矩陣／rank-gauss／symlink，只含 source、契約、環境、資料 manifest 與 C0+ 參考鍵。來源必須在指定根內，manifest逐檔與ZIP壓縮比／邏輯大小均核對。

將兩檔上傳 `MyDrive/MarketMamba/p4a/02_formal`，開 notebook，選 T4 GPU，從上往下執行。第0格只需確認 RUN_DIR／DATA_DIR，預設 FORMAL；另可選 SPEED_TEST。無手貼 hash、無時數／CU設定、無 pip check。

流程：掛載 Drive → 找ZIP並自動核對內嵌size/SHA → 安全解壓並 `p4.prepare_source(root)` 清理執行快取 → 驗資料manifest並複製至 `/content/data` SSD → 重用／建立 Drive rankgauss_cache → 環境自檢 → wheel安裝或略過 → 冒煙 → Q1–Q5 → 打包完整results zip並下載。矩陣不從 Drive FUSE mmap 訓練。

wheel 優先 `MyDrive/MarketMamba/wheels`，回退 `MyDrive/Mamba_Wheels_T4_new`；兩個 wheel 都不存在才執行官方sdist編譯備援格。部分缺失、多版本、cp313／ABI不符即 STOP。編譯格是交付程式碼，本次未執行網路、GPU或Colab。

## 進度與模式帳本（後續交付）

訓練 cell 使用 `Popen` 合併 stderr、逐行轉印並 flush，子程序 `python -u`；每 50 steps／每 epoch 顯示 fit 編號、Q 階段／臂／折／種子／窗／LR、loss、FORMAL 驗證 IC、耗時／ETA、GPU 溫度／時脈與檢查點時間。無完成 fit 計時時 ETA 為「預估中」；本 fit 按最大 epochs 外推（早停可能縮短），佇列按本次已完成 fit 的實際耗時均值外推，Q5 額外耗時未知，不承諾完成時間。

每約 60 秒按時間心跳原子替換 `EXPERIMENT_DIR/progress/formal/{PROGRESS.txt,progress.json}`；speed-test 使用 `progress/speedtest/`，不含 IC／績效。寫入失敗會提示檢查 Drive／空間並繼續訓練。兩檔各自原子替換，不保證成對交易一致；以 JSON 與 `observed_at` 判讀新鮮度，強制中止可能留下舊快照。Colab 單一 kernel 正在訓練時無法再執行其他格，請主要用手機／Drive 網頁查看 TXT；完成或中斷後再執行「目前進度摘要」。

FORMAL 結果與帳本在 `EXPERIMENT_DIR/budget/`，speed-test 在 `EXPERIMENT_DIR/budget_speedtest/`，跨交付延續各自帳本；CLI `--drive-root` 仍填正式帳本根目錄，runner 在 speed 模式自動選 sibling。不要把參數先填成 speed 路徑而重複加後綴。既有 chain／identity／缺帳保護不變，不自動搬移、清除或重置歷史帳本。訓練格結束（含失敗）列出結果目錄、manifest 與進度位置；失敗摘要含步驟／期望／實際回傳碼／下一步。

以上只有 CPU 合成工程驗證，Colab／Drive 現場 **UNVERIFIED**。正在執行的舊 02_formal 保持原交付，新版須另行重建 source identity；不要中途替換 ZIP。

## 中斷與結果

每個 validation 完成的 epoch 先本機fsync、Drive複製重讀hash，再原子更新 latest／previous／best pointer。重跑同 notebook、同 source／contract／matrix identity，恢復完整epoch的optimizer、scheduler與RNG；未完整epoch重做。已完成fit逐檔驗證後跳過。每fit即寫Drive。遺失／損壞pointer、identity不符、只有orphan均STOP，保留證據，不自動升格orphan。

只有紀錄帳本，無時間／CU上限或25%reserve；>=85°C仍硬體STOP，必須查明原因，不自動清除帳本。未知中斷間隔計至重啟並明記，不等同GPU訓練秒數。Notebook最後下載JSON、parquet與權重；完成下載及檢查後由Human自行關runtime，沒有自動關機或排程。

下載解壓後在本機評估（C0+完整分數留本機，不在交付ZIP）：

```bash
PYTHONPATH=V6 python -m research_lab.p4_eval \
  --run-dir /path/to/download/run-HASH \
  --c0plus-dir /path/to/verified/p1-scores \
  --data-root /path/to/processed_v6 --output /path/to/fresh-p4-evaluation.json
```

使用既有volume provenance gate與P2主格，輸出H-A/H-B/H-C、各折描述與共同列交集／排除比例；要完整64fits與雙seed。只報歷史比較，通過CI/Holm亦非未曝光確認性。資料價格／PIT／下市限制延續P1/P2，不放寬gate。

## ABI／版本衝突處理決策樹

先看 `python -c "import torch; print(torch.__version__, torch.version.cuda, torch._C._GLIBCXX_USE_CXX11_ABI)"`，再比對[OBSERVED runtime契約](../../environments/p4a-colab-runtime.json)的Python3.13.16／cp313、torch2.14.1+cu130、CUDA13.0與wheel標籤。CUDA無效→STOP；標籤不同→STOP；部分wheel→STOP；兩者不存在→備援編譯；完全相符→略過安裝。不得自行升降級 torch 或放寬FORMAL環境契約。SPEED_TEST只記錄差異，結果不可與正式環境比較，CUDA仍必須可用。

## 未驗證

T4長窗效能／OOM、Triton額外編譯、Python3.13整批、Drive FUSE的fsync／rename／斷線持久性、wheel編譯、實際資料完整性／上傳、Q5現場推論、瀏覽器自動下載皆UNVERIFIED。本機合成CPU證據不替代Colab acceptance。

## 跨版本接續：4e56e9d → 來源相容交付候選

狀態：UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED；CPU 工程證據，Colab 現場 UNVERIFIED。原 02_formal notebook 仍可原地接續；不要在執行中的 kernel 換碼。斷線後改用新版時，交付者沿用前述 deliver 指令，額外指定：

```text
--resume-from-commit 4e56e9d --run-name 03_formal_resume
```

`--resume-from-commit` 省略 `--run-name` 時也預設 `03_formal_resume`。工具從本機 git objects 重算舊版所有來源（含文件）身分，檢查固定數值／完成驗證／帳本函式，產生 `source/tasks/p4a-source-compat-<HEAD短碼>.json` 並放入 ZIP manifest；未 commit 的程式以逐檔 bytes 為準，宣告明標 UNMERGED。模型／配方／環境／資料／佇列／預算有差異時禁止宣告；目前只認證 4e56e9d，其他基準需先新增跨版測試及審查。每個差異檔包含完整 unified diff、舊新 SHA 與允許的理由分類。

Human 斷線後：開新 notebook，將同批 zip＋ipynb 上傳 `MarketMamba/p4a/03_formal_resume/`，從上到下執行。第 0 格 `EXPERIMENT_DIR=RUN_DIR.parent` 維持 `MarketMamba/p4a`；保留舊 `02_formal/p4a-bundle.zip`、`budget/` 及一次上傳的資料。新版 ZIP 使用自身 hash 隔離本機解壓路徑，避免殘留舊 bundle 衝突。rank-gauss 仍遵守原 implementation／input hash gate；來源改變可能觸發快取重建，不放寬驗證。

Runner 掃描 `budget/run-*/run_identity.json`。舊身分只存 manifest 的 size／SHA，新版讀取實驗下既有交付 ZIP 的有界 manifest，先匹配舊 run 的 manifest identity，再逐檔核對宣告 from／to／完整差異集合及所有非來源輸入（含參考鍵）。契約／64-fit 佇列／DATASET_MANIFEST／環境／record_only 預算全相同且只有一個可採用 lineage 才繼續；沿用舊 run_hash，所以舊 complete.json 及 attempt-active 的 epoch pointer 可直接驗證。歷史 artifacts 不遷移、不改寫。run 目錄新增 `source-upgrade-*.json`，ledger 追加 `SOURCE_UPGRADE`，記錄來源清單／雜湊、宣告身分、時間、執行環境與原／新 binding。

缺檔、雜湊或身分不合會 STOP 並說明預期／實際及下一步。保留帳本與舊 ZIP，修正交付或回原 notebook；不要刪 `budget/`。找不到舊 run 才建立全新 run；明確要新批次時才設第 0 格 `ALLOW_NEW_RUN=True`（runner `--allow-new-run`），此操作不自動清除 STOP／帳本。多個不同 lineage 保守拒絕，需另案盤點，不自選其中一個。升級宣告不是任意跨版許可，每次需 CPU GRU／crossstock 位元比較、舊 checkpoint 及完成 fit 接續測試與獨立審查。規則以 [Human 持續需求第 8 節](colab-human-requirements.md) 為準。

## 事故後必做：上線前排練（候選，2026-11-23）

長跑前先對真實資料排練；這是 [Human 需求第 9 節](colab-human-requirements.md) 的新 gate。以下指令在已備妥資料的本機 CUDA／Colab runtime 執行（`PYTHONPATH=V6`；`--bundle` 指已解壓目錄，可代替 `--reference` 自動選 keys.parquet／rank_average.parquet）：

```bash
python -m research_lab.p4_colab --rehearse --data-dir /path/matrix_v1 --reference /path/reference/keys.parquet --queue Q1
python -m research_lab.p4_colab --rehearse --data-dir /path/matrix_v1 --reference /path/reference/keys.parquet --queue Q3 --window 120
python -m research_lab.p4_colab --rehearse --data-dir /path/matrix_v1 --reference /path/reference/keys.parquet --queue Q3 --window 250
python -m research_lab.p4_colab --rehearse --data-dir /path/matrix_v1 --reference /path/reference/keys.parquet --queue Q4
python -m research_lab.p4_colab --rehearse --data-dir /path/matrix_v1 --reference /path/reference/keys.parquet --queue Q5
```

`--arm` 可選該佇列支援的臂；不指定時選 frozen queue 的第一筆（Q1／Q3 是 2021 mamba2 seed17，Q4 是 2022 mamba2 seed17、lr=1e-4）。Q5 固定 W60 crossstock＋mamba2，會先各做一個排練 fit，再用其 checkpoint 執行 validation／evaluation、10 次排列診斷。預設 `--steps 1` 且固定 1 epoch，保留原 validation／evaluation 支援，不以縮減列數藏掉尾端 NaN 或上市歷史不足。可直接傳單一 fold 目錄，Q4 需給 2022 折。

輸入矩陣只讀；快取及產物都在 `/tmp/p4-REHEARSAL-*`，不需 `--drive-root`、不接觸正式 run 或 ledger。保留資料 realpath、manifest/reference identity、配方、source identity 與 runtime 版本；輸出標示 `REHEARSAL / 不是研究結果`。1 epoch 的 `REHEARSAL-complete.json` 通過相同輸出／pointer／BEST／history 檢查，但正式 `resume_fit` 不認此檔，也不放寬 frozen recipe。失敗時先讀終端列出的暫存路徑保留診斷，排練通過不等於完整研究或 Human acceptance。

本次 4e56e9d 升級須保留舊 ZIP、36 份 Q1／Q2 complete 及其所有檢查點／輸出；逐一通過 resume_fit 才採用原 lineage。長窗發布 reference 取共同列，NaN 標籤只作列身分比對；±inf／標籤不一致仍拒絕。Q5 的 IC 明列成熟標籤支援與排除量，完全無成熟支援／不足 10 名／常數相關仍 STOP。Q4 的不同 LR 與 2022 折已由 CPU 排練覆蓋；真實 CUDA 記憶體、官方 mixer、T4 runtime、Drive FUSE 及完整 64-fit＋Q5 現場仍 UNVERIFIED。正在執行的 kernel 不換碼，獨立審查與現場排練後才採用新交付。


### 正式長跑前自動排練閘門（2026-11-24 候選）

新版 deliver notebook 將閘門放在正式 runner 同一格前段：Q3 W120 mamba2 → Q3 W250 mamba2 → Q5（既有 crossstock＋mamba2 prerequisites），全部 `--rehearse --steps 1`。沿用前格 Drive→SSD 的 `/content/data`、bundle reference、官方 mixer；排練產物只寫 `/tmp/p4-REHEARSAL-*`，不寫正式 outputs／budget。任何非零或中斷均不啟動正式 runner；保留上方輸出的確切暫存路徑，將整格輸出貼給 Claude，修正後重跑。

第 0 格 `REHEARSE_FIRST = MODE != 'SPEED_TEST'`：FORMAL／03_formal_resume 預設啟用，SPEED_TEST 預設關閉；可明設 False。全部成功印出「排練通過（不是研究結果）」與各步秒數，將身分綁定的 `rehearsal_passed.json` 寫入模式的 progress 資料夾；同一 run command、已驗 bundle source、prepared matrix／rankgauss、資料 realpath 與實際 Python／torch／mamba／CUDA／GPU／ABI 等 environment snapshot 相符才略過。遺失／壞 JSON／來源、資料、環境或 run command 改變即重做；重做失敗不保留舊成功標記。這只是工程觀測，不能採用為正式 fit 或研究結果；正式 36 個 completed-fit 採用 gate 與固定 64-fit queue SHA 完全保留。

W≠60 在 SeqData 配置及 split 索引前檢查 sessions≥W、eligible 訓練日≥val_dates+20；不足會顯示 fold、W、需要與實際及下一步。W60 不走新增檢查。成功排練輸出 `torch.cuda.max_memory_allocated` bytes；每個子程序先 reset。**真實 W250 windows 張量／CUDA 峰值與首次完整推論耗時仍 UNVERIFIED**；1 step 不等於全部只做一個 batch 的推論。CPU 合成／mixer 替身只能證明工程控制流程，現場官方 mixer／CUDA／Colab／Drive FUSE 與 Human acceptance 均未驗證。此候選為 UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED，不解除獨立審查 NO_GO。

## 自動排練結果驗證（第 12 次複審修補候選）

正式前依序 Q3 W120 mamba2、Q3 W250 mamba2、Q5，各 1 step。FORMAL／resume 預設開啟；SPEED_TEST 預設關閉，明設 REHEARSE_FIRST=False 的既有行為保留。返回碼 0 後，閘門仍逐階段讀獨立 JSON，核對請求／輸入身分、Q3 1 fit／Q5 2 fits、Q5 validation／evaluation 完成，以及既有 completed-output validator 的 scores／reference／checkpoint／complete 證據。結果與每個排練 fit 的契約綁定本次請求，空輸出、吞例外、缺檔或不符都 STOP，顯示階段、期望／實際與修正重跑的下一步。

三步均驗證後才寫 progress 的 rehearsal_passed.json。快取要求完整 schema、恰好三階段 identity／耗時／REHEARSAL evidence／fit 數及內容 digest；空 stages、TAMPERED、型別錯誤、額外欄位或 digest 不符會重做。TEMPORARY 結果仍在 `/tmp/p4-REHEARSAL-*`，只將驗證後 identity 摘要存入 Drive marker，重連略過不依賴暫存檔仍存在。digest 為誤改檢查，非惡意竄改防護。此 marker 不是研究或正式 completed-fit authority；36 fits 採用條件及 frozen queue 不變。修補尚待獨立複審與現場 CUDA／官方 mixer／Colab／Drive 驗證，不等於 Human acceptance。
