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
