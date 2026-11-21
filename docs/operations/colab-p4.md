# P4 Colab 操作（工程候選）

UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED。研究規格仍為 [DRAFT](../../tasks/spec-p4a-colab-20261121.md)，正式入口在契約凍結前會拒絕；speed-test 只量測速度，不能產生研究結論。本文件的 Colab／pip 儲存格供 Human 之後手動執行，本 worker 未啟動或驗證任何 GPU、網路或 Drive。

1. 本機確認 checkout、source commit、dirty diff（未提交 source 也逐檔 hash）、原始 fold 及 P1 manifest 身分，使用足夠空間的獨立輸出路徑。不要在訓練中改 source／commit。於 repository root 執行：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=V6:. /home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python -m research_lab.p4_bundle pack \
  --folds-root /ABSOLUTE/EXISTING/FOLDS \
  --years 2021 2022 2023 2024 2025 \
  --p1 /ABSOLUTE/P1/FULL/SCORES \
  --lock /home/frank/.venvs/marketmamba-gpu.requirements.lock.txt \
  --output /ABSOLUTE/NEW/p4a.tar.gz
sha256sum /ABSOLUTE/NEW/p4a.tar.gz
```

輸出為單一 tar.gz，內含 research source、P4契約／指南、9個原始fold檔、P1分數及manifest、原樣 exact lock、逐檔 size/SHA manifest。不含 `X_rankgauss.npy`。不要用 P3 bundle、cache、result 充當 P4 identity。真實大型 bundle 未由本 worker 打包；須由 Human 核對磁碟／上傳時間及 manifest。此處 self-contained 指 source/data/lock；套件 wheel 仍需取得，並非離線 wheelhouse。

2. 上傳 `p4a.tar.gz` 到 Drive 的 `MyDrive/MarketMambaP4/inputs/`。另將受信任 checkout 中的 `V6/research_lab/p4_bundle.py` 上傳同處作標準庫 bootstrap；在執行它前，以本機 SHA 核對。這是 bundle 內相同 source 的副本，避免先對未驗證 tar 做一般解壓。建立 Colab，選 GPU runtime（高 RAM 視供應）；不可直接假定有70–80GB RAM。先掛載 Drive：

```python
from google.colab import drive
from pathlib import Path
import hashlib, importlib.util, subprocess, sys

drive.mount('/content/drive')
base = Path('/content/drive/MyDrive/MarketMambaP4')
bootstrap = base/'inputs/p4_bundle.py'
assert hashlib.sha256(bootstrap.read_bytes()).hexdigest() == '貼上本機p4_bundle.py完整SHA256'
spec = importlib.util.spec_from_file_location('p4_bootstrap', bootstrap)
p4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p4)
archive = base/'inputs/p4a.tar.gz'
root = Path('/content/p4a')  # 首次必須不存在，拒絕覆寫；重連見下方
p4.extract(archive, root, expected_sha256='貼上本機tar.gz完整SHA256')
```

3. 檢查 runtime GPU／Python／RAM，產生安裝指令。舊 `archive/notebooks/V5_5_Training.py` Cell1 使用先 causal-conv1d 再 mamba-ssm 預編譯 wheel；`archive/MarketMamba_V3.ipynb` Cell1 使用 Drive wheel。沿用 wheel 策略，實際版本／URL／wheel SHA 以 bundle exact lock 為準（torch2.10.0+cu128／mamba2.3.2.post1／causal1.7.0），不用舊 notebook 的1.6.0／2.3.0。此處沒有重新 clone、資料抓取、production 或 scheduler 步驟。

```python
import torch
assert torch.cuda.is_available(), 'STOP: no CUDA'
assert sys.version_info[:2] == (3, 12), 'STOP: cp312 wheel requires Python3.12'
subprocess.run(['nvidia-smi'], check=True)
print(Path('/proc/meminfo').read_text().splitlines()[:3])
commands = p4.installation_commands(root)
print('\n'.join(commands))
```

Human 核對列出的 exact lock 安裝程序後，另一儲存格執行；這會使用網路／套件安裝，僅由 Human 在獲授權的 Colab 執行：

```python
for command in commands:
    subprocess.run(command, shell=True, check=True)
```

安裝使用 `--only-binary=:all:`，無相符 wheel 即 STOP，不臨時編譯或猜測 ABI。若需要 restart runtime，先重連 Drive，再匯入 bootstrap；不要重複解壓至已有 root。若 pip check 失敗或 exact lock 任一套件無可用 wheel，需 Human 決定相容 runtime／重新審查 lock，不能放寬版本。cp312/linux_x86_64/cu12/torch2.10/cxx11abiTRUE 必須相容；A100、L4、T4 等實際 kernel 支援與 forward/backward smoke 仍須現場確認，CPU 測試不能證明。

4. 在新 Python 程序先跑 speed-test。同一個持久 budget root 跨 speed／正式共用，不能換資料夾重置帳本。以下預算數字是佔位符，須先由 Human 核定總小時、CU上限、該GPU保守CU/hour及每fit預估；至少25%餘裕由程式強制。費率及預估必須為正數；`planned-hours-per-fit` 也作每個驗證／rankgauss stage的保守預估，應取足以涵蓋最慢stage的值。Bundle／runtime驗證和rankgauss前處理同樣入帳。上傳、pip安裝、人工閒置不在runner計時範圍，須另外對照Colab UI核計並預留總額，不能把runner ledger稱為完整Google帳單。CU 是估算帳本，另須記錄 Colab UI 實際扣點／費率，不能稱已串接 Google billing API。

```python
import os
training_env = dict(os.environ, PYTHONPATH=f'{root}/source/V6:{root}/source', PYTHONDONTWRITEBYTECODE='1')
common = [sys.executable, '-m', 'research_lab.p4_colab',
    '--bundle', str(root), '--drive-root', str(base/'budget'),
    '--max-hours', 'HUMAN_HOURS', '--max-units', 'HUMAN_CU',
    '--units-per-hour', 'HUMAN_CONSERVATIVE_CU_PER_HOUR',
    '--planned-hours-per-fit', 'HUMAN_ESTIMATE_HOURS', '--reserve', '0.25']
subprocess.run(common + ['--speed-test'], env=training_env, check=True)
```

Speed-test 固定首個 bundled fold、seed17、三臂各2epoch×100步，JSON提供每步／每epoch秒數、GPU型號、逐次溫度／時脈／功率與總耗時。epoch含validation/probe，step含資料窗與前反向／更新；guard precheck開銷不含在step計時內。額外 nvidia-smi 探針可能增加 wall。JSON為 ENGINEERING_SPEED_ONLY，沒有研究判定。溫度>=85°C立即STOP，不暫停等待，guard_wait_seconds=0。

5. 檢查所有 speed JSON、checkpoint hashes及實際環境；Human／Claude先凍結草案、完整config／多重比較及預算，再重新pack含凍結契約的正式bundle；source或bundle改變產生新run_hash，不能冒充原run。正式執行命令是同一 common 移除 `--speed-test`（換為經核定正式bundle/root），不需新增排程。DRAFT下執行將明確拒絕。

6. 每 fit 的 epoch generations、latest/previous/best pointer、`best.pt`、scores、metrics、speed JSON（僅速度模式）均立即寫入 Drive，fsync及重讀SHA後才發布完成檔；fit完成後才跳過。啟動前重驗bundle原始檔、runtime/source/contract、rankgauss inputs/cache/tag、fit outputs及預算鏈。已有 `/content/p4a` 時不要重解壓，runner允許三個已驗derived cache檔；新runtime則重解原bundle並重建rankgauss。輸入identity相同才重用，partial cache／tag不符STOP，不刪除未知artifact自動修補。

若 fit 中斷，下次會保守扣未知時間至重啟並鎖STOP，不自動繼續optimizer；保留attempt／latest／previous／best。缺pointer、ledger、hash不符一律STOP。此候選尚無帳本人工clear／rebuild介面；需要獨立盤點與明確後續修復，不能換root或刪帳本繞過。完整fit可續用，不承諾mid-epoch恢復。

7. GPU釋放前核對run manifest全部outputs、complete fit及best權重；將整個 `budget/` 連同inputs的archive SHA下載回本機（Drive UI下載即可，不用修改舊.artifacts）。本機驗證bundle：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=V6:. /home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python -m research_lab.p4_bundle verify /ABSOLUTE/FRESH/EXTRACTED/BUNDLE
```

下載的fit結果可在本機用 `resume_fit` 驗證，無須CUDA：

```python
from pathlib import Path
import json
from research_lab.p4_colab import resume_fit
for marker in Path('/ABSOLUTE/DOWNLOADED/budget').glob('run-*/*/complete.json'):
    saved = json.loads(marker.read_text())
    resume_fit(marker.parent, saved['contract'])  # 重驗contract hash及每個output SHA/size
```

這證明下載bytes／內部identity一致；研究准入仍須比對獨立核定的run_identity／bundle／frozen契約與完整grid，不能只信下載檔內自報contract。P4 portfolio evaluator／研究結論不在本輪範圍。
