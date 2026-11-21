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
# extract 已完整 verify；在匯入任何 source 模組前，清除 executable cache。
p4.prepare_source(root)
p4.verify(root)
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
# 由已核對 SHA 的標準庫 bootstrap 執行；此時不可先 import bundled research_lab。
p4.prepare_source(root)  # 重連時也先清除所有 source __pycache__ / .pyc / .pyo
p4.verify(root, allow_derived=True)
training_env = dict(os.environ, PYTHONPATH=f'{root}/source/V6:{root}/source', PYTHONDONTWRITEBYTECODE='1')
common = [sys.executable, '-m', 'research_lab.p4_colab',
    '--bundle', str(root), '--drive-root', str(base/'budget'),
    '--max-hours', 'HUMAN_HOURS', '--max-units', 'HUMAN_CU',
    '--units-per-hour', 'HUMAN_CONSERVATIVE_CU_PER_HOUR',
    '--planned-hours-per-fit', 'HUMAN_ESTIMATE_HOURS', '--reserve', '0.25']
subprocess.run(common + ['--speed-test'], env=training_env, check=True)
```

Speed-test 固定首個 bundled fold、seed17、三臂各2epoch×100步，JSON提供每步／每epoch秒數、GPU型號、逐次溫度／時脈／功率與總耗時。epoch含validation/probe，step含資料窗與前反向／更新；guard precheck開銷不含在step計時內。額外 nvidia-smi 探針可能增加 wall。JSON為 ENGINEERING_SPEED_ONLY，沒有研究判定。溫度>=85°C立即STOP，不暫停等待，guard_wait_seconds=0。

5. 檢查所有 speed JSON 及實際環境；speed 不產生 checkpoint、scores、metrics、IC 或研究結論欄位，總 manifest 與 stdout 同樣隔離；Human／Claude先凍結草案、完整config／多重比較及預算，再重新pack含凍結契約的正式bundle；source或bundle改變產生新run_hash，不能冒充原run。正式執行命令是同一 common 移除 `--speed-test`（換為經核定正式bundle/root），不需新增排程。DRAFT下執行將明確拒絕。

6. 正式 fit 的 epoch generations、latest/previous/best pointer、`best.pt`、scores、reference rows 與 metrics 均立即寫入 Drive，fsync及重讀SHA後才發布完成檔；完成 schema 重驗三個 pointer、BEST tensor 一致性、metrics、scores 行數／有限值／reference 列與 labels。speed fit 僅發布 speed JSON 與計時完成標記，不保存 epoch state 或預測。fit完成後才跳過。啟動前重驗bundle原始檔、runtime/source/contract、rankgauss inputs/cache/tag、fit outputs及預算鏈。已有 `/content/p4a` 時不要重解壓，runner允許三個已驗derived cache檔；新runtime則重解原bundle並重建rankgauss。輸入identity相同才重用，partial cache／tag不符STOP，不刪除未知artifact自動修補。

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


## ABI／版本衝突處理決策樹（UNVERIFIED）

在安裝以及 speed-test 前各執行環境自檢；下列是供 Human 在 Colab 執行的指令，本輪沒有執行 Colab、下載或安裝：

```bash
python -c 'import sys, torch, subprocess; print("python", sys.version); print("torch", torch.__version__); print("cuda", torch.version.cuda); print("cxx11abi", torch._C._GLIBCXX_USE_CXX11_ABI); print("available", torch.cuda.is_available()); print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE"); subprocess.run(["nvidia-smi"], check=True)'
python -m pip check
```

1. Python 不是 3.12、torch 不是 2.10.0+cu128、torch CUDA 不是 12.8、cxx11abi 不是 TRUE 或無可用 GPU：STOP，保存自檢 stdout／stderr、driver、GPU 型號與 lock SHA，回報 Human。不得自行升降級 Python／torch／CUDA、重寫 lock、改 ABI 或現場編譯。
2. 基礎 runtime 符合：Human 才可依已核定 exact lock 執行指南中的 only-binary 安裝。wheel 不存在／下載失敗／pip check 衝突：STOP 並保存原始錯誤；歷史 notebook 成功不能證明此版本組合相容。
3. 安裝後重跑自檢與 p4_colab.environment 的 exact package 驗證；任何不符仍 STOP。版本符合僅證明 identity，尚須獨立授權的官方 GPU forward/backward、kernel 與 speed-test，不能由 CPU 合成測試替代。
4. Driver／GPU 架構相容性、Drive FUSE 的 fsync／atomic rename／斷線持久性沒有在本輪驗證。失敗保存 artifact／帳本，不換 root、不刪 STOP、不自動重試；由 Human 決定相容 runtime 或重新審查 lock。新的 lock／recipe 必須有新 identity 與重新審查，不沿用舊 run 的完成證據。

Bundle 資源限制：單檔最多64 GiB、解壓總量（含 manifest）最多256 GiB、成員最多10,000、manifest最多4 MiB；實際 tar header 大小須逐檔等於 manifest 宣告，且總量等於宣告總和加 manifest 大小。所有檢查在建立目的目錄前完成；sparse member、.pyc／.pyo／__pycache__ 均拒絕。這些是工程安全上限，不是已驗證大型 bundle 能放入現場 RAM／磁碟的承諾。
