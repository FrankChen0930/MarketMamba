# P4 Colab 操作（工程候選）

UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED。研究規格仍為 [DRAFT](../../tasks/spec-p4a-colab-20261121.md)，正式入口在契約凍結前會拒絕；speed-test 只量測速度，不能產生研究結論。本文件的 Colab／pip 儲存格供 Human 之後手動執行，本 worker 未啟動或驗證任何 GPU、網路或 Drive。

## 主流程：一個 ZIP ＋一個 notebook

Human 2026-11-22 核准的交付介面是 `p4a-bundle.zip` ＋ `p4a-colab.ipynb`。沿用既有 `p4_bundle`／`p4_colab`，不另建流程樹。在可信任 checkout 用同一條 CLI 產生兩檔（獨立輸出 namespace，拒絕覆寫）：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=V6:. /home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python -m research_lab.p4_bundle deliver \
  --folds-root /ABSOLUTE/EXISTING/FOLDS --years 2021 2022 2023 2024 2025 \
  --p1 /ABSOLUTE/P1/FULL/SCORES \
  --lock environments/p4a-colab.requirements.lock.txt \
  --output-dir /ABSOLUTE/NEW/DELIVERY
```

只需在 Colab 開啟產生的 notebook，選 GPU runtime，將 ZIP 上傳 Files 的 `/content/`，或 Drive `MyDrive/MarketMambaP4/inputs/`，從上到下執行。所有預算與路徑集中於第0格；預設 SPEED_TEST、6小時／100 CU、10 CU/hour、每 fit 0.5小時、25%餘裕，均為保守操作預設而非現場計費觀測。Drive掛載仍需 Google 登入授權。無須手貼 SHA、另傳 bootstrap 或手查版本。

ZIP 是真正 ZIP64 格式；fold 的 `X.npy` 以 `(bytes,SHA-256)` 內容去重，manifest `files` 保存完整邏輯清單，`aliases` 記錄重複 fold → 實體成員。解壓重讀該成員並還原每個 fold 的一般檔案，逐份驗證，不建立 symlink。只在打包端允許 fold 輸入 symlink，先解析一般檔案；解壓端仍拒絕所有 links。既有 tar.gz 保留相容，不做 alias 去重。ZIP 邏輯還原大小（不是壓縮大小）仍受256 GiB上限；單檔64 GiB、10,000成員、4 MiB manifest、白名單、逐檔 size/SHA、拒絕 executable cache 與 descriptor 安全寫入維持強制。

Notebook先用標準庫 `zipfile` 只取固定 bootstrap／manifest，驗兩者大小／類型、重複成員及 bootstrap 內部 hash，再由 bootstrap 完整安全解壓與驗證。SHA僅印供記錄；兩檔必須來自可信任的交付來源，內部hash不能自證來源。重連驗原ZIP與現有root manifest bytes一致，完整驗證後才重用，partial不自動覆寫。

環境先印Python／torch／CUDA／GPU／套件與 nvidia-smi。已安裝套件及runtime符合環境契約即跳過wheel安裝；否則檢查指定Drive目錄，僅以 `--no-deps --no-index --find-links` 安裝兩個wheel。Human提供的OBSERVED環境是Python3.13.16／torch2.14.1+cu130／CUDA13.0／Tesla T4，使用 `MyDrive/Mamba_Wheels_T4_new/` 的 `mamba_ssm-2.3.2.post1-cp313-cp313-linux_x86_64.whl`（349592341 bytes）與 `causal_conv1d-1.7.0-cp313-cp313-linux_x86_64.whl`（207063960 bytes）。cp313正向／cp312不符STOP有CPU標籤測試；此處沒有讀取或驗證Drive實際wheel bytes。缺其中一個、多版本或標籤不符STOP；CUDA不可用一律STOP。

只有兩個wheel均不存在才執行編譯備援格：官方PyPI sdist SHA核對，mamba setup.py全部 `-std=c++17` → `-std=c++20`、`MAMBA_FORCE_BUILD=TRUE`、`MAX_JOBS=4`、T4 sm75，wheel輸出Drive。需GPU，估30–90分鐘或更久；時間／CU不在runner帳本，須額外預留。安裝或略過後均跑 Mamba2 `[8,60,64]` 前向／反向有限值冒煙與 pip check。這不等於專案正式GPU smoke。FORMAL仍由既有DRAFT gate拒絕；SPEED_TEST只產生計時與telemetry，結果格摘要秒／epoch、GPU、溫度與時脈，不產生IC／績效。

Runner直接將speed JSON、run manifest與持久budget帳本寫Drive；下載格匯出JSON到 `p4a-results.zip`，不包含正式權重。重連沿用相同Drive根目錄、ZIP與設定，completed-fit驗後略過，未知中斷保守扣帳並STOP；不得換root／刪帳本重置。Colab現場ZIP解壓、編譯儲存格、Drive掛載／寫入／下載均 **UNVERIFIED**；本輪只有CPU合成測試，沒有真實大型bundle、GPU／研究結果或Human acceptance。

## 備援：舊手動逐格流程（歷史操作範例）

以下 tar.gz／另傳bootstrap／舊cp312步驟只作備援，主流程以上方兩檔介面與目前OBSERVED環境契約為準。舊段落的TBD／需Human填版本描述已被目前 `environments/p4a-colab-runtime.json` 的Human觀測取代；不代表本worker執行過Colab。

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

2. 上傳 `p4a.tar.gz` 到 Drive 的 `MyDrive/MarketMambaP4/inputs/`。另將受信任 checkout 中的 `V6/research_lab/p4_bundle.py` 上傳同處作標準庫 bootstrap；不需手動貼 bootstrap SHA。這是 bundle 內相同 source 的副本，避免先對未驗證 tar 做一般解壓。建立 Colab，選 GPU runtime（高 RAM 視供應）；不可直接假定有70–80GB RAM。先掛載 Drive：

```python
from google.colab import drive
from pathlib import Path
import importlib.util, subprocess, sys

drive.mount('/content/drive')
base = Path('/content/drive/MyDrive/MarketMambaP4')
bootstrap = base/'inputs/p4_bundle.py'
spec = importlib.util.spec_from_file_location('p4_bootstrap', bootstrap)
p4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p4)
archive = base/'inputs/p4a.tar.gz'
root = Path('/content/p4a')  # 首次必須不存在，拒絕覆寫；重連見下方
p4.extract(archive, root)  # 自動印出 archive SHA；內部逐檔 size/SHA 仍自動核驗
# extract 已完整 verify；在匯入任何 source 模組前，清除 executable cache。
p4.prepare_source(root)
p4.verify(root)
```

3. 切換 Colab runtime 版本＋Drive wheel 路徑：Human 可在 Colab 的 runtime 版本選項選擇與 wheel 相容的歷史版本（例如 2026 年 3 月；實際選項依現場為準），然後選 GPU runtime。這是手動選擇 runtime，不是在儲存格自行升降級 Python／torch。Human 提供 Drive `MyDrive/Mamba_Wheels_T4/` 內兩個 2026-03-04 建立的檔名：`mamba_ssm-2.3.0-cp312-cp312-linux_x86_64.whl`、`causal_conv1d-1.6.0-cp312-cp312-linux_x86_64.whl`。這些資訊不是目前已安裝環境的觀測；實際 Python／torch／CUDA／ABI／套件版本仍待自檢與 pip list。

```python
import torch
assert torch.cuda.is_available(), 'STOP: no CUDA'
subprocess.run(['nvidia-smi'], check=True)
print(Path('/proc/meminfo').read_text().splitlines()[:3])
commands = p4.installation_commands(
    root, wheel_dir='/content/drive/MyDrive/Mamba_Wheels_T4')
print('\n'.join(commands))
```

這個模式列出找到的兩個 wheel 檔名、版本、cp／ABI／平台標籤及 SHA（只記錄），並與目前 Python 的支援標籤核對；不符、缺檔或有多個版本皆 STOP 並說明。只有每個套件恰好一個相容 wheel 才產生 `pip install --no-deps --no-index --find-links <dir> <兩個wheel路徑>`；不重裝其餘套件、不升降級 torch，也不編譯。標籤匹配不能證明 torch/CUDA ABI 或 kernel 相容，仍須現場測試。

Human 核對指令後，在已獲授權的 Colab 儲存格執行：

```python
for command in commands:
    subprocess.run(command, shell=True, check=True)
```

最後會執行 `pip check`；依賴不滿足就 STOP，不自動安裝其他套件。若缺必要套件，由 Human 決定後續相容環境。若要使用原本本機 P3 exact lock 的完整 only-binary 安裝路徑，可改用 `p4.installation_commands(root)`，該路徑會重裝 lock 套件、可能改動 torch，與 Drive 模式不同；需另行核對相容性。原 lock 保持不變。

安裝後保存 `python -m pip list --format=json`、下方 runtime 自檢、driver／GPU 資訊。Human 提供輸出後，由實作者填入獨立 [Colab 環境契約](../../environments/p4a-colab-runtime.json) 的實際版本、`observed_at` 與 observation source，確認完整後設 `status=OBSERVED`，重新 pack 以綁定新的 identity。初始 TBD 不代表已知相容組合，正式模式會 STOP。

P4a 三臂 mamba2／crossstock／GRU 在同一個 Colab 環境 identity 跑才可作臂間比較。與本機 P3（mamba-ssm 2.3.2.post1）的跨環境比較必須揭露套件／runtime 差異，不宣稱等價。speed-test 允許 lock／契約版本不符，會在 stdout、run binding 與每個 speed JSON 保存 `environment_mismatch` 的每項 expected／actual 及「結果不可與正式環境比較」。這不放寬正式模式或 CUDA gate。

4. 在新 Python 程序先跑 speed-test。同一個持久 budget root 跨 speed／正式共用，不能換資料夾重置帳本。以下預算數字是佔位符，須先由 Human 核定總小時、CU上限、該GPU保守CU/hour及每fit預估；至少25%餘裕由程式強制。費率及預估必須為正數；`planned-hours-per-fit` 也作每個驗證／rankgauss stage的保守預估，應取足以涵蓋最慢stage的值。Bundle／runtime驗證和rankgauss前處理同樣入帳。上傳、pip安裝、人工閒置不在runner計時範圍，須另外對照Colab UI核計並預留總額，不能把runner ledger稱為完整Google帳單。CU 是估算帳本，另須記錄 Colab UI 實際扣點／費率，不能稱已串接 Google billing API。

```python
import os
# 由受信任 checkout 上傳的標準庫 bootstrap 執行；此時不可先 import bundled research_lab。
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
python -c 'import sys, torch, subprocess; print("python", sys.version); print("torch", torch.__version__); print("cuda", torch.version.cuda); print("cxx11abi", torch._C._GLIBCXX_USE_CXX11_ABI); import sysconfig; print("platform", sysconfig.get_platform().replace("-", "_")); print("available", torch.cuda.is_available()); print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE"); subprocess.run(["nvidia-smi"], check=True)'
python -m pip check
python -m pip list --format=json
```

1. CUDA 不可用：speed／正式模式一律 STOP。不得自行升降級 Python／torch／CUDA、改 ABI 或現場編譯來繞過；Human 可先手動切換相容 runtime 版本，再核對 Drive wheel 標籤。
2. Drive wheel 的 Python／ABI／平台標籤不符、缺失／多版本或 pip check 失敗：STOP 並保存原始輸出。cp312 wheel 需要相容的 Python 3.12；檔名不含 torch/CUDA ABI 資訊不能推定相容。預設完整 lock 路徑保留 Python3.12 gate。
3. 安裝後保存完整自檢與 pip list。speed-test 遇版本不符會繼續並完整記錄 `environment_mismatch`，標記「結果不可與正式環境比較」；正式模式要求 Colab runtime 契約為 OBSERVED、所有列出的套件與 runtime 完全一致，TBD／缺套件／版本差異均 STOP。舊本機 lock 差異另存 `local_lock_mismatch`，不拿本機 P3 lock 取代 Colab 正式契約。
4. Driver／GPU 架構相容性、Drive FUSE 的 fsync／atomic rename／斷線持久性仍 UNVERIFIED。保存 artifact／帳本，不換 root、不刪 STOP、不自動重試；新的契約／bundle 必須有新 identity，不沿用舊 run 完成證據。CPU 合成測試不證明 GPU 可執行。

Bundle 資源限制：單檔最多64 GiB、解壓總量（含 manifest）最多256 GiB、成員最多10,000、manifest最多4 MiB；實際 tar header 大小須逐檔等於 manifest 宣告，且總量等於宣告總和加 manifest 大小。所有檢查在建立目的目錄前完成；sparse member、.pyc／.pyo／__pycache__ 均拒絕。這些是工程安全上限，不是已驗證大型 bundle 能放入現場 RAM／磁碟的承諾。


## Bundle 信任與解壓邊界

Archive SHA 與 bootstrap SHA 不需 Human 手動貼入。`extract`／CLI 省略 `--sha256` 時只計算並印出 archive SHA 供記錄，仍自動逐檔核對 manifest 的 size/SHA；提供 `--sha256` 時必須嚴格相符，否則在寫入前 STOP。內部 manifest 驗證確認 bytes 一致性，不能自證來源；bootstrap 仍從受信任 checkout 上傳，合法 source 的安全性仍需程式審查。此段依本輪 Human 明確決策 supersede 舊版指南的外部 SHA 必填要求，資源上限、symlink 防護與白名單不變。

白名單只允許 `source/V6/research_lab/**/*.py`、`source/V6/experimental/**/*.py`（排除 tests／cache）、指定 P4 contract／spec／操作指南／`source/environments/p4a-colab-runtime.json`、`folds/fold_2021` 至 `fold_2026` 的九個 MATRIX_FILES、P1 reference 的兩個指定檔、`requirements.lock.txt` 與 `bundle-manifest.json`。任何位置的 sitecustomize.py、usercustomize.py、conftest.py、setup.py、*.pth、*.so、*.pyd 均拒絕，非白名單檔也拒絕；manifest 一致不授予額外路徑權限。

在 tarfile 開啟之前，以最多1 MiB的單次讀取逐一預檢 physical header；extended payload 最多64 KiB、拒絕連續 extended headers以避免遞迴解析。每次掃描累計受256 GiB＋32 MiB硬上限與 manifest 宣告總和＋4 MiB＋32 MiB限制。Manifest 一出現立即收緊；若在末尾則事後核對此前用量，仍先受全域硬上限與32 MiB metadata overhead限制。共三次 gzip 串流掃描（physical preflight、解析 inventory、寫入），沒有全量 RAM／暫存解壓副本；真實大型 bundle 的 CPU／I/O 時間未量測。

Linux 寫入從 `/` 開始以目錄描述符逐層 openat 方式開啟，目錄使用 O_DIRECTORY|O_NOFOLLOW，檔案使用 O_NOFOLLOW|O_EXCL。Symlink 替換會拒絕，已開啟的 descriptor 不會跟隨替換後的連結。Colab bootstrap、本機帳號與 parent 目錄權限必須可信；這不是可防任意同權限惡意程序搬移已開啟目錄、修改 archive inode 或改寫 Python 程序的 sandbox。失敗時保留 partial destination 供盤點，不會自動刪除／重試。
