# V7：Colab 2026.04 對齊環境

> Source preservation review (2026-09-22): `requirements.lock.txt` is byte-identical to `feature/v7-e1-rolling-origin-refresh@044eef7` (SHA-256 `cebc665d6e4a4cfc5cd53113c7a52d594c7595b09ae0633903845f175cac1af5`). This directory records environment identity; the E1 frozen experiment contract remains authoritative for the formal run. `verification.json` describes a local 2026-09-12 check, with `colab_gpu_executed=false`; it is not E1 GPU-run evidence. Local `.runtimes` and Colab availability must be re-observed before executing these historical instructions.

本設定選擇 Colab 的 **2026.04 GPU runtime**。先前交付僅準備相容方案，這次另外建立實際本機環境。對齊的是 V7 所需依賴，不是複製 Colab 所有預裝服務。

| 項目 | 固定基準 |
|---|---|
| Python | 3.12.13 |
| PyTorch | 2.10.0+cu128 |
| CUDA runtime | 12.8 |
| Mamba | 2.3.2.post1，官方 cp312／cu12／torch2.10／CXX11 ABI TRUE wheel |
| PyG | 2.7.0（V7 額外依賴） |
| NumPy / pandas | 2.0.2 / 2.2.2 |
| PyArrow / SciPy | 18.1.0 / 1.16.3 |

本機使用 Ubuntu 24.04 WSL2，Colab 使用 Ubuntu 22.04.5；作業系統、驅動及 GPU 不會完全相同。官方 Linux x86_64 wheel 與 Python 套件版本相同，可減少環境差異，但不代表已完成 Colab GPU 實測。預編譯 wheel 避免自行建置 Mamba wheel；第一次 SSD 執行仍可能需要 Triton kernel JIT。

## 本機

舊的 MarketMamba/.venv 與 MAS 環境保留。新環境位置：

~~~bash
source /home/frank/projects/MarketMamba/.runtimes/colab-2026.04/.venv/bin/activate
python --version
~~~

本機 MAS 的專案設定已指向這個環境，解析與依賴檢查通過。後續 V7 測試請使用這個 Python。不要直接降版或覆蓋舊 V6 環境。

## Colab

1. 在「執行階段 → 變更執行階段類型」選擇 GPU，將「Runtime Version」設成 **2026.04**。
2. 將交付包的 MarketMamba 資料夾改名為 MarketMamba-source，放在 Drive 根目錄，依主 Notebook 第一格掛載 Drive。
3. 主 Notebook 已提供「1. 安裝固定環境」儲存格：設 INSTALL_LOCKED_ENVIRONMENT=True 並執行。它會執行下列完整鎖檔安裝，無須自行新增一格：

~~~python
import sys, subprocess
subprocess.run([
    sys.executable, "-m", "pip", "install", "--only-binary=:all:",
    "-r", "/content/drive/MyDrive/MarketMamba-source/environments/v7-colab-2026.04/requirements.lock.txt"
], check=True)
~~~

4. 安裝後選 Restart session（不是 Disconnect and delete runtime），同一台環境內安裝套件會保留。再掛載 Drive、重新執行變數設定，並將 INSTALL_LOCKED_ENVIRONMENT 設回 False。不要再啟用主 Notebook 的一般 base 安裝或可選 PyTorch 降版格。
5. 主 Notebook 的官方 wheel resolve 可用來產生 manifest/runtime metadata；套件已安裝，不必再啟用 INSTALL_REVIEWED_WHEEL。再依資料檢查 → 小量 smoke → 完整資料放行後訓練的順序執行。

若無法選到2026.04，先停止並重新核對可用版本，不要默默沿用最新 runtime。Colab 過去版本依官方政策有供應期限。

## 檔案用途與驗證界線

- requirements.txt：與官方 Colab 快照對齊的 V7 直接依賴。
- requirements.lock.txt：本機實際解析安裝後的完整依賴鎖檔，Mamba／Torch 指向官方 wheel。
- profile.json：所選 runtime、官方快照與 wheel 雜湊。
- verification.json：實際安裝與測試結果；不能把未執行的 GPU 測試視為已通過。

原始資料的 OHLC 與來源覆蓋問題不會因套件對齊而消失，仍應先看交付資料報告。

來源：[Colab runtime FAQ](https://research.google.com/colaboratory/runtime-version-faq.html)、[官方2026.04 GPU套件快照](https://github.com/googlecolab/backend-info/blob/77d5dbef56d73b96db5efef2280679cb548c9bd9/pip-freeze.gpu.txt)、[Mamba官方release](https://github.com/state-spaces/mamba/releases/tag/v2.3.2.post1)。
