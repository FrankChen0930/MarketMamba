"""Build the B4 Colab deliverables: code bundle zip + thin notebook. RESEARCH_ONLY.

python tools/research_env/make_colab_b4.py --out .local-data/colab_b4
Refuses to overwrite. The notebook embeds the bundle SHA-256 and verifies it before import.
"""
import argparse, hashlib, json, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FILES = ['__init__.py', 'matrix_baselines.py', 'significance.py', 'colab_runner.py', 'seq_models.py', 'gpu_env.py']
E5 = '/content/drive/MyDrive/MarketMamba_V7_Corrected_E5_Replication/matrix'
E1 = '/content/drive/MyDrive/MarketMamba_V7_E1_Rolling_Origin_Refresh/matrix/v7_e1_rolling_origin_refresh_v1'
RESULTS = '/content/drive/MyDrive/MarketMamba_V7_B4_baselines'
WHEELS = '/content/drive/MyDrive/MarketMamba_GPU_Wheels'


def build_bundle(path):
    with zipfile.ZipFile(path, 'x', zipfile.ZIP_DEFLATED) as z:
        for name in FILES:
            info = zipfile.ZipInfo(f'research_lab/{name}', date_time=(2026, 11, 15, 0, 0, 0))  # deterministic bytes
            z.writestr(info, (ROOT/'V6/research_lab'/name).read_bytes())
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cell(src, kind='code'):
    c = {'cell_type': kind, 'metadata': {}, 'source': src.strip('\n').splitlines(True)}
    if kind == 'code':
        c.update(execution_count=None, outputs=[])
    return c


def notebook(bundle_name, bundle_sha):
    cells = [
        cell('# Market Mamba B4：矩陣基線（Rules / Ridge / LightGBM + 配對顯著性）\n'
             '`RESEARCH_ONLY`。薄型 Runner：只做掛載、驗證、執行與輸出；模型邏輯在 `research_lab`。'
             '矩陣唯讀；結果寫入 Drive 的獨立資料夾，已完成的實驗不會被覆寫，中斷後重跑會從已完成的模型續跑。\n\n'
             '**操作**：先把 `' + bundle_name + '` 放進 Drive 的 `MarketMamba_V7_B4_baselines/`，'
             '選 CPU 執行階段（高 RAM 較穩），改第 2 格的 `EXPERIMENT`，然後「全部執行」。建議順序：E5 SMOKE → E5 完整 → E1 SMOKE → E1 完整。', 'markdown'),
        cell(f'''
# 2. 設定（只需改這裡）
EXPERIMENT = "E5"     # "E5" 或 "E1"，兩者結果各自獨立
LABEL = "y5"          # "y5" 或 "y10"
SMOKE = True          # True: 5 萬列 / 50 棵樹 / 200 次 bootstrap；完整跑請改 False
SEED = 17
MATRIX_DIRS = {{"E5": "{E5}",
               "E1": "{E1}"}}
RESULTS_ROOT = "{RESULTS}"
BUNDLE_NAME = "{bundle_name}"
BUNDLE_SHA256 = "{bundle_sha}"
STAGE_DIR = "/content/b4_stage"   # 本地暫存（SSD），避免在 Drive FUSE 上做 memory-map
'''),
        cell('''
# 3. 掛載 Drive、確認資源與套件
import os, sys, shutil, subprocess
from google.colab import drive
drive.mount("/content/drive")
os.makedirs(RESULTS_ROOT, exist_ok=True)
print("RAM GB:", round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1), "| CPUs:", os.cpu_count())
print("free disk GB:", round(shutil.disk_usage("/content").free / 1e9, 1))
try:
    import sklearn, lightgbm
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "scikit-learn", "lightgbm"])
'''),
        cell('''
# 4. 載入程式 bundle 並驗證 SHA-256（不符就停止）
import hashlib, zipfile
bundle = os.path.join(RESULTS_ROOT, BUNDLE_NAME)
if not os.path.exists(bundle):
    from google.colab import files
    up = files.upload(); bundle = BUNDLE_NAME
    assert BUNDLE_NAME in up, "請上傳 " + BUNDLE_NAME
got = hashlib.sha256(open(bundle, "rb").read()).hexdigest()
assert got == BUNDLE_SHA256, f"bundle hash mismatch: {got}"
shutil.rmtree("/content/b4_code", ignore_errors=True)
zipfile.ZipFile(bundle).extractall("/content/b4_code")
sys.path.insert(0, "/content/b4_code")
from research_lab import colab_runner as cr
print("code identity:", cr.code_identity())
'''),
        cell('''
# 5. 執行（身份驗證 → 本地暫存 → 記憶體評估 → 基線 → 顯著性 → 存檔）
src = MATRIX_DIRS[EXPERIMENT]
result = cr.run_experiment(src, RESULTS_ROOT, EXPERIMENT, label=LABEL, smoke=SMOKE, seed=SEED,
                           stage_dir=os.path.join(STAGE_DIR, EXPERIMENT))
'''),
        cell('''
# 6. 結果摘要
import pandas as pd
print("評估期:", result["semantics"]["evaluation"], "| 共同 session:", result["common_sessions"], "| 證據:", result["evidence_tier"])
display(pd.DataFrame(result["models"]).T[["ic_mean", "ic_std", "icir", "days", "seconds"]])
display(pd.DataFrame(result["vs_reversal_ret5"])[["model", "mean_diff", "t_hac", "p_hac", "p_hac_holm", "p_boot_holm"]])
print("LightGBM vs Ridge:", result["lightgbm_vs_ridge"] and {k: result["lightgbm_vs_ridge"][0][k] for k in ("mean_diff", "t_hac", "p_hac")})
print("結果目錄：", RESULTS_ROOT, "（含 identity.json / config.json / daily_ic/*.csv / result.json）")
'''),
    ]
    return {'cells': cells, 'metadata': {'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
            'colab': {'name': 'b4_matrix_baselines.ipynb'}}, 'nbformat': 4, 'nbformat_minor': 5}


def probe_notebook(bundle_name, bundle_sha):
    cells = [
        cell('# Market Mamba B7：官方 Mamba2／Samba 環境（一鍵）\n`RESEARCH_ONLY`，只做環境安裝與極小 forward/backward 驗證，**不訓練**。\n\n'
             '## 怎麼用（只有兩步，每次開新的 Colab 都一樣）\n'
             '1. 執行階段選 **GPU（T4）**，按「**全部執行**」。第一次會把 torch 換成 2.10 並**自動重啟**（畫面短暫顯示中斷／重新連線是正常的），這一輪到此為止。\n'
             '2. 重啟後再按一次「**全部執行**」。這次會安裝 mamba-ssm 與 causal-conv1d（官方預編譯版，約半分鐘）、跑驗證，**通過後自動把 wheel 存到 Drive**。最後一格會印出結果。\n\n'
             '把最後一格輸出中的 `status`、`saved`、`steps` 貼回來即可。`status` 要是 `verified`。\n\n'
             '前置：把 `' + bundle_name + '` 放在 Drive 的 `MarketMamba_V7_B4_baselines/`。', 'markdown'),
        cell(f'''
import os, sys, json, zipfile, hashlib, shutil, subprocess, time
from google.colab import drive
drive.mount("/content/drive")
RESULTS_ROOT = "{RESULTS}"
BUNDLE = os.path.join(RESULTS_ROOT, "{bundle_name}")
assert hashlib.sha256(open(BUNDLE, "rb").read()).hexdigest() == "{bundle_sha}", "bundle hash mismatch"
shutil.rmtree("/content/b7_code", ignore_errors=True)
zipfile.ZipFile(BUNDLE).extractall("/content/b7_code")
sys.path.insert(0, "/content/b7_code")
import importlib
for _m in [k for k in sys.modules if k == "research_lab" or k.startswith("research_lab.")]:
    del sys.modules[_m]   # 同一執行階段重跑時，丟掉舊版模組快取，確保用的是剛解壓的版本
importlib.invalidate_caches()
'''),
        cell('''
# （可選）固定 torch：官方 mamba-ssm／causal-conv1d 預編譯 wheel 目前只到 torch 2.10（cu12/cu13）。
# Colab 預設 torch 若更新（如 2.14+cu130）就沒有現成 wheel，只能花約 40 分鐘編譯。
# 預設 PIN_TORCH=True：torch 不是 2.10 時，這格會安裝 torch 2.10.0+cu128 並自動重啟執行階段（畫面短暫顯示重啟是正常的）；重啟後再按一次「全部執行」，此格就會略過。
import torch
PIN_TORCH = True
if PIN_TORCH and not torch.__version__.startswith("2.10.0"):
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch==2.10.0", "--index-url", "https://download.pytorch.org/whl/cu128"], capture_output=True, text=True)
    print("pip rc", r.returncode, r.stderr[-600:])
    if r.returncode == 0:
        print("torch 2.10.0 installed; restarting runtime - then Run all again"); os.kill(os.getpid(), 9)
'''),
        cell(f'''
# GPU 環境入口（B7／B8 共用）：已安裝→Drive 相容 wheel→（必要時）建置；通過真實 Mamba2 forward/backward 後才保存 wheel
from research_lab import gpu_env
report = gpu_env.setup("{WHEELS}", allow_build=True, allow_compile=False)   # 無官方預編譯 wheel 時預設停止；要現場編譯才改 allow_compile=True\n# 已安裝但想重新取得並保存 wheel：force=("mamba-ssm", "causal-conv1d")（用 --no-cache-dir 重新取得官方預編譯版）
'''),
        cell('''
# 結果：status 必須是 "verified"；report["saved"] 為空代表本次沒有保存任何 wheel（不會假裝成功）
print("status:", report["status"]); print("saved:", report["saved"]); print("verify:", report.get("verify"), report.get("verify_error"))\nprint("purged stale modules:", report.get("purged_stale_modules")); print("traceback:", report.get("verify_traceback")); print("steps:", *report["steps"], sep="\\n  "); print("diagnostics:", json.dumps(report.get("diagnostics"), indent=1))
'''),
    ]
    return {'cells': cells, 'metadata': {'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
            'colab': {'name': 'b7_mamba_probe.ipynb', 'gpuType': 'T4'}, 'accelerator': 'GPU'}, 'nbformat': 4, 'nbformat_minor': 5}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--bundle-name', default='b4_research_lab_bundle.zip')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    bundle = a.out/a.bundle_name
    nb = a.out/'b4_matrix_baselines.ipynb'
    nb7 = a.out/'b7_mamba_probe.ipynb'
    if bundle.exists() or nb.exists() or nb7.exists():
        raise FileExistsError('refusing to overwrite')
    sha = build_bundle(bundle)
    nb.write_text(json.dumps(notebook(bundle.name, sha), ensure_ascii=False, indent=1))
    nb7.write_text(json.dumps(probe_notebook(bundle.name, sha), ensure_ascii=False, indent=1))
    (a.out/(a.bundle_name+'.sha256')).write_text(f'{sha}  {bundle.name}\n')
    print(json.dumps(dict(bundle=str(bundle), sha256=sha, notebook=str(nb))))


if __name__ == '__main__':
    main()
