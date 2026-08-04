# head20d 實驗執行交接（2026-08-04 夜）

> **給接手的人／下一個 session**：這份是「跑到一半換手」的交接單。
> 訓練在 Colab 上跑，**不依附任何 Claude Code session**，關掉視窗也會跑完。
> 先讀「一分鐘現況」，再看要做什麼。

---

## 一分鐘現況

| 項目 | 值 |
|---|---|
| Colab session 名稱 | `head20d`（A100-SXM4-40GB） |
| h10（控制組，第二顆頭學 `Alpha_10d`） | **2026-08-04 22:19:32 開跑**，每 epoch ≈ 17.7 分、10 epochs → 約 **01:20 完成** |
| h20（實驗組，第二顆頭學 `Alpha_20d`） | 由接力腳本自動接續，約 **04:50 完成** |
| 接力腳本 | `~/h20_chain.sh`（WSL，`nohup` 脫離、PID 見 log），跑完**會自動 `colab stop`** |
| 進度 log | `/tmp/h10.log`｜`/tmp/h20.log`｜`/tmp/h20_chain.log`（皆在 WSL） |
| 結果落點 | **Drive** `MyDrive/MarketMamba_V6/`：`status_short_H_h10.json`／`_h20.json`、`head20d_ablation_result.json`、checkpoint `v6_short_H_h10.pt`／`_h20.pt` |
| 本機同步 | Drive 桌面同步在 `G:\我的雲端硬碟\MarketMamba_V6\` → **不需要 `colab download`** |

⚠️ **最重要的一件事**：沒有 `colab stop` 的 session 會**一路燒 compute units 到 24 小時上限**。
接力腳本結尾一定會 stop（連逾時分支也會），但**接手時務必自己確認一次**。

---

## 常用指令（WSL，直接複製）

```bash
# 進度
wsl -d Ubuntu -- bash -lc "tail -20 /tmp/h20_chain.log"     # 接力腳本在做什麼
wsl -d Ubuntu -- bash -lc "tail -20 /tmp/h10.log"           # h10 訓練輸出
wsl -d Ubuntu -- bash -lc "tail -20 /tmp/h20.log"           # h20 訓練輸出

# session 狀態（BUSY / IDLE）
wsl -d Ubuntu -- bash -lc "~/miniconda3/envs/colabcli/bin/colab status -s head20d < /dev/null"

# 接力腳本還活著嗎
wsl -d Ubuntu -- bash -lc "ps -eo pid,etime,args | grep '[h]20_chain'"

# ⚠️ 手動停（確認沒在跑、或要中止時）
wsl -d Ubuntu -- bash -lc "~/miniconda3/envs/colabcli/bin/colab stop -s head20d < /dev/null"
wsl -d Ubuntu -- bash -lc "~/miniconda3/envs/colabcli/bin/colab sessions < /dev/null"   # 確認已無 session
```

### 如果接力腳本掛了、h10 已完成但 h20 沒起來

```bash
wsl -d Ubuntu -- bash -lc "~/miniconda3/envs/colabcli/bin/colab exec -s head20d --timeout 3600 \
  -f /tmp/launch_h20.py < /dev/null > /tmp/h20.log 2>&1"
```
⚠️ 前提是 **kernel 還活著**（`df` 與已安裝的套件都在 kernel 記憶體裡）。
若 session 已被 stop，必須從頭重來（見下方「從零重建」）。

---

## 跑完之後要做什麼

1. **先判讀 `status_short_H_h10.json` / `_h20.json`（Drive）**
   - 腳本印的 `val IC` 是**第 0 顆頭（5d）**的，兩組本來就該接近
     → 那是**配對是否乾淨的檢查點**，不是本實驗的結論。兩組差很多＝配對有問題。
2. **產分數 → 過 portfolio_lab**（本機，**Windows 端跑**，WSL 的 pandas 3.0 會炸）
   ```
   # checkpoint 從 G:\我的雲端硬碟\MarketMamba_V6\ 複製到 D:\Downloads\
   # 在 V6/experimental/score_mamba_local.py 的 ARMS 加入這兩組
   python V6/experimental/score_mamba_local.py --arms <arm> --head 10d
   python V6/experimental/portfolio_lab.py --sweep
   ```
   **第二顆頭一律用 `--head 10d` 讀**，不論它學的是 10d 還是 20d（那是輸出的第 1 欄）。
3. **看 N=50 / k=1.5 / 20 日那一格**，與既有八模型表並列。
   對照基準：`v2_kg_nomacro` = **+38.0% / Sharpe 1.713**（同窗同格）。
4. **若 h20 明顯勝出** → 落地規格從「預測 5d、每 20 日再平衡」改成「**預測 20d、每 20 日再平衡**」。
   本機 Ridge/GBDT 已達判讀門檻（+6.8pp / +10.6pp），Mamba 是最後一塊拼圖。

---

## 實驗設定（供判讀時核對）

```
SEED=20260730  dropout=0.2  epochs=10  early_stop=5
use_gat=True   kg=knowledge_graph_v2.npz   zero_macro=True
purge_horizon=20  embargo=20
train 2,651 天 (2013-01-02 → 2023-11-03)
val     582 天 (2024-01-02 → 2026-06-02)   ← 與 2×2 最佳格完全相同，可直接並列
矩陣：Drive 的 V6_Feature_Matrix.parquet（2026-07-30 建、7,990,686 列、2,245 支、59 維）
```

**⚠️ 刻意不用今天（08-04）補齊的 MOPS 財報資料。** 那會讓資料基礎與 2×2 最佳格不同、
多一個變因。要比較就必須同基礎。

**已修的設計缺陷**：`head20d_ablation` 原本**完全沒有設 seed**（`SEED` 只寫進輸出 JSON），
直接呼叫 `train_short_model`、繞過 `groupd_ablation._train_one_arm` 的 seed 重設
→ 兩組會是髒配對。已於 commit `aeeb134` 修好，跑前用 `inspect.getsource` 斷言確認生效。

---

## Colab CLI 的坑（這次實測踩到的，寫新指令前先看）

| # | 坑 | 解法 |
|---|---|---|
| 1 | `colab exec --timeout` **預設 30 秒**，而且是「**等待輸出的間隔**」不是總時長 | 長跑一律 `--timeout 3600` |
| 2 | 背景／非互動執行**必須 `< /dev/null`** | 否則 client 卡在等 stdin EOF、**根本沒送出**，但程序看起來活著 |
| 3 | 不要把輸出接 `\| tail` | tail 等指令結束才吐 → 看起來像沒動靜。改成 `> /tmp/x.log 2>&1` |
| 4 | kernel 啟動後才建立的路徑無法 import | 需 `importlib.invalidate_caches()`（負面查找被快取） |
| 5 | 模組快取：upload/pull 後仍用舊版且**不報錯** | 先刪 `sys.modules` 再 import，並用 `inspect.getsource` 斷言關鍵字 |
| 6 | client 逾時斷開後 **kernel 照跑** | 這是好事：載 2.5 GB 那次就是逾時後才完成的 |
| 7 | `colab upload` **不需要 TTY** ✓ | 需要 TTY 的只有 `repl` / `console` / `auth` / `drivemount` |

---

## 從零重建（若 session 已死、要重跑）

```bash
COLAB=~/miniconda3/envs/colabcli/bin/colab
$COLAB new -s head20d --gpu A100          # 1) 開 session
$COLAB drivemount -s head20d              # 2) ⚠️ 需要你本人在終端機（TTY）
```
接著依序 exec（檔案都在 WSL `/tmp/`，若沒了就照下面重建）：
1. **setup**：clone repo → `importlib.invalidate_caches()` → 從 zip 抽 `knowledge_graph_v2.npz`（82 KB，不必解壓整包 598 MB）→ 直接讀 Drive 的 `V6_Feature_Matrix.parquet` 成 `df`
2. **deps**：`pip install torch-geometric` + 從 Drive 裝 mamba wheel
   `/content/drive/MyDrive/MarketMamba/mamba_wheels/cu128torch2110cp312/*.whl`
   （**注意是 `MarketMamba` 不是 `MarketMamba_V6`**；torch 2.11.0+cu128 正好有現成的，65 秒裝完，不必編譯）
3. **launch**：`H.run_head20d_ablation(df, arms=("h10",), drive_dir="/content/drive/MyDrive/MarketMamba_V6")`

**跳過原 notebook 的 Cell 2（3 GB 解壓）是刻意的**——Drive 的 `V6_Feature_Matrix.parquet`
已經是 `clean_and_scale` 之後的成品，直接讀就好。

---

## 這一輪已完成並 push 的東西（commit `9a39d49`、`aeeb134`）

- MOPS 財報整批直連（財報覆蓋斷崖已解，四項驗證全過）
- `trading_status` 增量 + 接進每日流程
- `Free_Cash_Flow` 兩層 bug 修正（走 `fundamentals_v2`）
- `head20d_ablation` 的 seed 修正

細節見 `CLAUDE.md` 的「最近完成」與「決策紀錄」。
