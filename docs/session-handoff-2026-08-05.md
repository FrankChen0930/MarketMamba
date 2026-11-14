# Session 交接（2026-08-05 晚）

> 給下一個 session：這兩天（08-04 ~ 08-05）做了很多事，**全部已 commit + push**。
> 這份是「一分鐘上手」，細節都在 CLAUDE.md 與各自的 docs。
>
> **沒有任何東西正在背景執行。** 不需要檢查 Colab session、不需要等任何跑批。

---

## 現在在哪一步

**V6.2 上線主線：規格已定案，卡在「推論端接線」——而那要先列計畫給使用者確認（規則 2）。**

規格與推論端的四個落差寫在 `CLAUDE.md` 的「下一步」→「★ V6.2 上線規格（2026-08-05 定案）＋ 推論端落差」。
**下一個 session 的第一件事就是把那份落差變成實作計畫給使用者看。**

一句話版本：上線 `v2_kg_nomacro` 的 **`5d/20`**（5 日頭、每 20 個交易日再平衡、N=50、k=1.5），
checkpoint 已在 `D:\Downloads\v6_short_GD_no_macro_gatv2.pt`，**不需要重訓**。

---

## 這兩天最重要的三個發現（會改變怎麼讀既有結果）

### 1. 組合層有 ±6pp 的雜訊底線，而 decile spread 幾乎不受影響

- Top50 年化的 run-to-run σ = **2.68pp**、Sharpe σ = 0.126
- **decile spread Sharpe 的 σ 只有 0.019（穩定 40 倍）**
- → **比較模型優劣用 decile spread，不要用 Top50 年化**
- → 連帶好處：**用 decile 就不必每個模型都跑多 seed**

**八模型表裡小於 6pp 的差距不該當數。**

### 2. GRU 的舊數字是錯的（v1 資料 + 無 purge）

`+31.3% / 排第 2` → 修正後 `19.6% / 排第 4`（−11.7pp）。
**教訓：資料層做過修復後，所有既有結果都要標記「待重跑」，不是預設仍可用。**

### 3. 標籤 horizon 在 Mamba 上沒有複製 Ridge/GBDT 的結果

Ridge/GBDT 是「標籤越長越好」；Mamba 是倒 U 形（10d 最好、20d 掉回去）。
**但同設定兩次抽樣在 20 日欄就差 6.2pp → Mamba 那個結論本身也在雜訊邊緣。**
→ **不要把上線規格改成 20d 標籤**（不是證明它差，是沒有證據支持改動）。

---

## 使用者訂的記法（請沿用）

**`<預測頭>/<再平衡天數>`**，例如 `5d/20` = 5 日頭 + 每 20 個交易日再平衡。

---

## 使用者提出、還沒做的設計（等他回來討論）

1. **中間 19 天顯示什麼**（20 日才換一次股，其他天要看什麼）
   我的建議已給：持倉 + 倒數 + **漂移度**（目前持倉還有幾檔在模型當前 Top50 內），
   刻意**不顯示「今日選股」**——那會誘導每日換股，而回測證明每日再平衡 −19.9%。
2. **多 Mamba 變體並行跑每日推論**（作品集 + 實戰可比較性）
   使用者已同意「先把 V6.2 主線收完再說」。成本確認過：**加 N 個模型的邊際成本幾乎是零**
   （特徵矩陣建構才是成本，且共用）。真正的價值是**累積真實 out-of-sample 前瞻紀錄**——
   那是 Colab 的錢買不到、只能用時間換的東西。
3. **前端**：預設頁顯示最佳模型，其他變體做成可點選按鈕；實驗結果用表格，
   支援三種排序（依預測日期／依再平衡日期／依模型）。使用者說骨架已改過、不需大改。

---

## 可以直接動手、不需要決定的小事

- **`result/scores/` 裡有 8 個 `{model}__common.parquet`**（08-05 為統一 panel 產生的）。
  它們**會被 `--sweep` 一併掃到**，每次多跑 8 組。要嘛留著當正式對照集、要嘛刪，需決定。
- `Data/processed_v6/*_backup_before_mops_20260804.parquet`（79 MB）與
  `trading_status_raw_backup_20260804.parquet`：MOPS 補齊已穩定跑過，可刪。

---

## 這兩天的 commit（都已 push）

```
fed3c68  portfolio_lab: run_config 加 reb_idx 參數
（標記慣例 5d/20 + 標籤×再平衡完整矩陣）
ef4a2ed  GRU 修正 + 同一把尺稽核：decile spread 才是比較模型的正確量尺
d4b6cef  baseline_rnn: 加 --seed / --purge / --tag
2b93dca  標籤 horizon Mamba 版：20d 未勝出 + 首次量到組合層雜訊底線
26e4eed  head20d 執行交接單 + CLAUDE.md 指標
aeeb134  head20d_ablation: 補上每組的 seed 重設（原本是髒配對）
9a39d49  MOPS 財報直連 + trading_status 增量 + FCF 兩層修正
```

相關文件：
- `docs/head20d-ablation-result-2026-08-05.md` —— 標籤 horizon 完整判讀
- `docs/head20d-run-handoff-2026-08-04.md` —— Colab CLI 首跑失敗的根因（WSL 重啟殺掉 keep-alive）
- `CLAUDE.md` 開頭「標記慣例」—— 記法 + 標籤×再平衡完整矩陣

---

## 環境備忘（這兩天新增的）

- **Colab CLI 已裝好可用**（WSL2 的 conda 環境 `colabcli`，CPU/T4/A100 都測過）。
  但**不適合整夜跑批**——keep-alive 依附 WSL 程序，WSL 一重啟就全毀（08-04 實際發生）。
  長訓練用 Colab 網頁版。詳見 CLAUDE.md「Colab CLI」章節的五個坑。
- **背景任務會被砍**：長工作用 `nohup` + 在同一條指令裡 `sleep` 撐住父程序才能真正脫離
  （只有 `nohup`、或只有 `setsid` 都不夠——實測過）。
- **`set -u` 與 conda 不相容**：`conda activate` 會引用未設定變數 → 腳本靜默退出、連 log 都沒有。
