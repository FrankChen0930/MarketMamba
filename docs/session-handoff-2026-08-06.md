# Session 交接（2026-08-05 深夜 → 08-06 早上）

> **一句話**：V6.2 上線鏈路全部做完並驗證通過，排程已設好，**週一 2026-08-10 上線**。
> 早上第一件事：看 Colab 的 47 維結果。

---

## 早上第一件事：看 47 維結果

Colab 的 `dim47_ablation` 昨晚 23:00 左右開跑（約 3.6h）。結果在 Drive 的
`dim47_ablation_result.json`，腳本會自己印判定表。

**判準已經寫死在腳本裡，不要事後改**：重評 IC 要 ≥ 控制組（**0.1145**）**+0.009**
才算勝出。達不到就**維持 59 維 + Group D 歸零**——那是已經被組合層驗證過的設定
（+38.1%／Sharpe 1.697），而換成 47 維要把推論端接線**整套重做一次**
（`INPUT_DIM`／`FEATURE_GROUPS`／3 組 embedding 都要改）。

⚠️ **不可以因為「參數比較少、比較乾淨」就採用**——那不是證據。

---

## 現在的狀態：全部做完、已 push

| 元件 | 檔案 | 狀態 |
|---|---|---|
| 推論 | `V6/run_v62_inference.py` | ✅ 582 天驗證通過 |
| 組合層 | `V6/v62_portfolio.py` | ✅ replay 對 `portfolio_lab` 一致（年化差 0.002pp） |
| 每日 orchestrator | `V6/run_v62_daily.py` | ✅ 自帶抓資料、當日檢查、Telegram 告警 |
| 進度視窗 | `V6/progress_window.py` | ✅ 中文 fallback + 主執行緒 Tk |
| 隱藏黑窗 | `V6/scripts/run_hidden.vbs` | ✅ 實測 exit code 正確回傳 |
| 後端 | `app/backend/routers/v62.py` | ✅ 已部署 |
| 前端 | `app/frontend/src/pages/BreadthPortfolio.jsx` | ✅ `/breadth/portfolio` |
| 排程 | Task Scheduler | ✅ 已設定（見下） |

commit：`03669d9` → `28357b3` → `eb1df02` → `6dd3ff5` → `a9f6f1d` → `d16a4f0` → `d38943e` → `b457ca9`

---

## 排程（已生效，2026-08-05 設定）

| 工作 | 時間 | 內容 |
|---|---|---|
| `PersonalOS_Daily` | 19:30 → **21:30** | V6.1 + 雙模型 |
| `MarketMamba_V62` | **22:15**（平日） | `run_hidden.vbs` → `v62_daily.bat` |

`MarketMamba_V62` 開了 `StartWhenAvailable`（錯過會補跑）＋ `WakeToRun`（睡眠喚醒）。

**為什麼是 21:30 而不是 19:30**：實測 19:35 時 TWSE 的 `margin`/`daytrade`
還「尚未公布」，21:11 才有當日資料。跑太早會靜默 ffill 昨天的值。

**⚠️ 沒能做到的一件事**：使用者要「不用登入也能執行」，但
① Claude Code 的 shell **不是系統管理員**，建不了 S4U 工作（Access denied）
② 更重要：那個模式跑在 Session 0，**WSL2 需要使用者 session 才能啟動**
→ 極可能整條鏈壞掉。**這一點沒能實測**（因為建不了那種工作），
只能依既有記載與原理判斷。要試的話用提權 PowerShell，指令在 CLAUDE.md。

---

## 週一上線前還沒做的

1. **裝中文字型**（需 sudo 密碼，使用者自己跑）：
   `wsl -d Ubuntu -- bash -lc "sudo apt-get update && sudo apt-get install -y fonts-noto-cjk"`
   沒裝的話進度視窗會**自動改用英文標籤**並印出這行指令（不會出現豆腐方塊）。
2. **`V6/.env` 加 Telegram 兩行**（選配）：`TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID`。
   值可從 `PersonalOS/scripts/.env` 複製。沒設的話告警只印 log。
3. **週一手動跑第一次建倉**（不要靠排程，要在旁邊看著）：
   `V6\scripts\v62_daily.bat --first-day`

---

## 這一輪最重要的四個發現（會改變怎麼讀既有結果）

1. **資料修正對模型的影響是雜訊**。base matrix 建於 07-30、Group C 之後被改過
   （FCF 修正 + MOPS 補齊），逐欄比對顯示 `Free_Cash_Flow` ρ 只有 0.09。
   但用現在的資料重評 582 天：年化 38.0→**38.1%**、decile 4.999→**5.007**、
   mean IC **0.1139→0.1139**。→ **原訂的「重建矩陣＋全部重訓（≈29h A100）」取消。**
2. **margin 不是 T+1**，是 19:35 抓太早（CLAUDE.md 原本記錯，已更正）。
3. **EPS 截斷是局部缺陷不是全歷史**（整數比例逐年 1.0~1.8% 平穩）。
4. **單日驗證挑在窗緣＝專挑最差的一天**。我挑 2026-06-02 → ρ=0.9398 未過；
   整窗 median 0.9940、569/582 天過。分界線精確落在 2026-05-15
   （＝2026Q1 財報 `available_from`）。

---

## 待決定 / 待辦（不擋上線）

- **`v2_kg_nomacro__live.parquet` 與 8 個 `__common.parquet`** 都在
  `V6/experimental/result/scores/`，**會被無過濾的 `--sweep` 一併掃到**。要留要刪需決定。
- **V6.1 / 雙模型退役**：使用者已同意拆，但**要等 V6.2 連續跑順幾天**。
  V6.2 已能自己抓資料（`fetch_data()`），過渡期兩者排程錯開即可。
- 舊備份可刪：`Data/processed_v6/*_backup_before_mops_20260804.parquet`（79 MB）、
  `trading_status_raw_backup_20260804.parquet`。
- `trading_status` 仍未接進每日流程（現行 `build()` 是整檔重建，需先加增量路徑）。

---

## 使用者的下一階段規劃（已同意的方向）

- **多模型並行累積實戰紀錄**。三個模型已就緒（`v2_kg_nomacro` 5d 頭 /
  `head10d` / `head20d`），**`{5d,20d}` 不需訓練**——`v6_short_H_h20.pt` 就是。
- **模型集合要在起跑日定案**：晚加入的模型少了那段紀錄，無法公平並列。
  若 47 維要進來，就要在週一之前一起上，不能之後替換
  （替換＝紀錄斷掉，與「累積真實 out-of-sample」的目標衝突）。
- **再平衡率是組合層參數、不是模型參數**——一份分數可同時跑 5/10/20 日，零成本。
