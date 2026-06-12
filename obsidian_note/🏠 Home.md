# MarketMamba 筆記中心

> 個人台股量化投資自動化系統 — 思考與發想紀錄
> 最後更新：2026-06-12

---

## 📍 快速導覽

| 分區 | 說明 |
|------|------|
| [[00 背景/關於我與這個專案]] | 做這件事的動機、備案路徑、未來目標 |
| [[01 系統現況/現況整理]] | 系統現在在哪、做了什麼、下一步 |
| [[02 問題追蹤/已知問題清單]] | Agent 分析找出的 bugs & 改善點 |
| [[03 架構筆記/模型架構]] | Mamba + GATv2 設計理念筆記 |
| [[03 架構筆記/推論流程]] | 每日推論管線的細節 |
| [[03 架構筆記/前端網頁]] | Web Dashboard + PersonalOS 架構、頁面清單、已知問題 |
| [[03 架構筆記/訓練紀錄]] | 每次重訓的動機、改動、結果 |
| [[04 發想/投資策略框架]] | 個人操作邏輯與選股流程 |
| [[04 發想/發想空間]] | 未整理的想法、突然冒出的點子 |

---

## 🎯 現在最重要的事

> [!important] 當前優先任務（2026-06-12）
> **系統穩定性** — inference 不穩定，第一輪強化已完成（暖機檢查、逾時自動重啟重試、Telegram 告警、git push retry），進入觀察驗證階段
> 詳見 [[02 問題追蹤/已知問題清單#推論管線風險]]、[[01 系統現況/現況整理]]

---

## 📅 最近動態

- 2026-06-12：推論穩定性強化第一輪完成 — WSL2 暖機檢查 + 逾時自動重啟重試、Telegram 後台告警（SYS-07）、AdminOps 階段卡片化、git push 失敗 retry 3 次（SYS-08）
- 2026-06-09：PR 3 持倉四層退場 checker 完成（使用者已驗收）
- 2026-06-07：新增「前端網頁」架構筆記（Web Dashboard + PersonalOS）
- 2026-06-07：.gitignore 加入 market_mamba_note/、.claude/
- 2026-06-05：建立 CLAUDE.md 兩層架構（靜態規則 + 動態狀態）
- 2026-06-05：V6.2 Zero-Padding Mask 完成（Long branch）
- 2026-06-05：Scale Gate 監控強化（訓練時 print + 圖表第 4 欄）
- 2026-06-05：訊號系統 V6.2 整修（signal_conditions.py、pattern_scanner.py、sim_engine_v3.py）

---

## 🗺️ 系統全貌（一眼看懂）

```
每日 17:00 自動觸發
  Windows Task Scheduler
    → WSL2 Ubuntu
      → run_daily_inference.py
          [1] 資料更新（yfinance + FinMind）
          [2] 特徵矩陣（56 因子）
          [3] Mamba+GATv2 推論 → Alpha 訊號
          [4] LLM 市場報告（Claude API）
          [5] 歸檔
          [6] 訊號掃描 → action_signals.json
          [7] git push → GitHub
              → Render 後端自動更新
              → Vercel 前端展示
```

---

## 🔗 外部連結

- 前端：https://marketmamba.vercel.app
- 後端：https://marketmamba-api.onrender.com
- GitHub：FrankChen0930/MarketMamba
