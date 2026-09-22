# V7 研究輸出契約 v1

日期：2026-09-15。此契約供逐日排名、研究報告與模擬組合共同讀取；排序分數不是報酬率或上漲機率。

## 逐日預測列

每列由 `date + stock_id + model_id` 唯一識別：

| 欄位 | 型別 | 規則 |
|---|---|---|
| date | YYYY-MM-DD | 訊號日期 |
| stock_id | string | 保留前導零 |
| score_5d / score_10d | float or null | 模型排序分數；不可補 0 |
| valid_5d / valid_10d | bool | 僅在對應 score 有限時為 true |
| label_5d / label_10d | float or null | 真實排序標籤；未成熟時必須 null |
| label_5d_mature / label_10d_mature | bool | 成熟狀態，不由數值 0 代替 |
| model_id | string | 架構／窗口／seed 身份 |
| data_id | SHA-256 identity | 矩陣 files 與 protocol fingerprint |
| seed | integer | 單模型 seed；集成列另以模型身份表示 |

Parquet 的浮點 null 讀回可能成為 NaN；讀取邊界須正規化回 null。Infinity、未成熟但非缺值的標籤、重複 identity 均硬失敗。

## 排名與集成

每一日期、每一頭分開計算。三 seed 先各自在共同有效股票集合轉成百分位，再做等權平均。任何 seed 缺少的股票不進該頭集成；不把缺值當 0。同分以 `stock_id` 升冪穩定決勝，確保 API、報告與組合重跑一致。

A/B 比較只使用三 seed 皆有效且標籤成熟的同樣本，保存雙頭每日有效數、單 seed／集成 IC、Top20／Top50 seed 重疊率、年度／季度／市況彙總。小於要求 N 時另列 effective_n，不能把小樣本的 100% overlap 當 Top50 證據。

## 實驗摘要

摘要至少包含：schema/revision、設定、train/selection/evaluation 區間、30 交易日 purge、checkpoint identity 與 SHA、data_id、source hashes、runtime versions、雙頭指標、完成狀態、停止原因與限制。selection 選 checkpoint；evaluation 不參與早停。完成輸出若 identity 或任何已承諾分片 SHA 不符，禁止續用。

## 組合邊界

模型規格與組合規格分離。下游只能把 rank／score 當相對排序訊號；權重、持股數、再平衡、緩衝、產業限制、成交與成本由獨立組合規格決定。研究篩選器不得改寫已凍結組合，未成熟標籤不得產生假績效。

