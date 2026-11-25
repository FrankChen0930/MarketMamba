# P4b 結果：神經分數與 C0+ 的 rank-average 混合（已曝光、描述性）

證據等級：E2 RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED。規格：[spec-p4b-blend-20261124.md](../tasks/spec-p4b-blend-20261124.md)（FROZEN，Human 2026-11-24 對話核准）。工程：[v7_p4b_engineering.md](v7_p4b_engineering.md)；獨立審查 GO（`a8822c7`）。

## 輸入與身分
- 36 個 P4a Q1／Q2 W60 fits（`.artifacts/p4/p4a-results-3/run-6fbaf815…`，來源 zip SHA `b9060b74…`）與 P1 C0+ 三種子原始分數（`.artifacts/p1/full-scores-ace2eef`）。
- 價格快照於任何 P4b 績效產生前凍結：`.artifacts/p4b/snapshot-prices-20261125/`（prices_raw SHA `29433822…`、market_value_raw SHA `ccc1f4d1…`，volume 單位 shares）。
- 輸出：`.artifacts/p4b/eval-full-20261125/evaluation.json`；manifest SHA `beee0bd2ca9f07f0509fe06733bd8e96dffaefebb16816122d3ef93c7df3a898`；2,300,620 列；支援門檻全通過。

## 結果（混合 − 單獨 C0+，日均淨超額；P2 主要格）
| 混合 | 2021–2023（Holm m=3） | 95% CI | p_boot／Holm | 全六折 | 判定 |
|---|---|---|---|---|---|
| B1＝C0+＋Mamba2 | −0.0079% | [−0.024%, +0.007%] | .36／.43 | −0.0095% | 未證實；非候選 |
| B2＝C0+＋三神經臂平均 | −0.0171% | [−0.043%, +0.004%] | .16／.43 | −0.0191% | 未證實；非候選 |
| B3＝C0+＋GRU（對照） | −0.0165% | [−0.041%, +0.004%] | .14／.43 | −0.0182% | 未證實；非候選 |

逐折：三個混合在六折中只有 2022（B2、B3）與 2026（B1）方向為正，其餘為負。B2、B3 在全六折合併的未校正 CI 已全部在 0 以下（描述性，未校正）。依凍結規則，**沒有任何混合成為前瞻候選**（priority_candidates 為空）。

## 多樣性診斷
- 神經分數與 C0+ 的每日截面秩相關平均約 0.76–0.77（中位數約 0.83–0.85），三臂平均（R_all）0.79。神經模型學到的排序與 C0+ 大部分重疊，剩餘的差異沒有帶來組合層增益。

## 解讀
- 加入神經分數**沒有改善**單獨 C0+，方向一致偏負；混合比例越偏向神經（B2、B3）越差。
- 與 P3／P4a 一致：在目前輸入資訊與未調參配方下，神經模型沒有提供 C0+ 以外的有用資訊。
- 限制：已曝光資料、描述性；神經臂未調參且兩種子、C0+ 三種子並調整過（偏向 C0+）；raw 價格無股利調整；非 PIT 市值；下市覆蓋未驗證。這不代表在調參或新資訊下混合必然無益（P4d 將檢驗調參與容量／種子／資料量）。
