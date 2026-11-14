# 方向二 Step 3：階 2 GBDT Baseline（LightGBM）結果報告

> 日期：2026-07-13
> 協定：`docs/baseline-experiment-protocol-draft-2026-07-11.md` v1.0（凍結，特徵=附錄 A 同一份 300 維）
> 程式：`V6/experimental/baseline_gbdt.py`（共用 `baseline_common` 資料層+評估層）
> 完整數字：`V6/experimental/result/baseline_gbdt_result.json`；5d 模型存 `result/gbdt_5d.txt`
> 引用紀律：依 `docs/baseline-ic-diagnosis-results-2026-07-13.md` §5——**headline IC 必須配分層數字一起引用**

---

## 核心結果（test = 2024-01-01 ~ 2026-06-02，580 個交易日，與階 1 / Phase 3 harness 同窗）

### 5d（主 horizon）

| 模型 | 全市場 IC | 高流動組 IC | ICIR | IC>0 | t(NW) |
|---|---|---|---|---|---|
| **GBDT（300 維）** | **+0.1098** | **+0.0802** | 0.98 | 86% | 13.9 |
| Ridge（300 維，階 1） | +0.1015 | +0.0705 | 1.22 | 90% | 16.1 |
| Mamba v6_short（同 harness 重跑） | +0.0870 | — | — | — | — |

分層 IC（GBDT 5d）：流動性 小/中/大 = **+0.1431 / +0.1087 / +0.0802**——與階 1 相同的梯度結構，headline 被小型股墊高（排查已證非資料錯誤，但含未量化的存活者偏差）。

### 20d（副 horizon）

| 模型 | 全市場 IC | 高流動組 IC | ICIR | IC>0 | t(NW) |
|---|---|---|---|---|---|
| GBDT | +0.1004 | +0.0685 | 0.77 | 76% | 5.1 |
| Ridge（階 1） | +0.1081 | — | — | — | — |
| Mamba v6_trend（歷史峰值 0.0961，**非同 harness 重跑、epoch 峰值選擇偏鬆**，僅供脈絡） | (+0.0961) | — | — | — | — |

20d 上 GBDT **沒有**贏過 Ridge（+0.1004 vs +0.1081）——非線性在較長 horizon 沒有換到東西。

### 網格與選參（誠實記錄，共 4 組控制多重測試負擔）

5d：4 組 val IC 範圍 +0.1115~+0.1140（極平坦）→ 選 leaves=127/min_leaf=2000/107 輪。
20d：+0.1390~+0.1463 → 選 leaves=127/min_leaf=2000/84 輪。val IC 對超參數不敏感，「訊號來自特徵、不是調參」的旁證。

## 訊號是什麼（SHAP top10，LightGBM 原生 pred_contrib、30k test 列抽樣）

- **5d**：`Return_1d`（短反轉）、`Return_1d_rstd20/60`（低波動）、`EPS`、`ATR_14`、`RSI_14_rmean60`、`Return_1d_rmean5`、`Foreign_Buy`、`Return_1d_lag1`、`Investment_Trust_Net`——與階 1 相同的「反轉+低波+籌碼+基本面」分散結構，無單一主宰特徵。
- **20d**：`ATR_14`、`EPS`、`RSI_14_rmean60`、`Return_1d_rstd20/60`、`Volume_rstd20`、`Foreign_Net_rmean20`、`Revenue_MoM_lag20`、`PBR`——波動與基本面權重上升，反轉權重下降（合理：horizon 拉長反轉衰減）。

## 組合層（協定 §7：Top50 等權、5 日再平衡、買 0.15%/賣 0.45%）

| | 年化 | Sharpe | MDD | 換手/次 |
|---|---|---|---|---|
| GBDT 5d（成本 ×1） | +10.8% | 0.62 | −31.8% | **84%** |
| GBDT 5d（成本 ×2） | **−13.6%** | — | — | — |
| Ridge 5d（階 1 對照） | +18.7% | 1.00 | −24.2% | 77% |
| 等權宇宙基準（不排序） | 總報酬 +44.1%（對 TWII −87.2%） | — | — | ~0 |

**值得注意的反差：GBDT 訊號層 IC 更高（+0.008），組合層卻明顯更差**（年化 −7.9pp、換手 +7pp、MDD 更深）。合理解讀：GBDT 的 IC 增量更集中在小型股與排名中段（對 Spearman IC 有貢獻、對 Top50 頭部無益），且頭部名單日間翻動更大 → 換手成本吃掉更多。對 TWII −99% 依排查結論不單獨引用（基準錯配：等權宇宙本身 −87%）。

## Walk-Forward（協定方案 B，expanding、6 個月窗、3 個月步進；2026-07-14 完成回填）

**46 folds（2015–2026）全部 IC 為正**，mean +0.1285、std 0.0241、範圍 [+0.0686, +0.1665]（對照 Ridge WF：46/46 正、mean +0.1141、範圍 [+0.054, +0.138]）。GBDT 在每個 fold 上系統性高於 Ridge 約 +0.014——非線性增量在時間軸上穩定，不是特定 regime 的產物。最弱段與 Ridge 相同落在 2022-10~2023-07（升息熊市後段，IC ~0.069）。注意：params/輪數沿用主切分 val 選定值（同階 1 慣例，早期 fold 對此有輕微樂觀性）；fold IC 以 stride-2 抽樣日評估。

## 誠實限制

1. 承階 1 全部限制：對 Mamba 偏鬆（epoch 峰值）、存活者偏差 D3（絕對數字偏高、四階相對公平）、train/test 邊界無 purge、test 窗與 Phase 3 harness 重用（B9 慢性）。
2. 網格僅 4 組 + lr/正則固定——GBDT 沒被調到極限；但 val IC 平坦顯示邊際有限。
3. label=rank + L2 regression（同一 label 鐵律）；沒試 lambdarank/Huber 等變體（會破壞同場對照）。
4. 組合層結論對成本假設極敏感（×2 即轉負），與階 1 相同屬短週期高換手策略固有屬性。

## 對計畫的意涵（初步，待階 3 補完）

- **「效能 vs 可解釋性」光譜到目前為止**（5d 全市場/高流動）：
  Ridge +0.1015/+0.0705（係數可讀）→ GBDT +0.1098/+0.0802（SHAP）→ Mamba +0.0870/—（低可解釋）。
- 階 2 確認階 1 的方向：**這個 harness 上，扁平模型 + 工程化時序特徵 ≥ 序列深度模型**；GBDT 對 Ridge 的增量小（5d +0.008、20d 反而 −0.008），非線性紅利有限、訊號本質是分散弱訊號的線性疊加。
- 但組合層提醒：**IC 排名 ≠ 落袋排名**。若以「照 Top50 買會不會賺」為準，Ridge 目前仍是最強 baseline。階 3（LSTM/GRU）跑完後光譜才完整。
