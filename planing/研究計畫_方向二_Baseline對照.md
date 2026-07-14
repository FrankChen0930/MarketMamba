# 方向二：業界標準 ML Baseline 對照

**性質**：研究＋展示　**用途**：讓 IC 有對照基準，量化「用可解釋性換到多少效益」
**掛載位置**：結果嵌入方向一頁面，同時作為 conviction 線篩選層方法論的佐證素材
**是否需要 Colab**：**Ridge/Lasso、GBDT 完全不需要**（CPU 本機即可）；**LSTM/GRU 建議先試本機 RTX 3060**，模型規模遠小於 Mamba+GATv2，除非訓練時間過長才考慮上 Colab

---

## 設計原則：公平性是核心，優先於一切

對照階梯（效能對照＋可解釋性光譜）：**線性(Ridge/Lasso) → GBDT(LightGBM/XGBoost) → 簡單 DL(LSTM/GRU) → Mamba+GATv2**

GBDT 是**必備** baseline（橫斷面選股業界主力），LSTM/GRU 的作用是隔離「贏是因為 Mamba 本身，還是任何序列模型皆可」。

**鐵律（務必遵守，這是公平性核心，不可妥協）**：
- 同一 universe、同一 walk-forward 切分、同一 label、同一成本模型
- GBDT 吃扁平特徵向量、Mamba 吃序列 → **必須**為 GBDT 工程化 lag/rolling 特徵補上時序資訊
- **不可**只給 GBDT 當下時點特徵卻給 Mamba 完整序列——那是讓 Mamba 贏的作弊對照，做出來的對照表沒有可信度，這種對照表比不做還糟

---

## To-do 拆解

| # | 工作項目 | 相依於 | 優先級 | 狀態 |
|---|---|---|---|---|
| 1 | 撰寫「實驗協定規格書」：明定 universe、walk-forward 切分方式（沿用現有折數設定）、label 定義（Exp_Alpha_5d/20d/60d 擇一或全部）、交易成本模型（沿用現有 0.15%/0.45% round-trip） | 無 | **高** | ✅ 定案凍結（2026-07-12，`docs/baseline-experiment-protocol-draft-2026-07-11.md`：單一切分為主+便宜階WF為輔 / rank label / 5d 主 20d 副 / Top50 等權 5 日再平衡） |
| 2 | Ridge/Lasso baseline：用現有 56/59 維特徵訓練線性模型，作為可解釋性地板 | 1 | 高 | ✅ 完成（2026-07-13，`docs/baseline-step2-ridge-lasso-2026-07-13.md`）：5d 全市場 IC +0.1015／高流動組 +0.0705，WF 46/46 fold 全正；引用需分層（排查定案見 `docs/baseline-ic-diagnosis-results-2026-07-13.md`） |
| 3 | GBDT baseline（LightGBM/XGBoost）：先做特徵工程——把序列資訊轉成 lag/rolling 特徵（例如過去 5/10/20 日動能、rolling mean/std），確保不因輸入資訊量不對等而處於劣勢 | 1 | **高（必備）** | ✅ 完成（2026-07-13，`docs/baseline-step3-gbdt-2026-07-13.md`）：5d 全市場 IC +0.1098／高流動組 +0.0802，小勝 Ridge（+0.1015/+0.0705）；20d 反輸 Ridge（+0.1004 vs +0.1081）。46/46 fold WF 全正、每 fold 系統性高 Ridge 約 +0.014，增量穩定非僥倖。**但組合層 Top50 年化只有 +10.8%（Sharpe 0.62）明顯輸 Ridge 的 +18.7%（Sharpe 1.00）**——IC 增量集中在小型股/排名中段、頭部換手更高（84% vs 77%），成本吃更多 |
| 4 | 簡單 DL baseline（LSTM/GRU）：輸入格式與 Mamba 對齊（同樣吃完整序列），確保是「模型結構」的對照而非「輸入資訊量」的對照；**先在本機 RTX 3060 嘗試，模型規模小，大機率不需要 Colab** | 1 | 中 | ✅ 完成（2026-07-15，`docs/baseline-step4-rnn-2026-07-15.md`；GRU 擇一 + window 60，本機 3060 實跑 12.2 小時、未動用 Colab）：**5d +0.1113／高流動 +0.0867 四階最高**、20d +0.1081 平 Ridge；組合層 Top50 年化 +22.8%（Sharpe 1.13）四階最強、成本×2 仍轉負；~49K 參數 GRU 同特徵/同 window 勝 v6_short（+0.0870）→「任何序列模型皆可」成立、Mamba 架構紅利不成立 |
| 5 | 產出「效能 vs 可解釋性」對照表：線性→GBDT→LSTM/GRU→Mamba+GATv2，含 IC/ICIR/Sharpe/MDD 及可解釋性描述（係數／SHAP／attention 或無） | 2, 3, 4 | 高 | ✅ 完成（2026-07-15，`docs/baseline-comparison-table-2026-07-15.md`：訊號層+組合層+可解釋性並排、含引用紀律與判讀）——Ridge/Lasso 的 IC 異常值懷疑已排查完畢（`docs/baseline-ic-diagnosis-results-2026-07-13.md`）：**非 bug/髒資料，但對照表必須用分層數字引用**（全市場 +0.1015 / 高流動大市值組 +0.0705 / 純籌碼基本面 +0.0717），存活者偏差對絕對水位的墊高尚未量化，須註明；組合層基準改用等權宇宙、TWII 只做脈絡陳述 |
| 6 | SHAP 分析套用在 GBDT baseline，作為可解釋性光譜中段的具體展示素材 | 3 | 中 | ✅ 已隨 Step 3 完成（SHAP top10 記錄於 GBDT 報告「訊號是什麼」節，用 LightGBM 原生 pred_contrib） |
| 7 | 結果同步交付給方向一 Step8（嵌入說明頁）與 conviction 線篩選層文件 | 5 | 低 | ✅ 完成（2026-07-15，已嵌入底稿 §5.5 + 前端 `/pipeline` 頁；conviction 線篩選層文件屆時直接引用對照表定稿） |

---

## 備註

- Step 1 是整個方向二（甚至間接影響方向一與方向三的可信度）的地基，建議最優先處理，且一旦定案盡量不要中途更動 label 或切分方式，否則後面所有 baseline 要重跑。
- Step 3 的 lag/rolling 特徵工程如果做得好，其實也能反過來檢視現有 Mamba 特徵工程有沒有遺漏的簡單特徵，算是附帶的效益。
- Step 4 如果時間有限可以先做 LSTM 或 GRU 擇一，不用兩個都做，重點是「有一個非 Mamba 的序列模型當對照」，不是窮舉序列模型種類。
- **Step 2 完成後追加的 IC 異常值排查（2026-07-13）**：診斷結果見 `docs/baseline-ic-diagnosis-results-2026-07-13.md`，建議把「分層 IC（依流動性/市值）、等權宇宙基準、成本敏感度表」直接納入 `baseline_common.py` 的標準評估慣例，Step 3（GBDT）與 Step 4（LSTM/GRU）跑完後應沿用同一套評估輸出，不要每階各自寫一套，也不要只引用單一 headline IC 數字。
- 排查過程額外發現三個資料衛生問題（Close=0 損壞列、超限報酬、還原/未還原價接縫），與方向二本身無關但已轉列進 `planing/02_不需要Colab清單.md`。
- **GBDT 結果帶出一個 Step 5 對照表要保留的重要教訓：「IC 排名 ≠ 落袋排名」。** GBDT 訊號層 IC 比 Ridge 高，但實際 Top50 等權組合報酬明顯更差（+10.8% vs +18.7% 年化）。做最終對照表時不能只放 IC 一欄就下結論「GBDT 較強」，一定要把組合層報酬/Sharpe/換手率並排放，否則會誤導讀者。
- **Step 4（GRU）後更新（2026-07-15）**：以「照 Top50 買會不會賺」為準的最強 baseline 由 Ridge 換成 **GRU**（年化 +22.8% > Ridge +18.7% > GBDT +10.8%），且 GRU 沒有 GBDT 的 IC/落袋反差（IC 與組合層報酬同向）。「IC 與組合層並排」的教訓不變。另：階 3 回答了 LSTM/GRU 的設計問題——「贏是因為 Mamba 本身，還是任何序列模型皆可」→ 後者成立（~49K 參數 GRU 勝 1.66M 參數 v6_short +0.024），Step 5 對照表的敘事軸心應是「訊號來自特徵/label/廣度，模型形式只在 ±0.01 內移動」。

---

> 本檔案原建立於 Cowork 討論階段，2026-07-11 併入 repo 的 `planing/` 資料夾統一管理。
