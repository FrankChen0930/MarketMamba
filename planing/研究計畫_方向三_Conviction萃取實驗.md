# 方向三：從 breadth 模型萃取 conviction

**性質**：研究實驗（**非**作品集展示門面，結果之後才摘要進展示）
**用途**：展示研究野心——用「假設→方法→結果」框架做一項實驗，不是又做一個模型
**已排除**：素樸版（直接訓練 DL 輸出少數集中持股）——樣本數太少、必過擬合，不做
**是否需要 Colab**：三個子方案預期都不需要，皆屬輕量分析或小模型，詳見各子方案標註

---

## 三個子方案總覽

| 子方案 | 核心邏輯 | 新鮮度 | 工程成本 | 優先級 |
|---|---|---|---|---|
| A. 事件驅動當核心 | conviction 定義在可重複事件（財報超預期/併購/分拆），breadth 來自歷史同類事件 | 較已知 | 中（需先盤點台股事件資料量是否足夠） | 中，待可行性評估 |
| B. Meta-labeling | 既有 breadth 模型當 primary（給方向），訓練 secondary ML 判斷「該不該下注、下多大」 | 中 | 中（需確認歷史交易次數是否夠訓練 secondary model） | 中 |
| C. 不確定性驅動集中【主推】 | 檢驗「訊號強且不確定性低」子集是否表現得像高信念組合 | 最新 | **低**（現有 MC-Dropout Uncertainty、Signal_Quality 可能已經是現成資料） | **高，建議優先做** |

> 落地順序建議：雖然文件把三者列為 A→B→C（按新鮮度遞增），但實際工程成本上 **C 反而最低**，且已標記為主推方案，建議優先驗證 C，A/B 視資料可行性評估結果再排入。這只是執行順序建議，不影響三個方向本身都要做的決定。

**Colab 需求細部說明**：A（事件研究＋簡單模型）與 B（secondary model 用 Logistic Regression/GBDT）規模都很小，本機可訓練；C 完全是既有資料的統計分析，不涉及任何新模型訓練。三者目前都不需要 Colab；只有在 C4b 走向「需要 deep ensembles」（即多個 Mamba checkpoint 重訓）時，才會真的碰到 Colab 需求（見 `planing/01_需要Colab清單.md`）。

---

## C 子方案：不確定性驅動集中（優先做）

**核心價值**：這實驗在實測「模型的信心是否校準到準確度」。既有 `docs/uncertainty-calibration-2026-06-13.md` 已對 P0 修復前的舊管線做過一輪類似分析（結論：U 與誤差相關 +0.30、Signal_Quality Top50 五日超額 +1.66%/日），這裡要做的是延伸到「conviction 子集」層級的專用分析，且需等 P0 後新管線累積 20+ 天再重跑確認結論維持。

| # | 工作項目 | 相依於 | 優先級 | 狀態 |
|---|---|---|---|---|
| C1 | 確認現有 `df_kelly.csv` 歷史歸檔（`V6/results/{date}/`）與 `history_index.json` 是否已包含足夠長度的 Uncertainty 與 Signal_Quality 歷史序列 | 無 | 高 | ✅ 完成（2026-07-11，50 天歸檔、資料足夠；報告見 `docs/conviction-c-analysis-2026-07-11.md`） |
| C2 | 若 C1 資料足夠：切出「高 Signal_Quality（強訊號＋低不確定性）」子集 vs 全體，比較命中率／IC／Sharpe | C1 | 高 | ✅ 首輪完成（腳本 `V6/experimental/conviction_c_analysis.py` 可重跑） |
| C3 | 建立信心－準確度校準曲線（calibration curve）：模型講的「有把握」是否對應更高準確度 | C2 | 高 | ✅ 首輪完成（**20d 校準大致成立、5d post-P0 反向但僅 14 天**） |
| C4a | 若校準良好 → 寫入 conviction 線「模型預測結果」頁的 DL 輔助訊號設計依據 | C3 | 中 | 首輪結論：20d SQ 連續值可當輔助訊號（低 U 硬門檻交集反而較差）；待 7 月底複驗後定稿 |
| C4b | 若不校準 → 誠實記錄為研究發現（本身即有價值），評估是否需要換成 deep ensembles 或 conformal prediction 提高校準品質 | C3 | 中 | 5d 反向已誠實記錄；**deep ensembles/conformal 暫不做**，等 post-P0 樣本 ≥20 天（約 7 月底）重跑再定 |
| C5 | （視 C4b 結果）導入 deep ensembles / conformal prediction 取代或補強 MC-Dropout——**deep ensembles 需要重訓多個 checkpoint，屬於需要 Colab 的項目**；conformal prediction 是推論後處理，不需要 Colab | C4b | 低 | 視結果決定 |

**失敗模式提醒**：高把握 ≠ 高準確，模型常過度自信——這是預期中可能出現的結果之一，不是實驗失敗，記錄下來就是研究成果的一部分。

---

## A 子方案：事件驅動當核心

| # | 工作項目 | 相依於 | 優先級 | 狀態 |
|---|---|---|---|---|
| A1 | 資料可行性盤點：財報超預期 vs 併購 vs 分拆，分別評估台股歷史樣本數是否足夠（併購/分拆案例在台股可能過少，需先確認再投入） | 無 | 中 | 待開始 |
| A2 | 若財報超預期資料最乾淨 → 建立 event study 框架，抓歷史財報公布日＋市場反應 | A1 | 中 | 待開始（視 A1 結果） |
| A3 | 訓練/驗證事件驅動 ML 模型，計算 base rate 與賠率（規模小，本機可行） | A2 | 中 | 待開始（視 A2 結果） |

---

## B 子方案：Meta-labeling（López de Prado）

| # | 工作項目 | 相依於 | 優先級 | 狀態 |
|---|---|---|---|---|
| B1 | 用現有 breadth 模型（V6）方向性預測當 primary signal | 無，可與 A 平行 | 中 | 待開始 |
| B2 | 設計 secondary model 特徵（可用現有 Uncertainty、Signal_Quality、法人籌碼等） | B1 | 中 | 待開始 |
| B3 | 確認樣本數是否足夠：secondary label 是「歷史交易次數」的對錯標記，需先確認歷史交易次數（而非持股數）是否夠訓練，避免重蹈「素樸版」樣本不足的過擬合問題 | B2 | 中 | 待開始 |
| B4 | 訓練 secondary model（建議先用簡單模型如 Logistic Regression/GBDT，本機可行） | B3 | 中 | 待開始（視 B3 結果） |

---

## 評估標準（三子方案共用）

無論哪個子方案，最終都要能回答：「高把握子集」vs「全體」的命中率／IC／Sharpe 差異，以及信心是否校準到準確度。正反結果都要誠實記錄——這是研究，不是選股工具的行銷素材。

---

> 本檔案原建立於 Cowork 討論階段，2026-07-11 併入 repo 的 `planing/` 資料夾統一管理。
