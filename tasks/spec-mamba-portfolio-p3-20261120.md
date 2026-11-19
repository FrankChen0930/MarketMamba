# 預先登記 P3：Mamba2 與 S-Mamba 風格模型在組合層協定下對 C0+

登記日期：2026-11-20，於任何 P3 神經網路分數與結果出現之前；本節登記後只追加、不修改。證據等級上限 E2，RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED。承接 [P1](../reports/v7_p1_portfolio_evaluation.md)、[P2](../reports/v7_p2_portfolio_evaluation.md) 與 pilot09 roadmap 12.20／12.24／12.27（序列模型與 C0+、LightGBM 在 IC 上無顯著差異；跨股混合無增益）。目的：回答使用者的核心假設——Mamba2＋S-Mamba 是否在**扣成本的組合層**勝過 C0+——而不只看 IC。

## 模型（沿用先前已定的設定，禁止本研究內調參或換設定）
- `mamba2`：官方 `mamba_ssm` Mamba2，d_model 64、2 層、headdim 8、d_state 16、回看 60 日，rank-gauss 輸入、純 MSE（中心化排名，5d:10d＝1:0.5）、lr 3e-4、每批 8 個交易日×256 檔、8 epoch×400 步、no-conv。
- `crossstock`：S-Mamba 風格跨股雙向 Mamba（arXiv 2403.11144 系，非 Microsoft Samba），n_cross=1、每日 256 檔 token，其餘同上。
- 種子 17、29（每折每模型 2 個種子；種子間以當日截面 percentile rank 平均）。標籤 y5。總經欄位歸零（沿用先前處理）。
- 評估列與 P1 分數的列集合完全相同（`SeqData.eligible` 與 split 2），以便與 C0+ 配對；C0+ 分數沿用 `.artifacts/p1/full-scores-ace2eef/`（3 種子平均，不重新擬合）。
- 折：2021–2026 滾動折；2021–2023 為確認性，2024–2026 為描述性。先完成 2021–2023 再做 2024–2026。

## 確認性假設（乾淨折 2021–2023）
組合層使用 P2 固定的主要格（top 50、20 日再平衡、緩衝到第 100 名、等權、成交值門檻 2e7、折扣 0.6、最低手續費 20、滑價 10bp、資金 1e6、raw 價格、基準只收比例成本），模型分數換成各神經網路。
- H-P3a：Mamba2 對 C0+ 的逐日淨報酬差，區塊 bootstrap（區塊 20、1000 次）95% CI 下界 > 0 → 「優於 C0+」；上界 < 0 → 「劣於 C0+」；其餘「未證實」。
- H-P3b：crossstock 對 C0+，同法。
- Holm 校正（family=2）。兩者各自對 P2 基準的淨超額為描述性。
- IC：同評估列的逐日 Spearman IC 對 C0+（配對 HAC＋區塊 bootstrap）為描述性，不得與組合層結論混同。

## 後續階段門檻（本規格內先固定，避免看結果後再決定）
僅當某一架構在乾淨折上「未被判定為劣於 C0+」且描述性六折平均淨超額不低於 C0+ 時，才另行預先登記 P3b：回看長度 {20, 60, 120}、預測天期頭 {5, 10} 的消融。否則停止該架構的放大，並明載「在此資料、此規模、此協定下無優勢證據」。

## 資源與安全
本機 RTX 3060 Laptop（6 GB），不使用 Colab。每個 fold×model×seed 為一次擬合，完成即寫入不覆寫的獨立檔案（可續跑）。GPU 溫度守衛：偵測到 ≥ 80°C 暫停並等待降至 ≤ 72°C 再續；≥ 85°C 停止並回報，不自行恢復；不得宣稱 NVIDIA 長時間訓練認證。預計 24 次擬合（6 折×2 模型×2 種子），單次約 10–25 分鐘；GPU 總時間上限 14 小時，超過則停止並回報已完成範圍。若 2021–2023 完成後已足以做確認性判定，2024–2026 可視時間補跑。不抓資料、不連網、不改 `Data/`、不讀 `.env`；不寫入外部矩陣目錄（rank-gauss 快取與工作檔置於 `.artifacts/p3/work/`，以唯讀連結引用原矩陣）。

## 必須誠實標示
延續 P1／P2 全部限制（模擬標籤、無 adj_close、下市覆蓋與 PIT 市值未驗證、觀測日曆）；神經網路僅 2 個種子且訓練預算小（8 epoch），先前亦無調參，不能據此排除架構在更大預算下的潛力；單一規模（d_model 64）；C0+ 使用 3 種子；樹模型與神經網路以相同評估列比較。主要格設計來自 P1／P2 的觀察，對新增的神經網路比較則為預先登記的檢定。

## 工程要求
- 新程式放 `V6/research_lab/`（例如 `rolling_seq_scores.py`，重用 `seq_runner.train_one`／`predict` 與 `cross_stock.py`），輸出與 `rolling_scores.py` 相同的 parquet schema 與 manifest 形式（含來源 commit、環境、輸入 SHA-256、種子、折、參數、GPU 型號與溫度紀錄），並讓 `portfolio_eval.py` 以最小擴充支援 P3 模式（三系統：c0plus 取自 P1 分數目錄，其餘取自 P3 分數目錄；P2 主要格固定；配對檢定與 Holm）。不得改變 P1／P2 預設行為與既有結果。
- 回歸與 mutation 測試：評估列集合與 P1 不一致應報錯、種子間 rank 平均 PIT 安全、訓練列不得含評估期或標籤未成熟列（purge 與 rolling_runner 一致）、配對檢定只用共同交易日；每個測試在壞行為下必須失敗。
- 分數生成期間不得修改 `V6/research_lab/*.py`（會使 manifest 被拒）。
