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

## 修訂 1（2026-11-20，於任何 P3 神經網路分數與結果出現之前；追加，不改動上文）
兩位獨立審查（皆 NO_GO，CPU 工程驗證通過）指出本規格有矛盾與缺口，修訂如下：
1. **模型契約（取代上文「no-conv」）**：上文「no-conv」與「沿用先前已定設定」互相矛盾——先前實驗（roadmap 11.7／12.20）實際使用官方 `mamba_ssm.Mamba2`，單股 mamba2 為官方預設卷積，跨股混合層 `d_conv=2`（見 `cross_stock.py`），mixer_kw 為 d_state=16、headdim=8、expand=2。P3 一律使用**先前實際用的配方**以保持可比，不含「no-conv」消融。
2. **有效設定凍結**：完整有效設定（含 `DEFAULT_CFG` 繼承的 weight_decay=.01、patience=3、val_max_rows=80,000、val_dates=126、attn_window 等，以及 epoch=8、dates_per_batch=8、stocks_per_date=256、steps 400）必須寫成明確的凍結字典、寫入 manifest，並由測試與執行前閘門確認與其逐項相等；`DEFAULT_CFG` 日後變動不得悄悄改變 P3。
3. **判定規則（操作化）**：「優於 C0+」＝95% CI 下界 > 0 且 Holm p < .05；「劣於 C0+」＝上界 < 0 且 Holm p < .05；其餘「未證實」。
4. **GPU 環境**：WSL 內原無 GPU 版 torch／mamba-ssm 環境。新建 `~/.venvs/marketmamba-gpu`（Python 3.12.3、torch 2.10.0+cu128、causal-conv1d 1.7.0、mamba-ssm 2.3.2.post1 官方預編譯 wheel，其餘套件與 pilot09 環境相同版本），鎖檔 `~/.venvs/marketmamba-gpu.requirements.lock.txt`；正式執行前須有官方 Mamba2 GPU 前向／反向冒煙證據。
5. **預算與安全**：14 小時預算須以持久帳本記錄每次嘗試（含失敗或被中止的 fit）的實際耗時，跨程序累計；啟動時若存在 STOP 檔必須拒絕續跑，需人工移除；溫度紀錄保存。
6. **purge 檢查**：`checked_splits` 的訓練列 purge 與標籤成熟規則須以真實折的 metadata 對照 `rolling_runner` 的實際切分（測試使用真實折目錄的 `splits.npy` 或其雜湊），而非只用合成切分。
7. 確認性路徑的解鎖程序（contract_status 由 UNRESOLVED 變為 RESOLVED 的條件）須寫入工程報告，並有端到端測試涵蓋 `p3_main`（含神經分數目錄載入與 `confirmatory_ready` 閘門）。

## 修訂 2（2026-11-20，於任何 P3 神經網路分數與結果出現之前；追加，不改動上文）
第二輪審查（Claude NO_GO 含一項 CRITICAL；Codex GO）後追加：
1. **准入證據不可偽造（取代「human_reviewed: true」布林）**：官方 GPU 冒煙證據 JSON 必須由執行器自己在指定環境產生，內含實際量測值（前向／反向的有限性檢查結果、參數量、梯度張量數、峰值顯存、GPU 型號與溫度、耗時）與程式／環境逐檔雜湊，不得手寫。「Human 審核」改為記錄結構化審核者資訊：`reviewed_by = {kind, job_id, verdict, evidence_sha256}`，其中 `kind` 為 `independent_worker`（另一個獨立 worker 的審查作業，記錄 Ruflo job id 與判定）或 `human`；並載明 `standing_authorization`：Human 在 2026-11-19／20 的對話中授權主控自行接續本機（RTX 3060）實驗、不得上 Colab。此授權可由 Human 日後稽核；主控不得在證據內聲稱 Human 已逐項審核。
2. **`p3_main` 必須重新驗證准入證據**：不得只讀 manifest 的 `admission.status`。必須重新核對證據檔路徑與 SHA-256、證據內容與目前程式／環境逐項相符、預算帳本雜湊與 `spent_seconds <= 14h`、24 個 fit 的 run_hash 與 sidecar 完整；任一不符即不得標為 CONFIRMATORY。
3. **預算帳本**：單一全域帳本 `.artifacts/p3/budget-global.jsonl`（含雜湊鏈與單一寫入鎖），不依賴 `--out-dir`；同 namespace 已有 fit 或 run_identity 但帳本缺失或鏈不符即拒絕；帳本 SHA 寫入 manifest。STOP 須先寫入並 fsync，再寫 END；清除 STOP 須以明確 CLI 子命令在帳本留下 CLEARED 事件（含原因與時間），僅刪檔不算清除。`collect()` 公開入口同樣受帳本與 STOP 約束，或在文件明示僅供內部使用。
4. **凍結字典補齊**：`dropout`、`window`（60）、`normalize`（rankgauss）、`train_probe_sessions`，以及寫死常數（grad clip 1.0、warmup 與 cosine 排程、AdamW betas、cross 層 d_ff 與 FFN dropout）須列入凍結字典或由 source hash 於 `p3_main` 重驗；Mamba2 官方其餘預設（ngroups、chunk_size、dt_min／dt_max、A_init_range、rmsnorm 等）須讀出並寫入 `mixer_parameters`。

## 修訂 3（2026-11-20，於任何 P3 神經網路分數與結果出現之前；追加，不改動上文）
第三輪審查（Codex NO_GO：CRITICAL）指出，預算帳本的雜湊鏈只能偵測事件被修改，**無法偵測合法前綴的截斷或清空**；截斷會抹除已耗時間與 STOP，已耗盡 14 小時後換 out-dir 仍可重新啟動。修訂如下：
1. **帶外錨點**：每次帳本追加事件後，原子性寫入獨立錨點檔 `.artifacts/p3/budget-anchor.json`（欄位：事件序號、帳本頭 hash、累計 spent_seconds、時間戳；寫入使用暫存檔加 fsync 加原子替換）；每個 fit sidecar 與 manifest 亦記錄當時的（序號、帳本頭 hash、spent）。啟動時若帳本事件數少於錨點序號、帳本頭 hash 與錨點不符、或 spent 小於錨點，一律拒絕（含帳本被清空或截斷到較短的合法前綴）。
2. **錨點缺失的處理**：同 namespace 已有 fit、sidecar 或 run_identity，但全域帳本或錨點任一缺失，即拒絕，需人工以明確 CLI 子命令重建並在新帳本留下 REBUILT 事件（含原因與已知 sidecar 的 spent 下界），重建後的可用預算須扣除所有 sidecar 記錄的 spent 最大值。
3. **範圍聲明**：此保護針對疏失與誤操作，不防禦同時刪除帳本、錨點與全部 sidecar 的刻意破壞；報告須如實寫明。
4. 其餘修訂 1、2 不變。測試須涵蓋：清空帳本、截斷為合法前綴、換 out-dir、錨點缺失、錨點與帳本不符、REBUILT 路徑；每項在壞行為下必須失敗。
