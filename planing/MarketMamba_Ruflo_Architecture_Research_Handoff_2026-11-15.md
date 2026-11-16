# Market Mamba — 給 Ruflo 的原始模型架構研究交接與下一階段任務

> **用途：** 可以直接將本文件全文交給 Ruflo／Claude Coordinator，作為 Market Mamba 長期研發的架構與研究方向補充。
>
> **編寫日期：** 2026-11-15（Asia/Taipei）
>
> **性質：** 依 2026-08-28～08-30 的原始論文研討、後來 V7 研究交接及 2026-11-15 的 Hybrid/B4/B7 進度整理。**歷史模型偏好 ≠ 已證明最優 ≠ 當前程式已實作。**
>
> **閱讀優先序：** 最新使用者指示／真實程式與數據 > 最新 `tasks/roadmap-20261115.md` 與 Git 狀態 > 本交接的歷史設計意圖 > 舊交接與早期研究筆記。不要把本文件誤當成允許修改正式環境的授權。

---

## 0. 你的任務與產品目標

你是 **Market Mamba 長期研究及工程協調者**。使用已驗證的 **Ruflo 3.77.1 + `@claude-flow/codex` 3.0.8 + Claude Code + 官方 Codex CLI** Hybrid 路線：Claude 只負責薄協調、重大研究與跨模型獨立審查；Codex Pro 是大部分資料工程、模型開發、實驗、測試、回測與前後端的主力。

產品最終目標是面向**台灣上市／上櫃全市場**的 AI 投資研究與決策支援系統：時點正確的數據、可信的樣本外選股排序、可重現的模擬投資組合、風險／績效、API／Web UI 和可長期迭代的研發體系。**目前不做實際下單、不保證投資收益。**

這次交接要補足一個關鍵背景：**當初最看好的 Market Mamba 模型不是單純 Mamba-2，也不是直接照搬 Microsoft Samba；而是一個「長時序 Temporal Mamba-2 + 顯式圖 GATv2 + 隱式跨股票 Bi-Mamba」的候選雙路徑架構。** 請理解其由來並作為重要研究候選，**但不得未經實驗就宣布它是最佳模型或直接取代既有 B7。**

---

## 1. 2026 年 8 月真正提出的「最看好」模型架構

### 1.1 歷史脈絡（請按時間順序理解）

- **2026-08-28：** 研究《Is Mamba Effective for Time Series Forecasting?》（S-Mamba）、《MambaTS》與《TimeMachine》。我確認 S-Mamba 的 Bi-Mamba 主要是**跨變量／通道資訊混合**，而不是取代自己最看重的長時序建模。對 Market Mamba 而言，它更可能**取代或補強 GATv2**，不是取代 Temporal Mamba。
- **2026-08-28～30：** 三篇研究整理出的首選候選為 **Channel-Independent Temporal Mamba-2 → [GATv2 ∥ S-Mamba/Bi-Mamba] → Feature Fusion／Gating → Ranking Heads**。理由是台股可用的顯式公司關係可能不完整，GATv2 提供可驗證的顯式先驗，Bi-Mamba 探索圖以外的隱性全市場關聯；兩者互補比完全依賴密集 Cross-Stock Attention 更值得試。
- **重要保留：** 我當時說的是「看完三篇論文後理論上最合理的候選」，**沒有宣稱一定最好**。尤其**先時間後股票（Temporal→Cross-Stock）還是先股票後時間、雙路是否有增益、股票掃描順序如何影響結果**，都須用實驗決定。
- **TimeMachine 的取捨：** 我不優先將分鐘線／日線做成雙層金字塔。當前產品首先是**日線全市場排名**；盤中預測是另一個資料、標籤、延遲與交易需求都不同的研究任務。TimeMachine 多尺度觀念保留作未來候選，而非現階段必加模組。
- **MambaTS 的取捨：** VAST／ATSP（變量重排、先做跨變量交錯掃描）是可研究的**替代設計**，而不是必須與後段 Bi-Mamba 同時堆疊。TMB 的卷積／Dropout 思路值得檢驗，**但早期筆記關於官方 Mamba-2 可直接 `d_conv=0` 的敘述不可照單全收；請以當前官方實作與 GPU 實測為準。**
- **2026-08-30 後續：** 讀 StockMamba 與 MCMamba，加入 **Market-Regime Mamba-2 → Dynamic Factor Gating**、多尺度與較穩定跨股票關係的**可選研究方向**。**U-shape／Rank-Position Loss 放後面**：應先確認模型、排名與投組如何使用中段股票，才決定是否強化頭尾樣本。

### 1.2 首選候選：Hierarchical Dual-Path Market Mamba

```text
  日線台股歷史資料 X: [B, N_stocks, L_days, F]
  （歷史時點可投資股票池 + 當時已知特徵）
                              │
                              ▼
     [A. Temporal Mamba-2 — 時間軸、每檔獨立／參數可共享]
     - 在各股票「過去 L 日」上學長期時序依賴
     - 起步沿用 L=60；L=120/250 為後續獨立消融
     - 不讓不同股票的特徵意外在 temporal convolution 中混合
                              │
                   每檔股票向量 H: [B, N, D]
                              │
               ┌──────────────┴──────────────┐
               ▼                             ▼
  [B1. GATv2: Explicit Graph]    [B2. Bi-Mamba: Implicit Cross-Stock]
  - PIT-safe 產業/關係邊            - 對股票 token 正反向掃描
  - 結構化、局部先驗                - 探索超出圖先驗的關聯
  - 不因圖缺漏而假裝已完整          - 要驗股票順序／股票池變動敏感性
               │                             │
               └──────────────┬──────────────┘
                              ▼
            [C. Fusion / Dynamic Gating / Residual]
            - 比較 GAT-only、BiMamba-only、Both
            - 融合不等於一定優於單分支
                              │
                              ▼
                  [D. Ranking / Forecast Heads]
                   例如 5d、10d 橫斷面排名
                              │
                              ▼
                 [研究評估 → 模擬投組 → API/UI]
```

**各部分的真正分工：**

1. **Temporal Mamba-2** 是時間特徵主幹：目標是利用 Mamba 的長時序能力，而不是把 Mamba 只當 Linear／LSTM 替代品。先採已實作的日線設定，不為了追求「超長序列」就跳過可重現性或訓練收斂。
2. **S-Mamba／Bi-Mamba** 借鑑的是 *inter-variate mixing*：將單股時間模型輸出的**每檔股票 Embedding 視為跨股票 token**，而不是將 raw 48／59 個 feature channel 誤認為 48／59 檔股票。S-Mamba 原論文使用 Mamba-1；如要換 Mamba-2，需自行設計並驗證。
3. **GATv2** 反映有來源依據的顯式關係，可能是產業／集團／持股／供應鏈，但只允許用**當時已知**的圖與節點；沒有可用歷史圖時，GAT branch 應明確標示缺失，不可自行使用今天的圖回填歷史。
4. **Fusion/Gating** 是待證明的互補假說。雙路並行**不是工程上必然勝出**；至少要有無 Cross-Stock、GAT-only、BiMamba-only、Both 的比較。
5. **Prediction / Ranking** 的目標是全市場橫斷面選股而非即時下單。先用定義一致的排名／回歸損失；不要預設 U-shape、成本正則化一定會提高 IC 或可交易績效。

### 1.3 可以研究但「不列為首輪強制」的擴充

- **Market Regime Scanner + Dynamic Factor Gating：** 借自 StockMamba。市場路徑可能影響因子有效性，可在確保宏觀資料實際可得時做獨立 ablation。不要混淆 regime scanner 與負責每檔股票時間表徵的主幹 Temporal Mamba-2。
- **Multi-scale temporal representations／較穩定的 global correlation prior：** 借自 MCMamba；先檢查同日線不同窗口尺度是否真有增益，不能因為論文用了多尺度就引入分鐘線需求。全球相關性矩陣需只使用訓練／當時可見資料。
- **VAST／ATSP ordering：** 借自 MambaTS；先研究排序敏感性，再決定是否值得加上昂貴或靜態的最適排序機制。前後都做同一種 cross-stock mixing 可能冗餘。
- **TMB／卷積改造：** 是可測試的模型細節；不要直接修改官方 CUDA fused kernel 或認定 `d_conv=0` 能安全跳過官方 Mamba-2 convolution。
- **U-shape RPL／IC Loss／成本感知 Loss：** 等模型本身、Baseline、Top-K／Long-only／Long-short 策略需求明確，再用獨立實驗衡量其效果。
- **盤中分鐘線＋日線多尺度：** 暫非現階段必要的產品規格；若未來明確要做 intraday，另行制定資料與推論 latency 契約。

### 1.4 要特別驗證的模型風險

- **Cross-Stock Bi-Mamba 並非 Permutation-Invariant。** 正反向掃描不代表對股票順序完全不敏感。固定股票代號排序、依當時產業分組、PIT-safe 關係排序、隨機順序對照可做 order ablation；避免用未來收益或未來產業分類決定當日順序。
- 股票池隨時點變動，需決定**缺股、停牌、轉板、下市、代號重用、batch 截面一致性**與排序機制，否則在訓練與推論可能使用不一樣的 cross-section context。
- 時序**雙向**與股票軸**雙向**不是同一件事：股票軸的 Bi-Mamba 可同時讀取同一決策時點其他股票已知特徵；不得偷讀決策時間之後的行情或財報。
- 全市場混合資訊可能造成噪音或過度平滑，GAT 的錯誤／缺漏圖也可能使排名變差；把高效率視為設計優勢，**不當成預測效能保證**。

---

## 2. 早期設計、V7 實作與今天 B7 的關係

| 時期／版本 | 真正的內容 | 不應誤認 |
|---|---|---|
| **2026-08-28～30 首選研究架構** | `Temporal Mamba-2 → (GATv2 ∥ Cross-Stock Bi-Mamba) → Fusion → Ranking`；Regime Gating 可選 | 不是已證明最優，也不是「純 Microsoft Samba」 |
| **後續 V7 研究實作** | 歷史記錄顯示曾採每股 `Temporal Mamba-2 + Cross-Stock Mamba-2` 正反掃描 + GATv2，約 60 日回看、5/10 日預測頭。部分舊實驗用 `N×60×59`、MSE+ListNet | 不等於今天每個資料版本都真的有啟用 GATv2；需檢查權重、輸入、圖與 Runtime |
| **E5／E1 既有真實矩陣** | 7,423,927 列 × 48 features，X/dates/stock_ids 等相同，但 y5/y10、split 不同；`HISTORICAL_SIMULATION_PROXY`，無產業中性。已確認有嚴重股票池倖存者偏差；E5/E1 IC 不可直接互比 | 和舊 V7 `59 features` 不能視為同一資料契約；不能據此聲稱全市場無偏績效 |
| **2026-11-15 B4 基線** | 相同 E5 split／y5：LightGBM IC 約 **0.1014**、Ridge 約 **0.0806**；E5 y10 LightGBM 約 **0.0929**；同版本研究結果有樣本內選模歷史與存活者偏差 | IC 顯著 ≠ 投組可以打敗基準、也不是新近未使用過的真正 holdout |
| **目前 B7 `seq_models.py`** | `SeqRanker('mamba2'|'samba')`，官方 `mamba_ssm.Mamba2`；Samba 候選為 Mamba→MLP→Sliding Window Attention→MLP，先驗形狀／因果性測試；Colab GPU forward/backward Probe 執行中（請以現況為準） | 主要是**時序候選基線**，**不是**當初同時有 GATv2 + Cross-Stock Bi-Mamba 的完整雙路模型 |

**最新工作位置（需現場核對，不要直接硬切 branch）：**

```text
原專案： /home/frank/projects/MarketMamba       （保護、唯讀盤點，不能擅改）
研發區： /home/frank/projects/ruflo-evaluation/pilot09/marketmamba
分支：   rd/hybrid
近期 commit：718f713（B4/B5）、43249a7（PIT/B7）
路線圖： tasks/roadmap-20261115.md（本交接不覆蓋最新版）
B7：     V6/research_lab/seq_models.py
B4：     V6/research_lab/matrix_baselines.py
```

截至 2026-11-15 的使用者回報：**96 passed、1 CUDA-only skipped**；已完成獨立 Git bundle；原始專案沒改；B7 Colab GPU Probe 已由使用者開始執行，尚待返回實際 JSON。**不得把 CPU 替身 mixer 的成功當作官方 Mamba-2 GPU 證據。**

**PIT 警告必須貫穿所有實驗：** 2005 年後的 402 個股票型下市事件中，323 個完全未進入當前樣本；76 個 TPEx→TWSE 轉板在轉板前的歷史缺漏；另有代號重用案例。原樣本的 IC、Top-K／投組績效為帶偏差研究證據，不能直接視為無偏 OOS 投資績效。`Alpha_5d` 命名原本會誤導，它實質是簡單報酬；目前 B4 已使用每日置中名次作預設訓練目標，評估 contract 必須清楚標示。

---

## 3. 希望你完成的具體研究與工程工作

### 第一步：讀真實程式與證據，做一份可核查的 Architecture Gap Map

先讀當前工作樹的 Git HEAD／dirty state、`tasks/roadmap-20261115.md`、`tasks/next-conversation-handoff.md`、`knowledge/`、Pilot09 evidence、`V6/research_lab/seq_models.py` 與相關 V7 `V6/experimental/*` 模型定義／有效 Checkpoint／資料 contract。**不要因為舊交接提及某模組，就假定今天仍存在或已接上圖分支。**

將模組逐一標示：`已實作且實測 / 僅有程式定義 / 歷史 V7 曾實作但未接 B7 / 尚未實作 / 受資料契約限制`。

要回答：
1. V7 舊的 Temporal Mamba2、Cross-Stock forward/reverse Mamba2、GATv2、Fusion 如何接線？在同一資料 contract 下哪些可直接重用？
2. B7 Mamba2／Microsoft Samba 的時間軸建模與上述雙路架構差在哪裡？是否應保留做**獨立 baseline**？
3. 在 E5/E1 扁平 `(rows,48)` 矩陣中建立 `[session, stock, lookback, feature]` 序列所需的 stock_ids、dates、masks、訓練/評估配對與 memory plan 有哪些限制？是否可以從現有 V7 matrix builder 重用而不重造資料？
4. GATv2 是否有真正可信的**歷史**圖來源？沒有就先跑 graph-disabled baseline 與明確缺漏報告；不可默默用今天的圖回填過去。
5. 已有 Colab GPU Probe、Wheel cache、訓練環境與 Checkpoint 哪些可以重用？避免反覆 40 分鐘編譯 `mamba-ssm`；優先保存相容 Wheel 與 Python/Torch/CUDA/ABI/Hash。

### 第二步：制定「能回答研究問題」的有界消融順序

**建議主線（不要求一次做完）：**

| 層級 | 候選 | 該實驗回答的問題 |
|---|---|---|
| C0 | 同一 V7 contract 的 LightGBM／Ridge、適當 GRU | 強而公平的 baseline |
| C1 | Temporal Mamba-2 only（5d/10d） | 沿時間軸使用 Mamba 是否比強基線好？ |
| C2 | Temporal Mamba-2 + Cross-Stock Bi-Mamba | 隱性全市場資訊交換是否有增益？ |
| C3 | Temporal Mamba-2 + PIT-safe GATv2 | 顯式關係圖是否有增益？ |
| C4 | Temporal Mamba-2 + GATv2 ∥ Bi-Mamba + Fusion | 使用者**當初最看好**的互補雙路架構是否優於單路？ |
| C5 | 目前 B7 Microsoft Samba 時序候選 | 局部 SWA 是否改善時間建模？需避免與 C4 混為一談 |
| C6（後置） | Market-Regime Gating、MCMamba 多尺度／Correlation Prior | 哪一個附加機制實際改善樣本外排名？ |
| C7（後置） | 先 Cross-Stock 再 Temporal、scan-order、VAST／TMB、RPL | 原始設計上的順序與 loss 假說是否成立？ |

**核心要求：** 同一 PIT／標籤 contract、明確 train/validation/test 期間、相同可見股票集合、相同特徵可見性、相同指標與每日對齊，至少記錄 Seed、收斂、參數、耗時、VRAM；使用 B5 HAC／block bootstrap／Holm 配對檢驗，**以當前 LightGBM 為必要對照**。若當前資料僅為存活者偏差代理資料，要明確標示為方法／工程驗證，並與 B3 PIT 修復並行，不假裝可投資。

**避免預設結論：** 早期的「最合理」是研究假說；如果 LightGBM、單 Temporal Mamba2 或單 Cross-Stock 分支勝出，應忠實保留結果而不是強行讓 C4 勝出。

### 第三步：用漸進整合而非另起爐灶

- 不要為了恢復早期架構就重寫整個 V7、直接移植舊 MAS，或另造多 Agent 協調框架；優先沿用 Pilot09／B4／B5／B7 可用的工程與資料契約。
- 統一模型介面與評估，而不是把所有新模組一次接上再大訓練。先完成官方 Mamba2 GPU Probe，核對 `mamba-ssm` 官方 fused kernel 對 `headdim`／CUDA 的真實要求，再實作／驗證 C1。
- 將 cross-stock mixing 單元化、可關閉、可獨立消融；研究 stock order 與 variable universe coverage，不要把跨股票股票 ID 排序順序當自然因果。
- 保留模型 checkpoint、配置、manifest、程式版本、逐日預測／IC、研究批次的不可覆寫證據；不干預既有 W38/W39 的已到期排程／原始 Shadow 產物。
- 本機 RTX 3060 6GB 做小型 Smoke，Colab 用於真正大型模型；相容 wheel 優先重用。除非已授權且符合資源預算，不得自行啟用大量 GPU 訓練或額外付費 API。
- **PIT 修復與模型工程可以並行**：B3 月底歷史快照 12 個月試做、TPEx source 完整性、`security_master`（含轉板/代號重用）、基本面公告時點，可與 C1/C2 的工程驗證平行推進。

### 第四步：階段級交付，不把小型測試當成整體完成

交付以下內容並持續更新既有路線圖：

1. **Architecture Gap Map**：當初首選架構 vs. 後來 V7 vs. 現有 B7，每個元件的實際程式檔、證據與缺口。
2. **Candidate Design**：一張正確資料流／維度圖（特別是 Temporal vs. Stock axes、GAT edges、PIT 時點與 Fusion）。
3. **Ablation Matrix 與算力安排**：哪個問題最值得先測、預期比較、必要數據、停止與成功條件；先 C0→C1，隨證據擴充。
4. **實際工程成果**：在不影響現役程式與資料的隔離分支中，完成至少一個下一步能用的模組／資料契約或測試，記錄真實 Pass/Fail／未驗證範圍；不能只交一頁設計宣告完成。
5. **下一階段 Backlog 與 Resume**：Git 狀態、Commit、Artifact 路徑、研究結果、阻礙、接手方式。未授權部署、交易、正式排程變更或付費 API 必須排除。

請用**繁體中文**整理研究與決策。不要在每一個一般技術細節都反覆要求 Human 核准；已授權的隔離研究與程式實作應能自行持續往下一項高價值任務推進。

---

## 4. 對應論文與官方來源（已核對連結）

### A. 當初形成首選架構的核心論文（優先讀）

1. **S-Mamba／Simple-Mamba — *Is Mamba Effective for Time Series Forecasting?***（2024）  
   https://arxiv.org/abs/2403.11144  
   角色：跨變量 token 化與 **Bi-Mamba inter-variate mixing**；論文的 temporal 部分主要透過 FFN，移植 Market Mamba 時不能誤以為它直接替代 Temporal Mamba-2。
2. **MambaTS — *Improved Selective State Space Models for Long-term Time Series Forecasting***（2024）  
   https://arxiv.org/abs/2405.16440  
   官方程式：https://github.com/XiudingCai/MambaTS-pytorch  
   角色：VST、VAST／變量排序、Temporal Mamba Block、Conv/Dropout 的消融參考；不是首選架構必備組件。
3. **TimeMachine — *A Time Series is Worth 4 Mambas for Long-term Forecasting***（2024）  
   https://arxiv.org/abs/2403.09898  
   角色：四 Mamba、多尺度與 CI／CM；當初認為對當前**日線**模型不是高優先，盤中模型另議。
4. **Mamba-2 — *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality***（ICML 2024）  
   https://arxiv.org/abs/2405.21060  
   官方實作：https://github.com/state-spaces/mamba  
   角色：Temporal 主幹與擬移植的 Cross-Stock mixer 的官方基礎；引用的加速倍數不是 Colab 台股任務的實測保證。
5. **GATv2 — *How Attentive are Graph Attention Networks?***（ICLR 2022）  
   https://arxiv.org/abs/2105.14491  
   角色：顯式圖分支的 Attention 改良與先驗圖建模；必須保證圖的歷史時點。

### B. 後續增加的可選設計論文

6. **StockMamba — *State-Space Gated Stock Transformer with Rank-Aware Optimization***（Mathematics 2026）  
   https://doi.org/10.3390/math14111859  
   角色：Mamba-2 市場狀態掃描、動態因子 Gate、cross-stock attention、U-shape Rank-Position Loss。當時的決策是**Gating 值得研究、U-shape 後置**，不必照搬整套 Attention。
7. **MCMamba — *A Multi-scale Correlation-aware Model with Mamba for Stock Price Forecasting***（Expert Systems with Applications 2026）  
   https://doi.org/10.1016/j.eswa.2026.132868  
   角色：multi-scale intra-stock Mamba 和 global correlation matrix 輔助 inter-stock modeling。當時只能充分利用摘要等可得內容；請勿把未讀到的付費全文技術細節當成已驗證。

### C. 名稱容易混淆、與現在 B7 有關的論文

8. **Microsoft Samba — *Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling***（ICLR 2025）  
   https://arxiv.org/abs/2406.07522  
   官方程式：https://github.com/microsoft/Samba  
   角色：**Mamba + Sliding Window Attention** 的混合層；現有 B7 `samba` 更接近這條路。**不是** 2026-08 首選的 S-Mamba／Cross-Stock Bi-Mamba。
9. **金融 SAMBA — *Mamba Meets Financial Markets: A Graph-Mamba Approach for Stock Price Prediction***（2024 預印本）  
   https://arxiv.org/abs/2410.03707  
   角色：金融 Mamba＋圖關係的相關研究；與 Microsoft Samba 和 S-Mamba 皆不同，不可混稱。

**閱讀策略：** 必讀 A1、A4、A5 及實際程式；A2/A3 為當初取捨脈絡；B6/B7 是可選增益假說；C8/C9 用於釐清 B7 模型定位及相關比較。先做有價值的實作與證據，不需先全文精讀九篇才開始工作。

### 原始研究討論備註

- 2026-08-28～30 三篇時序論文研討／模型藍圖（Gemini 討論）：https://gemini.google.com/app/fcdd84dd1b92796e
- 2026-08-30 StockMamba／MCMamba 研討與新候選（Gemini 討論）：https://gemini.google.com/app/48abf00c1050be41

若 Ruflo 無法登入上述私人 Gemini 對話，以本文件為可用的來源摘要，或在可讀檔案可取得時補讀對話匯出筆記；**不得假裝已直接讀過無法存取的私人來源**。

---

## 5. 最後的研究原則（請不要偏離）

> **我當初最看好的不是某一篇論文的成品，而是「Mamba-2 把長時序學好，Bi-Mamba 在股票維度學隱性關係，GATv2 保留可信顯式關係」的互補模型。這是研究起點，不是既定最終答案。**
>
> 做實驗的目的，是辨別**哪個部分真正增加有樣本外可信度的預測能力與實際投組研究價值**。若簡單的 LightGBM 較好，就保留它；若雙路模型沒有增益，就不要因為它很漂亮而留下複雜性。用真實資料、嚴格版本與時點契約、可重現的比較來決定架構，讓 Market Mamba 持續向產品前進。
