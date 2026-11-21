# P4a Colab 實驗草案 — 2026-11-21

**DRAFT / 未凍結 / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED**。Human／Claude 批准後才凍結，不得用工程測試宣稱結果。機器候選契約：[p4a-colab-contract-20261121.json](p4a-colab-contract-20261121.json)。本輪僅離線程式、CPU 合成測試及文件，不啟動 Colab。

目的為檢驗同配方神經模型是否增加 P2 主要格的組合價值。臂為官方 Mamba2、既有 S-Mamba 風格 `crossstock`、同配方 GRU，參考為 P1 凍結 C0+ 分數及 manifest。三臂使用相同資料窗 60、rank-gauss、d_model=64、2 層、split/purge、損失、AdamW 與現有 `FROZEN_CFG`；GRU 為兩層 nn.GRU，共同 embedding、末端 RMSNorm 與雙頭；不宣稱整個模型具有相同 normalization。參數數量由執行器記錄，近似不表示精確等量。歷史分析折 2021–2023（承襲 P3 的舊確認性折名稱，不代表 P4a 未曝光確認性），描述性 2024–2025，2026 是否納入為 OPEN；種子 17、29。歷史資料已曝光限制保留，不稱新的獨立未曝光檢驗。

沿用 [P2 主要格](spec-lowturnover-p2-20261119.md) 與 [P3 判定規則](spec-mamba-portfolio-p3-20261120.md)：優於必須 CI 下界 > 0 且 Holm p < .05；缺主格、種子或 identity 不作確認性結論。P3 21/24 擬合及既有未證實結果不修改、不補標為完整，也不與 P4 擬合混合。

保存每個 fit 的 BEST 權重供 P4c 排列穩健性診斷；epoch generation 保存 model/optimizer/scheduler/RNG/best_sd，latest、previous、best 指標只在 durable copy hash 驗證後發布。恢復只跳過完全驗證的 completed fit。中斷 fit 保留 forensic checkpoints；目前不承諾 mid-epoch／optimizer 精確續跑，不自動升格 orphan 為 best 或正式結果。

Colab 預算使用牆鐘小時及運算單位雙上限；Human 提供保守 CU/hour 估計，非冒充供應商實際扣點。每次開跑要求剩餘容量涵蓋預估耗時／單位與至少 25% 餘裕，執行中持續檢查。記錄實際 wall、guard wait、active hours、估計 CU；溫度守衛等待不計入訓練扣帳。Colab 不做等待／降頻暫停，溫度 >=85°C 即 STOP，所以 guard wait=0。中斷未知時間保守扣到重啟並 STOP，不能刪帳本重置。預算根目錄跨 speed/formal 共用，不能用新 out-dir 規避。

Human 已在本任務交付中陳述核准 Colab 的原因：筆電熱降頻，乾淨環境每 epoch 約110–130秒、GPU80–100%、84–87°C，正式訓練因守衛暫停慢2–3倍；這是 Human 提供的背景，非本 worker 新 runtime observation。Colab RAM 70–80GB 是可用性背景，仍須在實際 runtime 檢查。

尚需 Human 決定：

- 等預算調參的範圍、總 CU／小時上限、GPU 費率與每 fit 預估，是否將 speed-test 和調參計入同一總額（本工程預設共用）。
- 是否加入神經＋C0+ 混合的描述性臂，以及預先固定的混合權重。
- 是否獨立補跑 P3 缺的2026三次；若做須符合原 P3 身分／預算／准入，不能由 P4 程式繞過。
- 凍結完整 P4 config／多重比較 family、確認性完整性規則、run-ready GPU wheel smoke 與獨立審查程序。現階段正式入口遇 DRAFT 即拒絕。

Reconnaissance：沿用 `V6/research_lab/{seq_runner,seq_models,cross_stock,rolling_seq_scores,rolling_scores}.py`、既有 tests、tasks、docs/operations。新增 `p4_bundle.py` 是因現有 P3 adapter 無 self-contained safe-extract/lock 封裝；新增 `p4_colab.py` 是因 P3 collect 綁定凍結14h全域帳本及官方准入，不宜拿 P4 CU 預算削弱它。共用 `train_one`、purge 與 row checks，沒有另造模型訓練系統。`environments/p4a-colab.requirements.lock.txt` 原樣保存本機GPU lock，因既有2026.04 Colab lock屬舊E1環境、不能承載P4指定版本；SHA-256為 `2c39a3fb9567d1f9fc8ee83369c7b6c90cd19a8e679e32e4c8a6c1d75c5782c9`。新增測試／契約／操作文件分別承載 P4 隔離邊界與尚未凍結的研究問題；未新增 top-level tree。


## 凍結前模型差異揭露（DRAFT）

以下為審查提供的官方模型 CPU 建構數字（48 features、width64、2 層），不是 CUDA 驗證；凍結時必須以固定版本與配置重新記錄。

| 項目 | GRU | 官方 Mamba2 |
|---|---|---|
| 參數量 | 53,250 | 60,706（約多 14%） |
| 初始化 | PyTorch nn.GRU 預設初始化 | 官方 Mamba2 專屬初始化；不能視為等同 GRU |
| residual／pre-norm | GRU recurrent layers；沒有 Mamba residual pre-norm block | residual pre-norm blocks |
| 內部 gated normalization | 無 Mamba gated norm | 官方 mixer 內部 gated normalization |
| 共同末端 | embedding、末端 RMSNorm、雙 head | embedding、末端 RMSNorm、雙 head |
| dropout | 此路徑無 dropout | 此 temporal 路徑無 dropout |

共同的是 60-session、rank-gauss、split/purge、損失、AdamW、排程及雙 head；「近似寬度與參數量」不構成架構公平性或同 normalization 的證明。crossstock 的 dropout 與專屬 cross mixer 也須在凍結配置另列；FROZEN_CFG 的 dropout 值不是 GRU／temporal Mamba2 實際啟用 dropout 的證據。

## 研究決策清單 — 全部 OPEN

| 決策 | 待 Human 決定的選項與必要記錄 | 狀態 |
|---|---|---|
| 搜尋／選擇規則 | 固定單配方，或等預算搜尋；後者須列有限搜尋空間、每臂 trials／seeds、validation-only selection metric、tie-break、停止規則與候選失敗處理，禁止用外折結果選模型 | OPEN |
| 多重比較 family | 選項 A：H-P4a 三個神經臂各對 C0+，Holm m=3；選項 B：事前固定階層式 family／gatekeeping，須定義各層假設、alpha 分配及未通過時行為；新增混合臂或其他比較須重新計 family。不得直接沿用 P3 m=2 | OPEN |
| 完整性門檻 | 固定臂×折×seed 全部完成才准入，或事前限定可接受缺失及處置；需列 paired support、缺 seed／fold、失敗 trial、optional 2026 的角色。工程不能自行放寬 gate | OPEN |
| 預算 | 先定 GPU 保守費率、總 wall／CU、每 fit／stage 估時、搜尋及 speed-test 分配與至少25% reserve；數字尚未核定，工程預設所有 stage 共用同一帳本 | OPEN |
| 確認性期間與統計規則 | 新資料／前瞻期間、成熟 label、選擇截止、假設與 family、CI／bootstrap、門檻、資料／模型凍結 identity，均須事前登記 | OPEN |

P3 結果已知，P4a 在 2021–2025 折上不是未曝光資料。舊機器契約的 confirmatory_years 是歷史沿用名稱，不能單靠改 status 為 FROZEN 授予新確認性解讀；研究凍結必須同步審查機器契約。

可確認的路徑：Human 事前核定搜尋／選擇與比較規則，凍結模型及資料／統計契約，再使用未參與設計的新資料或前瞻期間，待 label 成熟且完整性／provenance 全通過後進行預先登記的確認性檢驗。目前這條路徑的所有選項仍 OPEN。

只能描述性的路徑：在已曝光 2021–2025 折上比較 GRU／Mamba2／crossstock、事後調參與混合權重、展示 CI／Holm 或補齊歷史支持。即使數字通過原門檻，也只能報告歷史描述性／探索性結果，不能稱為新的未曝光確認性證據。2026 的曝光與准入尚未核定，optional 不代表自動取得確認性地位。


## 本輪 Human 核准的 Colab 操作修訂（2026-11-21）

Archive／bootstrap 外部 SHA 改為可選；extract 自動印 archive SHA，提供時嚴格核對，manifest 內逐檔 size/SHA、資源上限、symlink 防護與白名單全部保留。Drive wheel-dir 模式只以 no-deps／no-index 安裝兩個本地 wheel，核對 Python／平台標籤，SHA 只記錄；不重裝其餘套件或 torch。

既有 exact lock 是本機 P3 參考，不代表已觀察的 Colab 環境。新增 `environments/p4a-colab-runtime.json`，因本機 lock 無法承載尚未知且不同版本的 Colab runtime；artifact identity 由 bundle 逐檔 SHA 綁定，初始 TBD／UNOBSERVED，待 Human pip list 與 runtime 自檢後填實際版本。P4a 三臂共用相同 Colab 環境 identity；與 P3 跨環境比較需揭露差異。speed-test 記錄每項 environment_mismatch 並標記結果不可與正式環境比較，正式入口仍要求 OBSERVED 完全相符；CUDA 不可用一律 STOP。研究問題與其他 OPEN 決策未凍結，沒有新研究結果。
