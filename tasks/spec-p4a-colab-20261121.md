# P4a Colab 實驗草案 — 2026-11-21

**DRAFT / 未凍結 / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED**。Human／Claude 批准後才凍結，不得用工程測試宣稱結果。機器候選契約：[p4a-colab-contract-20261121.json](p4a-colab-contract-20261121.json)。本輪僅離線程式、CPU 合成測試及文件，不啟動 Colab。

目的為檢驗同配方神經模型是否增加 P2 主要格的組合價值。臂為官方 Mamba2、既有 S-Mamba 風格 `crossstock`、同配方 GRU，參考為 P1 凍結 C0+ 分數及 manifest。三臂使用相同資料窗 60、rank-gauss、d_model=64、2 層、split/purge、損失、AdamW 與現有 `FROZEN_CFG`；GRU 為兩層 nn.GRU，加相同 embedding/norm/雙頭。參數數量由執行器記錄，近似不表示精確等量。確認性折 2021–2023，描述性 2024–2025，2026 視預算；種子 17、29。歷史資料已曝光限制保留，不稱新的獨立未曝光檢驗。

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
