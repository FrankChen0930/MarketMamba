# P4a Colab 實驗規格

**FROZEN — 2026-11-22，在任何 P4a 結果出現前凍結。UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED。** Human 已核准本批固定網格與工程實作；凍結不是 Colab 現場驗證或 Human acceptance。機器權威：[p4a-colab-contract-20261121.json](p4a-colab-contract-20261121.json)，環境權威：[OBSERVED Python 3.13／cp313 契約](../environments/p4a-colab-runtime.json)。本工作僅 CPU 合成工程證據，無 P4a 研究結果。

## 研究身分與不對稱

P3 結果已知；P4a 的 mamba2／crossstock 主格是新環境的已曝光 replication，仍納入指定 Holm family，不能稱未曝光確認。GRU 與回看窗是新假設，但資料仍已曝光。可確認的路徑：唯一獨立確認性路徑仍是未曝光的前瞻資料（P2 主格），不因本批而改變。只能描述性的路徑：本批歷史分析、2024–2026、學習率、排列診斷及 P4b rank-average 混合；歷史門檻通過也不升格為獨立確認。2026 是部分年度。

本批神經臂不調參，C0+ 曾調整，這個不對稱偏向 C0+，結果必須註明。另案 P4d 在本機 CPU 做 C0+ 調參預算對等比較；這是唯一保留 OPEN 的研究決策，不阻擋本批固定配方。P3 21/24 與其研究結論保持原身分，不合併 P4 fits、不補標完整。沿用 P1/P2 的 raw 價格、模擬標籤、未驗證下市覆蓋／PIT 市值等限制；historical simulation 不等於可交易。

## 凍結配方與執行順序

共用 d_model=64、2 層、rank-gauss、60 日窗、5d:10d 損失 1:0.5、AdamW lr=3e-4、weight_decay=.01、預設 betas=(.9,.999)、grad clip=1.0、15% 線性 warmup 後 cosine、8 epoch×400 步、dates_per_batch=8、stocks_per_date=256、patience=3。只用驗證 IC5 選 BEST，禁止 LAST。完整設定在機器契約 recipe；重用 `seq_runner.train_one`，不修改 P3 凍結 adapter 或 portfolio_eval。

| 佇列 | 固定內容 | fits |
|---|---|---:|
| Q1 | mamba2／crossstock／gru × 2021–2023 × seeds 17、29 | 18 |
| Q2 | 三臂 × 2024–2026 × seeds 17、29；描述性 | 18 |
| Q3 | windows 120、250 × mamba2／gru × 2021–2023 × seeds 17、29 | 24 |
| Q4 | lr 1e-4、1e-3 × mamba2／gru × 2022 × seed 17；描述性 | 4 |
| Q5 | crossstock BEST，2021–2023、seed17，驗證／評估各 R=10 隨機排列 | 0（前向推論） |

較長窗要求同股票完整歷史；新上市／缺歷史列不合格。Family C 在所有比較系統、W60 與 C0+ 共同可用列交集上重算分數排名與組合，輸出交集列數、各系統輸入列數、排除列數與比例。seed 缺列、標籤不一致、重複鍵一律拒絕；不得以靜默 dropna 變更支持。主格仍 require_same_rows。

Q5 每次排列後還原股票身分，輸出每日分數 Spearman、IC 變化、top50 持股重疊（同分以 code 決勝）。固定 R=10；各 split 的排列間平均 IC 標準差（ddof=1）≥ abs(crossstock 與同折 seed17 mamba2 的平均 IC 差距)，即記「順序敏感」。這是描述性診斷，記錄後保留結果，不作優勢宣稱；零差距亦照原規則判斷。各模型對驗證／評估完整 eligible rows 推論，沒有重新訓練。

## 多重比較（組合層）

P2 主格：top50、20 日再平衡、buffer100、等權、成交值2e7、佣金折扣.6、最低費20、滑價10bp、資金1e6、raw，基準只收比例成本。折 2021–2023，配對逐日淨報酬差，circular block bootstrap block20／1000，CI 下界>0 且 Holm p<.05 才通過登記的「優於」門檻；本批只標歷史門檻通過，不聲稱獨立確認。缺完整 fold／seed 或不足20配對日為工程未完成，不能通過。

| Family | 假設 | 校正 |
|---|---|---|
| A | H-A1 gru vs C0+；H-A2 mamba2 vs C0+（P3 已曝光 replication）；H-A3 crossstock vs C0+（同） | Holm m=3 |
| B | H-B1 mamba2 vs gru；H-B2 crossstock vs mamba2 | Holm m=2 |
| C | H-C1 mamba2 W120 vs W60；H-C2 mamba2 W250 vs W60；H-C3 gru W120 vs W60；H-C4 gru W250 vs W60 | Holm m=4，各用顯式共同交集 |

三個 family 分別校正，非階層式 gatekeeping。其餘一律描述性，不擴增確認性 family。組合統計使用不變的 `portfolio_eval` 公開函式；P4 evaluator 是新增 adapter，因 P3 身分禁止修改，沒有另造交易引擎。

## 架構差異揭露

以下既有 CPU 建構數字以48 features為例；正式 runner 逐 fit 記錄參數數量，不能當同容量證明。

| 項目 | GRU | Mamba2 | crossstock |
|---|---|---|---|
| 參數量 | 53,250 | 60,706 | runner 實際記錄（額外 cross layers） |
| 初始化 | PyTorch GRU 預設 | 官方 mixer 專屬初始化 | temporal 與雙向 cross mixer |
| residual／pre-norm | recurrent layers | residual pre-norm | temporal residual、cross LayerNorm／FFN |
| 內部 gated normalization | 無 Mamba gated norm | 官方 mixer 有 | 官方 mixer 有 |
| dropout | temporal 路徑未啟用 | temporal 路徑未啟用 | cross FFN .1，與 P3 一致 |

共同末端 embedding／RMSNorm／雙頭不代表全部 normalization 相同。Mamba2 使用官方 temporal 預設卷積；cross d_conv=2、d_ff=4*d_model、n_cross=1，沿用 P3，不做 no-conv。crossstock 推論通常2個排列平均；Q5 每個隨機排列 n_perm=1，與通常結果比較。

## 資料與接續契約

Drive 一次上傳 `MarketMamba/data/matrix_v1/{shared,folds}` 與 CLI 產生的 DATASET_MANIFEST.json。shared 八檔、folds2021–2026各 splits.npy，共14檔逐檔 size/SHA，禁止符號連結。X 固定1425394112 bytes／9778392d4c337d1f9cb5768002340a0d1aec13f7689b2f601b77bd2329b55363。SSD `/content/data` 先驗後複製，不從 Drive FUSE mmap 訓練；只有 Colab 本機 fold 可建立 links。rank-gauss 與 tag 在 shared 計算一次，Drive rankgauss_cache 以 X／end_indices／實作 identity 與輸出 hash 驗證後重用，不符重建。

每次 ZIP 只交付 source、研究／環境契約、參考 lock、dataset manifest、C0+ fold/date/code/label_y5 鍵檔與 provenance。完整 C0+ 分數留本機評估使用，不傳209MB全分數。矩陣 manifest identity 與 bundle/source/environment identity 納入 run_hash。

每個完整 epoch validation 後保存 model／optimizer／scheduler／epoch／BEST／bad／history／numpy global RNG／batch generator RNG／torch CPU與CUDA RNG：先本機 serialize/fsync/hash，再 Drive exclusive copy/fsync/re-read hash，最後 atomic pointer；latest／previous／best 全保留。只從 pointer 指向且契約／矩陣／原始碼／seed 相同的完整 epoch 恢復；重放中斷的未完成 epoch。缺 pointer 有 orphan、corrupt/missing checkpoint 或 identity mismatch 一律 fail closed；orphan 不自動升格 best。已完成 fit 必須逐檔與 schema 驗證後跳過，未完成 fit 在同一 attempt 接續。

P4 僅記錄每次嘗試 wall 秒數、GPU、溫度、時脈，不設時間／CU上限、不設25% reserve；廢除舊 P4 預算上限，P3規則不變。中斷未知間隔保守記錄到重啟（含離線時間，不冒稱訓練時間）。>=85°C 硬體 STOP 仍鎖定，需人工處理，不自動等待或清除。移除 pip check，正式環境仍 exact OBSERVED 契約核對；speed 不符只記錄，但 CUDA 必須可用。

## T4 耗時外推（不是實際完整批量測）

Human speed-test 熱機步速：mamba2 .456s、crossstock .462s、GRU .129s；Mamba2 首步含 Triton 一次編譯約285s。

| 範圍 | 訓練步耗時外推 |
|---|---:|
| 主格36 fits，各臂12×3200步 | 11.17小時（4.864＋4.928＋1.376） |
| 回看24 fits，假設步速隨窗長線性增加 | 19.24小時（120/60、250/60） |
| LR四 fits | 1.04小時 |
| 合計64 fits | 約31.45小時，另加validation／evaluation／I/O／編譯／Q5 |

長窗線性是假設，非T4長窗量測；crossstock不同形狀可能額外編譯，Q5無完整推論步速、無可靠估時。early stopping可能縮短。沒有時間限制，做完驗證並下載後由 Human 自行關 runtime。

## 唯一 OPEN

| 決策 | 狀態 |
|---|---|
| P4d 本機 CPU C0+ 調參預算對等比較；本批固定神經配方不調參，明列偏向 C0+ 的限制 | OPEN |

Reconnaissance：擴充既有 p4_bundle、p4_colab、seq_runner；沿用 tests、tasks、docs/operations、reports、canonical knowledge。新增 p4_eval 因 P3 portfolio_eval 身分不可修改，以公開函式重用。沒有新 top-level tree、runtime、資料來源或 production authority。
