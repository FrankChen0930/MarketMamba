# V7 P0 接手盤點

日期：2026-09-15。工作區：`/home/frank/projects/MarketMamba`；分支 `main`；HEAD `6cde79c6f9e9e192ba9a4f57c8deaa83e7817e69`。工作樹原有 README 修改與大量未追蹤 V7 程式／研究文件，均保留；未 reset、clean、stash 或重建空專案。

## Runtime 與測試

- ML Python：`.runtimes/colab-2026.04/.venv/bin/python`
- 實測：Python 3.12.13、torch 2.10.0+cu128；CUDA 可用，GPU 為 NVIDIA GeForce RTX 3060 Laptop GPU。
- 既有基線：`v7_experiment_test v7_confirmation_test v7_depth222_test`，2026-09-15 實跑 15 tests / OK。
- 新 stability 測試另覆蓋輸出契約、三 seed 集成、三窗 purge、逐日匯出續跑、壞 checkpoint、診斷與 window suite。

## 已完成結果（唯讀）

| 結果 | suite identity | summary SHA-256 | 狀態 |
|---|---|---|---|
| capacity-v1-fp32 | `049a5834ae53430682c5dcac30b51c2bc552d219b871791f4555dbab08f0c578` | `ab34d47c89bc3db01f4aafcd9e1131d8e98ed81ad77643d9355675deae561462` | complete，winner E5-state32 |
| confirmation-v1-fp32 | `77f549349a225f20093287362b9c070c695ef1abf17e1e2d53b9cdfd87cb4951` | `0076bb8fcda7f0f920d761c9da995f0b01c3ba45b200ba2910657ac3a31a769b` | complete |
| depth222-v1-fp32 | `191e9f1db72d86c74c9ed5a29172b5fcb560d198b50ca8611f3a2f464c2aab0c` | `f325a1993351c3e29867dbc7996a8e234112c17aa3375e44eab08a8d3d3bfb17` | complete |

E5 seed17 best checkpoint SHA 為 `85b4a1cc0d776cc0a4e43a2e0ef18ef86bc354e45dbb5eb707c5c6a773030829`；seed29 為 `d2665b94085f92df56e097a086e2f4701926d9861e9e965604af36d44dfd678d`；seed43 為 `b63777d29f9d2f489c495b048e0eb4ae68245c44f8ef88a4ed8232e9fb452344`。不重跑 capacity、confirmation 或 depth222。

## 矩陣

本機診斷矩陣 `.artifacts/v7-local-matrix/prepared-smoke` 存在，完成標記 SHA `eee4f1d9e947ba809f431c4a1a4377921d48c29ccf24b90ffa3c5c2b94a20ca8`，3,903 列、七股、train 2 日／validation 1 日，只可作工程驗證。

Colab 完整矩陣預定位置：`/content/drive/MyDrive/MarketMamba_V7/prepared-local-handoff-20260911`。本機未重新下載或掃描其大檔；交付會再次驗完成標記與各檔 SHA。特徵為每日截面轉換，macro expanding z-score 明確 shift(1)，所以三窗可重用矩陣並另建 split identity。限制：產業分類是凍結累積快照，並非歷史 point-in-time 分類；知識圖也是單一凍結 CSR，沒有逐年 edge snapshot。兩者可重用作回溯穩定性工程，但不能宣稱完整歷史 PIT，交付與 C 判讀都必須揭露。

## 已具備與仍待外部執行

已具備：三個 E5 best checkpoints、矩陣 identity 驗證、warmup／早停／best/latest／RNG／batch／validation 精確續跑、逐日匯出分片與 A/B 診斷、C 三窗設定與評估匯出。

仍待 Colab：完整矩陣上的三 seed 全日期 A/B 推論與診斷。只有來源與數值檢查通過後，使用者才開啟已備妥的 C 三窗完整訓練。C 不因程式已就緒而自動核准。

