# Colab 訓練交付：Human 的持續有效需求（單一參考檔）

狀態：CANONICAL（Human 的操作偏好與驗收條件）；來源為 2026-11-21／22 與 Human 的對話。之後任何要上 Colab 訓練的工作（P4a 之後的 P5、P4d 等），**只需參考本檔**；逐步操作細節見 [colab-p4.md](colab-p4.md)，安全與穩定契約見 AGENTS.md「訓練與 Colab 穩定契約」。與本檔衝突時，以 Human 最新明確指示為準，並更新本檔。

## 1. 交付形式
- 每次交付 **一個 zip＋一個 ipynb**（加一個兩行的 README.txt）。Human 只把這兩個檔案放進 Drive 指定資料夾，用 Colab 開啟 ipynb，從上到下按執行就能開始。
- **不要求 Human 手貼任何 hash／SHA、手動上傳 bootstrap、手動查版本、手動改路徑或參數**。第 0 格只放少數必要開關，預設值可直接用。
- 完整性驗證由程式自動做（zip 內逐檔 size／SHA、資料集 manifest）。威脅模型是「Human 自己打包、自己上傳、自己的 Colab」：保留擋『自己程式 bug』的檢查（解壓上限、符號連結防護、白名單、identity），不加只對惡意第三方有意義的儀式。zip 的 SHA 可只印出供記錄。
- 不使用 `pip check` 當閘門（Colab 預設環境本來就有與本專案無關的相依衝突）。

## 2. Drive 結構（資料只上傳一次）
```
MyDrive/MarketMamba/
├── data/matrix_v1/            ← 矩陣資料集（一次上傳；shared/、folds/、DATASET_MANIFEST.json、rankgauss_cache/）
├── wheels/                    ← 預編譯 wheel（回退 MyDrive/Mamba_Wheels_T4_new）
└── p4a/                       ← 實驗資料夾（之後依實驗命名）
    ├── budget/                ← 帳本與結果，整個實驗共用；每個實驗階段不可共用同一本帳造成契約衝突（speed-test 另用獨立帳本）
    └── 01_speedtest/、02_formal/ …  ← 每次交付一個小資料夾：zip＋ipynb
```
- 矩陣**建一次就好**，存 Drive 重用；不為了每天的新資料重建。新資料只在做前瞻驗證時另案擴充（現有矩陣涵蓋 2005-01-03 至 2026-09-11，尚未確認建置程式是否支援增量追加）。rank-gauss 快取建立一次後存回 Drive。
- 不替 Human 刪 Drive 檔案；舊的資料夾保留，Human 自行決定。

## 3. 執行環境
- Colab Pro，T4。使用**目前最新的 Colab runtime**，不追舊版；套件版本如實記錄於 `environments/p4a-colab-runtime.json`（OBSERVED），版本差異需在報告中註明（例如與 P3 本機的 torch／CUDA 不同）。
- mamba-ssm／causal-conv1d 用 Drive 內自行編譯的 wheel（cp313、T4/sm75）；缺 wheel 時 notebook 內有編譯備援（官方 sdist，將 `setup.py` 內所有 `-std=c++17` 改為 `-std=c++20`，`MAMBA_FORCE_BUILD=TRUE`、`MAX_JOBS=4`；編完做 Mamba2 小型前向／反向冒煙）。Colab 預設 torch 變動時需重編。
- 已安裝套件符合環境契約則跳過安裝。CUDA 不可用一律停止。

## 4. 訓練功能（必備）
- **斷點接續**：每個 epoch 完成後原子寫檢查點（model、optimizer、scheduler、epoch、best 權重、RNG 狀態），Colab 斷線或重開後重新從上到下執行即可續跑，已完成的 fit 驗證後跳過；續跑結果需與一次跑完位元一致。
- Warm-up（15% 線性）＋cosine 衰減、梯度裁剪 1.0、AdamW、early stopping（patience）、保留 BEST 權重。
- **不設時間／運算單位上限**（只記錄耗時、GPU、溫度、時脈）；Human 做完會自己下載結果並關閉 runtime。硬體保護溫度閘門（85°C 停止）保留。
- 一次執行把目前看得見的實驗全部放進同一個佇列，依優先序處理（先確認性，再描述性，再消融），完成一個存一個到 Drive。
- 結束時自動打包結果 zip 並觸發下載。

## 5. 進度可見性（必備，P4a 之後的交付要有）
- 訓練過程要**即時顯示在 notebook 輸出**：當前 fit（第幾個／共幾個、臂、折、種子、窗、學習率）、epoch／step、loss、驗證 IC、本 fit 已用時間與預估剩餘、整個佇列預估剩餘、GPU 溫度／時脈、最近一次檢查點時間。不得讓子程序輸出被吞掉（以串流方式逐行轉印並 flush）。
- 同時每約 60 秒寫一份人類可讀的 `PROGRESS.txt`（與機器可讀 JSON）到 Drive 實驗資料夾，讓 Human 可用手機或 Drive 網頁查看，不必守著 notebook。
- 出錯時訊息要可行動：說明哪一步、期望值與實際值、下一步該做什麼，不只丟出 traceback。

## 6. 分工與流程
- 實作交給 Codex；Claude 做規劃與獨立審查（Claude 額度較少）；審查者與實作者不同 session。審查沒過不交付；審查者只做防禦性程式碼審查，不需自行構造攻擊載荷（避免被安全分類器擋下）。
- 預先登記：規格在任何結果出現前凍結；結果出現後不得修改規則。已知偏向（例如神經臂未調參）要明寫。
- 完成通知要主動檢查，不只依賴完成通知（曾漏通知）。
- 誠實回報：測過什麼、沒測什麼（Colab 現場行為在 Colab 實測前一律標 UNVERIFIED）。

## 7. 已知待改進（尚未實作）
- speed-test 與正式實驗共用同一本預算帳本，契約改變後會 `ledger identity/chain mismatch`（2026-11-22 遇到）；應改為 speed-test 使用獨立帳本。
- 進度可見性（第 5 節）尚未在 02_formal 版本實作（該次 notebook 以 `subprocess.run` 執行，輸出被吞掉）；正在跑的 02_formal 因 run identity 綁定原始碼，不要中途換新版 zip。
- 矩陣增量追加能力未確認。
