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
    ├── budget_speedtest/      ← speed-test 專用帳本，跨交付延續
    ├── progress/formal/、progress/speedtest/ ← PROGRESS.txt／progress.json
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

## 7. 已實作與仍待確認
- **已實作（後續交付候選，UNMERGED／NOT_DEPLOYED／NOT_ACCEPTED）**：speed-test 使用 `budget_speedtest/`，FORMAL 使用 `budget/`；各模式跨交付延續自身帳本。既有缺帳／chain／identity 保護維持，不自動遷移或重置歷史帳本。
- **已實作（同上）**：第 5 節的串流逐行 flush、每 50 steps／每 epoch 進度、每約 60 秒 Drive 快照、可行動錯誤摘要與完成／中斷後摘要格。FORMAL 才顯示驗證 IC；speed-test 保持隔離。接續 fit ETA 只用本次啟動起點後的新 steps／耗時，至少 10 steps 且 1 秒才估計，否則顯示「預估中」。佇列目前 fit 優先用本次 step ETA，後續 fit 用本次已完成 fit 的固定實測耗時均值（至少 1 秒）；cached skips 不納入樣本，無可靠資料顯示「預估中」，Q5 耗時另列未知，不以 fit 計時冒充。略過也更新 fit 編號、Q 與完整識別欄位，進度行及 TXT／JSON 顯示「已完成（略過）」；最後 fit 完成時 fit／queue ETA 為 0。此兩項複審修補仍待獨立複審，不代表已解除 NO_GO。
- 驗證層級：**CPU 合成測試**（含權重／預測 bytes 與 torch RNG 一致、串流大量輸出與非零退出、原子寫入／失敗繼續、模式帳本獨立與變異測試）；**Colab 現場 UNVERIFIED**，不是 Human acceptance。細節見[工程報告](../../reports/v7_p4a_colab_engineering.md)。
- 正在跑的 02_formal 仍是原先版本；不要在執行中的 kernel 換碼。斷線後的新版接續依第 8 節來源相容升級候選處理；原 notebook 原地續跑仍是保底。
- Colab 單一 kernel 執行訓練時不能同時執行摘要格；主要從手機／Drive 網頁查看固定 `progress/formal/PROGRESS.txt` 或 `progress/speedtest/PROGRESS.txt`，並核對更新時間。
- 矩陣增量追加能力未確認。

## 8. 跨版本接續（來源相容升級）

- 斷線後可改用經審查的新交付，沿用舊 run 目錄／run_hash、已完成 fit 與完整 epoch 檢查點。**只限數值中立的進度顯示、日誌、speed-test 帳本分離或文件變更**；模型、optimizer、RNG、研究配方、資料／矩陣、環境與預算契約不得變動。
- 每次升級都要由工具產生逐檔 from／to SHA、差異與理由分類的宣告檔，隨 ZIP 交付；必須有舊版對新版的 CPU 位元一致及舊 checkpoint／completed-fit 接續測試。以後所有 Colab 交付都必須保留這個能力；新的來源基準先新增測試與獨立審查，再擴充認證範圍，不能只換 commit 參數。
- 本次候選認證來源為 `4e56e9d`。舊 run_identity 只有 bundle manifest identity，因此必須保留 `02_formal/p4a-bundle.zip` 作逐檔證據；新版自動讀取，不需 Human 手貼 hash。缺少宣告、舊 ZIP 或任一身分不合一律 STOP，列出預期／實際與下一步；不得悄悄新開批次。只有第 0 格明設 `ALLOW_NEW_RUN=True` 才允許新批次，此開關不修復／重置原帳本，也不略過既有完整性驗證。
- 斷線後操作：開新 notebook，將新版 zip＋ipynb 放到 `MarketMamba/p4a/03_formal_resume/`，從上到下執行；`EXPERIMENT_DIR` 仍指向 `MarketMamba/p4a`，資料不用再上傳，舊 ZIP／budget 不刪不搬。不要在仍執行訓練的 kernel 中更換程式。
- 在 run 目錄與 hash-chain ledger 記錄 `SOURCE_UPGRADE`，含 from／to 來源清單雜湊、宣告身分、時間及環境。歷史 run_identity、完成標記與 checkpoint 不改寫。候選目前只有 CPU 工程證據，Colab／T4／Drive 現場仍 UNVERIFIED；仍需獨立審查與 Human acceptance。

## 9. 上線前排練（2026-11-23 事故後新增）

- 每種佇列項目的第一個擬合，必須先用**真實 fold 資料與同一程式路徑**在本機 CUDA 或 Colab 排練；至少涵蓋 Q1、Q3 的 120／250 窗、Q4，以及以排練 checkpoint 執行的 Q5。CPU 合成測試不能代替此現場 gate。
- `python -m research_lab.p4_colab --rehearse --data-dir ... --reference ... --queue Q3 --window 120 --arm mamba2 --steps 1` 預設 1 epoch／1 step；入口沿用 SeqData、checked_splits、train_fit／train_one、predict、發布與 completed-output validator。沒有 Drive 輸出／預算帳本，所有產物在系統 `/tmp/p4-REHEARSAL-*`，明標 `REHEARSAL / 不是研究結果`，不能納入正式 fit 或推論研究結論。
- 第 8 節來源相容範圍針對本事故擴充：允許有專項證據的窗≠60 對齊錯誤修正、新增排練，以及 Q5 成熟標籤診斷。不是任意研究配方變更許可。`common_rows` 容許相同位置 NaN，拒絕 ±inf 與標籤不一致；Q5 IC 僅使用所有模型共用的成熟標籤列並回報排除量，未知／不足仍 STOP。組合績效不使用 label_y5。
- 本事故升級採用條件：舊 run 的 **36 個 Q1／Q2 W60 completed fits 全部通過 resume_fit**，保留 bytes 並略過，從尚未完成的 Q3 接續。缺任一完成證據或損毀即 STOP；不重跑前 36 筆來湊數。本機 GRU／crossstock 位元一致、未呼叫修改路徑、36 完成接續整合、宣告重產與獨立審查都是必要工程證據；本機真實 CUDA／Colab 現場與 Human acceptance 仍需另外完成。

- `deliver` notebook 在正式 runner **同一格前段**自動依序執行 Q3 W120 mamba2、Q3 W250 mamba2、Q5（既有兩個 prerequisite fits），各 `--rehearse --steps 1`，使用前格 Drive→SSD 準備的 `/content/data`、同一 bundle reference 與官方 mixer。任一步非零／中斷即停止；顯示階段、`/tmp/p4-REHEARSAL-*`、期望／實際及貼完整輸出給 Claude／修正重跑的下一步。全部通過才啟動正式 runner，摘要明標「排練通過（不是研究結果）」及各步耗時。
- 第 0 格 `REHEARSE_FIRST = MODE != 'SPEED_TEST'`：FORMAL／resume 預設 True，SPEED_TEST 預設 False，可明設開關。同一 run command／已驗 bundle source、prepared matrix／rankgauss identity 與實際 environment identity 全相同時，以 `progress/formal/rehearsal_passed.json` 略過；缺少、損毀或任一身分改變即重做。標記不是完成 fit／研究 authority；失敗不寫成功標記，重檢前移除過期標記。保留第 9 節其他類型首次擬合現場排練要求。
- W≠60 在 train_fit 前檢查 sessions≥W、eligible 訓練日≥val_dates+20；不足報 fold／W／需要與實際數量及下一步，禁止縮窗／降低門檻。W60 不進入新增檢查，CPU 跨版權重／預測／RNG 位元驗證不得取代 CUDA evidence。
- 排練成功輸出 `torch.cuda.max_memory_allocated` bytes（每個子程序先 reset，Q5 包含兩個 fits／診斷）。**W250 真實 windows 張量與 CUDA 峰值、全量推論耗時仍 UNVERIFIED**；1 step 只縮短訓練，完整驗證／推論可能仍慢，首次排練耗時未實測。標記用於避免相同身分的重複完整推論。
