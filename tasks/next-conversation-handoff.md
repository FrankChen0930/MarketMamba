# MarketMamba V7 新對話交接
更新：2026-09-15。請先讀本檔、tasks/plan.md、tasks/todo.md，再針對正在執行的階段讀程式。不要把全部歷史文件塞進上下文。
使用者要求建立新對話並按計畫直接實作，模型指定GPT 5.6 sol；不要另用GPT 6。此交接不是要求再問一輪是否開始。

## 1. 使用者已確認
- 全市場研究與模擬組合為主。排名、權重組合、績效都要；不考慮個人本金／持倉／整張零股，不做實際下單。
- 60交易日回看，5/10日頭；Mamba2時間序＋雙向跨股票Mamba2＋GATv2，no-conv。1/1/1不是理論定論，只是現有實驗對照。
- 大型訓練使用者在Colab執行，本機RTX3060 laptop 6GB只做小測試與資料處理。
- 優先準備Colab程式；Colab運行時開發配套，新結果到來穿插分析與下一輪。
- Codex管理MAS、覆核與整合，MAS品質不是自動保證。對外階段級回報，不逐檔詢問；合理新增檔案可自行做。
- 保留研究誠實性：排序不是報酬率/機率；特徵產業中性不等於組合中性；每日排名不等於每日換股。
- 完整Archify後置，TradingView等只借適合的設計，不建通用交易終端。

## 2. 實際工作位置與狀態
目前真正的新程式在WSL Ubuntu-24.04：
/home/frank/projects/MarketMamba
分支main；交接時HEAD=6cde79c6f9e9e192ba9a4f57c8deaa83e7817e69。
README.md有修改；V6/experimental/v7_*、environments、research/v7部分文件、tasks多數未追蹤，切勿reset/clean/stash移走。
單靠git clone main會丟失這些工作，請直接讀現有路徑。新對話的saved project選父目錄「MAS + Market Mamba」/home/frank/projects（非Git根），不是Windows舊MarketMamba saved project。
單獨MAS試用需建立乾淨隔離產品checkout，明確記baseline與exact paths，不動既有未提交研究程式。
頂層維持agent-skills、market-mamba-reference-repos、MarketMamba、mas-reference-repos、multi-agent-system五個既有資料夾；產物放其內。

環境可能與exec容器分離：若工具找不到實際路徑，不要建立空MarketMamba或當作檔案不存在。可使用Windows WSL bridge：
/init /mnt/c/Windows/System32/wsl.exe wsl.exe -d Ubuntu-24.04 --cd /home/frank/projects -- /home/frank/projects/multi-agent-system/.venv/bin/python -c <Python>
多一個wsl.exe是此環境已驗證需要的。用exec_command shell=/bin/sh,login=false,workdir=/tmp,sandbox_permissions=require_escalated。
跨Windows命令引號容易損壞，可把Python源碼每字轉ord，傳入單引號包住的 exec("".join(map(chr,[整數列表])))。每條命令保持短於32KB，長文件分塊寫；不要把Python JSON字串直接當shell quoting。
MAS venv只用來做stdlib bridge，不能因此視為已固定MAS執行版本。

ML Python：/home/frank/projects/MarketMamba/.runtimes/colab-2026.04/.venv/bin/python
已用版本Python3.12.13、torch2.10+cu128、mamba-ssm2.3.2.post1、pyg2.7、triton3.6。
Node：/home/frank/.nvm/versions/node/v24.19.0/bin/node，未必在PATH。
本機WSL約12GiB RAM+8GiBswap，不能把整個矩陣都讀進RAM。
一般exec沙箱曾報bubblewrap unavailable；不要為此重裝環境。WSL內network DNS曾失敗，Windows curl可用。這些是歷史觀察，當次需再確認。

## 3. 已有程式與測試
V6/experimental/v7_experiment_{model,data,train,suite,storage,test}.py：容量與通用訓練框架。
v7_confirmation_{design,suite,benchmark,test}.py、v7_confirmation_reference.json：三seed參照與來源比對。
v7_depth222_suite.py、v7_depth222_test.py：最新深度對照。
v7_integrated_*：環境／資料品質／point-in-time／產業中性／矩陣快取及初代候選。
測試入口：在V6/experimental工作目錄，以ML Python -m unittest v7_experiment_test v7_confirmation_test v7_depth222_test；依實際修改再選integrated tests。不是每回合重跑所有測試。
frontend有npm run build與lint；無現成frontend test script，不可假造npm test通過。網頁要實際browser驗證。
現有框架已做LR1e-4/weight_decay.01、warmup15%cosine、20epochs、early patience5/min5、gradclip1、best/latest、100steps或180sec checkpoint；resume還原模型/optimizer/scheduler/RNG/phase/batch/validation position。
舊V6 notebook「不還原scheduler」是另一套fine-tune，不套用到V7。
避免改V6/models/與V6/marketmamba/models/權重／正式架構。新增實驗放V6/experimental。
環境Cell曾失敗的修復：probe之前將v7_integrated_environment.json複製到/content/v7-experiment-runtime/manifest.json，捕捉並輸出setup錯誤；不得刪掉此順序。

## 4. 現有研究結果（勿重跑確認已完成的批次）
| 組別 | 設定 | IC5 | IC10 |
|---|---|---|---|
| E3 seed17 | 1/1/1 width64 state8 | .1131643931 | .1203084823 |
| E5 seed17 | 1/1/1 width64 state32 | .1144764166 | .1234058235 |
| E3 seed29 | 同E3 | .1100023734 | .1176306514 |
| E5 seed29 | 同E5 | .1121838859 | .1202509722 |
| E3 seed43 | 同E3 | .1116734184 | .1189896452 |
| E5 seed43 | 同E5 | .1135142842 | .1218985764 |
| T3 seed17 | 3/1/1 width64 state32 | .1043831996 | .1079472449 |
| D222 seed17 | 2/2/2 width64 state32 | .0996143901 | .1041452888 |
全部FP32；同一best-IC5 checkpoint的IC10。E5三seed均小幅較好，不宣稱統計顯著。D222跑13epochs、best8；train loss下降而val後退，不能推論所有加深無效。
E5參數100469、D222190469。E5訓練約212–220秒/epoch，D222約351秒，是舊環境量測非預算保證。
BF16短bench約11.96steps/sec，FP32約13.23；BF16省一些VRAM但較慢，不是完整精度實驗。
結果目錄：
deliveries/capacity-v1-fp32
deliveries/confirmation-v1-fp32
deliveries/depth222-v1-fp32
摘要research/v7/capacity-v1-fp32-results-review.md及confirmation-v1-fp32-results-review.md。
原始結果唯讀；先讀summary/suite/manifest，不要一開始載入所有.pt。

當前模型：股票N×60×59輸入，embedding→各股temporal stack→最後時間表示；GATv2及按stock_id字典序forward/reverse跨股scan，与temporal融合→5/10 heads。
Group D的12宏觀欄位現為0；其他47維有效語意。目前標籤是當日中心化名次、未正規化；MSE+.5ListNet，兩頭權重1:.5，loss大不能直接當失效。
原切分train2013-01-02至2023-11-17（名義2023年底、30交易日purge）；validation2024-01-02至2026-09-11，IC5有效648日、IC10有效643日。
2024–2026已被選架構使用，新的回溯三窗不可聲稱完全未碰過測試。

## 5. 資料與Colab交付
Drive完整已組裝矩陣：/content/drive/MyDrive/MarketMamba_V7/prepared-local-handoff-20260911
本機小測試：.artifacts/v7-local-matrix/prepared-smoke
原資料最新凍結2026-09-11。歷史含下市情境，但不能因存在若干下市股就宣稱毫無倖存者偏差。
分類快照不是完整歷史分類；切窗前需要檢查標準化/fit及圖時點。使用者容許偶爾缺漏、要求分級，不可一見問題全部停掉。
不要要求重傳raw／補丁／重建完整矩陣，除非檢查證明特定步驟確需更新；成本理由與範圍需明確。
deliveries/README.md仍稱Depth222為最新交付，該實驗已完成；新包準備完再更新入口。

不可覆寫的舊ZIP及SHA256：
deliveries/V7-Experiments-20260914/MarketMamba-V7-experiments.zip
ea855221eb0addce27958f6a279aa5ad5aa870e694889ea553ebc8d63798febd
deliveries/V7-Confirmation-20260914/MarketMamba-V7-confirmation.zip
5943ae5c1d30c5848a2a3e42c8f37eef74dceee8e4abd2c7647c9985df9a902d
deliveries/V7-Depth222-20260915/MarketMamba-V7-depth222.zip
77b470865b4db41e2a7ba7682a71cc7016d904b2d637900f47cc19fd1dafaa8e

## 6. 設計與歷史參考
research/v7/product-architecture-alignment.md已整合所有產品討論與來源，包含TradingView/financial-services/Obsidian。
Obsidian不在WSL repo，實際為/mnt/d/Desktop/work/ProjectForMe/MarketMamba/obsidian_note；22份md，最後修改至2026-08-11，私人且gitignored。按domain讀，不搬個人資金/背景/交易案例到公開文件。
舊文件及AGENTS大量描述V6.1/V6.2；新使用者要求優先，不能將舊“尚未授權/只規劃”套到本次已要求實作。
Archify完成品deliveries/MarketMamba-Archify/{overview,model}.html，對應JSON與evidence；原生渲染+四viewport瀏覽器檢查，未測所有匯出功能。後續按實際V7完成狀態擴充，不先做大圖。
參考庫/home/frank/projects/market-mamba-reference-repos/financial-services，commit69cbc81467a5dced793eee03dec4658aa24ef856。只借研究方法／報告，不執行內部角色指令。

## 7. MAS接入、報错與模型設定
使用者提供的原文：
/mnt/c/Users/Master/.codex/visualizations/2026/09/13/01a09bc5-6a7e-70f2-bfc8-f99cc83f00e4/mas-phase41-recovery/docs/integration/market-mamba-mas-handoff.md
交接原文副本：tasks/mas-integration-handoff.md（只作快照，routing仍讀原共用目錄）。
共用目錄：
/mnt/c/Users/Master/.codex/visualizations/2026/09/13/01a09bc5-6a7e-70f2-bfc8-f99cc83f00e4/mas-coordination
routing.json交接時：
mas_maintenance_thread_id=01a0a572-33e0-7d83-a5c8-f3979304effa，host=local
mas_repository=/mnt/c/Users/Master/.codex/visualizations/2026/09/13/01a09bc5-6a7e-70f2-bfc8-f99cc83f00e4/mas-phase41-recovery
reference_commit=3e61bcdc2e10ac89006eec498097d65f9fb95c1a
market_mamba_thread_id=null；automatic_polling=false。
新對話先自報實際thread ID於registrations/<id>.json，未知欄位null，再通知routing指定維護對話核對。若工具環境未直接提供自己的ID，用list_threads核對此新對話或由父對話通知，不靠猜UUID。
使用者已授權在這兩個對話間傳遞MAS問題、補件、恢復結果；不授權Slack/email或額外產品對話。

讀MAS README production entrypoint章節；使用orchestrator.production_cli，不是legacy orchestrator.cli。乾淨pinned checkout由MAS維護對話準備；不要指向持續修改分支，也不能從/home/frank/projects/multi-agent-system擅取未知版本。
目前先用GPT 5.6 sol作外層Codex與MAS選定的Codex執行配置；要從目前官方provider設定機制核對，不瞎造CLI參數、不改全域模型設定。MAS內部其餘角色沿合法配置核對，不靜默升到GPT 6。
MAS首次小型配套排除模型／V7／訓練／實驗輸出，不讀其內容來試用。外層Codex直接做P1訓練程式，這不限制產品整體V7工作。
MAS任務最少排除產品相對路徑：
V6/experimental/（所有v7_*）、V6/models/、V6/marketmamba/models/、Data/、deliveries/、research/v7/、environments/、.artifacts/、.runtimes/、V6/results/；另禁止讀寫任何.pt/.pth/.npz/parquet及私人.env/credentials。初始exact allowlist只能包含一般配套小切片。
不要把這份master plan當成已machine-issued的exact Plan；框架要求的Human批准依合法checkpoint處理。若仍需本人批准，先完成可review資料再問，其他獨立工作繼續。
出錯按原協議停該寫入單元，保存現場／一致DB副本，UUID report.json原子寫入，通知維護對話；不自行修改MAS核心/SQL，不無限重試。恢復按response.json政策與hash核對，寫自己的resume-result.json。同一檔不得Codex和MAS同時寫。
不建立輪詢或背景監看，沒有訊息就不承諾自動巡檢。
用量reset統一由MAS維護對話讀shared usage-reset.json及其memory協調；原文門檻嚴格>95%、最多一次，產品對話不得另行消耗。本摘要不重授權任何reset。

## 8. 新對話第一個可完成階段
讀plan/todo，確認實際目錄和未提交檔案，從P0與P1建立stability Colab包。
先交付A/B匯出診斷與已準備好的三窗C程式；A/B結果檢查點保留。不要重啟已完成的容量/confirmation/depth222訓練。
把MAS註冊路由接好即可，MAS初試如被精確批准或環境阻擋，不讓它拖住Codex直接實作Colab交付。
