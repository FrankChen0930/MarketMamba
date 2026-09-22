# V7 下一階段：訓練效率、實驗控制與模型容量
日期：2026-09-13 計畫；2026-09-14 更新交付狀態。
**新實驗入口已完成本機功能驗證**：[操作說明](../../deliveries/V7-Experiments-20260914/README.md)。
70 項回歸測試通過；實際交付 ZIP 在 RTX 3060、官方 SSD/GAT 上，FP32 與 BF16 各六組小樣本流程完成，含停止／續跑與完成跳過。
這是程式功能驗證，不是完整資料集的容量效果比較。

## 實際交付與原計畫差異
- 新增 v7_experiment_{data,model,storage,train,suite,test}.py；舊固定快照與 V6 權重不改。
- 目前已提供資料向量化、單批預取、pinned/nonblocking 傳輸、GPU 使用率快照／峰值與分段計時；尚未提供自動多日期 batch 或可靠全組 ETA。
- 採用 warmup＋cosine LR，而非 OneCycle 動量循環；15% warmup、clip=1、patience=5、min_epochs=5。
- 固定使用 5 日驗證 Rank IC 做選擇，不在缺失時偷偷換成 10 日；兩頭結果皆記錄。
- 預設 FP32；BF16 經小樣本 CUDA 驗證，可用新系列啟用，尚未宣稱 A100 效能收益。
- 每 100 個 batch 或 180 秒後的 batch 邊界存檔，訓練及驗證都可續跑，保留最新／前一份／最佳。
- 已組裝矩陣以完成標記及檔案 SHA 校驗後重用；新入口不重新建構。
- CPU 1400 檔模擬工作量的資料準備中位數改善約 7.09 倍；不代表整體訓練加速倍率。
- 詳細證據：.artifacts/v7-experiments-verification-20260914/verification.json。
以下保留原始稽核與實驗設計背景；「現況」指舊固定學習率版本。


## 已確認的現況
稽核對象為 V6/experimental/v7_integrated_train.py，以及本次中間交接包的固定程式快照。
- 時間 Mamba 2 一層；跨股票正向一層、反向一層，兩方向參數獨立。
- d_model=32、d_state=8、expand=2、head_dim=8；GATv2 一層四頭。
- AdamW；LambdaLR 恆回傳 1，實際固定學習率。沒有 warmup，也沒有 early stopping。
- 每 100 個更新及 epoch 邊界保存 checkpoint；已有模型、optimizer、scheduler、RNG、epoch/batch/step 與來源指紋，並保存驗證最佳 .best。
- 存檔目前直接寫目標檔，尚無先寫暫存再提交的機制。
- 有逐輪 validation 計算，但無清楚的逐輪指標事件；沒有 AMP 或 gradient clipping。
- V6 現行 trainer 使用 OneCycleLR、warmup、early stopping、AMP、gradient clipping。現行 config 的 d_model=256、d_state=32，warmup 比例 0.15、patience=10、clip norm=1.0。多尺度層數為 [2,3,3]。這是現行原始碼設定，尚未核對使用者特定 V6.2 checkpoint 的實際架構。

## 優先順序 1：效率量測與必要訓練控制
先保留模型及資料／損失定義，量測資料組裝、圖構建、H2D、forward/backward、optimizer、validation 與存檔時間。
輸出 steps/s、epoch 耗時／ETA、GPU utilization、顯存 allocated/reserved、每批股票／邊數；只在診斷模式加入必要同步，避免量測本身拖慢正式訓練。

依實測瓶頸依序處理窗口與圖的向量化／有界快取、CPU 預取與 pinned/nonblocking 傳輸、減少逐參數 GPU 同步。
保留必要非有限值防護；先確認結果等值再採用效能修改。快取須有記憶體上限，不能展開所有日期的 60 日窗口。
checkpoint 先本機完整寫入再提交 Drive；latest/best 分離，顯示保存成功及耗時，保留合理中斷恢復粒度。
多日期 batching 排在後面，日期之間不得連圖或混算排名；梯度累積與有效 batch 改變須另列實驗，不宣稱和原更新路徑完全等值。

必要控制：
1. 真正的 warmup＋衰減排程。起始方案沿用 V6 的 OneCycle 類型與 15% warmup 作候選，先驗證小步數邊界；schedule 總步數與硬性停止上限分開保存。
2. validation Rank IC early stopping，明定監控頭／fallback、patience、min_delta、min_epochs。初始規劃 patience=5、min_delta=0、min_epochs=5；是待驗證的工程起點，不是最佳值。只有完成整輪驗證才更新耐心計數。
3. 每輪輸出與落盤：train loss、兩頭 Rank IC／有效日期數、股票及 target 數、LR、梯度 norm、耗時、best 指標及 patience 計數。
4. gradient clipping 起始上限 1.0；AMP/BF16 在 CUDA SSD/GAT 的 forward/backward 非有限值檢查及數值對照通過後才啟用，保存 scaler 狀態（若使用）。
5. 可恢復且防半寫 checkpoint：完整 optimizer/scheduler/RNG/游標/early-stop/precision 狀態和設定指紋。中斷與不中斷的更新路徑須有回歸測試。
6. loss 尺度診斷記錄股票數及零預測參考值，避免將不同股票數日期的 raw rank MSE 當成相同尺度。效率改寫不偷偷變更排名標籤或 loss 權重。

驗收：非有限值被擋下、早停與續訓計數正確、warmup/衰減與續跑一致、latest/best 完整、驗證事件可見；同條件局部基準有實测吞吐，沒有測量就不宣稱加速倍數。

## 優先順序 2：容量可設定化
時間、跨股票正向、跨股票反向層數分別可設定，預設保留 1/1/1。
d_model、d_state、expand、head_dim、chunk_size 均明確記錄；新增堆疊的 residual/norm 結構要先定義並驗證，不能只變層數標籤。
設定連同參數量、精度及訓練控制進入實驗／checkpoint 指紋。增加層數或維度要新開實驗，不能當成同一 checkpoint 的普通續訓。

## 優先順序 3：循序容量實驗主題
| 順序 | 主題 | 時間／正向／反向層數 | d_model／d_state |
|---|---|---|---|
| E0 | 新訓練控制下的小模型對照 | 1／1／1 | 32／8 |
| E1 | 加深時間分支 | 3／1／1 | 32／8 |
| E2 | 加深雙向跨股票分支 | 較佳時間深度／2／2 | 32／8 |
| E3 | 增加表示寬度 | 沿用較佳深度 | 64／8 |
| E4 | 再增加表示寬度 | 同上 | 128／8 |
| E5 | 增加狀態容量 | 較佳深度與寬度 | 較佳 d_model／32 |

初期正反向層數對稱，以節省組合；需要時才拆開研究非對稱。若 128 或 d_state=32 有效益，再考慮 d_model=256 或 d_state=64，不一開始全部排列組合。
資料、切分、種子、圖、heads、loss 與訓練控制保持一致。計算實際參數量、GPU/CPU 時間與成本。
短跑先比較吞吐與穩定性；不能拿尚未走完 warmup 的結果下效益結論。表現較佳候選再多種子確認，並兼顧 5/10 日指標、成本與後續組合回測；不可因幾個噪聲點決定加深有效。

## 正在跑的 Colab 與交付界線
正在跑的 1/1/1、32/8、固定 LR 版本保留作歷史小模型紀錄，不能直接當作更換排程後容量實驗的公平對照。
不改寫現行中間包固定程式，不重新建矩陣，不改原始資料或受保護 V6 模型。新 trainer 在隔離開發版本完成後另行交付。
新控制與實驗自動排程已完成上述功能驗證；完整資料容量比較仍待使用者在 Colab 執行，不能宣稱一層足夠、三層較好或 A100 可加速特定倍數。
