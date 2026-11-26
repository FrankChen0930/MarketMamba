# P2 前瞻重訓靜態 Human 清單（WP5 候選）

SYNTHETIC_CPU_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED。
本清單與 [WP5 腳本](../../tools/forward_wp5.py)依
[不可修改規格 WP5／封盲條款](../../tasks/spec-forward-p2-primary-20261124.md)供獨立審查。
目前只有合成演練，並非可啟動的 Human 封盲 runner；不得藉工程 PASS 解除依賴 STOP。
不建立提醒、排程、wakeup 或通知。owner 固定既有 Human 一人，管道另待簽署。

## 凍結與每版欄位（全部私有）

每個 R_j 固定一列，不依績效改日期；真實 calendar 未合格以前不填造日期。

| 必填欄位 | 凍結規則／來源 |
|---|---|
| 版／階段／排定 origin／實際切換 session | T0／T1／後續更新分開；R_j 不因重試變動 |
| 人工核對日 | R_j−5 sessions；既有 Human 手動核對，不增加服務 |
| 資料快照日／截止 | R_j−1 session 18:00 Asia/Taipei，WP1 calendar identity |
| 訓練末日 | 含 origin−21，非重試日−21 |
| 當日／最晚驗收 | R_j、R_j+1、R_j+2 各18:00；最晚 R_j+2 18:00 |
| 程式與規格 | 完整 frozen source SHA、dirty source bytes、規格 SHA、靜態清單 SHA；重試重驗同一 pin |
| 輸入 | immutable source／publication／vintage／calendar／matrix／mask／labels／合格與抽樣 keys hashes，原 seed17／29／43 |
| 環境／資源 | 每 fit 前 WP4 environment／lock、CPU threads／layout／native binaries；實際秒數與峰值 RAM，不填估計作實測 |
| reference | 已曝光 reference identity、兩個獨立新程序、keys／X／mask／dtype／scores／ranks bytes；近似或只 rank 相同都 STOP |
| 模型／切換 | 三seed各 bytes／size／SHA、load→predict、完整 pointer／前版 identity；不混搭、不重設持倉／NAV／相位 |
| 事故／報告 | 私有事件、排定及實際時間／delay sessions、非績效判定、report SHA；不刪壞日、不補日、不換起點 |

## 每次手動核對

- 啟動前確認 WP1～WP4 及當階段依賴已獨立驗收、Human 接受；WP2 seed17 不相等仍 STOP。
- 以原輸入／calendar／seeds、原凍結腳本驗一次本版；初次成功不代表以後自動通過。
- 固定快照完整、尚未產生／切換版本前的可重建 IO／程序中斷才是 RETRAIN_TRANSIENT。
- 遲到沿上一版；依 R+1、R+2 順序同快照重試。identity／壞模型／非有限／來源事故立即失效，不當 transient。
- 初始 M0／F0 沒有可沿用的已准入前版；失敗即不啟動，依非績效延期政策重新凍結 T0／T1，不沿已啟動後重試政策開始。
- 全驗收完成後僅當日18:00授權三seed整體切換，下一session open生效；錯過不可回填。
- R+2超時：PRESTART 不啟動／STOP，FORMAL INVALID_FOR_CONFIRMATION；不得跳過等下半年。
- 私有重訓 STOP 不影響來源摘要固定次曆日12:00與原終點；來源摘要在模型／分數／持股讀取前決定。
- 公開只准 WP4 封閉白名單；禁止自由訊息、股號、分數、NAV、持股缺價、私人錯誤／重試進度、nonce、私有digest／尺寸。
- 淨化失敗不得發布替代摘要；按 BLIND_INTEGRITY 失效並停止不安全輸出，保留私有事故。
- 偷看／壞輸出／缺日立即失效；換 owner、刪檔、重設 T1、補日均不恢復確認性。

## 分階段截止與唯一開盲

- T0 前：初始非績效驗收、每日健康／缺日／封盲事故演練及最低 WP1～WP6，未過不啟動。
- T1 前20 sessions：每次重訓 R+1／R+2／超時、正式開盲合成演練完成，source SHA 凍結；T1 前另取得連續20排定sessions完整日鏈。
- T1 前5 sessions：WP6 精確252端點、初始成本／正式重置／切換連續性獨立驗收；全部簽署。
- Q2及次年年報 R_j 前5 sessions：人工再核快照、origin−21、期限、owner；不建立提醒服務。
- 開盲前以盲態驗完整252收益sessions、最後session+11及必要 publication、事故帳／private chain／外部head；已失效不得確認性開盲。
- Human 手動一次觸發，記批准者、UTC／台北時間、spec／source／data／model hashes及盲態判定；階段0預設一起描述性揭曉。
- 此候選的 `manual_unblind` 只寫合成audit record，沒有讀取／釋出真實結果或實作WP6。

## 合成重跑

使用新 namespace，不覆寫證據；無真實資料路徑參數。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=V6:V6/research_lab/tests:. \
/home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python -m tools.forward_wp5 \
  --synthetic-out .artifacts/forward/wp5-UNIQUE-IDENTITY/rehearsal
```

`frozen.json` 是候選 bytes pin，非 Human 簽署。由 Human 獨立保存並簽署完整 pin／來源與日曆，才可另案整合；目前沒有 production authority、OS隔離、durable retry queue 或正式前瞻啟動授權。
