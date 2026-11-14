# Window16 — Real Resource Coverage / Immutable Proof

WINDOW_COMPLETE / STOP / UNMERGED / NOT_DEPLOYED。無新study/validator/collector。

| Dataset／provider | HTTP／Encoding | 實際日期範圍／sessions | Rows／distinct securities | Bytes／基本QC |
|---|---|---|---|---|
| 11549 TWSE | 200 `text/csv;charset=utf-8`／UTF-8 | 2026-10-22至同日／1 | 1383／1383 | 137956；1368 PASS、15缺OHLCV |
| 11371 TPEx | 200 `application/csv;charset=utf-8`／UTF-8 | 2026-10-22至同日／1 | 1017／1017 | 141480；990 PASS、27缺OHLCV |

每resource僅1次資料GET、0retry/redirect/fallback。兩次catalogue HTML核對確認CSV href與
W15 matrix一致且OGDL1.0標示仍在；另4次官方網頁open核對授權/服務文件，無歷史API請求。
日期是CSV欄位民國1151022轉換，**不據此宣稱已認證交易日曆或「最近應有交易日」**。
無duplicate date/security keys；缺OHLCV保持缺失、未補0。基本QC只含schema/encoding、
finite正OHLC、非負整數volume、high/low bounds。成交股數按欄位標示為股，未轉lot；
價格幣別/調整/交易段/各security類型及可交易性未認證，未直接把全部證券視為普通股universe。

**兩者本次都只提供單日Snapshot；Historical Training Coverage = BLOCKED。**
已停止查詢其隱藏歷史能力，沒有試日期參數或替代API。無真實RankIC、fit、GPU或研究准入。
可作未來Forward Vintage的raw resource候選，但本次receipt不是2026-10-22原始準時vintage，
publication/revision/最早availability仍UNKNOWN；Collector activation仍NOT APPROVED。

[Coverage／精確source-license/request/retrieval/schema/QC/hash與raw路徑](../research/v7/window16-resource-coverage.json)。
原始bytes及manifest：本工作樹 `.artifacts/window16-resource-proof/snapshots/`。
[獨立restore證據](../research/v7/window16-restore.json)：另一Python invocation無網路，
將兩份bytes+manifests還原至 `.artifacts/window16-resource-restore/`，279436 raw bytes
逐檔SHA相同、v1 verify_snapshot及CSV rows重驗通過。不是僅以Git hash宣稱可還原。
**兩個copy都仍在同一隔離worktree，未追蹤Git，未認證跨裝置備份；移除tree可能同時遺失。**

W15固定27dcb36b5431b70e8aa831edb6cb9234cb3a40b3已由4da1bcf本機ff-only整合；
25targeted CPUtests PASS3.141s、既有W15獨立review及保護hash復核，未重跑175全套。
整合後source identity不變；另3項Snapshot focused tests PASS0.207s。W14 raw/W15 metadata/
synthetic bytes都保留原tree，未隨Git移入main。W06/W12不合併，Frozen/gpuHALTED不變。

[Fallback Policy](../research/v7/window16-fallback-policy.json)只是PROPOSED：限定次數/時間的
retry需另批；區分HTTP/permission/schema/unit/session/temporal錯誤與endpoint/provider故障。
只可用另經授權且語意相容的fallback，白名單目前空；不自動切換、不碰Backup Provider。
missing-session保留gap，後續合法backfill保留實際晚到身份，不能冒充準時Forward。

下一Human動作只需選：另授權探索合法歷史來源，或準備限定scope的Forward operation
（頻率、時計/QC/calendar、archive/backup/capacity/缺漏/通知仍須批准）；不能直接訓練。
來源/授權沿[11549](https://data.gov.tw/dataset/11549)、[11371](https://data.gov.tw/dataset/11371)
與[OGDL1.0](https://data.gov.tw/license)，顯名：金融監督管理委員會證券期貨局／TWSE、TPEx，
2026開放資料集11549/11371。本輪並未對外發布原始資料或啟用每日擷取。
StrictSTOP/E1BLOCKED/H2NOT_APPROVED/ExploratoryPROPOSED/CollectorNOT_APPROVED/
GPU_SAFETY_HALTED保持；最終review/HEAD/恢復見本輪evidence與唯一research-program Window16。

本輪唯一獨立review：Critical0/Required0，0修復；coverage/QC由本地CSV重算、原始及restore
bytes/hash吻合。所有新成果僅在W16 branch；STOP，不自行開始下一輪。
