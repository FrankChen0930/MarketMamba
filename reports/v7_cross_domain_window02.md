# V7 跨領域窗口02 — 工程與研究準備證據

單一 backlog：[research program](../research/v7/research-program.md#cross-domain-window-02)。
基線 main `c1179fe1ee8b66c768391b5f22f8aaebfc42d646`（已含批准的f7af348 Charter）。
分支 `development/v7-cross-domain-window-02`；UNMERGED / NOT_DEPLOYED。
此報告是有界工作的證據，不是第二套 roadmap／memory 或研究批准。

## W02-DATA：官方來源離線診斷

Authority：Human本次Data Engineering方向、Data domain的時間／provenance邊界，
以及既有fetcher的source-specific日期與欄位語意。沒有修改V6 production。
測試直接編譯固定source的六個原始AST function bodies，避開import fetcher會建立
raw/cache目錄的副作用；HTTP與HTML輸入均為synthetic。這是函式診斷，不宣稱
整個production模組或外部網站現在的行為已認證。

| Case | 合法對照 | 異常fixture的觀察 | 判斷與下一步 |
|---|---|---|---|
| TWSE MI_INDEX 日期 | 日期20260716，101列標記2026-07-16 | 回傳20260717仍產出101列，全部標記請求日2026-07-16 | 已重現日期未綁定風險；P1 production修復需另授權，不得修寫歷史正式資料 |
| TPEx daily quote schema | Open10、Close11 | 欄名與值一起重排Open/Close，parser產出Open11、Close10 | 已重現固定index schema drift風險；需在原parser採欄位契約拒絕/解析，不能以新平行pipeline遮掩 |
| TPEx 日期 guard | 同日正常解析 | 異日明確拒絕 | 既有正控制有效；不是所有官方parser都有相同缺陷 |
| MOPS monthly revenue完整性 | 上市＋上櫃2列 | OTC transport失敗時仍回上市1列，沒有完整性欄位 | 覆蓋不足風險，不自行決定monthly應全拒絕；需Human決定domain完整性政策。季度財報已有缺一市場就拒絕，不能靜默套用到monthly |
| MOPS month/PIT | 2026年5月營收→Date2026-06-01，100千元→100000元 | create_time空字串 | Date是既有cache月份口徑，不是publication timestamp；不可當V7 PIT evidence或把它改成新研究日期規則 |

七組對照／異常fixtures通過診斷oracle（不是production correctness PASS）。
外部請求0；正式資料讀寫0。完整數值、source hash與重現命令在
[window02 evidence](../research/v7/cross-domain-window02-evidence.json)。

重現：在該分支root以既有recovery venv執行
`python -B research/v7/cross-domain-provider-probe.py`；只輸出JSON，沒有資料落盤。
Probe是GENERATED diagnostic evidence的generator，不是source certifier。
Source固定於c1179fe；若source hash不同，必須把新觀察另記，不能覆寫本輪歷史。
新真實response/發布時間/資料版本與每個partition provenance仍需合格證據；
本輪不做大量抓取或用mutable cache補成PIT。
