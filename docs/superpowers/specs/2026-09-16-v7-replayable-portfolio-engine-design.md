# V7 可重放組合引擎設計

日期：2026-09-16
狀態：已核准
核准依據：使用者審閱「狀態機、成交／成本規則、可校驗重放」及 unknown fail-closed 原則後，明確要求繼續執行。

## 目標與排除

建立不依賴模型結果、淨值由 1 起算的研究用組合狀態機。它把收盤後訊號轉成下一個明確可交易時點的持倉，保留現金、成本、未成交與部分成交，並可在斷線後從不可竄改的 JSONL 日誌重放到相同狀態。

本階段不接個人資金、券商帳戶或實際下單；不修改 V6.1、既有回測器、模型、權重、資料檔、A/B/C 結果或排程。

## 方案

採用獨立的事件溯源式 V7 引擎。輸入事實、狀態轉移與保存責任分離；斷線後可重建；重送用 event id 去重；舊 V6 行為不變。多一層事件契約是防止重複扣成本所需的最小複雜度。

未採用：

1. 擴寫 `v62_portfolio.py`：其名單狀態與同日收盤研究口徑已服務既有流程，混入新成交語意會混淆結果。
2. 完整交易所／券商模擬器：目前沒有逐筆委託簿與可靠交易狀態資料，撮合模型只會把假設包裝成精確結果。

## 邊界與檔案

- `V6/experimental/v7_portfolio_contract.py`：版本化規格、事件、交易狀態與 Decimal 序列化。
- `V6/experimental/v7_portfolio_engine.py`：純 CPU 狀態機；規劃目標、套用市場時點、成本、部分成交與公司行動。
- `V6/experimental/v7_portfolio_journal.py`：加鎖 append-only JSONL、hash chain、event-id 冪等與重放 CLI。
- 對應三份 `*_test.py`：用手算 fixture 驗證公開行為。

不新增第三方依賴。金額、價格、數量、費率及分數使用 `decimal.Decimal`，在 JSON 中以十進位字串保存，避免重放漂移。

## 組合與再平衡

`PortfolioSpec` 固定 holdings count、buffer multiple、rebalance session 間隔、head，以及比較假設買進成本 `0.0015`、賣出成本 `0.0045`。成本只沿用舊研究比較口徑，不描述目前法規、折扣或真實費率。journal 建立後 spec 不可變。

訊號含唯一 id、帶時區收盤後時間、head 與每股分數。head 不符直接拒絕。排序固定為分數降冪、stock id 字典序升冪。

到期再平衡：

1. 保留排名不差於 `buffer_multiple × holdings_count` 的現有持股。
2. 從 Top-N 依固定排序補到 N；候選不足時只用實際候選。
3. 訊號只建立 pending target，不在同一時點成交。
4. 最早只在 `occurred_at > signal.as_of` 的市場 session 嘗試成交。
5. 下一週期仍未完成時，新訊號可明確取代舊 pending target。

## 可交易性與成交

每個 quote 明確提供價格、狀態及買賣兩側 `fill_ratio ∈ [0,1]`：

- `OPEN`：可買可賣。
- `BUY_BLOCKED`：不可買，可賣。
- `SELL_BLOCKED`：可買，不可賣。
- `HALTED`：兩側皆不可。
- `UNKNOWN`：兩側皆不可，是預設 fail closed。

引擎不由日報酬或還原價格推定漲跌停／停牌。理想化研究必須顯式送入 `OPEN` 和 fill ratio 1，假設會留在 journal。

每個 session 先標記價格，再賣後買。費用只對 filled notional 扣一次。買進受現金限制；不足、被擋或 fill ratio 小於 1 時保留 pending。沒有 quote 等同 UNKNOWN。

目標使用等權與 fractional quantity。整張 1,000 股、盤中零股流動性及個人資金門檻留到實盤介面，不在淨值 1 的研究引擎中假裝精確。

## 公司行動與估值

除權息不由價格跳動反推。上游必須提供 quantity multiplier、每股現金及 action 後參考價格。引擎依顯式事件調整數量、現金與最後價格。拆分 fixture 維持總價值；現金股利以原持股數一次入帳。缺事件時不補造。

任何狀態滿足：

`net_value = cash + Σ(quantity × last_price)`

成本只使淨值下降，cash 不得無故為負。

## 日誌、冪等與斷線接續

每行 record 含連續 seq、event id、kind、時間、payload、前筆 hash 與本筆 SHA-256。寫入使用檔案鎖、flush、fsync。

- 首筆是帶完整 spec 的 `GENESIS`。
- 同 event id、相同內容重送：不新增，狀態不變。
- 同 event id、不同內容：衝突停止。
- seq、prev hash、record hash、JSON 或時間順序不合法：fail closed。
- crash 若發生在落盤後但回應前，相同 event id 重送後 replay，不重複交易或扣費。
- engine schema/version 不符時拒絕重放。

CLI 只讀重放，輸出事件數、最後 session、淨值、現金、持股數、累計成本與 pending 狀態。

## 錯誤處理

拒絕非有限數字、負價格、非法費率、無時區時間、空 id、未知 enum。未知交易狀態保守阻擋；日誌錯誤不自動修補。錯誤時保留原檔，沒有 destructive recovery。

## 驗收

1. 手算買賣案例現金＋持倉守恆，買賣成本各扣一次。
2. 同日訊號不成交；下一個明確可交易 session 才成交。
3. UNKNOWN、停牌、單側阻擋、缺 quote、部分成交與現金不足有測試。
4. tie-break、buffer、頻率與 head mismatch 有測試。
5. split 與 cash dividend 使用顯式事件，價值與入帳符合手算。
6. replay 與逐步執行狀態相同；同事件重送不增加交易或成本。
7. journal 衝突、竄改與截斷均被拒絕。
8. CLI 顯示人類可讀數字。
