# 前瞻 WP5 事故狀態機與 WP2 缺值 oracle 設計（Claude 規劃）

狀態：DESIGN／依 P2 規格（`tasks/spec-forward-p2-primary-20261124.md` 第118、280–290行），不改規格。
起因：WP5 的事故與開盲邏輯經五輪獨立審查，每輪都找到新繞過，其中兩項是前一輪修補造成的回歸（見 `.artifacts/papers/notes/fwd-r{3,4,5}-review*.md`）。逐項打補丁不收斂，改用單一狀態機，並以窮舉比對做驗收。

## 1. 事件分類（直接取自規格錯誤表）

| 類別 | 事件 | 對確認性開盲的影響 |
|---|---|---|
| NO_EFFECT | SOURCE_OPTIONAL、PRIVATE_EXPECTED | 無影響，也不改變任何狀態 |
| RETRYABLE | RETRAIN_TRANSIENT | 僅在同一 R_j 的重試窗內可恢復，見 §2 |
| IRREVERSIBLE | SOURCE_MISSING、SOURCE_COVERAGE、SOURCE_UNIVERSE、SOURCE_INVALID、RETRAIN_IDENTITY、PRIVATE_MARK_EXECUTION、BLIND_INTEGRITY、UNKNOWN_ERROR | 立即且永久失效 |

未列在上表的事件名稱一律視為 UNKNOWN_ERROR。

## 2. 重訓排程 R_j 的狀態（跨 session，不限同一 session）

狀態：`PENDING → (SWITCHED | RETRYING(k) | INVALID)`

- 在 R_j 的 18:00 前通過全部驗收 → `SWITCHED`。
- 發生 RETRAIN_TRANSIENT，且 k < 2 → `RETRYING(k+1)`。於 R_j+k+1 session 的 18:00 重試，必須使用**相同 origin、snapshot、seeds**；任何一項不同即屬 RETRAIN_IDENTITY。
- `RETRYING(k)` 在期限前成功 → `SWITCHED`，整段**無影響**。
- 已過 R_j+2 的 18:00 仍未成功 → `INVALID`。
- 任何 IRREVERSIBLE 事件 → `INVALID`。

## 3. 先記錄、後執行（封住「吞掉例外」類繞過）

1. 每次 attempt 在做**任何**檢查或計算之前，先寫入 `OPEN(attempt_id, R_j, k, request, snapshot, origin, seeds)` 紀錄並封存。
2. 結束時寫入 `CLOSE(attempt_id, outcome)`。outcome 只能取 §1 表中的值，或 `SUCCESS`。
3. 若 `OPEN` 沒有對應的 `CLOSE`（例外被呼叫端吞掉、程序被殺、改名、遺失），一律等同 UNKNOWN_ERROR。
4. 所有 OPEN／CLOSE 紀錄依封存清單納入雜湊鏈。清單外的檔案、改名或缺件 → BLIND_INTEGRITY。

## 4. 單一判定函式

`verdict(sealed_chain) -> ACCEPTED | REJECTED(reason)` 是純函式，只讀封存鏈。`run_attempt`、每日驗收與開盲**都必須呼叫同一個函式**，不得各自實作。

判定 ACCEPTED 須同時滿足：
- 不含任何 IRREVERSIBLE 事件；
- 每個 R_j 的狀態都是 `SWITCHED`；
- 沒有未關閉的 attempt；
- 此研究 identity（規格 SHA、sessions 鏈、external head）尚未開盲過。

唯一開盲的鍵只由研究 identity 決定，路徑、檔名、calendar metadata 都不影響。

## 5. 驗收方式：窮舉比對參考 oracle

- 依 §1–§4 寫一個約 50 行的參考判定函式，直接放在測試檔中，不 import 產品程式。
- 窮舉事件序列，與產品實作逐一比對結果完全相同：
  - 事件字母表：§1 全部事件，加上 SUCCESS、未關閉的 OPEN、deadline 逾時、origin／snapshot／seed 不同；
  - 每個 R_j 最多 4 個事件，最多 2 個 R_j，且跨 session 排列。
- 必須涵蓋的已知反例（五輪審查）：
  - wrapper 重新標記狀態；
  - 複製 store；
  - 改 metadata 後二次開盲；
  - journal 改名；
  - 同 session 與跨日的合法 RETRY 後成功；
  - 吞掉的 RETRAIN_IDENTITY；
  - NO_EFFECT 事件經 `run_attempt` 處理後仍為無影響。

## 6. WP2 缺值 oracle

不再手挑擾動欄位，改用隨機擾動加上固定 seed 集合：
- 對每個決策時點 t，隨機選取 t 之後的格子子集，**涵蓋全部特徵欄與 labels**，雙向擾動：有限值 → NaN、NaN → 有限值、有限值 → 其他有限值；
- 至少包含一組**跨欄聯合**擾動，例如多欄同時改動，以攔截跨欄 availability 差值 mutant；
- 至少 20 個 seeds；
- 檢查 ≤t 的全部 scores、fit_idx、target 與模型輸入雜湊都不變；
- 五輪審查的 WP2 mutant 全部列為必殺測試。
