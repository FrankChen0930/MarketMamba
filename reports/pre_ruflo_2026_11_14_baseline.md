# Pre-Ruflo 開發基線 — 2026-11-14

狀態：本機保存；UNMERGED / NOT_DEPLOYED；遠端依 Human 指示暫停。
此為開發原始碼基線，不是完整 Disaster Recovery。

完整清單：[Manifest](pre_ruflo_2026_11_14_manifest.json)；[憑證唯讀審核](pre_ruflo_2026_11_14_credentials.json)。
source main `4bfaa7b3c7447ea54de67013b8cb3a9aab13003d`；GitHub 公開 main
`50c5b785e8c373913c3301a056497eb955f36681`；本機超前117 commits。
snapshot branch 與 annotated tag 均為 `pre-ruflo-2026-11-14`。
完整 snapshot SHA 由 `git rev-parse pre-ruflo-2026-11-14^{commit}` 解析，並在 commit 後寫入本機 attestation。

77 個原始 branch、4 個既有 tag、31 個未整合 branch 完整列於 manifest。
81 個獨立 `preservation/pre-ruflo-2026-11-14/` refs 額外固定原始 branch、
Windows HEAD 及3個 dirty source snapshots；未合併成果不升格 main 或 production。
未提交原始碼涵蓋兩個 run-status JS 檔與現有 Python tests；staged/unstaged patches
另存本機。env/auth/credentials 未納入 snapshot，原檔不動。
Windows main `8b82a681e43148ff5ee6389c8cb4f6394f5360e5` 乾淨；沒有可見的未同步 tracked/untracked source。
Ignored 資料、env、models、logs 不能由 Git status 推定已備份。

## 憑證判讀與遠端方案

4個不同 HS256 JWT 均已存在於2026-09-22已公開的50c5b785歷史；其中3個仍在公開tip。
有帳號claims且無exp，非明顯placeholder；沒有驗簽／查API，不宣稱目前有效。
Human 說明8/11–12已輪替並清理3個現役程式；未將每個舊token的失效狀態獨立重證。
Windows現役V6/.env JWT與4個舊JWT皆不同；全可達Git歷史精確搜尋0命中。
WSL V6/.env 未找到JWT，不據此推定正式流程不用其他憑證。
本次沒有證據證明新的現役token外洩，不因既有殘留自動重做輪替。
檔案、path aliases、完整含該blob的commit與變更commit、fingerprints見審核JSON；無token或個資值。

最適合先另建私有GitHub封存repository，審核範圍後明確推送保存refs/tag，
保留既有完整SHA與研究authority，不改現有公開repo歷史。
如不接受向GitHub再次傳送未逐一確認失效的舊字串，可先將完整bundle加密，
上傳私有repo的Release並把解密金鑰另存；下載解密後驗hash、refs、smoke。
兩者均可完成真正遠端備份；目前未建立repo、未上傳、未push，須另授權。
公開歷史清理可另案；GitHub建議優先處理仍有效的已曝憑證，單純新commit刪檔不清歷史。
參考：https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository

## 還原

保存容器：`/home/frank/projects/MarketMamba/.artifacts/pre-ruflo-2026-11-14/`。
先依attestation比對bundle SHA256，再在新空目錄還原：

```bash
cd /home/frank/projects/MarketMamba/.artifacts/pre-ruflo-2026-11-14
sha256sum -c SHA256SUMS
git bundle verify pre-ruflo-2026-11-14.bundle
git clone --branch pre-ruflo-2026-11-14 pre-ruflo-2026-11-14.bundle /tmp/marketmamba-restored
git -C /tmp/marketmamba-restored rev-parse 'pre-ruflo-2026-11-14^{commit}'
git -C /tmp/marketmamba-restored fsck --full
```

在還原目錄執行manifest列出的28個Node smoke tests及knowledge health。
其他分支保存在`origin/preservation/pre-ruflo-2026-11-14/<原branch>`，需個別建立工作分支；不整批合併。
不在原工作樹執行reset、checkout、clean；不帶入正式env、Windows資料symlinks或啟動排程。
bundle含歷史敏感殘留，限本機私有保存；不要公開上傳。

## 驗證與限制

初始28/28 Node smoke、5份Python AST、7份依賴檔hash；knowledge health WARNING、無structural FAIL。
commit後的實際bundle／tag／所有refs／clone檔案hash／smoke結果另存attestation.json，避免自指commit SHA。
未重新安裝完整依賴、build、模型推論或訓練，不宣稱完整環境可重現。
1,534個既有local restore檔已讀取SHA256；可讀不等於key集合／source coverage完整或offhost備份。
Data、deliveries、raw/checkpoints/predictions/ledgers、Windows模型及logs、環境與cloud state仍需獨立備份。
manifest固定外部盤點hash；盤點本身不是資料副本。
Data根目錄是WSL實體目錄，4個子目錄symlinks指向Windows；舊prose不可解讀為根目錄symlink。
2026-11-14T14:21:47.3180008Z唯讀Task觀測：fetch仍指向Windows8b82a681；
W19/W36/W38/W39各指向自己的WSL worktree；Ready與NextRun=null不等於已停用。
觀測不授權執行或改Task，亦不證明Task內過往run使用當前磁碟版本。

Ruflo僅能在獨立clone、無正式credentials/data/services與共用Git寫入權限的條件下做原始碼評估；
本機保存容器不得放進它可修改的範圍。尚無offhost復原驗證，建議先完成私有遠端備份再大改。
另在既有外部audit root新增`/mnt/d/Desktop/work/project-audit/market-mamba/pre-ruflo-2026-11-14/`副本；實際hash驗證見attestation。
此namespace避免覆寫舊audit，與專案根隔離，但未證明異機／異碟備份。
本輪停止，不啟動Ruflo或MAS整合。
