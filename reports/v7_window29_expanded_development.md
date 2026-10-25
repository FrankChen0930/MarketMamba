# Window29 — Expanded Cross-sectional Ranking Development Pilot

狀態：COMPLETE / STOP；獨立 Replay PASS／Review GO（Critical0／Required0），UNMERGED / NOT_DEPLOYED。僅 DEVELOPMENT_ONLY，非正式 Data/Research Admission、Independent Test 或投資績效。
研究問題：W24–W28 的正向 Ridge RankIC，是否也出現在事先固定、不同且較大的股票樣本？

## Integration、Cohort 與預先固定

W28 `1d9a2a7a1ee57c455ba58250f63681efbbdd9050` 已由 main `39e2fed…` 本機 ff-only 整合；35 pins/restore、6tests、ReviewGO驗證。
W29 branch `research/v7-window29-expanded-development`，base同固定SHA。Cohort/Recipe/allowlist/隔離測試在新GET前 commit `746ca04`；Panel/Coverage在Fit前 commit `16ea18d`。
未整合 W06/W12/W19。

[固定 Cohort](../research/v7/window29-cohort.json)：**68檔新股票，W24重疊0**。
既有官方18419 issuer/11543 delisting CSV bytes經hash確認；按產業round-robin及固定SHA順序選60檔pre2021 current issuers，另4檔期間上市及4檔期間下市。
沒有使用價格、RankIC、事後Label有效性或目前資本額排序，下載後不更換股票。新上市可能包含轉板，歷史前市場不推定。
4檔下市候選起日／普通股歷史身分UNKNOWN，保留Raw診斷但**不產生合格Predictions**；其餘64檔為conditional候選，不是完整歷史成分股。
Current survivors／current industry構成hindsight與存活者偏差，不能宣稱Historical Universe Verified或產業/市值代表性。

## 真實 API、保存與故障修復

精確FinMind TaiwanStockPrice endpoint，2021-01-01–2023-12-31，每檔一次；**68 GET／0 Metadata**，cache檢查無可重用資料。
全部HTTP200，Raw **8,387,840 bytes**；80total/70price未耗盡，unused已CLOSED。無token/付費/redirect/fallback；未重送成功查詢。
單並發、起始間隔至少5秒；local共享ledger納入既有W21/W22/W24/W26已知請求，另設known-local100/hour保守上限。
其他外部匿名用量UNKNOWN，不宣稱provider-wide餘額已知或有全域排他權；沒有讀取W19 Task/Artifacts。
FinMind非商業／顯名／禁止Raw再散布限制沿用既有W20/W21官方證據；Raw只留private root，不提交Git。

第一筆成功後，本機將immutable snapshot writer誤用於mutable rate/CPU counters，第二筆在GET前停止。
保留原Trace、原計數器與第一筆Raw；以既有E1 atomic_json修復mutable counters，失敗CPU階段保守扣600秒，沒有清零預算。
Trace與code位置證明seq2尚未GET，修復後僅發送其第一次GET及剩餘固定名單，不Retry seq1、不清除任何provider STOP。
[修復證據](../research/v7/window29-source-recovery.json)、[Network](../research/v7/window29-network.json) 保存請求／版本與預算身份。

## Feature-only Universe 與 Label Join 隔離

既有W24 engineering_sample同時檢查未來entry/exit才回傳Features，因此不能直接作as-of prediction介面。
W29只拆出**原兩個公式**到最小research helper，沒有新增Feature、Trainer、Registry或V6修改：
close[t]/close[previous shared session]−1、(high[t]−low[t])/close[t]。

Prediction eligibility只看logical decision時的membership interval、previous/current OHLC與已觀察到的feature價格跳變；不讀entry/exit、持有路徑、Label值或maturity。
保存Feature-only Predictions之後，才進行事後Label Join；2022/2023尾端沒有成熟Label仍保留Predictions。
Synthetic regression證明修改／刪除Label值或整個Label record可以改變評分Coverage，不能改寫既有Prediction keys/scores。

W27 safety保持：不補零/ffill、不壓縮個股缺漏列成交易日、不混Raw/Adjusted、entry=t+1 open／exit=t+h close、exit嚴格早於下一split。
另在metrics前固定2倍/0.5倍相鄰有效觀察close跳變作**疑似**單位轉換marker；不是公司行動證明或依RankIC挑閾值。
本次觸發0，confirmed event0（2321不在新Cohort）。UNKNOWN actions/suspension/availability/vintage風險仍保留。

## Coverage 與模型

48,061 Raw OHLCV rows；44,899 Feature-only rows、4,673 feature rejections。
940核心日期×Horizon的N **58–64**，無低於40；日曆729個TWSE sessions、無未解extra session。Label安全/Missing/Zero/maturity原因詳見[Coverage](../research/v7/window29-coverage.json)。
Prediction models為L0正向Momentum、L0反向Momentum（明列post-W25 development control）、Ridge alpha1。
Ridge各horizon一次fresh fit，**5d Train N14,113／10d13,711**，TRAIN-only scaling，0search/selection/refit。
LightGBM在下載前略過：既有固定recipe/guard僅SYNTHETIC_ENGINEERING_ONLY≤64/toy features，沒有本Panel可直接採用的既定Recipe，不繞過guard。

182,178筆完整Predictions含不可評分尾端；所有模型以相同Feature keys和合格Label Join成Common Support。
每模型共970個日×horizon請求，其中940可計算，30為預先定義tail/maturity（N0）而非降低min40；完整rejection ledger保留原因。

| Period/Horizon | 有效日 | 有效N | L0正 Mean/Median | L0反 Mean/Median | Ridge Mean/Median |
|---|---:|---|---|---|---|
| 2022/5d | 241 | 59–63 | −.010740/−.004514 | .010740/.004514 | .037333/.033014 |
| 2022/10d | 236 | 58–63 | .013257/−.007658 | −.013257/.007658 | .012073/−.004079 |
| 2023/5d | 234 | 60–64 | −.025797/−.027658 | .025797/.027658 | .047138/.050515 |
| 2023/10d | 229 | 59–64 | −.008323/−.001215 | .008323/.001215 | −.008580/−.002417 |

Paired mean Ridge−L0正：.048074／−.001183／.072935／−.000257；Ridge−L0反：.026593／.025330／.021340／−.016903，順序同表。
沒有Bootstrap／LOSO／正式顯著性或勝者宣告。全部2022/2023已曝光，反向Factor沒有回頭改寫W24原方向。
[完整結果](../research/v7/window29-results.json)、[Run/IDs](../research/v7/window29-run.json)。

## 舊新 Cohort 的現象

[只讀舊結果對照](../research/v7/window29-old-new-comparison.json)，沒有重新計算W24/W28 metrics：
Ridge四組W24原mean為 .037016/.043930/.080310/.054056；W28舊Cohortclean-trained為 .034682/.044561/.080407/.054248。
新Cohort5d兩年Mean/Median仍正，2023均值較舊Cohort小；**10d一致正向沒有延續**，2023mean/median轉負，2022median亦負。
不能宣稱原四組皆正已具普遍性。Cohort、來源、mask介面、N門檻不同，此為Development contrast，不是嚴格OOS改善或失敗的因果歸因。

## Replay、資源、保存與下一步

獨立Replay重新由Raw產生Feature rows/Label masks，重算固定模型Predictions／Common Support／Daily與paired，0額外Fit。
私有根 `/home/frank/projects/MarketMamba/.artifacts/window29-expanded-development/`；同機restore `/home/frank/projects/MarketMamba/.artifacts/window29-restore/`。
所有Raw Bytes/manifest/hash與舊W24–W28證據保留；BACKUP_PENDING，不宣稱獨立媒體備份。
研究程序OS CPU limit每stage600秒，總ledger3600秒含60秒verification reserve；真實失敗保守charge600秒保留。新增artifact＋restore由1GiB guard限制，無GC。
外部其他程序CPU或provider匿名用量不在本地全域強制範圍；不能把Goal counter當成本硬上限。
18 targeted tests PASS；Independent Replay 182178 predictions／2910 daily／2910 paired PASS（0fit）。
Review GO：Critical0／Required0，1主要輪／0 review rework；Reviewer獨立驗證原件/restore、prediction keys/support、train-only scaling及無額外solve的normal-equation residual（1.776e−15／2.776e−17）。
299 immutable files／74,612,661 bytes/份還原PASS；artifact+restore149,226,490 bytes，CPU計帳647.740401秒（含failed600）＋60reserve<3600。
[Evidence](../research/v7/window29-evidence.json)、[Review](../research/v7/window29-review.json)、[Resource observation](../research/v7/window29-resource-evidence.json)。

下一個值得申請的研究：保留**既定5d primary／10d secondary**與固定L0±/Ridge，先補強歷史membership/ordinary identity與事件未知政策，
再對事前固定、曝光狀態已查核的新樣本/新時間或prospectively collected資料驗證；不依本輪結果調Alpha/方向或宣稱已選出模型。
需Human另行批准資料/universe/Label與時點用途、Exposure-aware evaluation及精確API/CPU/storage/backup上限。
StrictSTOP／FormalAdmissionNOT_APPROVED／W12SAFETY_HALTED／E1BLOCKED／H2NOT_APPROVED，W19完全隔離。
完成後STOP，不整合W29、不Push/Deploy、不自動開下一窗口。
