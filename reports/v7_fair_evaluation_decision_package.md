# V7 Fair Evaluation — Research Decision Package

**PROPOSED / NO_NEW_EXPERIMENT_APPROVAL / NO_TRAINING**。
Authority：[Human integration/preparation scope](../knowledge/02_History/Decisions/20261021-engineering-integration-fair-preparation.json)。
唯一Backlog：[Research Program](../research/v7/research-program.md#fair-evaluation-preparation-window-04)。
本包延伸[Window03 fairness review](v7_data_evaluation_window03.md)，不重算或重新分類
E5/E1結果；新增data-exposure ledger、樣本與fit-count需求、成本限制與可重用readiness。
15筆固定source identities、算式輸入與驗證見[證據](../research/v7/fair-evaluation-preparation-evidence.json)。

## 1. 目前可以決定什麼

Human可選擇後續要回答哪個問題、採何種公平比較設計，以及是否投入新資料／
訓練。此包沒有代選方案，沒有指定新的final test日期、threshold、target或label。
E1原hypothesis仍為temporal/regime staleness問題；其既有
PASS_REFRESH_HYPOTHESIS仍僅是HISTORICAL_SIMULATION_PROXY。
Strict Phase0 STOP、E1 matrix BLOCKED、H2/portfolio admission未批准。

## 2. 資料角色與歷史曝光帳

| 資料／時段 | 已知用途與角色 | 後續可以／不可以宣稱 |
|---|---|---|
| E5 targets 2013-01-02..2023-11-17 | gradient training；48 features、60-row sequence、固定兩heads/no graph | 不是E5驗證集；依PIT/provenance contract仍需實際資料證據 |
| E5 purge 2023-11-20..12-29 | 30-session target邊界；較早context按既有mask可用 | purge不等於另有untouched test，也不自動認證publication時間 |
| E5 evaluation 2024-01-02..2026-09-11 | split2每epoch validation→best RankIC5／early stopping→best final trainer evaluation | 已參與model selection；不可稱獨立holdout |
| E5 2026 extended/equal-window/head/occlusion診斷 | label extension依horizon成熟度；觀察退化／特徵組／heads，並提供E1研究方向背景 | 已用於研究方案選擇；occlusion不是重訓ablation，不批准刪features |
| E1 targets 2013-01-02..2025-11-18 | 原E5 evaluation的2024–2025部分進入refresh training，這是已批准split變數 | 不能把E5全期間summary與E1 2026直接比較成同support優勢 |
| E1 purge 2025-11-19..12-31 | 固定30sessions，eval從2026-01-02 | 日期與session數由原contract限定，本輪不改 |
| E1 evaluation 2026-01-02起；5d至09-04、10d至08-28 | shared trainer split2 validation／best-IC5 selection／final evaluation | model selection與final dataset重用；10d讀同best-IC5 checkpoint |
| E1 paired postrun 314780 prediction keys | 舊／新scores依共同keys、v2 labels評估；307309匹配、7471 missing label keys | support配對不是selection bias修正；各horizon有效樣本不等於314780 |
| E1 month／leave-one-month-out／epoch diagnostics | 已揭示July敏感性；epoch/regime仍INCONCLUSIVE | 不可看完後再把同區間叫新test；未證因果regime機制 |
| 其他／較後時段 | 是否被Human或別的研究看過沒有完整帳 | 日期較晚不自動等於未曝光；例如09-11之後仍須另核曝光紀錄與資料admission |

模型角色trace固定於9bb4247的`v7_corrected_e5_train.train_seed`：split1用於
loss/backprop；split2每epoch計算RankIC5，供`update_selection`與early stopping；
載入BEST後，`evaluate_model`再次使用同一dataset。E1 runner共用此流程。
所以「validation、model selection、final evaluation」是三個用途，並非三份資料。
E5正式完成17/29；seed43未完成不得納入。E1正式17/29皆best epoch1、last epoch6。

此ledger記錄可取得的程式／contract／diagnostic證據，不推測每位Human曾看過
哪些資料，也不宣稱精確知道所有歷史effective masks。來源缺失維持UNKNOWN。

## 3. 公平比較的方案與資料需求 — 全部PROPOSED

符號：T=training target sessions、V=selection sessions、E=final-evaluation
sessions、G=由批准的label/embargo規則推導之邊界間隔。新T/V/E/G均待Human
指定。既有30-session purge/60-row lookback只是原實驗事實，不自動搬成新方案。
日×股票panel必須以session切分，不能把相鄰股票列當作不同時間。

| 選項 | 需準備的區間／問題 | 優勢 | 限制與必須凍結的選擇 |
|---|---|---|---|
| A Temporal holdout | T→G1→V→G2→E；E在選模型/規則前保留。較早feature context按批准規則提供 | 最清楚分離選模與最後評估，訓練次數較少 | 單一E可能只代表一段市場；E長度/功效未知，反覆查看後修改會耗用holdout。須決定是否選模後refit、selection budget、E存取政策 |
| B Walk-forward | K組只看當時可得資料的T_k/V_k/E_k；選expanding或固定長度rolling；E_k的重疊與聚合事先指定 | 比較refresh在多時段的表現／穩定性 | K次重訓、overlap使fold不獨立，不能假定√K改善不確定性；觀察早fold後改策略將改變研究問題 |
| C Nested temporal | K個外層E_k只評估整個選模程序；各外層training部分再有J個時間順序inner selection folds | 適合比較多個設定或模型搜尋程序 | 訓練最多、早期可用樣本較少；outer不能回饋inner設計。須固定候選集合、inner budget及refit的epoch規則 |
| D 固定既有checkpoint後續觀測 | 不再挑checkpoint，對新admitted時段按預凍結方法觀測 | 可不新訓練；回答固定模型後續表現 | 不能回答重新訓練架構優劣；仍須新資料/inference/observation批准，不能自動部署或排程 |

A的總target跨度示意為T+G1+V+G2+E；可另外需要較早feature context與最後label
成熟時間。B/C需逐fold列出日期、交易session與每horizon有效labels，不能用
股票列數充當獨立時間樣本。原程式允許不足lookback的mask/padding；本包不
偷偷新增「必須滿60天」門檻。缺矩陣與新區間資料，無法報可信的power／最小E
數字，故sample-size sufficiency=UNKNOWN，需設計及資料盤點後才可決定。

## 4. Fit次數與GPU／Colab成本

M=要訓練比較的固定model recipes數（含incumbent），S=每recipe seeds數，
C=每recipe候選設定數（不搜尋時1），K=外層/forward folds，J=inner folds，
R=每recipe選模後額外refit次數（0或1須Human決定；通常nested需明確refit政策）。
既有17/29只屬原實驗；以下S=2例子不是新seed批准。

| 方案 | 訓練fit數，含incumbent | 純算術例子，不是批准的配置 |
|---|---|---|
| A | M×S×(C+R) | M2/S2/C1/R0 →4 fits |
| B | M×S×K×(C+R) | 同上，K3 →12 fits |
| C | M×S×K×(C×J+R) | M2/S2/K3/C2/J3/R1 →84 fits |
| D | 0，僅既有checkpoint reuse | 仍有inference／資料驗證／儲存成本 |

這些式子假設各recipe/fold獨立訓練；不把不相容的歷史checkpoint當免費refit。
若某baseline可合法reuse，須在批准設計中逐項扣除並證明其資料／selection
身份公平。Search早停、失敗重跑、更多seeds與額外diagnostics皆會增加工作量。

可用歷史觀察是`ef020e5:.../epoch-evaluation.json`中的每seed六個、共12筆E1 epoch紀錄，
本輪只加總JSON telemetry，沒有重新訓練或評估：

| Seed | Train秒 | Validation秒 | Checkpoint秒 | Logged epoch合計 |
|---|---:|---:|---:|---:|
| 17 | 4088.886 | 220.113 | 2320.055 | 6629.054秒 ≈1.8414小時 |
| 29 | 5661.857 | 270.083 | 2551.499 | 8483.439秒 ≈2.3565小時 |

此為歷史局部wall-time，**不是GPU billed hours、完整run時間或新fit預測區間**；
matrix build、startup/resume、final inference、transfer、GPU allocation與Colab
quota成本不在這個合計中或不明。不能把其他E5報告的6.94/7.82小時混成E1數字。
原recipe最高20epochs，而歷史E1停在6；新研究可能不會同樣早停。

成本規劃式：`H = Σ t_fit(model, seed, fold, config) + H_matrix + H_final_eval
+ H_transfer/startup + H_failure/retry`。`t_fit`需同硬體/precision/資料規模的
實測或另獲准benchmark；含checkpoint I/O時不能再重複加總。
若用付費資源，貨幣成本=`billable_gpu_hours × 當時批准單價 + storage/transfer`；
Colab compute units及可取得GPU型號目前UNKNOWN。**未查價、未購買、未訂資源；
本包不提供虛假的固定金額或工時承諾。** 任何benchmark/訓練也須另授權。

## 5. 每種設計都必須解決的偏誤

- 時間洩漏：以session和label entry/exit/maturity定義邊界；不允許future
  targets進入梯度/選模；past feature context與future label要分開判斷。
- PIT：publication/as-of、revision、universe membership、feature轉換都要
  有資格證據；purge與byte hash無法替代PIT admission。
- Selection bias：A只能在V選，B只能在每fold當時可得的inner資料選，C把
  整個選模程序鎖在inner；D不得重新挑最佳既有checkpoint。
- Test reuse：記錄看過的時段/metrics與決策；若據outer/E結果改模型或規則，
  必須記為探索，不能再稱同E是獨立confirmation。
- 公平support：incumbent/challenger共用有明確identity的dates/keys/有效labels，
  缺值、成本、ranking/portfolio admission不能由model score默認補出。
- 種子及不確定性：固定seed政策、失敗/缺seed處理、多重比較budget與fold
  aggregate/CI方法；不得在看到結果後保留好seed或調threshold。

有限樣本selection criterion也可能過擬合，performance comparison因此可能
偏向較易被選中的程序；此為一般方法限制，不是本輪量化E1 bias。
[JMLR作者摘要](https://www.jmlr.org/papers/v11/cawley10a.html)。
TimeSeriesSplit提供時間順序/gap概念，fold時長可比仍需等間距樣本；直接對
panel row使用gap不等於交易session purge。[官方文件](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)。

## 6. 缺少Frozen Matrix的明確限制

可支持：已固定result檔完整性、原contract、metadata身份鏈、原已admit結論、
目前source的selection/evaluation reuse、歷史logged telemetry摘要。
不可新增認證：原array bytes、實際完整split／feature值、逐epoch同support
replay、safe checkpoint embedded identity、bias大小、因果regime解釋或新設計power。
E1 frozen matrix仍a590b280…947f62 BLOCKED。epochs2–4未保留也是獨立依賴；
即使找回matrix也不自動補齊每epoch replay。不得生成替代矩陣或猜新位置重搜。

## 7. CPU readiness與Colab停止條件

已完成可重用命令（repo root；不會訓練、推論、讀payload或寫artifact）：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
/home/frank/projects/MarketMamba/.worktrees/v7-e1-colab-recovery/.venv/bin/python \
-B -m V6.experimental.v7_e1_readiness
```

固定eec63c5 manifest hash先驗，逐檔驗7 copied artifacts，再用既有validators
核48-feature/config、split、seeds17/29、result→best pointer、frozen acceptance
metadata。合法時status=VERIFIED_METADATA_ONLY、execution_ready=false；篡改／
缺檔／candidate split或feature-order漂移會拒絕。不是一般新實驗admission工具。

| 時點 | 可執行準備／需具備的條件 | 本輪狀態與停止點 |
|---|---|---|
| 研究批准前 | Decision package、metadata audit、synthetic fail-closed tests | COMPLETED；所有新選項PROPOSED |
| Colab啟動前 | Human凍結比較規格與compute；original/admitted matrix及data/feature/split身份；可信checkpoint payload政策 | BLOCKED／未批准；任一缺失即STOP，不以metadata PASS放行 |
| Bundle/environment | exact requirements lock與source/config identity、safe extract、逐檔hash；獨立runtime驗證、durable空間與namespace | lock沿用；本輪沒有生成新bundle或驗證GPU/runtime，不能宣稱launch ready |
| Resume前 | contract/matrix/seed一致、latest/previous/best關係、可信loader、corrupt/missing/orphan policy | fixtures已有回歸；歷史payload未載入。validated orphan不能自行升best |
| 訓練結束後 | terminal完成紀錄＋BEST selection身份、predictions/labels common support、maturity、hash/provenance、durable logs/eval reports | 僅準備流程；不得只看檔名存在或放寬gate |
| 離線評估／診斷 | 批准的frozen evaluator、paired support、selection/secondary metric、不改原threshold；postrun month/seed/support診斷分開證據 | 原de2b62e evaluator與ef020e5 diagnostic保持branch-qualified，不執行/rewrite；新設計先Human批准 |

Metadata、checkpoint或matrix identity mismatch、缺seed/有效label support、NaN gate
值、environment/bundle差異、unknown publication/universe、未批准新方案均不得
變成成功。CPU檢查只核其宣告範圍；不能由它解除StrictSTOP/H2或GPU授權。

## 8. Human下一次最小決策

1. 選擇要回答「固定模型未來表現」、「refresh穩定性」或「多候選模型搜尋程序」；
   對應D/A、B、C只是適配關係，不是Agent選定。
2. 凍結T/V/E或fold邊界、資料admission、baseline/feature/label/seed政策、selection
   與refit規則、有效support、uncertainty/acceptance；所有新日期與threshold仍待定。
3. 決定是否投入original-matrix recovery或新admitted資料、是否批准compute及其
   上限；未批准前不啟動任何訓練/大型推論/Colab。

MOPS月營收完整性與H2另有domain決策，不在本包靜默解決。Current State與唯一
Backlog記錄本輪完成及恢復位置；此包不是新Roadmap或新Project Memory。

## 9. 本輪完成／恢復 checkpoint

WINDOW_COMPLETE / STOP / UNMERGED / NOT_DEPLOYED（僅指本輪新增工具與決策包）。
Window02/03已整合main；新readiness工具commit2a9de32；本包與closure在包含
本段的commit。64項相關非訓練tests PASS，independent readiness/decision review
均0 Critical/Required；15 source hashes與算式核對通過。修復輪數：audit1，
決策包0。Knowledge health WARNING無FAIL，不提升研究／runtime authority。

新Session進入 `.worktrees/v7-fair-evaluation-decision`，核branch/HEAD/clean，
再讀Current_State、Authority_Map、唯一Program的Window04段、此包與evidence。
Main導航將pin closure SHA；不需重新盤歷史或重搜原matrix。新執行需新的有界
授權；本包未選定方案，沒有未經批准可直接啟動的研究／訓練工作。
