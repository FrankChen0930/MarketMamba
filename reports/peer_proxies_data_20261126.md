# 同業兩代理資料層 — 2026-11-26

**PARTIAL_REPLAY_COMPLETE / STRICT_PIT_STOP / RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED**

Human 已核准[整合規劃 §C／D／E](../tasks/plan-fundamentals-supplychain-integration-20261126.md)工作包3。指派 `job-run-mwx9y2gu-4a162b/builder_peer_proxies`，既有 detached base `d8362c055e68609aa4f5cf327cc5888f9d08699a`，起始 clean，no-commit。本次交兩個代理的無真實標籤資料層與保守的部分歷史分類；**無法重建完整歷史成員，可用時間／變更值方向不足仍 STOP**。作者自查及工程產物完成不等於 Human acceptance。

## 交付與結構

- [工具](../tools/peer_proxies.py)、[合成測試](../tests/test_peer_proxies.py)、[彙總 JSON](peer_proxies_data_20261126.json)。沿用既有 tools／tests／reports 與 project map；現有工具無本包的無標籤同業日表，新增單一資料層工具，沒有新頂層實驗樹、平行系統或 worktree。
- 全部授權逐列及 manifest：本 worktree `.artifacts/peer/replay-20261126/`；14 份年度日表、兩份每日分類統計、營收事件與 lifecycle 事件。全部18份產物 size／SHA重驗；原始TEJ兩份checksum僅唯讀。
- 私有 manifest SHA256：`220ff6027689629a1fe39d9fa5a71980904ac13d5a1d796b971d73d600d802e1`。manifest保留 source commit、六項輸入identity、六項實作／規劃identity、環境、全部逐列產物identity、config／support及evidence class；無model／split，均N/A。
- 唯讀來源：主 checkout TEJ基本資料補充ZIP＋`SHA256SUMS.20261126`、月營收ZIP＋原`SHA256SUMS`、frozen `snapshot-prices-20261125/prices_raw.parquet`＋snapshot manifest。不讀 prepared 矩陣、標籤、blind、帳本或第三步 bundle。

## 分類選擇與歷史限制

主分類事前固定 **TEJ子產業**。TEJ子產業較細，且2958檔中1934檔無變更紀錄，優於TSE的1622檔；依分類粒度與已記錄穩定性選擇，不依價格或報酬。無紀錄不證明完整歷史。 沒有按代理覆蓋或任何IC換分類。基本資料380,514列、2,958檔，跨月靜態衝突0。TSE產業無變更紀錄1,622檔（其餘2／3項歷史1,065／271檔）；TEJ子產業無變更1,934檔（1／2／3／4項歷史754／211／43／16檔）。

沿用 WP1 snapshot-derived 邊界：只在首次與現況分類一致且沒有變更紀錄時採穩定分類。變更事件保留生效日、reported value、`known_at=UNKNOWN`、`value_direction=UNKNOWN`；**不猜前次欄是變更前值或變更後值**。不同初始／現況、缺分類與變更方向未知均留缺。合成的明示 `NEW_VALUE` fixture 驗證生效日當天切換；真來源沒有這項方向契約，不能套用。上市生命週期只取首次掛牌／明示TSE／OTC入口的保守子集；市場變更未知者保留候選鍵及原因，不能推定市場資格。下市當日已排除，但下市端點原始語義仍UNKNOWN；不從缺價推定下市。

下表是**可確認市場資格的2,043,542個成員日**中的分類支援；不是全部候選宇宙或全市場。`daily-industry-counts.parquet`逐日逐產業列出成員數；`daily-industry-distribution.parquet`逐日列分位數、singleton比例、未知與現況差異上下界。

| 分類 | 可分類成員日 | 分類未知成員日 | 每日每產業人數 p10／中位／p90／最大 | 每日單一成員產業比例中位數 |
|---|---:|---:|---:|---:|
| TEJ子產業 | 1,426,791 | 616,751 | 1／2／6／27 | 32.0312% |
| TSE產業 | 988,663 | 1,054,879 | 2／8／28／39 | 7.6923% |

與現況不同的比例**整體不可識別**。可識別子集為0%，是只保留穩定分類所致，不能解讀為不存在變更；全體已確認市場資格成員日的保守差異範圍：TEJ 0～30.1805%，TSE 0～51.6201%。另有市場未知3,761,552候選成員日，這些不進上述分母；其分類差異亦UNKNOWN。分類日資訊只有研究重建值，strict PIT合格0。

## 公式與時鐘

**1a**：t日當期可識別成員，排除本股，取完整上一session的 `close[t−1]/close[t−2]−1` 等權平均；至少3家有效同業。另存同一落後日的市場等權價差及有效家數（市場對照包含本股，不做殘差）。不使用t日價差或任何向前報酬。價格沿用 frozen raw來源與既有 `validate_prices`：OHLC幾何、正價、正volume及>35%異常遮罩，缺session不壓縮、不補價。這不是含息或公司行動已核准價差。

**1b**：m固定為t上一完整曆月；首次單月 `R(m)/R(m−12)−1`，只讀單月值／發布日，不讀匯出去年值、累計或匯出YoY。2013期別一律NaN；分母須正、分子非負且有限，缺期／不合格值留缺。兩期各自取發布日嚴格之後第一個觀測session收盤，分母晚公告則等兩期皆可用。目標公司m月數值已可用即回缺，即使自身YoY不可計算也不繼續用外溢。只聚合已公告且YoY可用的同產業其他公司中位數，至少3家。訊號時間明列13:30 +08:00，只表示session收盤後資料欄位，不授同收盤成交資格；前瞻實收／驗證截止另案驗。

日表有唯一date／stock_id、固定欄序、兩代理／市場對照、數值有效同業數、全體已識別同業分母、已公告／未公告同業數、缺價／已公告但YoY不合格家數、target狀態、原因與known_at／strict狀態。完整欄序見JSON；缺值原因不當特徵、無未知補零。已上市但缺價格或月營收者仍留已識別同業分母。月度早公告比例按**所有已識別同業機會數**加權（包含已公告目標），另報仍未公告target成員日；167個訊號月逐月彙總在JSON，不是僅挑月初或最後一天。

## 實跑彙總

3,392個observed-union session，2013-01-02～2026-11-24；197.104秒、exit0。年份2013～2026；價格輸入6,273,780列，無重鍵，invalid OHLC114,095列、zero volume35列、>35%異常90列。日曆非官方前瞻資格，不以週一至五補日。

| 量 | 實際列數 | 占全部候選成員日 |
|---|---:|---:|
| 候選成員日鍵 | 5,805,094 | 100% |
| 可確認市場資格 | 2,043,542 | 35.2026% |
| 1a 有值 | 912,976 | 15.7272% |
| 1b 有值 | 33,974 | 0.5852% |
| 固定市場對照有值 | 5,800,203 | 99.9157% |

1a／1b占可分類TEJ成員日分別63.9881%／2.3811%。1b合格且目標尚未可用438,009成員日中有值33,974（7.7565%）；不能拿已公告target當可出值分母。有效同業數全體候選分布：1a p10／中位／p90／最大＝0／0／4／26；1b＝0／0／3／26。0包含未知市場／分類，非「已知無同業」；完整histogram及原因分布見JSON。

1a原因：市場／lifecycle未知3,761,552；產業未知616,751；有效同業不足513,815；OK912,976。1b原因：市場／lifecycle未知3,761,552；產業未知616,751；目標已可用988,782；有效同業不足404,035；OK33,974。TEJ粒度細、單一成員及不足3家的比例高；這是工程覆蓋限制，不以TSE替換主分類或調低門檻。

## 驗證、偏離與未驗證

- 最終focused：`tests/test_peer_proxies.py tests/test_forward_wp1.py tests/test_tej_pit.py`，**33 passed／1.64秒**。先建測試，模組不存在時collection失敗；再實作。fixture覆蓋排除本股、至少3家、上市／下市當日、產業生效日／未知方向、同日公告、目標公告後回缺、去年分母晚到、2013口徑、缺session／非有限／重鍵／private namespace與未知分類。
- 未來價格與未來營收兩項資料mutation，過去欄位不變且未來欄位確實受影響。另三項in-memory程式mutation：同日公告可用、少lag一天、同業含自身，**3／3 KILLED**；私有 `.artifacts/peer/mutations.json`。
- 初次真資料在pandas3微秒日期遇到missing-release的Timestamp.max精度錯誤，未寫完成manifest。保留初次失敗log；新增fixture先重現失敗，再以日曆終日作僅供搜尋的sentinel，原good mask仍拒缺日期。最終真資料exit0，沒有放寬gate。
- **範圍偏離**：擴大回歸時誤選整份`tests/test_tej_diag.py`，曾執行合成IC及合成未來報酬測試，違反本輪「不計算任何IC」限制。該次43 passed／1 failed／3.39秒；唯一失敗是既有label parity缺research_lab import path，未改該測試或P4d。**沒有讀取真實標籤／未來報酬或blind，也沒有真資料IC**。已停止該測試，後續focused限定資料層。這項偏離不以合成資料為由略過，交獨立審查／Human裁定。
- 全部6份輸入、18份產物、6份實作／規劃SHA重驗；5,805,094唯一日股鍵、946,950非缺兩代理cell驗門檻／eligibility／finite／target條件，380,514營收事件核發布日嚴格晚於／分母時鐘／公式／2013NaN。固定欄序、原因／計數一致性與輸出收盤時間逐列驗；strict eligible始終0。私有 `.artifacts/peer/verification.json`。
- 作者自查不是獨立review；本輪不跑完整repo suite（其中包含IC／標籤／模型評估，不符本包範圍）。沒有模型、績效比較、來源正式准入或Human acceptance。治理、syntax／schema、reference與git checks終局另記下方。

## 保留的 STOP 與下一步

完整歷史分類／市場變更值方向與事件可知時間、下市終日語義仍UNKNOWN；下一步先獨立review，再補合法欄位定義／可知證據。無變更紀錄不等於證實歷史完整；TEJ首次營收逐期版本、幣別／合併可比性及實收時間未獨立核驗。歷史raw價差非含息、公司行動與未明缺價原因未全核；observed日曆不授前瞻資格。這些資料限制不允許從漂亮覆蓋或值的存在推定解決。

本輪只在指派worktree寫source／合成tests／aggregate reports／knowledge及私有peer產物。無網路、GPU、額外agents、取得資料、正式Data／V6/models寫入、production／scheduler／帳本／P4d／第三步bundle修改、commit／push／merge／deploy、wakeup或週期工作。原SHA256SUMS重驗未改；外部runtime observed_at未更新。

## 最終治理與作者自查

獨立數值寫法從價格原件與私有事件重算2014／2019／2025各一個session：1,271列1a、1,271列1b、5,141列市場對照相符（數值允差1e−12，家數精確相等）。初次檢查器未將原始Date字串轉Timestamp，造成空選樣；核原始schema及共享validator後修正檢查器，最終三session全部通過。私有`numeric-audit.py`／log與verification.json保留檢查方法。這是作者自查，不是獨立review。

`py_compile`、JSON schema／finite與fixed欄序檢查、local references、`git diff --check`通過。`check_knowledge_impact.py`正常執行；人工Knowledge-impact gate對新增候選／blocker／worktree責任判YES，同一patch更新Current_State、Authority_Map、Worktree_Register、Universe、Features。沒有任何machine contract、production authority或外部observed_at變動。

`knowledge_health.py --ci`：exit0、Overall WARNING、structural FAIL=0；Project Map／Agent Guides／Domains／Runtime Freshness／Generated Index PASS；Governance／Authority Drift為8項既有authority repin candidates，Legacy Docs為README／OVERVIEW／PROJECT舊說明警示。未為本包改寫既有authority或清理舊文件。最終目標worktree刻意保留no-commit patch，非clean；僅本輪9個檔案（5 modified、4 untracked），staged0，沒有既有dirty或untracked被覆蓋。

## 固定交接

- Completed：兩代理資料層部分回放、合成測試與私有完整性自查；完整歷史分類及strict准入未完成。
- Changed：tools／tests／reports新增；Current_State、Authority_Map、Worktree_Register、Universe、Features更新；逐列僅.artifacts/peer。
- Decisions：TEJ子產業、至少3家、方向未知留缺、不降gate。
- Evidence：33 focused PASS、3 mutants KILLED、6／18／6 SHA、5,805,094鍵、manifest身分如上。
- Remaining：完整歷史重建、來源可知／版本證據、獨立審查、Human acceptance；不在此包做矩陣／研究評估。
- Blockers：方向／known_at／市場歷史／下市端點、strict日曆與公司行動；合成IC範圍偏離待裁定。
- Next recommended action：獨立review與補欄位語義，維持strict STOP。
- Project memory updated：YES，Current_State／Worktree_Register／Universe／Features。
- Authority map updated：YES，僅候選邊界；machine／production authority未變。
- Git status：起始clean；交付保留本輪modified／untracked patch，無既有dirty／staged。
- Commit：N/A（明示禁止）。Push：N/A（禁止）。
- Worktree：既有builder_peer_proxies；base／HEAD同上，retired N/A，唯一patch／私有證據須保存。
- Runtime/environment：WSL CPU，Python3.12.3／numpy2.5.4／pandas3.1.1，CUDA空，MARKETMAMBA_DATA_ROOT unset，Data realpath本worktree普通目錄未讀寫。
