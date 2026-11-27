# 第五步 5a：月營收表示精煉工程

**RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED。** 僅無標籤 CPU 工程；完成不等於 Human acceptance。實際 admission 結果見下方。fit＝0、IC＝0、評估標籤統計＝0。

## 凍結、重用與安全邊界

[規格](../tasks/spec-step5-revenue-refinement-h20-20261127.md) SHA `977ec654dd8018d9524259141e8b9d0b43d476b220ec289ef5fdfa960012d78c`；[凍結紀錄](../tasks/decision-step5-freeze-20261127.md) SHA `bb0e66fcad32e65a52874ab5458b41c29ff01ae6cb0db605b959db23d78224c2`。末尾 Human 凍結紀錄及本輪 EXECUTE_APPROVED 控制執行；保留規格前段起草時的文字。R-A：A＝第三步B的190欄，B追加四欄成194欄，12折×3 seeds×2臂＝72 logical jobs，歷史m＝1。前瞻另受 fixed-sequence gatekeeping，未在本輪收集。

既有 tools／tests／reports／knowledge 足以承載。原第三／四步 bundle 無法承載新的四欄及契約 identity，因此只在既有 `.artifacts` 下建立獨立第五步 namespace。第三步B的矩陣、抽樣、target、推論鍵、成熟標籤原樣複製，未讀模型／分數。標籤只以 bytes 複製與SHA驗證，不 `np.load`；不重新抽樣、排名或刪列。共用190欄只存一份，各臂以references組裝，新四欄另存。最大化沿用第四步的 axes／key_rows／baseline order、component bundle與admission模式，未改 tools/step3*.py 或 tools/step4*.py。

新檔：[step5.py](../tools/step5.py)、[step5_features.py](../tools/step5_features.py)、[step5_prepare.py](../tools/step5_prepare.py)、[step5_audit.py](../tools/step5_audit.py)、[test_step5.py](../tests/test_step5.py)。5b驅動不得以step5開頭；本輪沒有runner、fit或evaluation入口。

## 特徵與無標籤稽核

重用第三步 admitted_events 的受限歷史單位／範圍容差及精確來源證據；重播時僅2013/01起首次單月值，發布日嚴格之後第一個凍結session收盤可用。相同session依發布日／年月／版本／source row穩定排序並全部套用才計算。相同版衝突或多版無首次證據STOP；未知identity不互認可比；最新月缺值不退回舊月。audit從凍結的normalized monthly及cross-source pairs重建admitted inputs，再重播全部事件及矩陣鍵。

- age：最新合格月首次可用session起算，當日0；逐完整日曆session遞增；晚到舊月或分母不重設。
- acceleration：當月YoY減前連續三個月YoY算術平均；四筆有任一缺值即NaN，不跳月。
- signed streak：須追到零或反號才確定完整段長；中間缺月／不可比或歷史起點未見段首則NaN；零為0，不cap。
- seasonal MoM：當月MoM減前三個曆年同月份合格MoM的中位數，至少一筆；不向更早年補位。

每個新欄各折各seed fit／eval須≥2個有限且min＜max；audit記finite、NaN比例、population variance、variable，另列c左設限比例與e支援數0–3／UNKNOWN。支援數與左設限旗標不進模型。當月YoY本身未知時，c列為缺值；`streak_left_censored`旗標特指已知非零當月YoY但無法追到段首。缺值不補零、不改抽樣鍵。共同鍵、來源B byte identity、warmup及完整日曆purge `t+21<origin`均獨立核對；未來擾動重播不改過去。

## 執行與驗證

主checkout絕對輸出：`/home/frank/projects/MarketMamba/.artifacts/step5/features/`、`/home/frank/projects/MarketMamba/.artifacts/step5/bundle/`。新bundle的member路徑均相對於bundle，不寫成worktree artifact路徑；來源commit與程式snapshot仍保留真實開發identity。40,000,000,000 bytes硬gate及10%空間餘裕；超額STOP。source manifest與逐檔size／SHA、calendar／matrix／split／environment identities保留於私有manifest。僅目錄形式component bundle；沒有ZIP或解包動作，member路徑拒絕absolute／traversal／symlink。

首次普通sandbox執行因主checkout唯讀而停止，未產出。按本輪既有核准取得該指定輸出的sandbox escalation後實跑。首次寫manifest遇 `dict(path=..., **info)` 重複path參數，留下四個完整資料檔但無manifest，未視為完成。修正為明確path覆寫後，以 `--resume-verified-partial` 精確inventory驗四檔，重建來源與事件、比較calendar及全部7,423,927列block才新增manifest；未覆寫partial。partial驗證失敗或inventory不符即STOP。後續audit再次重播。

實際特徵：7,423,927列、354,866事件；estimated bytes＝434,588,316；features manifest SHA `86b34ca753188954fcce95b327d4976c2192baa29f44e491ab99782900ba60d3`。features manifest記 `recovered_partial=true`。

執行命令（所有外層shell輸出以 `2>&1 | head -c 3000` 限制）：

```text
PYTHONPATH=V6:V6/research_lab/tests:. CUDA_VISIBLE_DEVICES='' /home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python -m pytest -q tests/test_step5.py tests/test_step4.py tests/test_step3.py
PYTHONPATH=V6:V6/research_lab/tests:. CUDA_VISIBLE_DEVICES='' /home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python -u -m tools.step5_features --out /home/frank/projects/MarketMamba/.artifacts/step5/features --resume-verified-partial
PYTHONPATH=V6:V6/research_lab/tests:. CUDA_VISIBLE_DEVICES='' /home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python -u -m tools.step5_prepare --out /home/frank/projects/MarketMamba/.artifacts/step5/bundle
PYTHONPATH=V6:V6/research_lab/tests:. CUDA_VISIBLE_DEVICES='' /home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python -u -m tools.step5_audit --bundle-dir /home/frank/projects/MarketMamba/.artifacts/step5/bundle
```

初始RED：test_step5收集因尚無step5模組失敗。修正後指定回歸94 passed；包含四欄公式、age歸零／日增／晚到、零／負／左設限、季節中位數1／3支援及固定窗口、未知最新月／identity、發布當日／週末、亂序、版本衝突、未來不變性、matrix190／194、支援與磁碟STOP，兩項mutation（age歸零、左設限補零）均被oracle殺死。標準回歸含第三步既有小型合成fit；真資料fit仍0。

Prepare preflight：estimated bytes＝19,338,045,405（19.34 GB），limit＝40,000,000,000，free＝781,946,695,680，10%餘裕通過；未放寬gate。Prepare完成：72 jobs、`PREPARED_NOT_ADMITTED`、381.659680秒、peak RSS 1,360,024 KiB、member artifact bytes＝19,184,359,777；含preflight及bundle manifest的實際目錄bytes＝19,184,512,688。bundle SHA `5f2f4f2311aad64ed4ac13fdafa66c3461e21871e5c6c12393fddf996e24c30e`。

最後重跑指定回歸：**94 passed in 121.53s**；syntax compile及`git diff --check`通過，受保護第三／四步與Data／V6/models無diff。Audit完成：**ADMIT（僅歷史無標籤工程）**，stops＝[]，72 jobs／48獨立partitions／283個unique files，全量7,423,927列；最後signal＝2026-09-11。耗時220.048104秒，peak RSS 1,902,424 KiB。admission SHA `f4e495f84882e29a7c67c9d251e1f64e72979b738fdc3b45e4bfa41e62af03b4`。未來重播cutoff＝2015-11-11 00:00:00，比較65,205個事件，differences＝0。

Knowledge-impact checker涵蓋五個新程式／測試檔：PASS；涵蓋本報告及三份knowledge後為WARNING（canonical-knowledge要求review GOVERNANCE.md）。已人工核閱既有artifact分類／lifecycle與completion sequence，本輪沿用既有GENERATED研究產物規則，無須修改治理規則；current-state／下一步／實作導航有影響，已更新三份canonical knowledge。`knowledge_health.py --ci`：Project Map／Agent Guides／Domains／Runtime Freshness／Generated Index PASS；Governance／Authority Drift／Legacy Docs WARNING，Overall WARNING，無structural FAIL。既有warning未在本輪擴大修理。Runtime版本：Python3.12.3、NumPy2.5.4、pandas3.1.1。


逐折無標籤支援摘要：a,b,c,e依凍結欄序；fit欄為三seed的最小有限數／最小variance，eval為共用partition。完整48 partitions逐欄finite／NaN／variance／c左設限／e的0–3及UNKNOWN分布保存在私有admission.json，以下不載入任何標籤。

| 年折 | phase | rows | finite(a,b,c,e) | variance(a,b,c,e) | c左設限比例範圍 | e支援0/1/2/3（seed17或共用eval） |
|---|---|---:|---|---|---|---|
| 2015 | fit | 500000 | 473202/154600/128105/199642 | 36.248/3548.9/8.8663/25484 | 0.184738–0.184832 | 272970/200320/0/0 |
| 2015 | eval | 312782 | 312647/308970/285574/310408 | 37.377/28553/32.223/50731 | 0.079723–0.079723 | 1873/54223/256686/0 |
| 2016 | fit | 500000 | 482434/272596/240683/302823 | 36.592/5827.8/22.907/37316 | 0.151630–0.151868 | 179124/175631/127865/0 |
| 2016 | eval | 313060 | 312772/308704/299873/310237 | 36.527/54016/56.689/1.9598e+05 | 0.033345–0.033345 | 2481/5173/57310/248096 |
| 2017 | fit | 500000 | 486788/330290/302427/353052 | 36.563/30098/36.573/69526 | 0.121346–0.121498 | 133211/132032/129088/92729 |
| 2017 | eval | 332634 | 332465/328486/325124/330168 | 36.557/11369/53.946/3135.1 | 0.015272–0.015272 | 2123/4625/8876/317010 |
| 2018 | fit | 500000 | 489599/365362/342542/383999 | 36.652/25863/41.884/63430 | 0.098824–0.099110 | 105103/105477/103883/175370 |
| 2018 | eval | 336574 | 336454/332655/330965/333441 | 37.876/27265/66.003/10742 | 0.008759–0.008759 | 2300/4014/7011/323249 |
| 2019 | fit | 500000 | 491439/388251/369329/403961 | 36.837/20459/48.434/47948 | 0.082168–0.082528 | 86861/87520/87008/230275 |
| 2019 | eval | 337437 | 337393/332927/330570/334731 | 36.002/14729/72.798/490.31 | 0.012767–0.012767 | 1850/4962/6749/323876 |
| 2020 | fit | 500000 | 492760/404486/387642/418171 | 36.659/22948/52.985/35717 | 0.072320–0.072546 | 73885/75508/75213/268362 |
| 2020 | eval | 360897 | 360469/357080/355258/358478 | 36.697/37348/60.063/43510 | 0.008725–0.008725 | 1633/2937/7129/349198 |
| 2021 | fit | 500000 | 493708/416891/401949/428971 | 36.614/21793/54.388/28589 | 0.062944–0.063984 | 63654/65450/66610/298235 |
| 2021 | eval | 378547 | 378303/374297/372183/376764 | 37.105/92712/61.023/2.5314e+05 | 0.010894–0.010894 | 1359/2806/6300/368082 |
| 2022 | fit | 500000 | 494472/426947/413539/437930 | 36.666/39448/56.203/63711 | 0.056668–0.057258 | 55817/57937/59534/321407 |
| 2022 | eval | 385419 | 385013/382219/379636/382821 | 36.563/8.9491e+05/84.596/7.8937e+05 | 0.011377–0.011377 | 1003/2937/5544/375935 |
| 2023 | fit | 500000 | 494974/434670/422437/444527 | 36.615/1.1377e+05/60.826/89672 | 0.051554–0.052036 | 49675/51888/53264/340451 |
| 2023 | eval | 390606 | 388857/382621/380966/386382 | 35.533/87295/81.506/32545 | 0.013684–0.013684 | 2613/2652/8254/377087 |
| 2024 | fit | 500000 | 495341/440348/429141/449682 | 36.54/1.1675e+05/63.571/1.547e+05 | 0.047596–0.048576 | 45012/46881/48665/355232 |
| 2024 | eval | 418138 | 417128/401553/399088/408492 | 35.393/1.0199e+05/71.378/2.106e+05 | 0.019379–0.019379 | 8304/2886/8446/398502 |
| 2025 | fit | 500000 | 495609/444010/433593/453199 | 36.451/1.0646e+05/64.257/1.3544e+05 | 0.045122–0.045300 | 41761/42454/44935/367001 |
| 2025 | eval | 425682 | 423717/403722/400664/410889 | 36.379/3.3761e+06/60.541/2.5641e+05 | 0.027260–0.027260 | 13058/10635/8166/393823 |
| 2026 | fit | 500000 | 495747/446364/436476/455421 | 36.377/2.3345e+05/63.76/1.471e+05 | 0.043308–0.043908 | 39839/39601/41830/375187 |
| 2026 | eval | 302228 | 300522/283172/279587/290373 | 38.707/70181/69.287/26830 | 0.035529–0.035529 | 10320/10075/11251/270582 |

## 限制與交接

strict_pit_eligible=False：首次公告語義不等於完整逐期更正鏈或實收時間證據，歷史容差不升格strict前瞻。凍結calendar仍是觀測價格聯集；universe／lifecycle／停牌下市／tradability與selection偏誤未解。歷史已曝光、研究分岔與b和既有surprise的冗餘未消除；c左設限與e一筆基準的泛化風險保留。工程ADMIT不等於Human acceptance、研究確認或交易准入。

Completed：5a候選、features／prepare／audit實跑及指定回歸，限定歷史無標籤工程ADMIT；尚非Human acceptance。Changed：上述五新檔、本報告及Current State／Authority Map／Research導航。Decisions：沿用FROZEN R-A與40GB gate，沒有調門檻。Evidence：私有manifests、admission及本報告。Remaining：獨立審查、Human c／e支援驗收與5b；本輪未fit／IC／前瞻。Blockers：strict來源與交易gate仍STOP。Next recommended action：先獨立審查5a，按凍結identity實作非step5前綴的5b驅動。Project memory updated：YES。Authority map updated：YES，導航更新，不改production machine pin。Git status：開工clean，交接保留本輪modified／untracked，no-commit故不聲稱clean。Commit：N/A（禁止）。Push：N/A（禁止）。Worktree：builder_step5a，detached `261686c5628c58e18fe44d009d8330f07850a272`。Runtime/environment：WSL CPU、Python `/home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python`、MARKETMAMBA_DATA_ROOT unset；正式Data realpath `/home/frank/projects/MarketMamba/Data`，worktree Data為本地普通目錄；未改資料、V6/models、P4d、排程、production、TEJ raw或第三／四步產物，無網路／GPU／agents／push／merge／deploy。
