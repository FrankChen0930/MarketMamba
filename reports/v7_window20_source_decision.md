# Window20 Historical Source Decision Brief

PROPOSED / documentation sprint complete / STOP / UNMERGED。不是Dataset／Research Admission。
Base/main87695e4；W19 cf1a84f只作隔離身份參照，不修改／停用／觸發／整合Task或其worktree。
兩單元：來源矩陣；決策與下一次有限驗證規劃。無程式重構／新平台／行情API／登入／Token／RankIC。

A. **主要候選 FinMind TaiwanStockPrice（有條件）**。官方[技術文件](https://finmind.github.io/tutor/TaiwanMarket/Technical/)
稱1994-10-01起上市／上櫃日資料；這是DOCUMENTED聲明，不是真實多年份Coverage。
[方案](https://finmind.github.io/Pricing/)明示Free非商業個人／學術、匿名300/hr／註冊600/hr；
[Quickstart](https://finmind.github.io/quickstart/)列Token optional，匿名可用性仍未實測。
[條款](https://finmind.github.io/PrivacyPolicy/)與[資料授權](https://finmind.github.io/Disclaimer/)不容許原始資料再散布／轉售／鏡像API。
初期應限定私有非商業資料資格診斷；本地研究cache合理推論為INFERRED，永久保存年限／終止後處理UNKNOWN，
不能將軟體LICENSE或上游OGDL宣稱當所有FinMind資料再散布權。

B. **備用 富邦Historical Candles（有條件，現在BLOCKED）**。[文件](https://www.fbs.com.tw/TradeAPI/docs/market-data/http-api/historical/candles/)
稱上市櫃日資料2010起、每次區間<1年，dailyvolume為股，明示adjusted參數；
[登入範例](https://www.fbs.com.tw/TradeAPI/docs/market-data/http-api/getting-started/)需要帳號／憑證／行情權限，
[歷史配額](https://www.fbs.com.tw/TradeAPI/docs/market-data/rate-limit/)60/min。帳號權限、研究保存／再散布條件與費用UNKNOWN，
因此已停止深入；不註冊／登入，不假設免費。只在另批合法權限後作備選，不是自動Fallback。

W14官方歷史API目前不選作主備：TWSE[網頁](https://www.twse.com.tw/zh/trading/historical/mi-index.html)宣稱2004-02-11起，
TPEx[網頁](https://www.tpex.org.tw/zh-tw/mainboard/trading/info/pricing.html)主路徑2007-01起（另有更舊不同連結，未擴查）。
W14實際僅各一個2022交易日、合計337221rawbytes；網站範圍不是精確API完整性的證明。
[TWSE條款6/8](https://www.twse.com.tw/zh/terms/use.html)限制未同意的自動下載／重用、另有open-data例外；
TPEx[條款5/7](https://www.tpex.org.tw/zh-tw/gtsm_disclaimer.html)有同類規範（web工具403，另一次有界文件封存GET200；已保留兩種觀測，不繞過控制）。
精確W14routes尚未映射到W15兩CSV授權。這不是判定所有官方API違法；是不足以授予本專案歷史抓取資格。

C. **最小待證資格**：兩市場／長歷史／下市及轉板實際rows、日曆／生命週期、缺值與公司行動口徑、日期範圍完整性、
可合法保存的actualbytes／hash／restore。所有來源原始歷史availability／revision ledger仍UNKNOWN，不能稱StrictPIT。
FinMind額外風險：0可代表無公告價格（即使有量），不可當有效價格或自動ffill；興櫃open是前日均價，首輪應隔離，
但不能用今日市場別回填歷史；rawPrice與paidPriceAdj不同，後者回溯調整並可能沿用舊價。
[下市表](https://finmind.github.io/tutor/TaiwanMarket/Fundamental/)宣稱2001起，並不證明下市個股price仍可取或universe完整。

D. **建議下一次只批准FinMind無Token小型資格Pilot：最多22請求**（本輪未執行／未批准）：
6檔角色＝上市2、上櫃2、下市1、轉板1；身份須先由合格既有資料確認，角色可跨不同歷史時點，非研究Universe。
每檔3小窗口：2010年初、W14已有2022-09、經生命週期證明之公司行動／下市窗口，各最多10合格sessions。
2010未上市者預期空缺，不錯認缺資料；較晚IPO可於其合格上市區間驗資格，不擴張請求總額。
18price請求＋3Metadata（StockInfo／TradingDate／Delisting，各1）＋1固定請求重取比對＝22，零自動Retry／Fallback。
日期上限須明確end_date；總price rows理論最多180＋一次重取最多10（空缺更少），不是60lookback研究panel。
匿名若被拒或觸限就停止，不能改用Token／開帳戶。Metadata可能較大，建議每response2MiB硬限，不足即BLOCKED不翻頁。
備用富邦僅另批帳號及使用條款後：同6×3＝18candles＋1adjusted對照＋1重取最多20行情請求，
全部區間<1年；登入／instrument/calendar授權與操作額度須另列，不把它們藏進20或本輪。

E. **研究可行性**：L0／Ridge／LightGBM及Expanding Window有W07–W11工程可重用，本輪不改；
多年OHLCV「文件上可支援」，但5d／10d須另批Label／entry-exit／成熟與公司行動政策、跨市場session日曆、
歷史Universe、fold／purge／選模／公平比較／compute。這22請求只驗來源，不能訓練／RankIC、凍結研究參數或重現48-feature E5。
所有可能暴露過的期間不得稱untouchedfinaltest；目前沒有正式Research-ready資料集。

取得成本只作INFERRED算例：N=2000個歷史合格代碼（非實測名單），Y=16年2010–2025，T=4000合格sessions（假設250/年）。
FinMind全區間若完整約N=2000calls，需年切則N×Y=32000；匿名配額相當6.67／106.67 quota-hours，非牆鐘工時或吞吐實測。
富邦約32000calls／60每分鐘＝533.33 quota-minutes；權限費用未知。官方全市場日檔若成立約2T=8000calls，配額／使用資格未確認。
Metadata／分頁／失敗／調整版等額外，尚無bytes量或金錢總價的實測；不建議現在批量取得。

F. **Human下一步最少批准**：是否接受上述「無帳號、無Token、私有非商業、22requests上限」資格Pilot及保存範圍；
永久archive／商業／分享用途如需擴大須先補供應商權利證據。富邦帳號／憑證／行情權限必須另批；付費非首輪必要、費用仍未知。
來源推薦仍PROPOSED，沒有Human代為授予上游使用權。W19既有Collector保持隔離，本輪未查改WindowsTask或追加行情GET。

[Machine matrix](../research/v7/window20-source-matrix.json)、[來源索引](../research/v7/window20-evidence.json)、
[文件bytes](../research/v7/window20-document-archive.json)、[唯一Backlog](../research/v7/research-program.md#window20-historical-data-source-decision-sprint)。
LocalHTML只留本tree.artifacts，Git中的Hash不等於HTML可還原；不公開再散布。完成後STOP，等新授權，不merge/push/deploy。

一次獨立審查GO：Critical0／Required0／Optional0、0修復；8HTML／manifest／924253bytes與分類引用／預算核對通過。
沒有實作改動，不重跑Trainer／Evaluator套件。Knowledge／Artifact檢查另列window20-checks與knowledge-health。
