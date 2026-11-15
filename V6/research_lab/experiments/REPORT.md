# pilot09-w2 排序研究實驗

`RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED`。Yahoo 價格、150 個當期大型股／ETF 的存活者偏差 universe，沒有 fundamentals；不是投資或可成交證據。

HEAD `814002275d67e7d755b530fceb9285a227e3ba10`（實作未提交）；dataset bytes SHA-256 `64e6e1b4201643336817ebf72a31882b0e23ff0a3df047deb3c4ae34118c85f8`；manifest SHA-256 `99ff423678ae3e5dc138f945bc571bb6cd7642dd8eb521f9cce1f01b9fb491c1`。

CPU 完整實驗一次 233.49s，僅重評估既有 predictions 103.15s；2 torch/BLAS threads，無 GPU／網路／安裝。Python 3.12.3、torch 2.15.0+cpu、LightGBM 4.7.0、sklearn 1.9.1。

沿用 research_lab expanding split：min_train=504、test=252、purge=20，7 個有效 folds；首個候選 fold 因成熟標籤日期支援不足被既有 helper 略去；最後不滿 252-session 的區段不評估。所有模型共同要求完整60-session歷史，沒有依未來label篩預測。

11 個當日橫斷面 z 特徵，不跨日期 fit scaler；raw baseline 分別為20d return、負5d return、負20d volatility。Ridge alpha100；LightGBM rank-label regression，60 trees、leaves15、depth5、lr.05、min_child100、L2=10、seed17。

神經模型為雙頭 rank-label MSE、GRU width12、selective SSM width12/state4；input-selective delta/B/C、negative diagonal A、gated skip，純 PyTorch scan，不聲稱等同正式 Mamba。seeds17/29/43；每 fold 從頭重訓，2 epochs、Adam .003、gradient clip1；固定 seed 抽最多4096 fit／1024 validation rows。內層 validation 最多64 sessions（day>=maxday-63），前隔20 sessions，選 validation BEST；尚不足充分收斂比較。兩種集成均為逐日 seed percentile rank 等權平均。

下表日期等權；ICIR未年化，NW Bartlett lag=h-1、95% normal CI；保留完整 observed-session grid，不壓縮缺失IC日期。缺值 covariance pair 不貢獻、SE 分母為有效IC數；不對多重比較校正。spread 為 top-minus-bottom decile 的重疊 h-session報酬差，非投資組合損益；hit 為top decile正報酬比例。decile 各 floor(N/10)，平手按code遞增，top取末端。

|模型|h|IC|ICIR|NW t|IC 95% CI|spread|hit|有效IC日|
|---|---:|---:|---:|---:|---|---:|---:|---:|
|momentum20|20|-0.0103|-0.057|-0.71|[-0.0387, 0.0182]|1.93%|52.8%|1624|
|reversal5|20|0.0107|0.064|1.28|[-0.0056, 0.0270]|-0.83%|53.6%|1624|
|lowvol|20|-0.0225|-0.095|-1.12|[-0.0619, 0.0170]|-4.53%|56.0%|1624|
|ridge|20|0.0114|0.065|0.84|[-0.0151, 0.0379]|-0.36%|53.4%|1624|
|lightgbm|20|0.0271|0.198|2.67|[0.0072, 0.0470]|0.34%|54.9%|1624|
|gru|20|0.0181|0.105|1.41|[-0.0071, 0.0433]|1.22%|53.3%|1624|
|selective|20|0.0068|0.055|1.03|[-0.0062, 0.0198]|0.95%|52.8%|1624|
|momentum20|5|-0.0173|-0.093|-2.19|[-0.0327, -0.0018]|0.34%|48.1%|1639|
|reversal5|5|0.0255|0.150|3.95|[0.0129, 0.0382]|-0.00%|50.6%|1639|
|lowvol|5|0.0248|0.100|2.35|[0.0041, 0.0455]|-0.72%|52.3%|1639|
|ridge|5|0.0444|0.251|6.33|[0.0306, 0.0581]|-0.15%|51.7%|1639|
|lightgbm|5|0.0502|0.298|7.63|[0.0373, 0.0630]|0.06%|52.8%|1639|
|gru|5|0.0335|0.222|5.50|[0.0216, 0.0454]|0.13%|51.2%|1639|
|selective|5|0.0171|0.119|3.22|[0.0067, 0.0275]|-0.02%|50.7%|1639|

20d LightGBM 的平均 IC 最高；GRU 高於簡單基準但低於 LightGBM，selective 低於 reversal/ridge。5d selective 也低於 reversal、lowvol、ridge 與 LightGBM；沒有證據支持 SSM 優越性。模型間差異未做 paired 顯著性測試，不將單模型 CI 當勝出證明；不同 metric 的排序也不一致。

逐年度平均 IC（完整年度 spread/hit/CI、所有seed與fold支援見 JSON）：

**h=20**

|year|momentum20|reversal5|lowvol|ridge|lightgbm|gru|selective|
|---|---:|---:|---:|---:|---:|---:|---:|
|2019|-0.0448|0.0182|-0.0514|-0.0157|0.0344|-0.0335|0.0012|
|2020|-0.0315|0.0340|-0.0599|0.0226|0.0468|0.0556|0.0282|
|2021|-0.0027|0.0003|0.0143|-0.0353|0.0157|0.0142|0.0029|
|2022|-0.0971|0.0381|0.0187|0.0843|0.0540|0.0150|-0.0233|
|2023|0.0390|0.0134|-0.0538|0.0378|0.0273|0.0626|0.0187|
|2024|0.0284|-0.0107|0.0243|0.0250|0.0283|0.0201|0.0013|
|2025|0.0080|-0.0182|-0.0250|-0.0774|-0.0266|-0.0344|0.0105|
|2026|0.0850|-0.0288|-0.1085|-0.0216|-0.0192|-0.0036|0.0331|

**h=5**

|year|momentum20|reversal5|lowvol|ridge|lightgbm|gru|selective|
|---|---:|---:|---:|---:|---:|---:|---:|
|2019|-0.0262|0.0326|0.0255|0.0473|0.0600|0.0228|0.0090|
|2020|-0.0462|0.0283|0.0110|0.0545|0.0551|0.0460|0.0212|
|2021|-0.0064|0.0321|0.0196|0.0329|0.0478|0.0370|0.0118|
|2022|-0.0195|0.0046|0.0505|0.0358|0.0509|0.0411|0.0252|
|2023|-0.0364|0.0459|0.0212|0.0660|0.0537|0.0303|0.0166|
|2024|0.0025|0.0196|0.0320|0.0565|0.0553|0.0220|0.0119|
|2025|0.0150|0.0023|0.0387|0.0216|0.0340|0.0483|0.0336|
|2026|0.0084|0.0389|-0.0275|0.0075|0.0187|0.0101|0.0058|

三seed個別平均 IC：

|模型|h|s17|s29|s43|
|---|---:|---:|---:|---:|
|gru|20|0.0024|0.0226|0.0123|
|gru|5|0.0285|0.0166|0.0271|
|selective|20|-0.0067|0.0066|0.0144|
|selective|5|0.0159|0.0061|0.0079|

預測區間 2019-03-11～2026-06-11；每 model/horizon 214775 predictions；20d label coverage 97.15%。

重要限制：先去除未知label再做 decile，是事後可評估股票子集合，不能冒充事前可成交 portfolio；預測本身仍保存未知label rows。Yahoo vintage、raw-price corporate actions、非官方 observed-union calendar、當期股票選樣、特徵完整歷史篩選與label缺漏造成選擇偏差；無官方 PIT/universe/tradability、無成本、無 fundamentals。樣本不足10股日不算指標，缺失日留在NW時間軸。

未保存 fitted weights/checkpoints，不能續跑或提供權重級模型驗證；本次模型 identity 由 frozen input/source hashes、fold/date support、config、seed與prediction SHA限定，不宣稱 checkpoint artifact identity。原 run 的 source hashes 於結束時計算，模型/runner/metrics 當時未改；之後只修正評估／防覆寫邊界，原 source hashes 保留，evaluation_source_hashes 另列。

原始 [results.json](results.json) 保留初始評估；[results_evaluation.json](results_evaluation.json) 是本報告的完整session-grid校正評估與各檔 hash authority。不重訓、不修改既有 predictions；不變更 corrected V7／production authority。完整測試與治理結果見本報告後續驗證紀錄。

驗證紀錄：

```bash
PYTHONPATH=V6 /home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python -m pytest V6/research_lab/tests V6/experimental/v7_executable_labels_test.py V6/experimental/v7_temporal_contract_test.py -q
# 44 passed in 20.89s
PYTHONPATH=V6 /home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python -m compileall -q V6/research_lab
# exit 0
git diff --check
# exit 0
```

Artifact audit：26 files、5,584,150 total predictions、每檔214,775 rows，逐檔SHA／schema／string code／finite score／unique keys／共同code-date-fold支援全部PASS；42次neural fold-seed fit，無額外完整訓練。
獨立review的空評估schema、未知IC不抹除可算收益、缺session NW、防覆寫問題已補RED/GREEN測試並修正；複查無新增Important。
LightGBM／sklearn／scipy版本為同一runtime的**重評估時環境觀測**，原始results環境欄未記這三項，不追溯聲稱原run已保存。

Knowledge-impact focused source check PASS，Current_State／Authority_Map已更新本分支研究邊界。
`knowledge_health.py --ci --json` 實際 exit 1：Project Map／Domains／Governance／Authority Drift FAIL，原因既有`e3bce91`等authority commit objects在本checkout缺失；Agent Guides／Runtime Freshness／Generated Index PASS，Legacy Docs WARNING。
新報告／JSON連結存在；不重設或降低gate，未修既有歷史authority缺失。

Completed：模型／runner／一次真實實驗／校正評估／報告。Changed：research_lab模型、metrics、runner、evaluator、tests、README及兩份project map。
Decisions：維持RESEARCH_ONLY，所有模型共同coarse schedule；不主張SSM勝出。Evidence：44tests、26檔hash/schema、初始與校正JSON。
Remaining：充分收斂、多重／paired顯著性分析、權重artifact、嚴格PIT／universe／成交證據；本次不擴大。
Blockers：既有knowledge-health FAIL。Next recommended action：先檢查label coverage與支援選樣，再由Human決定後續實驗。
Project memory updated：YES，Current_State及Ruflo `pilot09-w2-models/w2-models-result`。
Authority map updated：YES，僅新增局部研究identity邊界，無production supersession。
Git status：依要求保留未提交改動；既有 `.claude-flow/policy/{ledger-anchors,state}.json`／`.swarm/agentdb-memory.db.ruvector` dirty與Ruflo untracked DB/index/runtime檔保留；CLI亦產生runtime metadata，不清理他人檔案。
Commit：NONE（明確禁止）。Push：NONE。Worktree：w2-models，HEAD814002275d67e7d755b530fceb9285a227e3ba10。
Runtime/environment：指定pilot09 venv、CPU、dataset唯讀main路徑；MARKETMAMBA_DATA_ROOT未設定，Data解析為本worktree路徑但非本次input。
Ruflo首次memory search宣告自動啟daemon，已執行stop（回報not running）；後續memory命令明設RUFLO_DAEMON_AUTOSTART=0。

Shared memory實際驗證：指定 `memory store --key w2-models-result --namespace pilot09-w2-models` 回報sql.js persistence警告；另開程序以`memory retrieve`成功讀到同key完整摘要。未重建native dependency，未安裝套件。
