# P3 bounded builder candidate — 2026-11-20

UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED。Detached base `06bc0c73a44d6bb725b36f136c7c953d4d199e14`，job `job-run-mwnteeta-4dea80`，既有 builder worktree。只實作本輪 P3 工程候選；沒有 commit、push、merge、GPU、Colab、資料 acquisition、production 或 scheduler 變更。

## 完成範圍與 STOP

新增 `V6/research_lab/rolling_seq_scores.py`：相容的分數欄位、逐 fit exclusive parquet／hash sidecar、run identity、輸入 SHA-256、來源／環境 identity、GPU 型號／peak allocation 欄位、每 batch 溫度紀錄、fit-level resume、逐日截面 percentile rank 種子平均。預設 dry-run。矩陣輸入只以 allowlist symlink 與 `np.load(..., mmap_mode='r')` 引用到 `.artifacts/p3/work/`，不引用外部 rank-gauss cache；derived cache 另以輸入／implementation／輸出 hashes 綁定。Symlink 本身不是 OS 存取控制：唯讀保證是此流程只讀輸入的程式行為，未變更外部權限。

正式 CLI 與無注入 mixer 的 library entry 均 **STOP_MODEL_CONTRACT_AMBIGUITY**。Frozen spec 同時要求「官方 mamba_ssm.Mamba2」與「no-conv」，但歷史 recipe 沒有指定 `d_conv`，而 `CrossStockRanker` 跨股 mixer 明確覆寫 `d_conv=2`。既有 `V6/experimental/v7_corrected_e5_model.py::Mamba2NoConv` 是另一路注入官方 SSD 的實作，不能靜默替代官方 Mamba2 class。未修改登記規格，也未新增可繞過 gate 的 CLI 參數。`run_jobs`、`collect` 目前僅允許明確注入 mixer 的工程驗證，**不是正式可訓練的 P3 交付**；解除 gate 與產生確認性分數尚未完成。

使用者本輪總限制含 `No GPU`，工作項 5 卻要求 GPU 冒煙。採限制較嚴的解讀：GPU 冒煙 NOT_RUN，GPU 型號／顯存／實際溫度 N/A。指定 pilot09 venv 是 Python3.12.3，實際 `mamba-ssm` package metadata 為 MISSING；沒有安裝或下載依賴。CPU injected mixer 的數字不可冒充官方 Mamba2 性能。

溫控透過可注入讀值與 sleep：低於 80°C 正常；達 80°C 暫停，到 72°C 以下才續；達 85°C／讀值未知／runtime deadline 即 latched STOP。預設使用 `nvidia-smi --id=0`；CUDA device 0 固定，不自行選別張卡。每次 windows 載入（訓練、驗證、推論）前檢查；單個正在執行的 kernel 中不輪詢。14 小時預算計入同 namespace 已完成 fit wall time 與本次 process wall time；**失敗／被 kill 的未完成 fit 時間未完整跨重啟累計**，不宣稱可靠的跨程序 GPU 時數會計。只支援完成 fit 續跑，不支援 mid-fit checkpoint；partial payload 缺 sidecar 時保留 orphan 並停止，使用新 namespace。

`portfolio_eval.py --mode p3` 使用獨立分支，P1／P2 原本 evaluator 與預設行為不變。三系統共用固定 P2 主要格（raw、top50、20 sessions、buffer100、discount0.6、min fee20、capital1e6、benchmark proportional only）。c0plus 從 P1 manifest/hash 驗證的 rank_average 讀取，其餘從 P3 讀取，每 fold 逐一檢查 keys 及 label 完全相同。輸出各系統逐折逐日淨報酬、描述性對基準淨超額、逐折 IC paired HAC／block bootstrap、H-P3a/b 共同日期差與 Holm family2。缺折／種子／正式 contract admission 或不是 1000 bootstrap 時不確認。沒有重新擬合 C0+。

新增檔案的理由：既有 rolling_scores 只支援 c0plus.run；seq_runner 的 result／scores 不具有 P1 parquet、完整輸入 hash 與 immutable fit contract，且會在矩陣旁產生 cache。因此新增薄 adapter，沿用現有 research_lab、reports、.artifacts，不新增 top-level tree 或 worktree。

## 偵察與重現設定

- `seq_runner.SeqData`：預設 60-session 同股完整觀測視窗；`split_rows` 的 eval 為 split2 & eligible，不因 y5/y10 缺失縮減評估列，與 P1 `matrix_baselines.Context.eval_idx` 一致。P3 在訓練前比對 P1 keys／labels。
- `rolling_runner.run_fold_seed`／`rolling_scores.collect`：使用 `SeqData(normalize=None)` eligibility 傳給 C0+，沒有神經網路訓練。outer purge 實際 owner 是 `rolling_folds.make_fold_splits`，train session `< eval_first - 20`；P3 對全部 split1 逐列驗證，不能只看被內層 fit 篩掉後的列。
- 神經網路內層：最後126個 eligible train sessions 作 validation，fit_last 為 `train_dates[-val_dates-20]`，fit／val 都須 finite y5 & y10。保留預設 validation cap80,000，固定 RNG0；eval 永不抽樣。預設 AdamW weight_decay0.01、warmup15%、cosine、gradient clip1、patience3、BEST by val IC5。若無有效最佳驗證狀態，P3 拒絕 LAST fallback。8 epochs 是上限，既有 patience3 可提前停止。
- `train_one`：傳入 `cfg={**DEFAULT_CFG, **RECIPE}`，其中 d_model64、n_layers2、mixer_kw(d_state16,headdim8,expand2)、lr3e-4、epochs8、steps_per_epoch400、dates_per_batch8、stocks_per_date256、listnet_w0、n_cross1。pure MSE targets 為逐日 centred rank，head5:head10=1:0.5。模型 `mamba2`／`crossstock`，seeds17/29；crossstock 訓練 stock token 不重複，推論使用既有 target256／n_perm2／seed0 分塊。
- `ensure_rankgauss`：每 session／feature 截面 rank-gauss；macro_columns 以固定 seed0 抽60 sessions 判定日內常數並歸零；其餘常數截面轉換也是0。這是歷史演算法，不能據此宣稱總經欄位有人工 frozen schema。P3 只改快取目的地／hash gate，不修改歷史轉換邏輯。
- pilot09 唯讀根：`/home/frank/projects/ruflo-evaluation/pilot09/marketmamba/.local-data/`。`local_b8/E5_seq_2fc9d1789f/result_mamba2_s17.json` 與 `E5_seq_048a4212ad/result_crossstock_s17.json` 的實際 config 是上述8×256、8×400、純MSE，後者 n_cross1。`E5_seq_1f512a7e3e` 是16日／n_cross1；`E5_seq_c66fc390d0` 是8日／n_cross2，不能混用。
- `colab_b8` 到 `colab_b8v8/b8_seq_baselines.ipynb` 實際執行 cell 皆呼叫 `sr.run_experiment(... kinds=KINDS, seeds=seeds, cfg=cfg, device='cuda', tag=...)`；smoke 為1 epoch×20步、8日×128檔、baseline50k。v3–v8 非smoke傳 epochs／steps，v1/v2 非smoke傳空 cfg；v6–v8 另由 RUN_TRAINING 控制。v7加入 rank-gauss 診斷，v8延續。僅靠 notebook override 不能重現本次8日設定，還必須明傳 dates_per_batch8／stocks256／MSE／n_cross1。local_b8 中找到的是結果與scores／daily_ic，沒有保存獨立 local launch .py/.sh；不捏造 shell invocation。偵察 identity 見忽略目錄 `builder-verification/reconnaissance.json`。

可重現的預檢命令（不訓練）：

```bash
PYTHONPATH=V6 /home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python -m research_lab.rolling_seq_scores \
  --folds-root /home/frank/projects/ruflo-evaluation/pilot09/marketmamba/.local-data/matrices/rolling \
  --p1-scores-dir /home/frank/projects/MarketMamba/.artifacts/p1/full-scores-ace2eef \
  --years 2023 --models mamba2 crossstock --seeds 17 --smoke \
  --out-dir .artifacts/p3/planned-smoke
```

folds-root 已對照 P1 manifest 的 input keys；本輪未讀取／重算完整矩陣 hash，也未聲稱正式矩陣已通過訓練准入。main worktree 有 P1 full-scores-ace2eef，本 worker 沒有隱式跨 checkout fallback。加 `--run` 會在建立輸出／讀矩陣／初始化 GPU 前停止並說明模型 contract 衝突。

## 驗證與反證

指定命令最終結果：**493 passed、1 skipped、4 warnings／108.28s**（首輪491 passed／114.80s之後新增best／resume回歸）。既有 skip 與 Pandas dayofweek deprecation 警告未修改。focused P3：18 passed／5.53s。syntax compile、`git diff --check` 通過；knowledge-impact PASS（自動規則未識別新 P3 語義，但人工仍更新 Current State／Authority Map／Research domain）。新增測試包含 immutable resume／輸入變更、work cache isolation、no-best拒絕、合成三系統 evaluator／part reuse／不完整 confirmation gate，以及兩個注入 mixer 的 CPU adapter 路徑。

五個 in-memory mutation 沒有改 repository source，每次單獨 pytest，均 exit1：

```text
rows     test_p3_exact_p1_rows                  1 failed in 0.05s
pit      test_p3_rank_average_pit               1 failed in 0.06s
purge    test_p3_training_purge_maturity        1 failed in 0.03s
pairing  test_p3_common_dates                  1 failed in 0.03s
thermal  test_p3_temperature_hysteresis_stop    1 failed in 0.03s
```

對應壞行為：no-op row gate、rank group移除date、直接跳過outer purge gate、union date並補0、no-op thermal gate。`mutation-*.log` 保存完整失敗 traceback，`mutations.json` 保存實際 exit code；沒有把預期失敗算成通過的測試。

本地 `.artifacts/p3/builder-verification/` 是 GENERATED／SYNTHETIC_CPU_ENGINEERING_ONLY。首輪 `cpu-injected-scores` validation cap120 導致不足10檔／日，出現 best=-9，已保留為不合格工程觀察，不能使用於模型／選模結論。修正 adapter 後重跑 `complete-validation/`，改為保留完整240 validation rows。注入 `nn.Linear` 非官方Mamba，2日×12檔、d_model16、1層、1epoch×2步；用固定60°C假讀值驗證 guard hook。實際 GPU 溫度與顯存皆未量測。實際 wall time：mamba2 adapter **2.1513s**、crossstock adapter **0.0966s**；各984 fit／240 validation／240 evaluation rows。三系統合成 portfolio **1.7584s**、各61 return days，H-P3a/b 的 p_holm=1、INCOMPLETE_ENGINEERING_ONLY。兩個 adapter 的首次初始化成本不同，不比較其速度。完整摘要見 `complete-validation/smoke-summary.json`，不是正式市場分數。

## 完整24次擬合估時與限制

沒有合格的本機 GPU 冒煙，**不能提供基於本輪實測的24 fits GPU估時**。規格既有10–25分／fit 僅推出4–10 GPU小時的預算先驗，非新測量；14小時上限不變。歷史local_b8 seed17（不同矩陣／非本次GPU確認）mamba2八epoch合計約21.21分、crossstock約20.52分，不能據此替代本輪外推。若日後合格smoke各20步取得完整job耗時 Tm/Tc 秒，粗略保守估計為 `12 × 160 × (Tm + Tc) / 3600` 小時；它把一次性的rank-gauss／評分／啟動也放大160倍，只可作保守包絡。較合理需獨立記錄每步訓練、每epoch驗證與一次性前處理，再用 `12 × [3200(t_m+t_c) + 8(v_m+v_c) + p_m+p_c+q_m+q_c]` 秒估算；其中 t是訓練每步、v是每epoch驗證、p/q是前處理／最終推論。

限制保留 P1/P2：模擬標籤、survivorship、無 adj_close、未驗證下市及PIT市值、觀測日曆；兩種子對C0+三種子，不升格成交易或production evidence。溫控單步輪詢不是NVIDIA長期訓練認證。未執行真實2023分數生成、完整24 fits、正式P3組合評估、GPU性能／估時或Human acceptance。
