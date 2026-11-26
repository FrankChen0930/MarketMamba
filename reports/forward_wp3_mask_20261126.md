# 前瞻 WP3 同版本 mask 重建 — 2026-11-26

**EXACT_X_REBUILT_MASK_CANDIDATE_PROVENANCE_LIMITED / CPU_ENGINEERING_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED。** 已由 prepared cache 在補零前重建 observation mask；重建 X 與指定 lagus X 完整檔案逐位元相同。真實完整 48／185 欄 immutable 工程 reference 候選已建立、重驗。原歷史轉換腳本仍未恢復；不是 Human acceptance、來源准入或 WP3 整包通過，每日延伸／WP1／WP2 STOP 不解除。

## 範圍與來源

既有 detached HEAD `3f94ae28a0dc73defe14d0b256d1c758cbf5a7a5`，起始 clean。依序核對 AGENTS／CLAUDE、Current State／Authority Map、[WP3 報告](forward_wp3_20261126.md)、corrected builder／tests、Features／Data／Governance／Operations 與 canonical checkout 的 `.artifacts/papers/notes/fwd-wp1-4-review.md`。本工作樹原無私有 artifacts，該筆記僅在 `/home/frank/projects/MarketMamba` 唯讀參考。沿用既有 tools／reports／knowledge／`.artifacts/forward/` 結構；指定新 namespace 隔離本輪重建，既有 WP3 STOP inventory 不適合覆寫。無新頂層框架或 worktree。

cache：`/home/frank/projects/MarketMamba/Data/v7_prepared_20260911/.prepare-cache`；universe：同 Data root 的 `v7_corrected_baseline_v1/universe`；目標／舊 E5：`/home/frank/projects/ruflo-evaluation/pilot09/marketmamba/.local-data/matrices/E5_fwd_lagus`／`E5`。這些均只讀。`manifest.json` 保存每一來源檔的 realpath、size／SHA 與程式／contract inventory；`validation.json` 再次完整計算輸入 hashes，確認未變。

| 來源 | 檔數／bytes | 完整 tree SHA-256 |
|---|---:|---|
| prepared Parquet | 2,244／2,285,803,074 | `3fbba16c77071bc65685afdedfe3b249bc4a65c38cd33b6daaba0923b90208cc` |
| 歷史 universe | 22／10,445,950 | `39caafb0d577f1ecd7a5beede6091a2de340678a13e2dee5a0045c420c560183` |
| cache JSON metadata | 2,245／811,807 | `23825bd5986c2adf90400657ccf5cb76a3e71fa6d1842495adda5e21c73cf3c1` |

前兩項精確符合舊 E5 manifest。feature tree 的 path 使用原程式的 `.prepare-cache/` 前綴；直接以 cache 子目錄為 root 的同份 bytes tree 為 `ce83c28cedaecf223bc76c6838065f3aed945a01bdef9aa87f3495eeb1278a37`。兩個 hash 的差別是相對路徑，沒有挑選另一批來源。

## 重建與逐位元 gate

逐股票讀 frozen 48 欄，直接呼叫既有 `V6/experimental/v7_corrected_e5_matrix.py::build_arrays`，讓原程式先 `np.isfinite(values)`、再補零。以真實歷史 universe 產生列，既有 reference keys 只作 memmap 寫入定位；重建鍵有 extra／duplicate／missing 即拒絕，最後 7,423,927 列全部覆蓋。labels 用空的 left join 輸入，沒有載入 y5／y10 或用 label 成熟度控制 feature 列；這是 mask-specific 重建，沒有宣稱完整上游 source-admission gate 通過。

在現行 repo、pilot09 repo 的 Python／shell source（含 ignored／hidden）與歷史 `git log --all -S lagus -- '*.py' '*.sh'` 搜尋，**未找到原 lagus 產生腳本**。現存可核對來源為 [roadmap §12.19](../tasks/roadmap-20261115.md)：`SPX_Return, VIX, TNX, Gold_Return, Oil_Return, USD_TWD, Fear_Greed` 取前一 observed session 值。私有 `rebuild.py` 僅為本輪診斷重現程式，不能冒稱原程式；其 SHA 見下表。先驗七欄同 session 值皆相同，再移動數值與同時重建的 mask，第一 session 保留來源值。此配方最後由指定 X 的完整 bytes 核驗，而非名稱或接近程度准入。未用舊 mask、全 1、或 zero-filled X 反推 mask。

| 驗證 | 實際結果 |
|---|---|
| 原 E5 X | file SHA `67358805f50b0dafaf70c29971def766191f537042cd1dc48518e2f45df18aed`；重建 base payload uint32 差異 **0** |
| lagus／重建 X | **1,425,394,112 bytes；SHA `9778392d4c337d1f9cb5768002340a0d1aec13f7689b2f601b77bd2329b55363`**；`cmp` exit **0**；uint32 差異 **0**，第一差異 **N/A** |
| 重建 mask | bool 7,423,927×48；356,348,624 bytes；C-layout SHA `302392f73512dc7f7b3337b582276282a922b752f83241c1c43d7cc9a5b235ab` |
| mask 缺觀測格 | **2,453,171**；mask=false 且 X 非零 **0**；X 全部 finite |
| 舊 mask 身分交叉核對 | 將重建 mask 以原 F-layout 串流序列化 hash 為 `004771ddb41ee24903b046e1ff377c6e42139dd3393f3e80abd2c7e5433dc6c5`，精確符合原 manifest；沒有讀取／套用舊 mask 檔 |
| lag 對 mask 的影響 | 本批七欄轉換前後 mask payload 完全相同，經完整 C-file hash 核對；不是預設不變 |

首輪 audit 自訂摘要錯把 C-layout 的 base X／mask file SHA 與原 F-layout 比較，也漏了 feature-tree 的相對路徑前綴，因而輸出誤導的 `STOP_X_MISMATCH`／exit2；同份 log 已明載 target SHA 完全相同與 0 格差異。保留該原始 manifest／log，另以 `validate.py`／`validation.json` 更正判讀，沒有改目標 gate、改來源或重跑挑結果。獨立驗證使用完整 `cmp`、target SHA、原 X payload、mask F-serialization hash 與前後 source hashes，exit0 才標候選。

frozen 欄序依 `research/v7/corrected-baseline-v1/feature-manifest.json`（file SHA `d3b2c2d23ccc071b6e0d49ca77d27f41ade085084ea8fd3952321f5352a32547`）；48 欄依序：

Open、High、Low、Close、Volume、Return_1d、Return_5d、Return_20d、MA_20、MA_60、RSI_14、ATR_14、RS_5d、RS_20d、RS_60d、Foreign_Buy、Foreign_Sell、Foreign_Net、Investment_Trust_Net、Dealer_Net、Margin_Purchase、Margin_Repay、Short_Sale、Short_Cover、Margin_Balance、Short_Balance、Day_Trade_Volume、KD_K、KD_D、OBV、Volatility_20d、Holdings_Large_Pct、Holdings_Large_Change、Securities_Balance、Foreign_Holding_Pct、Market_Cap_Log、TWII_Return、SPX_Return、VIX、TNX、Gold_Return、Oil_Return、USD_TWD、Futures_OI_Foreign、Options_PC_Ratio、Fear_Greed、Business_Signal、FED_Rate。

原 materializer SHA `b3735a7d8c01dd09bfdcd4e3a920ebd7ef0976a75195bbd3b8892310e639f671`；replication/config 程式 SHA `8c3a2b37950c5a24028cf9a2081fa6a3ea1cd89ce4be8e05b29ba1811c5d3af8`。本輪執行程式：

| 程式（均在私有 namespace） | SHA-256 |
|---|---|
| `rebuild.py` | `0399e394b2b4b2b24ef7f2fc1e9e6650ab496f8b99bd40a3b838195cf4f67bc4` |
| `validate.py` | `c7150092cf7aaf08dd19cae855300cc0c9a54b3806edac23b1e95d3a0db860ce` |
| `reference.py` | `1b11ddfd6c6424a6be9ce1c3f94d4ffa02d5410dacce4c5aac4e6f611f6b2a5c` |
| `reference_stream.py` | `2ef1fc68f7a73144512c617790fdcedf793db25c7964dd7a2168df2779acf0ac` |

## 真實完整 immutable 工程 reference

預設 C0+ 同時持有各欄與 hstack 的資料量約 10,987,411,960 bytes，另有 inputs／metadata／rolling scratch；未強行製造 OOM。首個列批次 adapter 保持完整 context，每批重新呼叫原 C0+，三種合成 chunk bytes 相等，但真實首 500,000 列後估算會超出本輪時間，故中止 exit130。`C185.npy` 是 **PARTIAL／TEMPORARY**，不能因 header 顯示全量 shape 當成 complete；原 log／空的 resource log 均保留，沒有捏造中止程序的 RSS。

最終 `reference_stream.py` 改用原 `dense_panel`／`lagged`／`rolling_mean` 與逐字相同 market aggregate expressions，完整歷史 axes 只計一次；逐欄寫 F-layout memmap，chunk 只控制寫入。沒有 monkeypatch 或修改 tracked 原程式。三種寫入 chunk（1／37／1,000）與原 C0+ 完整 350 列輸出逐位元相等（包括 NaN bits）；真實資料另取頭／尾及固定 seed 的 **256 列**，重呼原 `c0plus.build_features` 比較全部185欄 bytes。

沿用 P1 `full-scores-ace2eef` manifest 核對 fold_2026 的 X／dates／stock_ids／splits／end_indices；stock_idx 全列驗 roster 對應。`SeqData(labels=(), normalize='none', stock_idx=既有陣列)` 只取得原 W60 eligibility，不呼叫 rank-gauss cache、不載入 y5／y10。舊 `rank_average.parquet` 驗 hash 後整份作 opaque bytes 保存，沒有解讀 scores／labels 欄位、模型語義或計算 IC。

用 `tools.forward_wp3.validate_batch` 與既有 `publish`（等同 create_reference 的前置 gates，再傳已串流的 C185）封版，最後 `load_snapshot` 完整重驗每份 file SHA、payload SHA、schema／context／implementation identity。沒有繞過同版本 mask gate、來源日期 gate 或舊 bytes 門檻；source identity 只屬工程綁定，`forward_admitted=false`。

- 路徑：`.artifacts/forward/mask-rebuild-20261126/full-reference/`。
- identity：`e50d9db919f148f59685af598b345a40ab679e87434e3fa718016f126dc7098e`；source identity：`a679e849df6eeac5de658b2fc7ebd8ede07e9d67477bdfa2babaefd6aa4b6e2f`。
- shape：**7,423,927×185**；eligibility true **6,759,780**；splits：`{'0': 2089766, '1': 4971589, '2': 324395, '4': 38177}`。
- 封版檔案總計 **8,351,176,228 bytes**；完整重驗 **PASS**。不包含新 labels、fit／新 scores、真實每日 append 或 score keys／model 語義驗證。

## 命令、資源與驗證

共用環境前綴（下列每次實際執行皆使用，無網路）：

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=V6:V6/research_lab/tests:. \
/usr/bin/time -v /home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python \
  .artifacts/forward/mask-rebuild-20261126/rebuild.py
```

其後同前綴分別執行 `validate.py`、`reference.py`（中止）、`reference_stream.py`。原路徑為 no-clobber 證據，**不可直接覆寫重跑**；程式／命令／input identity 供審查，另一次重建須使用新 namespace 並保留新程式 SHA。

| 階段 | 實際秒數 | 峰值 RSS KiB | exit |
|---|---:|---:|---:|
| X／mask 重建＋比對 | 324.823（time wall 325.51） | 4,773,128 | 2（上述摘要誤判） |
| 獨立完整稽核 | 12.827578（time wall 13.13） | 1,782,052 | 0 |
| 完整 reference／重驗 | 395.472748（time wall 403.64） | 10,667,880 | 0 |

標準測試命令（從未使用 `--basetemp`）：

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=V6:V6/research_lab/tests:. \
/home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python -m pytest \
  tests/test_forward_wp1.py tests/test_forward_wp2.py tests/test_forward_wp3.py \
  V6/experimental/v7_corrected_e5_matrix_test.py \
  V6/research_lab/tests/test_rolling_folds.py V6/research_lab/tests/test_rolling_runner.py \
  V6/research_lab/tests/test_p1.py \
  .artifacts/forward/mask-rebuild-20261126/test_reference_adapter.py -q -p no:cacheprovider
```

結果 **198 passed／104.14秒／exit0**；其中原 WP3＋builder focused **52 passed／10.72秒** 是重疊驗證，不加總。另同標準環境執行 `test_reference_stream.py`：**3 passed／2.57秒／exit0**。兩個新 adapter 的最初 RED 都是缺模組 collection error，並非宣稱已得到數值反例。後續三個 chunk 的完整 bytes／NaN 等價與 no-clobber 反例有實際 assertions。未跑全 repo suite；作者自查不是獨立 review。最終 syntax／JSON／links／hash／git diff／knowledge 檢查補記於後。

## 限制與交接

這是 prepared cache 層的 finite observation mask；上游若以預設值掩蓋來源缺失，有限值不會自動恢復 raw observation／publication-time 真相。原轉換腳本與當年執行環境仍 UNKNOWN；新重建程式只證這份 source tree 與指定 X 的 bytes。來源 PIT、calendar／lifecycle／tradability、WP2 seed17、WP4 環境還原、每日延伸與 scores 語義仍獨立 STOP。snapshot immutable 是 no-clobber／manifest 重驗契約，非 OS 防惡意竄改；未測跨機／斷電／並行來源修改。所有歷史 artifact／inner manifest／source-admission／production authority 保留。

Knowledge-impact：major blocker／next action、候選 evidence 與 worktree responsibility＝YES；public semantics、authority、subsystem boundary、active external runtime、correctness thresholds、canonical supersession、deprecation＝NO。正常測試後執行 impact，再更新 Current State／Authority Map／Features／Worktree Register；Governance／Operations 已人工核閱，規則與 operational 語義未變。

- Completed：同版本 X／補零前 mask 重建、完整 byte gate、source tree 前後重驗、真實完整 reference 候選與資源量測；完成不等於 Human acceptance。
- Changed：本報告、`knowledge/00_Project_Map/{Current_State,Authority_Map,Worktree_Register}.md`、`knowledge/01_Domains/Features.md`；ignored 證據產物在指定 `.artifacts/forward/mask-rebuild-20261126/`。早期唯讀 Python import 另在 `V6/experimental/__pycache__/` 產生可重建 bytecode 快取，已盤點保留，沒有寫正式資料；tracked tools／tests 未修改。
- Decisions：原 X byte gate 不變；原程式未找到如實保留；新重建 mask 只標候選；F-layout／相對路徑差異明確分辨；partial 保留；既有 helpers 逐欄串流，不重設 rolling 起點。
- Evidence：完整 hashes／cmp／0 格差異、2,453,171 缺觀測格、mask 原 SHA 交叉核對、198＋3 tests、真實 256 列全185欄核對、snapshot 全量重驗與實測 RSS／time。
- Remaining：獨立 review／Human acceptance、原轉換程式來源、每日 append／scores keys／model 語義／前瞻來源與其他 WP gates。
- Blockers：上述 provenance／forward gates；不能以本候選解除 WP3 整包 STOP。
- Next recommended action：review 私有 validation／reference manifests 與程式 SHA；裁定 mask 候選可用範圍後，再做每日來源／延伸驗收，不啟用日鏈或排程。
- Project memory updated：YES，Current State／Features／Worktree Register。
- Authority map updated：YES，僅候選補證／STOP 導航，不修改 machine authority 或 supersede 歷史 artifact。
- Git status：起始 clean；交接保留4個 tracked 修改＋本報告1個 untracked，staged0；因明確 no-commit 不宣稱 clean。
- Commit：無（Human 明確禁止）。
- Push：無；merge／deployment 亦無。
- Worktree：既有 `run-mwxdbyz7-28c523/builder_fwd_mask`、detached `3f94ae2`；namespace `job-run-mwxdbyz7-28c523`；created UNKNOWN、接手2026-11-26、retired N/A。保留 patch、完整 private artifacts／inventory／source pointers，Human 另案退役前不清理。
- Runtime/environment：WSL CPU、pilot09 venv Python3.12.3／NumPy2.5.4／pandas3.1.1，CUDA空、OMP／BLAS1；MARKETMAMBA_DATA_ROOT unset，worktree Data realpath為本工作樹普通目錄；來源 main Data／pilot09 matrices 僅讀。無網路／GPU／資料取得／agents／blind／正式Data或V6/models寫入／TEJ原SHA256SUMS修改／P4d／第三步bundle／production／scheduler／wakeup／recurring jobs，無外部 observed_at 更新。

最終治理與檢查：`check_knowledge_impact.py` WARNING／exit0（需人工檢查Governance／Operations，已核閱）；`knowledge_health.py --ci --as-of 2026-11-26 --json` **WARNING／exit0**，Project Map／Agent Guides／Domains／Runtime Freshness／Generated Index PASS，Governance／Authority Drift／Legacy Docs WARNING，無structural FAIL、broken links=0。未為消除既有warnings改寫authority或刷新觀測。獨立驗證JSON內的1,781,924 KiB為輸出前自取RSS；上表列完整程序`time -v`峰值1,782,052 KiB。private程式AST、JSON、報告local links、程式／contract hashes、snapshot與mask的identity鏈、5份交付文件whitespace／`git diff --check`及Git狀態已做最終檢查，詳私有`verification-final.json`。完成時4 tracked修改＋1 untracked／staged0；no-commit授權下保留dirty候選，不宣稱clean。
