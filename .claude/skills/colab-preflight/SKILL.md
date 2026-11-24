---
name: colab-preflight
description: MarketMamba 上線前排練與 Colab 交付稽核；builder／verifier 準備或修改 bundle、notebook、正式接續、來源相容升級、進度及完成驗證時載入。
---

# Colab 交付前排練

本 skill 不覆蓋 AGENTS.md 與 Human 最新指示。這是核對清單，不授權 GPU、Colab、Drive 寫入、通知、排程或部署；沿用 advisor／ruflo-project 的角色與交接，不派新 agent。

Human 操作偏好與驗收條件的單一參考檔是 [colab-human-requirements.md](../../../docs/operations/colab-human-requirements.md)。任何 Colab 交付、接續、環境／進度修改均須先讀該檔，尤其第8節來源相容與第9節排練；本清單不複製交付形式、Drive 結構、runtime 版本與操作步驟。
工程依據：[P4a 報告](../../../reports/v7_p4a_colab_engineering.md)的 Q3 事故、來源相容升級、自動排練及第12／13次複審；[P3 報告](../../../reports/v7_p3_portfolio_engineering.md)的 source identity、獨立准入與預算教訓。候選 gate 仍可能 UNMERGED／NOT_DEPLOYED／NOT_ACCEPTED，先查 project map，不能把本文當現場 acceptance。

以下程式路徑相對 repository root；`test_*.py` 均在 `V6/research_lab/tests/`。使用既有核准 CPU runtime，設定 `PYTHONPATH=V6 CUDA_VISIBLE_DEVICES=''`，例如 `python -m pytest -q V6/research_lab/tests/test_p4_rehearsal_gate.py`。真資料／GPU／Colab 現場核對須有該任務授權，否則標 UNVERIFIED 並停止交付升格。

## 交付順序與檢查

1. **第一次真實執行就是上線，不是排練**。
   - 怎麼檢查：先列全部 queue 類型首 fit、長窗與診斷路徑；讀 `p4_colab.py` 的 rehearsal 入口與 [colab-p4 操作指南](../../../docs/operations/colab-p4.md)，使用正式相同資料、模型、sampler、train_one／predict、發布／完成驗證路徑。排練輸出僅 `/tmp/p4-REHEARSAL-*`，不得混入正式目錄或收益結果。
   - 出事案例：P4a `02_formal@4e56e9d` Human 報告36個 Q1／Q2 fits 後，Q3 W120 才撞尾端 NaN；正常合成 smoke 未覆蓋真實形狀。跑 `test_p4_rehearsal.py` 可核 CPU 路徑，不能替代現場官方 mixer。

2. **正式前閘門必須覆蓋 Q3 W120、Q3 W250、Q5**。
   - 怎麼檢查：讀 `p4_colab.py` rehearsal gate 與 `p4_bundle.py` notebook 產生器；FORMAL／resume 預設先排練，三階段依序各1 step，Q3 各1 fit，Q5 先2 prerequisite fits 再 validation／evaluation permutation。另核 Human 需求第9節的每類首 fit，三階段 gate 不取代全部覆蓋。
   - 核長窗 eligible sessions／dates、全量推論時間、host RAM／CUDA peak bytes、官方 kernel／ABI；失敗／中斷不啟動正式 runner，顯示階段、期望／實際、下一步。檢查 `test_p4_rehearsal_gate.py::test_failure_never_launches_formal`、`test_long_window_actionable_preflight`、`test_notebook_actual_runner_cell_control_flow`。
   - 出事案例：P4a 自動排練候選源於 Q3 事故；CPU W250 測試通過仍不能宣稱 T4 峰值或真 fold 推論通過。

3. **驗證不能只憑返回碼0或檔案存在**。
   - 怎麼檢查：gate 自行讀各階段獨立 result JSON，核請求 nonce、queue／window／arm／steps、fit數、source／matrix／reference／實際 environment，重驗 scores／reference／checkpoint／complete identity 與 bytes。空輸出、吞例外後 exit0、錯 selection、缺發布都 STOP。
   - 出事案例：第12次複審 NO_GO 明指只看返回碼會放行；核 `test_p4_rehearsal_gate.py::test_real_subprocess_zero_exit_requires_evidence`、`test_gate_independently_rejects_invalid_publications`。

4. **成功標記要驗封閉 schema 與完整支援**。
   - 怎麼檢查：讀 `p4_colab.py` 共用遞迴 schema validator；`rehearsal_passed.json` 須恰好三階段、各 identity／elapsed／REHEARSAL evidence／fit數與 digest。綁定 run command、bundle source、prepared matrix／rankgauss、data realpath 與 environment；不符即重排練，不要求舊 `/tmp` 產物仍存在。
   - Q5 核 R=10、permutation整數0…9、daily ISO日期支援一致且無重複、`excluded_dates` 為日期清單、凍結 rule、數值有限及值域；拒絕未知／缺欄位、bool充數、過大數值、遞迴損壞與 metadata symlink。無效 marker 回報原因後按既有程式重排練。
   - 出事案例：第12次發現有效 JSON 可被清空仍略過；第13次補 Q5 nested schema 與 producer session index→真日期。核 `test_marker_complete_schema`、`test_gate3_marker_each_field_types`、`test_gate3_q5_real_dates_preserve_numeric_diagnostic`（均在 `test_p4_rehearsal_gate.py`）。digest 防誤改，不是來源簽章；marker 不等於 completed fit 或研究 authority。

5. **來源相容升級只限已認證的數值中立變更**。
   - 怎麼檢查：依 Human 需求第8節與事故專項認證範圍，讀 `p4_source_compat.py`；工具產生 from／to逐檔 SHA／size／diff／理由宣告，隨 zip 交付。模型、optimizer、RNG、研究配方、資料／矩陣、環境／預算契約不能順便改；AST pin 不是數值證明或獨立審查。
   - 舊 `02_formal/p4a-bundle.zip` 是舊 manifest digest 的逐檔證據，不能靠記憶或新 metadata 重建舊 run identity。缺舊 zip／宣告、來源或任何非來源 identity 不符均 STOP，不默開新 run、不重置帳本；新來源基準須先有跨版數值／接續測試與獨立審查。
   - 核 `test_p4_source_compat.py::test_old_head_training_rng_and_epoch_resume`、`test_generator_rejects_non_neutral_changes`；採用保留原 run_hash、binding、checkpoint bytes，並核 source-upgrade audit／SOURCE_UPGRADE ledger。
   - 出事案例：P4a 來源相容升級報告 `20e6b54` 只認證 `4e56e9d`，不是任意 commit 通行證；Q3 專項擴充亦須重新驗證，不能把 NaN 修補理由泛化到改研究規則。

6. **36個完成 fit 必須逐一驗證後採用**。
   - 怎麼檢查：舊 Q1／Q2 W60 前36個全部走原 `resume_fit`，核 formal identity／job／run_hash 與完整產物，保留 bytes、不重訓；missing 36th、corrupt、錯 run_hash 均拒絕。跑 `test_p4_source_compat.py::test_adoption_requires_all_36_valid_completed_fits`、`test_run_old_36_fits_then_upgrade_starts_only_first_q3`。
   - 出事案例：Q3 事故後要求保留36個已完成 fits；本機 CPU artifact double 的36→37→64接續證據，不是 Drive 現場36個模型都驗過，也不降低正式64-fit完整性。

7. **訓練／驗證執行中不 commit，也不在執行中 kernel 換碼**。
   - 怎麼檢查：先辨目前 run 使用 checkout 還是解壓 zip；`source_commit` 在 identity 內，`rolling_scores.py:provenance` 還會 hash research_lab sources。啟動前凍結 source，執行中不得改會被讀取的 bytes 或 commit；驗證結束後若獲授權才 commit／重產交付 identity。文件任務明確 no-commit 時維持 patch。
   - Colab 以已驗 zip 的解壓 bytes 為準：本機文件 patch 不會自動更新現場，也不能重標舊 zip；這不授權本機 commit 或 hot reload。升級待斷線／停止後開新 notebook，依既有相容流程驗證，不覆蓋正在跑的 kernel。
   - 出事案例：P4a 來源相容報告全套測試中修改 source，實際 `1 failed, 771 passed` 並報 `P3 source changed during run`；此為正確攔阻，不能改 gate 消音。

8. **進度可見性與預算餘裕必須排練**。
   - 怎麼檢查：依 Human 單一需求檔核 notebook 串流／flush、Drive 人可讀與機器快照、fit／epoch／loss／驗證／ETA／GPU／checkpoint 欄位；中斷、cached skip、接續 ETA 未知應顯示未知，不能沿用前fit或把0當估算。核 `test_p4_progress.py::test_stream_is_live_before_child_exit`、`test_resumed_eta_uses_only_this_launch_steps`。
   - 核剩餘全佇列、rankgauss／setup／全量推論與至少25%餘裕、未知耗時／費率、閒置／編譯未計部分；遵 Human 最新 record-only／無時間及CU上限需求，不恢復舊硬上限。既有 ledger／thermal／corruption gate 不因預算估計好看而放寬；讀 `test_p4_full.py::test_record_only_ledger_recovers_interruption`。
   - 出事案例：P4a 進度修補曾吞子程序輸出、接續 ETA 錯起點；早期預設1小時／15CU不足保守2.5小時／25CU計畫，屬歷史設定教訓，不能照抄成最新契約。

9. **通知後仍主動核完成，不能把通知當完成證據**。
   - 怎麼檢查：本次已授權工作期間查實際會寫的 progress／logs，再核 completed artifacts、hash、durable predictions／metrics／telemetry／manifest／報告；無權讀 Drive 時列待核。不得為此新增收件人、喚醒、常駐監看或週期任務。
   - 出事案例：Human 需求第6節明記曾漏通知；P3／P4a 多次報告 exit0／loadable checkpoint 不等於 completed 或 Human acceptance。沿用既有通知機制，本 skill 不授權送出訊息。

## 交付判定

列出 source／zip／contract／matrix／environment identity、實際命令與數值、各階段證據、NO_GO／UNKNOWN／UNVERIFIED、獨立審查與 Human acceptance 待辦。CPU替身、現場真資料／CUDA、Drive持久性與正式研究結果分開報告；未驗項不升格，依 AGENTS.md 固定交接欄位結束。
