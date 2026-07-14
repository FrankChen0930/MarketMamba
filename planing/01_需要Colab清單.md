# 需要 Colab（GPU 雲端訓練，有算力成本）的待辦

> 掃描全專案（CLAUDE.md、docs/、obsidian_note/、本次 Cowork 討論）整理，2026-07-11。
> 判準：凡涉及 Mamba+GATv2 主模型（或其分身 v6_short/v6_trend）**重新訓練或需要多次訓練**才能拿到結果的項目，都算需要 Colab。單純用既有 checkpoint 推論、或訓練規模遠小於主模型的工作，不算在內（見 `02_不需要Colab清單.md`）。
>
> **現況提醒**：2026-07-06 使用者已決定**暫停所有模型實驗**，等真倉（V6.1）先驗證有沒有賺錢再繼續。以下項目是「已定案要做、但還沒做」，不是「現在要馬上排」——排序時請先確認暫停決定是否已解除。

---

## Phase 3：短線／趨勢模型抬升 IC 實驗（`docs/phase3-experiment-plan-2026-06-25.md`）

| # | 項目 | 狀態 | 優先級 | 備註 |
|---|---|---|---|---|
| 1 | 實驗 A 正則救峰值（dropout sweep） | ✅ 已完成（dropout=0.2 有效，已定案帶進趨勢模型） | — | 唯一已跑完的實驗 |
| 2 | 實驗 B listnet 權重 sweep | 檔案已寫好推 main，**尚未在 Colab 執行** | 中（暫停中） | `V6/experimental/phase3_b_listnet_sweep.py` |
| 3 | 實驗 C 趨勢單尺度簡化 | 檔案已寫好推 main，**尚未在 Colab 執行** | 中（暫停中） | `V6/experimental/phase3_c_trend_single_scale.py`，把握度高（預期持平即可砍多尺度） |
| 4 | 實驗 D 短線窗口 sweep | 檔案已寫好推 main，**尚未在 Colab 執行** | 中（暫停中） | `V6/experimental/phase3_d_window_sweep.py` |
| 5 | 實驗 E 多 seed 集成 | 尚未設計、尚未寫程式 | 中高（把握度高，方向明確：能穩定小升 IC 並降低脆弱性） | 需 N 倍訓練，推論端要載多 checkpoint 平均，上線複雜度會增加 |
| 6 | 實驗 F 特徵分離（短線快特徵/趨勢慢特徵） | 尚未設計，大工程 | 低（把握度最低，潛在升幅也最大，留最後） | 需重建 feature matrix，是這批實驗裡成本最高的一項 |

**恢復順序**：按計畫接著跑 B，再 C、D、E，F 留最後。

---

## Phase 4-A：知識圖譜豐富化進模型（`docs/phase4-industry-chain-fusion-plan-2026-06-27.md`）

| # | 項目 | 狀態 | 優先級 | 備註 |
|---|---|---|---|---|
| 7 | 爬 ic.tpex 產業鏈 → 建邊 → 併入 KG cache | 計畫定稿，尚未動工 | 中（待 Phase 3 完成後接續） | 爬蟲與建邊本身不需要 Colab |
| 8 | 下次重訓驗證加產業鏈邊後 IC 是否提升 | 待 7 完成 | 中 | **驗證這步驟需要重訓，落在 Colab** |

---

## 其他模型層待辦

| # | 項目 | 狀態 | 優先級 | 備註 |
|---|---|---|---|---|
| 9 | Scale Gate 均衡化（若後續仍偏 Long）：branch-level dropout / loss 正則化 | 待觀察，備案 | 低（目前傾向用 Phase 3-C 的「單尺度簡化」直接解決，可能不需要另外做） | 來源：`obsidian_note/04 發想/發想空間.md` |
| 10 | Walk-Forward 驗證例行化（U7） | 未做，目前只有 Quick Fold IC | 中 | 真正的滾動窗口 walk-forward 需要多次重訓，屬於 Colab 工作；來源：`docs/architecture-analysis-2026-06-12.md` |
| 11 | 不確定性升級：deep ensembles（3–5 個不同 seed checkpoint） | 未做 | 低（`docs/uncertainty-calibration-2026-06-13.md` 已顯示現有 MC-Dropout/Signal_Quality 排序力已經夠用，這項優先度可降低） | 對應方向三 C5；conformal prediction 路線不需要 Colab，見 `02_不需要Colab清單.md` |
| 12 | 若決定不走 5d 路線，切回 20d 重訓 | 待決策 | 低 | 目前 remote main 是 5d 實驗設定，回退需要重新跑訓練驗證 |
| 13 | V6.1 是否也要套用 D1 macro 修復（`macro_norm="ts"`）並重訓 | **狀態不明，需向你確認** | — | D1 修復已在雙模型（v6_short/v6_trend）上生效，但正式真倉線 V6.1（56 維）是否也要升級尚未定案；CLAUDE.md 舊的「V6.2 部署 checklist」條目可能已被雙模型上線取代，建議下次動工前先確認這條是否還適用 |

---

## 本次 Cowork 討論新增項目中，需要 Colab 的部分

`研究計畫_方向一/二/三` 與 `雙模型架構重整計畫` 裡的所有項目都不需要 Colab（見 `研究計畫_主檔.md` 開頭的速覽說明），僅在方向三 C5 走向 deep ensembles 時才會碰到（已列在上方第 11 項）。

| # | 項目 | 狀態 | 優先級 | 備註 |
|---|---|---|---|---|
| 14 | 資料基礎升級計畫階段三：用新資料基礎（v2.0 協定）重訓一版 Mamba baseline，對照舊 V6.1／雙模型 IC 確認新基礎沒有讓模型退化 | 待階段二（資料衛生修復＋序列輸出擴充）完成 | 中 | 見 `資料基礎升級計畫_baseline_common扶正.md` 階段三；不是要現在汰換 V6.1，只是驗證新基礎品質 |

---

> 本檔案為 2026-07-11 全專案掃描整理產出，來源涵蓋 `CLAUDE.md`、`docs/`、`obsidian_note/` 與本次 Cowork 討論。
