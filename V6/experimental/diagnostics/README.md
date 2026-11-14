# diagnostics/ — 一次性診斷腳本

**全部純讀**：只讀 `result/scores/*.parquet`、`result/*.json`、base matrix，
不寫任何檔案、不動 `V6/experimental/` 既有程式、不碰 production。

保留的理由：這些就是產出 CLAUDE.md 所記數字的程式碼本身，逐字保留才對得上。

⚠️ **路徑是硬寫的**（`ROOT` 指向本機 repo、2×2 那支讀 `D:\Downloads\`），
換機器要改。當時是為了一次性分析、沒有做成通用工具。

---

## GRU decile spread Sharpe = 2.846 的診斷（2026-08-02）

四階，依序跑。結論：**不是 bug、不是 panel 差異，是真的**。

| 檔案 | 回答什麼 | 關鍵產出 |
|---|---|---|
| `gru_decile_01_panel.py` | panel 對不對齊？Sharpe 高是分子大還是分母小？ | gru 與 ridge 的 (Date,stock_id) **逐對相同**；兩腳拆解顯示**分子大 64%**、分母反而最大 |
| `gru_decile_02_nature.py` | 是小型股/少數幾天/下市邊緣列造成的嗎？ | 高流動前 1/3 下 Sharpe **2.66**（優勢更大）；去最好 20 天仍 1.56；兩端 NaN 標籤 ≤0.02% |
| `gru_decile_03_common_ruler.py` | 用同一把尺重算 IC，並看十分位輪廓 | GBDT +0.1027 ≈ GRU +0.1016 > v2_kg +0.0989；GRU 輪廓**接近嚴格單調**、D9−D0 最大 |
| `gru_decile_04_robustness.py` | 換 decile 寬度／再平衡頻率還在嗎？顯著嗎？ | freq=1 時 GBDT 反勝，**freq 降低 GRU 就拉開**；配對 NW t +2.80~+3.48 |

## Group D × GAT 2×2（2026-08-02）

`groupd_gat_2x2_analysis.py` — 讀三個 ablation JSON，組 2×2 表 + 配對 Newey-West。

結論：兩效應**可加**（交互作用 −0.0032、t=−0.59）；最佳格 `no_macro + v2 圖`
**IC +0.1145 / ICIR 1.340**；**移除 Group D 讓 GAT 的配對比較離散度小 2.9 倍**。

需要的輸入檔（都在 Drive／`D:\Downloads\`，不進 git）：
`kg_ablation_result.json`、`groupd_ablation_result.json`、`groupd_ablation_result_gatv2.json`
