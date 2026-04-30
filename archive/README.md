# Archive — MarketMamba 歷史版本存檔

此目錄存放 V6 之前的所有版本，**僅供查閱，不可直接執行**。

## 版本演進

| 版本 | 目錄/檔案 | 主要技術 | 備註 |
|------|-----------|---------|------|
| V1 | `MarketMamba_V1.ipynb` | 基本 LSTM | 概念驗證 |
| V2 | `MarketMamba_V2.ipynb` | 改良 LSTM + 技術指標 | |
| V3 | `MarketMamba_V3.ipynb` | 加入基本面因子 | |
| V4 | `V4/` | Transformer 嘗試 | |
| V5 | `V5/` | Mamba 骨幹，單股 | |
| V5.5 | `V5.5_repo/` | + 型態學掃描、KG 雛形 | 含 pattern/scanner.py |
| **V6** | `../V6/` | Mamba+GAT, 46D, KG | **現行生產版** |

## 重要歷史資產

- **`V5.5_repo/marketmamba/pattern/`** — 傳統型態學掃描程式碼（已移植到前端展示）
- **`V5_code.py`** — V5 完整單檔版本，方便快速參考
- **`Get_Data/`** — 舊版 FinMind 抓資料腳本（V6 已有更完整的版本）

## 訓練資料位置

```
../Data/processed_v6/          ← V6 訓練用資料（全部保留）
  V6_Feature_Matrix.parquet    ← 2.5GB，2888支股票×46特徵×12年
  prices_raw.parquet           ← 原始日K價格
  institutional_raw.parquet    ← 三大法人
  margin_raw.parquet           ← 融資融券
  ...等 15 個 parquet 檔
```

## 下次重訓前需要做

1. 壓縮 `Data/processed_v6/` 成 zip → 上傳 Google Drive → 掛載 Colab
2. 執行 `V6/scripts/train_v6.py`（或相應的訓練腳本）
3. 訓練完後下載 `v6_best.pt` → 放到 `V6/models/`
