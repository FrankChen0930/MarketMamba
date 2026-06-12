# MarketMamba V6

> **Mamba SSM + GATv2 Knowledge Graph · Daily Alpha Signal Generation · Full-Stack Web Dashboard**
>
> A personal quantitative investment automation system for Taiwan stock market (~2,500 stocks), running daily inference at market close and surfacing results through a live web dashboard.

**Live Demo** → [marketmamba.vercel.app](https://marketmamba.vercel.app)

---

## What This Does

Every trading day at 17:00, an automated pipeline:

1. Fetches the latest price, institutional flow, and fundamental data
2. Rebuilds a 56-factor feature matrix for all ~2,500 listed stocks
3. Runs **MarketMambaV6** (Mamba SSM + GATv2) to predict 5d/20d/60d Alpha for every stock
4. Generates an AI market report via Claude API
5. Scans for entry/exit signals and chart pattern formations
6. Pushes results to GitHub → auto-deploys to Render/Vercel

---

## Model Architecture — MarketMambaV6

~4M parameters, trained on Google Colab A100, inferred locally on RTX 3060 (WSL2).

```
Input: (N_stocks, SEQ_LEN=252, INPUT_DIM=56)
  ↓
FactorGroupedEmbedding
  Group A: Price & Momentum     (12 dims) → sub_dim 54
  Group B: Institutional Flow   (20 dims) → sub_dim 94
  Group C: Fundamentals         (12 dims) → sub_dim 54
  Group D: Macro Environment    (12 dims) → sub_dim 54
  All projected → Concat → d_model=256
  ↓
MultiScaleMambaEncoder (3 parallel branches, adaptive fusion)
  Short branch: last  20 steps × 2 Mamba layers
  Mid   branch: last  60 steps × 3 Mamba layers
  Long  branch: full 252 steps × 3 Mamba layers  ← zero-padding mask applied
  Fusion: scale_gate = Softmax(Linear(d_model×3 → 3))
  ↓
GATv2 (Knowledge Graph-guided cross-sectional interaction)
  ~640K edges, CSR sparse matrix
  ↓
Gating Fusion
  gate = Sigmoid(Linear(d_model×2 → d_model))
  ↓
MultiHorizonHead
  3 independent Linear layers → pred_5d / pred_20d / pred_60d

Output: [Alpha_5d, Alpha_20d, Alpha_60d]
```

### Key Design Decisions

**Multi-Scale Mamba**: Three time horizons capture different market dynamics — short (20 steps ≈ 1 month) for momentum, mid (60 steps ≈ 1 quarter) for trend, long (252 steps ≈ 1 year) for structural patterns. An adaptive `scale_gate` fuses the branches dynamically per-batch.

**GATv2 Knowledge Graph**: 4 edge types — TWSE sector classification, corporate group affiliation, supply chain relationships (TPEX data), and daily-rolling Pearson correlation edges. Edges are rebuilt into a CSR sparse matrix each inference cycle, enabling O(1) batch-level subgraph extraction.

**Zero-Padding Mask (V6.2)**: The Long branch (full 252 steps) may contain zero-padded positions for stocks with shorter histories. Short/Mid branches always slice the last 20/60 real steps, so no mask is needed there. Padding positions are multiplied by 0 to cut gradient flow during training.

**Uncertainty Estimation**: MC-Dropout (N=30 samples) produces per-stock uncertainty. `Signal_Quality = Net_Alpha_20d / (Uncertainty + 1e-6)`, clipped to [−10, +10].

### 56-Dimensional Feature Groups

| Group | Dims | Content |
|-------|------|---------|
| Price & Momentum | 12 | OHLCV, MA5/10/20/60, 1d/5d returns, RSI(14), ATR |
| Institutional Flow | 20 | Foreign/Investment Trust/Dealer net buy (abs + ratio), margin balance, short balance, KD(9,3), OBV, volume ratio |
| Fundamentals | 12 | Monthly revenue YoY, EPS, P/E, P/B, ROE, market cap, dividend yield, FCF |
| Macro Environment | 12 | TAIEX/SPX/Gold returns, VIX, DXY, TWD/USD, PMI, Fed rate |

V6.2 training adds RS_5d/RS_20d/RS_60d (relative strength vs market), expanding to 59 dims.

---

## Signal System

### Entry Scoring (max 150 points)

| Condition | Points |
|-----------|--------|
| Rank stability (Top10 ≥2 days or Top50 ≥3 days) | 30 |
| Low uncertainty (< daily Q30 percentile) | 25 |
| Institutional consecutive net-buy ≥2 days | 25 |
| Relative low (RSI<40 or price<MA20) | 20 |
| Pattern score 60–74 | +20 |
| Pattern score 75–89 | +30 |
| Pattern score ≥90 | +40 |
| Double-confirmation (pattern≥60 AND alpha rank≤200) | +10 |

Entry threshold: ≥70 pts (normal market, TWII > MA60) / ≥90 pts (conservative mode)

### 4-Layer Exit Logic

| Layer | Trigger | Action |
|-------|---------|--------|
| 1 | Trailing stop hit / pattern failure price breached / foreign sell 3 days / M-top confirmed / held >30 days | Exit all |
| 2 | Rank drops out of Top50 for 2 days / Uncertainty doubles / RS_20d negative 3 days | Exit all |
| 3 | RSI>75 + fading momentum / return ≥+20% / Alpha_20d declining 3 days | Halve position |
| 4 | SQ rank in bottom 50% + new signals available + fully invested | Rotate |

**Trailing Stop (ratchet — only moves up):**

| Peak return | Stop level |
|-------------|-----------|
| < +5% | Entry − 5% |
| ≥ +5% | Entry + 2% |
| ≥ +10% | Entry + 6% |
| ≥ +15% | Entry + 10% |

### Chart Pattern Scanner

**Bullish (entry scoring boost)**: W-bottom, Spring W-bottom (false breakdown), Head & Shoulders Bottom, Converging Triangle Bottom, Ascending Flag

**Bearish (exit trigger)**: M-top (neckline break confirmed), False Breakout Down

Each bullish signal includes a `failure_stop` price fed directly into Layer-1 exit logic.

---

## Daily Inference Pipeline

```
17:00  Windows Task Scheduler
         └─ daily_inference.bat
              └─ WSL2 Ubuntu → run_daily_inference.py

  [1]  Data fetch      yfinance + FinMind API + TWSE direct HTTP
  [2]  Feature build   56-factor matrix + update_correlation_edges()
  [3]  Inference       MarketMambaV6 → df_kelly.csv, df_traj.csv
                       MC-Dropout ×30 → Uncertainty column
  [4]  LLM report      Claude API → market_summary.json
  [5]  Archive         Rolling 90-day; history_index.json (60 trading days)
  [6]  Signal scan     scanner.py + pattern_scanner.py → action_signals.json
  [7]  Push            git push → GitHub → Render cache refresh
```

Progress shown via tkinter window (WSLg). Auto-closes in 3s on success; stays on top on failure.

### Output Files (`V6/results/`)

| File | Content |
|------|---------|
| `df_kelly.csv` | Full-market Alpha rankings (~2,500 stocks) |
| `df_traj.csv` | Multi-horizon prediction trajectories |
| `action_signals.json` | BUY / EXIT / WATCH signals |
| `pattern_signals.json` | Pattern scan results with `failure_stop` |
| `history_index.json` | Daily Top-50 history (60 trading days) |
| `market_summary.json` | Claude LLM market analysis |
| `model_tracker.jsonl` | Daily append: date / val_ic / top50 / duration |

---

## Web Dashboard

**Frontend** (Vite + React) → Vercel: [marketmamba.vercel.app](https://marketmamba.vercel.app)

**Backend** (FastAPI) → Render: [marketmamba-api.onrender.com](https://marketmamba-api.onrender.com)

Data flow: GitHub `V6/results/` → Render pulls on startup (1hr TTL, `asyncio.Lock`) → REST API → Frontend

| Page | Route | Description |
|------|-------|-------------|
| Dashboard | `/` | Top-50 Alpha ranking table, sector heatmap, global market sidebar |
| Scanner | `/scanner` | Entry/exit signal cards with 4-condition breakdown per stock |
| Quant Analysis | `/quant` | 5 tabs: technicals, institutional flow, market breadth, model alpha, chart patterns |
| Portfolio | `/portfolio` | Real holdings via Sinopac Shioaji API, cross-referenced with exit signals |
| AI Daily Report | `/market` | Claude-generated market analysis + LLM Top-10 picks |
| Model Status | `/model` | IC/ICIR metrics, training curves, Walk-Forward validation results |
| Sim Robot | `/sim` | Backtest: Alpha robot (SQ Top-20) + Scanner robot (4-condition scoring) |

---

## Training History

| Run | Config | Best Val IC | Notes |
|-----|--------|-------------|-------|
| V6.1 | INPUT_DIM=56 | 0.0825 | GATv2 effectively inactive (edge indexing bug) |
| V6.2 R1 | +RS features (59 dims) | 0.0950 | scale_gate Long-dominant (0.989); RS features not fully utilized |
| V6.2 R2 | +Zero-Padding Mask | In progress | Targeting balanced scale_gate |

**Validated**: Look-ahead bias is protected — `_merge_financials()` uses `available_from = Date + 45 days` per IFRS reporting deadlines.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Sequence model | Mamba SSM (`mamba-ssm 2.3.0`) |
| Graph neural network | GATv2Conv (`PyTorch Geometric`) |
| Data sources | FinMind API + yfinance + TWSE direct HTTP |
| LLM reports | Anthropic Claude API |
| Frontend | Vite + React + Recharts |
| Backend | FastAPI + Uvicorn |
| Portfolio sync | Sinopac `shioaji` |
| Training | Google Colab A100 |
| Inference | RTX 3060 (WSL2 Ubuntu) |
| Frontend deploy | Vercel |
| Backend deploy | Render |

---

## Repository Structure

```
MarketMamba/
├── V6/
│   ├── marketmamba/
│   │   ├── config.py                  ← Global hyperparams & paths
│   │   ├── data/
│   │   │   ├── fetcher.py             ← FinMind + yfinance crawler
│   │   │   └── feature_engineer.py   ← 56-factor feature engineering
│   │   ├── models/                    ← ⚠️ Do NOT modify checkpoints
│   │   │   ├── architecture.py        ← MarketMambaV6 model definition
│   │   │   └── trainer.py             ← Training loop
│   │   ├── signals/
│   │   │   ├── scanner.py             ← Weighted entry/exit scanner (v1.2)
│   │   │   └── signal_conditions.py  ← 150-pt entry scoring + 4-layer exit
│   │   ├── quant/
│   │   │   └── pattern_scanner.py    ← 5 bullish + 2 bearish patterns
│   │   ├── knowledge/
│   │   │   └── graph_builder.py      ← KG construction
│   │   ├── llm/
│   │   │   └── report_generator.py   ← Claude API daily report
│   │   └── backtest/
│   │       └── sim_engine_v3.py      ← Stateful daily simulation robot
│   ├── run_daily_inference.py         ← Daily inference entry point (WSL2)
│   ├── results/                       ← Daily outputs (git-pushed)
│   └── models/                        ← ⚠️ Trained checkpoints
│
├── app/
│   ├── backend/                       ← FastAPI (Render)
│   └── frontend/                      ← Vite + React (Vercel)
│
└── archive/                           ← V3–V5.5 (reference only)
```

---

## Running Locally

```bash
# Clone
git clone https://github.com/FrankChen0930/MarketMamba.git
cd MarketMamba

# Daily inference (WSL2)
wsl -d Ubuntu -- bash -lc "
  source ~/miniconda3/etc/profile.d/conda.sh && conda activate mamba_env &&
  cd /mnt/d/Desktop/work/ProjectForMe/MarketMamba &&
  python V6/run_daily_inference.py"

# Frontend dev server
cd app/frontend && npm install && npm run dev   # → localhost:5173

# Backend dev server
cd app/backend && pip install -r requirements.txt && uvicorn main:app --reload
```

**Environment variables** (`V6/.env`):
```
FINMIND_TOKEN=...
ANTHROPIC_API_KEY=...
RENDER_BACKEND_URL=https://marketmamba-api.onrender.com
```

---

## License

MIT License © 2024–2026 FrankChen
