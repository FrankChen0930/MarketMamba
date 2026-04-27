"""
Mock data for development.
Mirrors app/frontend/src/mockData.js

Replace with real inference outputs when v6_final.pt is ready:
  - signals:     read from V6/results/df_kelly.csv
  - performance: read from V6/results/ic_history.json
  - portfolio:   call shioaji API
  - market:      call TWSE API
"""
import math
from schemas import (
    SignalItem, SignalsResponse,
    ICPoint, WFFold, CumRetPoint, PerformanceResponse,
    PortfolioItem, PortfolioResponse,
    TaiexStatus, MarketStatusResponse,
    TickerItem, TickerResponse,
)

# ── Signals ───────────────────────────────────────────────────────────────────

MOCK_SIGNALS = SignalsResponse(
    date="2026-04-27",
    model_version="V6",
    total_stocks=2888,
    signals=[
        SignalItem(rank=1,    stock_id="2330", name="台積電",   sector="半導體",
                   alpha_5d=0.142, alpha_20d=0.187, alpha_60d=0.231,
                   uncertainty=0.018, vol_ratio=1.32, signal="BUY",
                   suggested_weight=0.15, confidence="高信心"),
        SignalItem(rank=2,    stock_id="2454", name="聯發科",   sector="半導體",
                   alpha_5d=0.121, alpha_20d=0.162, alpha_60d=0.198,
                   uncertainty=0.022, vol_ratio=1.18, signal="BUY",
                   suggested_weight=0.12, confidence="高信心"),
        SignalItem(rank=3,    stock_id="2317", name="鴻海",     sector="電子製造",
                   alpha_5d=0.108, alpha_20d=0.148, alpha_60d=0.171,
                   uncertainty=0.031, vol_ratio=0.95, signal="BUY",
                   suggested_weight=0.09, confidence="中信心"),
        SignalItem(rank=4,    stock_id="2382", name="廣達",     sector="電子製造",
                   alpha_5d=0.101, alpha_20d=0.139, alpha_60d=0.162,
                   uncertainty=0.028, vol_ratio=1.05, signal="BUY",
                   suggested_weight=0.08, confidence="中信心"),
        SignalItem(rank=5,    stock_id="3034", name="聯詠",     sector="半導體",
                   alpha_5d=0.094, alpha_20d=0.131, alpha_60d=0.155,
                   uncertainty=0.034, vol_ratio=0.87, signal="BUY",
                   suggested_weight=0.07, confidence="中信心"),
        SignalItem(rank=6,    stock_id="2357", name="華碩",     sector="電子製造",
                   alpha_5d=0.088, alpha_20d=0.125, alpha_60d=0.148,
                   uncertainty=0.041, vol_ratio=0.92, signal="BUY",
                   suggested_weight=0.06, confidence="中信心"),
        SignalItem(rank=7,    stock_id="2308", name="台達電",   sector="電子零組件",
                   alpha_5d=0.081, alpha_20d=0.118, alpha_60d=0.141,
                   uncertainty=0.038, vol_ratio=1.11, signal="BUY",
                   suggested_weight=0.05, confidence="中信心"),
        SignalItem(rank=8,    stock_id="2379", name="瑞昱",     sector="半導體",
                   alpha_5d=0.074, alpha_20d=0.112, alpha_60d=0.131,
                   uncertainty=0.045, vol_ratio=0.78, signal="HOLD",
                   suggested_weight=0.04, confidence="中信心"),
        SignalItem(rank=9,    stock_id="6505", name="台塑石化", sector="化工",
                   alpha_5d=0.061, alpha_20d=0.098, alpha_60d=0.112,
                   uncertainty=0.052, vol_ratio=0.85, signal="HOLD",
                   suggested_weight=0.03, confidence="低信心"),
        SignalItem(rank=10,   stock_id="2881", name="富邦金",   sector="金融",
                   alpha_5d=0.054, alpha_20d=0.087, alpha_60d=0.101,
                   uncertainty=0.048, vol_ratio=1.02, signal="HOLD",
                   suggested_weight=0.02, confidence="低信心"),
        SignalItem(rank=2879, stock_id="1301", name="台塑",     sector="化工",
                   alpha_5d=-0.098, alpha_20d=-0.132, alpha_60d=-0.161,
                   uncertainty=0.041, vol_ratio=0.72, signal="SELL",
                   suggested_weight=0.0, confidence="中信心"),
        SignalItem(rank=2880, stock_id="9904", name="寶成",     sector="其他",
                   alpha_5d=-0.107, alpha_20d=-0.141, alpha_60d=-0.172,
                   uncertainty=0.038, vol_ratio=0.68, signal="SELL",
                   suggested_weight=0.0, confidence="中信心"),
        SignalItem(rank=2881, stock_id="2002", name="中鋼",     sector="鋼鐵",
                   alpha_5d=-0.118, alpha_20d=-0.155, alpha_60d=-0.188,
                   uncertainty=0.044, vol_ratio=0.61, signal="SELL",
                   suggested_weight=0.0, confidence="中信心"),
        SignalItem(rank=2882, stock_id="2603", name="長榮",     sector="航運",
                   alpha_5d=-0.131, alpha_20d=-0.168, alpha_60d=-0.201,
                   uncertainty=0.051, vol_ratio=0.58, signal="SELL",
                   suggested_weight=0.0, confidence="低信心"),
        SignalItem(rank=2883, stock_id="2615", name="萬海",     sector="航運",
                   alpha_5d=-0.142, alpha_20d=-0.179, alpha_60d=-0.218,
                   uncertainty=0.056, vol_ratio=0.54, signal="SELL",
                   suggested_weight=0.0, confidence="低信心"),
    ]
)

# ── Performance / IC History ──────────────────────────────────────────────────

_ic_pts = [
    ICPoint(epoch=i + 1,
            train_loss=round(2.8 - i * 0.08 + (0.05 if i % 3 == 0 else -0.02), 4),
            val_loss=round(2.5 - i * 0.05 + (0.04 if i % 4 == 0 else -0.01), 4),
            val_ic=round(min(0.10, 0.01 + i * 0.005 + (0.003 if i % 2 == 0 else -0.001)), 4))
    for i in range(20)
]

_cumret = [
    CumRetPoint(
        month=f"{'2024' if i < 12 else '2025'}-{str(i % 12 + 1).zfill(2)}",
        model=round(100 * math.exp(0.012 * i + math.sin(i * 0.5) * 0.02), 2),
        benchmark=round(100 * math.exp(0.006 * i + math.sin(i * 0.3) * 0.015), 2),
    )
    for i in range(24)
]

MOCK_PERFORMANCE = PerformanceResponse(
    ic_history=_ic_pts,
    wf_folds=[
        WFFold(fold="F01", period="2010-2012", ic=0.041, icir=0.52, sharpe=1.21, ret=0.148),
        WFFold(fold="F02", period="2012-2014", ic=0.058, icir=0.71, sharpe=1.87, ret=0.213),
        WFFold(fold="F03", period="2014-2016", ic=0.033, icir=0.44, sharpe=0.94, ret=0.087),
        WFFold(fold="F04", period="2016-2018", ic=0.062, icir=0.78, sharpe=2.14, ret=0.261),
        WFFold(fold="F05", period="2018-2020", ic=0.021, icir=0.29, sharpe=0.65, ret=0.041),
        WFFold(fold="F06", period="2020-2022", ic=0.074, icir=0.89, sharpe=2.43, ret=0.318),
        WFFold(fold="F07", period="2022-2024", ic=0.061, icir=0.76, sharpe=2.01, ret=0.234),
    ],
    cumret=_cumret,
    best_val_ic=0.0744,
    best_epoch=15,
    training_status="training",
)

# ── Portfolio ─────────────────────────────────────────────────────────────────

MOCK_PORTFOLIO = PortfolioResponse(
    total_pnl=86500.0,
    total_value=2_412_000.0,
    data_source="mock",
    last_updated="2026-04-27 15:30",
    positions=[
        PortfolioItem(stock_id="2330", name="台積電", qty=1000,
                      avg_price=830.0, current_price=915.0,
                      pnl=85000.0, pnl_pct=10.24, model_signal="BUY"),
        PortfolioItem(stock_id="2454", name="聯發科", qty=500,
                      avg_price=1180.0, current_price=1205.0,
                      pnl=12500.0, pnl_pct=2.12, model_signal="BUY"),
        PortfolioItem(stock_id="2317", name="鴻海", qty=2000,
                      avg_price=145.0, current_price=139.5,
                      pnl=-11000.0, pnl_pct=-3.79, model_signal="HOLD"),
    ]
)

# ── Market Status ─────────────────────────────────────────────────────────────

MOCK_MARKET = MarketStatusResponse(
    taiex=TaiexStatus(value=22381.4, change=87.3, change_pct=0.39),
    advancing=563,
    declining=412,
    model_ic=0.0744,
    last_run="2026-04-27 15:30",
    run_status="not_ready",
    training_epoch=5,
    training_status="training",
)

# ── Ticker ────────────────────────────────────────────────────────────────────

MOCK_TICKER = TickerResponse(items=[
    TickerItem(id="TAIEX", name="加權",    price="22,381", change="+87.3",  pct="+0.39%", up=True),
    TickerItem(id="2330",  name="台積電",  price="915.0",  change="+12.0",  pct="+1.33%", up=True),
    TickerItem(id="2454",  name="聯發科",  price="1,205",  change="+25.0",  pct="+2.12%", up=True),
    TickerItem(id="2317",  name="鴻海",    price="139.5",  change="-2.5",   pct="-1.76%", up=False),
    TickerItem(id="2382",  name="廣達",    price="278.0",  change="+6.5",   pct="+2.40%", up=True),
    TickerItem(id="2881",  name="富邦金",  price="82.5",   change="-0.8",   pct="-0.96%", up=False),
    TickerItem(id="2308",  name="台達電",  price="312.0",  change="+4.5",   pct="+1.46%", up=True),
    TickerItem(id="6505",  name="台塑石化", price="98.5",  change="-1.2",   pct="-1.20%", up=False),
])
