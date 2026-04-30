from pydantic import BaseModel
from typing import List, Optional


class SignalItem(BaseModel):
    rank: int
    stock_id: str
    name: str
    sector: str
    alpha_5d: float
    alpha_20d: float
    alpha_60d: float
    uncertainty: float
    vol_ratio: float
    signal: str          # BUY / HOLD / SELL
    suggested_weight: float
    confidence: str      # 高信心 / 中信心 / 低信心


class SignalsResponse(BaseModel):
    date: str
    model_version: str
    total_stocks: int
    signals: List[SignalItem]


class ICPoint(BaseModel):
    epoch: int
    train_loss: float
    val_loss: float
    val_ic: float


class WFFold(BaseModel):
    fold: str
    period: str
    ic: float
    icir: float
    sharpe: float
    ret: float


class CumRetPoint(BaseModel):
    month: str
    model: float
    benchmark: float


class PerformanceResponse(BaseModel):
    ic_history: List[ICPoint]
    wf_folds: List[WFFold]
    cumret: List[CumRetPoint]
    best_val_ic: float
    best_epoch: int
    training_status: str   # "training" / "completed"


class PortfolioItem(BaseModel):
    stock_id: str
    name: str
    qty: int
    avg_price: float
    current_price: float
    pnl: float
    pnl_pct: float
    model_signal: str      # BUY / HOLD / SELL / N/A


class PortfolioResponse(BaseModel):
    total_pnl: float
    total_value: float
    positions: List[PortfolioItem]
    data_source: str       # "mock" / "shioaji"
    last_updated: str


class TaiexStatus(BaseModel):
    value: float
    change: float
    change_pct: float


class MarketStatusResponse(BaseModel):
    taiex: TaiexStatus
    advancing: int
    declining: int
    model_ic: float
    last_run: str
    run_status: str        # "completed" / "running" / "not_ready"
    training_epoch: Optional[int] = None
    training_status: str = "completed"
    # Macro indicators (from yfinance)
    spx_change: float = 0.0   # S&P 500 % change today
    vix: float = 0.0           # VIX current level
    gold_change: float = 0.0   # Gold futures % change
    usd_twd: float = 0.0       # USD/TWD exchange rate



class TickerItem(BaseModel):
    id: str
    name: str
    price: str
    change: str
    pct: str
    up: bool


class TickerResponse(BaseModel):
    items: List[TickerItem]


class InferenceStatus(BaseModel):
    status: str
    message: str
