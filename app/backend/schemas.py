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
    freshness_warning: Optional[str] = None


class ICPoint(BaseModel):
    epoch: int
    train_loss: float
    val_loss: float
    val_ic: float


class ScaleGatePoint(BaseModel):
    epoch: int
    short: float
    mid: float
    long: float


class OnlineICPoint(BaseModel):
    pred_date: str
    future_date: str
    ic: float
    n_stocks: int


class OnlineICSummary(BaseModel):
    horizon: str           # "5d" / "20d"
    n_days: int
    mean_ic: float
    icir: float
    t_stat: float
    ic_gt0_pct: float


class TrainingInfo(BaseModel):
    """來自 V6/results/training_status.json（Colab 訓練時逐 epoch 寫出）"""
    model_version: str
    status: str            # "training" / "completed" / "early_stopped"
    started_at: Optional[str] = None
    updated_at: Optional[str] = None
    epoch: int
    epochs_max: int
    best_val_ic: Optional[float] = None
    best_ic_epoch: int = 0
    best_val_loss: Optional[float] = None
    early_stop_patience: Optional[int] = None
    config: dict = {}


class PerformanceResponse(BaseModel):
    training: Optional[TrainingInfo] = None      # None = 尚無 training_status.json
    ic_history: List[ICPoint] = []               # 訓練學習曲線（真實）
    scale_gates: List[ScaleGatePoint] = []       # Scale Gate 三分支 epoch 曲線
    online_ic: List[OnlineICPoint] = []          # 線上 5d IC 時序（ic_analysis.json）
    online_summary: List[OnlineICSummary] = []   # 線上 IC 統計摘要
    online_period: Optional[str] = None          # 線上 IC 統計區間
    data_sources: dict = {}                      # 各資料來源載入狀態（除錯用）


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
    spx_change: float = 0.0
    vix: float = 0.0
    gold_change: float = 0.0
    usd_twd: float = 0.0
    jpy_twd: float = 0.0       # JPY/TWD exchange rate (100 JPY → TWD)



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
