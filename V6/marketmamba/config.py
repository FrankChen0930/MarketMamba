"""
MarketMamba V6 — Global Configuration
=======================================
Single source of truth for all hyperparameters, paths, and feature definitions.
All other modules import from here; never hard-code values elsewhere.
"""

from pathlib import Path

# Load .env file before anything else so environment variables are available
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(dotenv_path=_env_path, override=False)  # override=False: system env wins
except ImportError:
    pass  # python-dotenv not installed — rely on system environment variables

# ============================================================
# Project Paths
# ============================================================
# Resolve absolute root regardless of where the script is called from
_THIS_FILE = Path(__file__).resolve()
ROOT_DIR   = _THIS_FILE.parent.parent          # .../V6/
DATA_DIR   = ROOT_DIR.parent / "Data"          # .../Data/  (shared with V5.5)
PROCESSED_DIR = DATA_DIR / "processed_v6"      # V6 uses its own processed folder
MODELS_DIR    = ROOT_DIR / "models"
RESULTS_DIR   = ROOT_DIR / "results"
KG_CACHE_PATH = PROCESSED_DIR / "knowledge_graph_cache.npz"
LLM_REPORT_PATH = RESULTS_DIR / "market_summary.json"
NEWS_CACHE_DIR  = DATA_DIR / "News_Cache"

# Auto-create directories on import
for _d in [PROCESSED_DIR, MODELS_DIR, RESULTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ============================================================
# Data / Universe Settings
# ============================================================
DATA_START_DATE = "2005-01-01"   # Maximum coverage: institutional data starts 2005-01-01
DATA_END_DATE   = None           # None = today

# Stock universe filters (applied during data fetch)
MIN_MARKET_CAP_TWD = 5e8        # 5億台幣，過濾微型股
MIN_AVG_VOLUME_5D  = 1e7        # 1000萬台幣日均量（實盤流動性門檻）

# ============================================================
# Feature Engineering
# ============================================================
SEQ_LEN    = 252    # One full trading year (V5.5 was 180)
INPUT_DIM  = 46     # Pure quant, no sentiment (V5.5 was 84)

# Factor groups for FactorGroupedEmbedding
# Indices are 0-based positions in the final feature tensor
FEATURE_GROUPS = {
    "price_momentum": [           # Group A — 12 dims
        "Open", "High", "Low", "Close", "Volume",
        "Return_1d", "Return_5d", "Return_20d",
        "MA_20", "MA_60", "RSI_14", "ATR_14",
    ],
    "institutional_flow": [       # Group B — 16 dims
        "Foreign_Buy", "Foreign_Sell", "Foreign_Net",
        "Investment_Trust_Net", "Dealer_Net",
        "Margin_Purchase", "Margin_Repay",
        "Short_Sale", "Short_Cover",
        "Margin_Balance", "Short_Balance",
        "Day_Trade_Volume", "KD_K", "KD_D", "OBV", "Volatility_20d",
    ],
    "fundamentals": [             # Group C — 10 dims
        "PER", "PBR", "Revenue_MoM", "Revenue_YoY",
        "EPS", "EPS_Surprise", "Gross_Margin", "ROE",
        "Market_Cap_Log", "Book_Value",
    ],
    "macro_environment": [        # Group D — 8 dims
        "TWII_Return", "SPX_Return", "VIX", "TNX",
        "Gold_Return", "Oil_Return", "USD_TWD", "Market_Closed",
    ],
}

# Flat ordered feature list (used by cleaner / scaler)
FEATURE_COLS: list[str] = (
    FEATURE_GROUPS["price_momentum"]
    + FEATURE_GROUPS["institutional_flow"]
    + FEATURE_GROUPS["fundamentals"]
    + FEATURE_GROUPS["macro_environment"]
)
assert len(FEATURE_COLS) == INPUT_DIM, \
    f"FEATURE_COLS length {len(FEATURE_COLS)} != INPUT_DIM {INPUT_DIM}"

# Group dimension sizes (used by FactorGroupedEmbedding)
GROUP_DIMS = {k: len(v) for k, v in FEATURE_GROUPS.items()}

# ============================================================
# Prediction Targets
# ============================================================
PRED_HORIZONS     = [5, 20, 60]    # Days — multi-horizon heads
PRED_MAIN_HORIZON = 20             # Primary evaluation horizon

# ============================================================
# Model Hyperparameters
# ============================================================
D_MODEL = 256
D_STATE = 32
N_HEADS_GAT      = 4
MAX_NEIGHBORS_GAT = 15      # Per-stock GAT neighbour cap (prevents attention collapse)
DROPOUT           = 0.1

# Multi-scale Mamba: number of layers per branch [short, mid, long]
MULTI_SCALE_LAYERS = [2, 3, 3]     # Total 8 Mamba layers
MULTI_SCALE_SEQLENS = [20, 60, 252]  # Must match PRED_HORIZONS[0], [1], SEQ_LEN

# ============================================================
# Training
# ============================================================
BATCH_SIZE     = 1          # Each sample = one full cross-section
LR             = 1e-4
EPOCHS         = 60
EARLY_STOP     = 10
VAL_RATIO      = 0.15
GRAD_CLIP_NORM = 1.0
WARMUP_PCT     = 0.1        # OneCycleLR: first 10% steps do linear warmup
AMP_ENABLED    = True       # FP16 mixed-precision

# Multi-horizon loss weights
LOSS_WEIGHTS = {
    "mse_20d":    1.0,
    "mse_5d":     0.3,
    "mse_60d":    0.3,
    "listnet_20d": 0.5,
}

# Cross-section sub-sampling during training (set None to use all stocks)
N_SAMPLE_TRAIN = None       # Set to e.g. 800 if A100 OOMs on full 2000-stock cross-section

# ============================================================
# Walk-Forward Validation
# ============================================================
WF_TEST_WINDOW_MONTHS = 6    # Test window per fold
WF_STEP_MONTHS        = 3    # Roll step
WF_MIN_TRAIN_YEARS    = 3    # Minimum training history before first test fold

# IC thresholds for model acceptance
IC_THRESHOLD   = 0.05
ICIR_THRESHOLD = 0.5

# ============================================================
# Data Source Priority
# ============================================================
# Order: try each source left-to-right, fallback if unavailable
DATA_SOURCE_PRIORITY = ["yfinance", "twse_direct", "finmind"]
MARGIN_FORWARD_FILL  = True   # Use yesterday's margin data if FinMind not yet updated

# TWSE direct API endpoint for institutional investors
TWSE_INSTITUTIONAL_URL = (
    "https://www.twse.com.tw/rwd/zh/fund/T86"
)
TPEX_INSTITUTIONAL_URL = (
    "https://www.tpex.org.tw/web/stock/3insti/daily_trade/3itrade_hedge.php"
)

# Tokens / API keys — loaded from .env (see .env file in V6/)
import os
FINMIND_TOKEN     = os.getenv("FINMIND_TOKEN", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")

# ============================================================
# Knowledge Graph
# ============================================================
KG_EDGE_WEIGHTS = {
    "conglomerate":   0.8,   # 集團從屬（鴻海族、台積電生態等）
    "supply_chain":   0.6,   # TPEX 產業鏈同產業
    "twse_sector":    0.5,   # TWSE 產業分類（基底）
    "rolling_corr":   None,  # 滾動相關性（動態計算，60天窗口）
}
KG_CORR_WINDOW   = 60    # Days for rolling Pearson correlation
KG_CORR_THRESHOLD = 0.7  # Minimum correlation to create an edge

# ============================================================
# Deployment
# ============================================================
GITHUB_RESULTS_KEEP_DAYS = 90   # Rolling window for df_kelly history

# LLM settings (loaded from env)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LLM_MODEL         = "claude-sonnet-4-6"
LLM_MAX_TOKENS    = 800

