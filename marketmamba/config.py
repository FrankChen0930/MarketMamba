"""
MarketMamba V5.5 — 全域設定模組
集中管理路徑、Token、特徵欄位、模型超參數等所有設定
"""

import os
import logging
from datetime import datetime
import pytz

# ==========================================
# 日誌設定
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('MarketMamba')


# ==========================================
# 環境偵測
# ==========================================
def is_colab() -> bool:
    """偵測是否在 Google Colab 環境中執行"""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def get_drive_base() -> str:
    """取得 Google Drive / 本機資料基礎路徑"""
    if is_colab():
        drive_path = '/content/drive/MyDrive'
        if not os.path.exists(drive_path):
            from google.colab import drive
            drive.mount('/content/drive')
        return drive_path
    else:
        # 本機開發時的資料路徑 (可自行修改)
        return os.path.expanduser('~/MarketMamba_Data')


# ==========================================
# 路徑設定
# ==========================================
DRIVE_BASE = get_drive_base()

RAW_DIR = os.path.join(DRIVE_BASE, 'MarketMamba_V4', 'Raw')
PROCESSED_DIR = os.path.join(DRIVE_BASE, 'MarketMamba_V5', 'Processed_Features')
MODEL_DIR = os.path.join(DRIVE_BASE, 'MarketMamba_V5', 'Models')
NEWS_CACHE_DIR = os.path.join(DRIVE_BASE, 'MarketMamba_V5', 'News_Cache')

RAW_SUBDIRS = {
    'Daily_Macro':          os.path.join(RAW_DIR, 'Daily_Macro'),
    'Daily_Market':         os.path.join(RAW_DIR, 'Daily_Market'),
    'Daily_Price':          os.path.join(RAW_DIR, 'Daily_Price'),
    'Daily_Dividend':       os.path.join(RAW_DIR, 'Daily_Dividend'),
    'Daily_GovBank':        os.path.join(RAW_DIR, 'Daily_GovBank'),
    'Weekly_Holdings':      os.path.join(RAW_DIR, 'Weekly_Holdings'),
    'Monthly_Revenue':      os.path.join(RAW_DIR, 'Monthly_Revenue'),
    'Quarterly_Financials': os.path.join(RAW_DIR, 'Quarterly_Financials'),
}

# Colab 本地暫存 (加速碎小 parquet 讀取)
LOCAL_RAW_DIR = '/content/Local_Raw' if is_colab() else os.path.join(DRIVE_BASE, 'Local_Raw')

# GitHub 部署用的 repo 產出路徑
def get_repo_output_dir() -> str:
    """取得 CSV/JSON 輸出路徑 (推送到 GitHub 的檔案)"""
    if is_colab():
        return '/content/repo_folder'
    else:
        # 本機：直接放在 repo 根目錄
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ==========================================
# API 設定
# ==========================================
FINMIND_TOKEN = (
    'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.'
    'eyJkYXRlIjoiMjAyNi0wMy0xMiAyMzozMToyMSIsInVzZXJfaWQiOiJGcmFua0NoZW4i'
    'LCJlbWFpbCI6ImEwOTY2NDY5OTY0QGdtYWlsLmNvbSIsImlwIjoiNDkuMjEzLjEzNy4y'
    'NyJ9.mXKyYSG2Gi-1FT8ODsRElrfnhEIKtEDLeOxz1GvCEXY'
)
FINMIND_API_URL = "https://api.finmindtrade.com/api/v4/data"
DATA_START_DATE = "2019-01-01"


def get_today_str() -> str:
    """取得台灣時區的今日日期字串"""
    return datetime.now(pytz.timezone('Asia/Taipei')).strftime("%Y-%m-%d")


def get_now_str() -> str:
    """取得台灣時區的現在時刻字串"""
    return datetime.now(pytz.timezone('Asia/Taipei')).strftime("%Y-%m-%d %H:%M:%S")


# ==========================================
# yfinance 台股國際盤 Ticker 對照
# ==========================================
US_TICKERS = {
    '^SOX': 'US_SOX',
    'QQQ':  'US_QQQ',
    '^VIX': 'US_VIX',
    '^TNX': 'US_TNX',
    'GC=F': 'Gold',
    'CL=F': 'Oil',
}

# ==========================================
# FinMind 籌碼資料集對照
# ==========================================
FINMIND_MARKET_DATASETS = {
    'Inst_BuySell':  'TaiwanStockInstitutionalInvestorsBuySell',
    'Margin_Short':  'TaiwanStockMarginPurchaseShortSale',
    'Day_Trading':   'TaiwanStockDayTrading',
    'Sec_Lending':   'TaiwanStockSecuritiesLending',
}


# ==========================================
# 特徵欄位定義
# ==========================================

# V5.0 原始 46 維量價籌碼特徵
FEATURE_COLS_V5 = [
    # 價量 (5)
    'Open', 'High', 'Low', 'Close', 'Volume',
    # 三大法人 + 籌碼 (7)
    'Foreign_Buy', 'Trust_Buy', 'Dealer_Buy',
    'Margin_Balance', 'Short_Balance', 'Day_Trading_Ratio', 'Securities_Lending',
    # 基本面 (3)
    'PER', 'PBR', 'DY',
    # 國際盤 + 總經 (7)
    'TWII_Close', 'US_SOX', 'US_QQQ', 'US_VIX', 'US_TNX', 'Gold', 'Oil',
    # 狀態標記 (3)
    'US_Market_Closed', 'Gov_Bank_Buy', 'Is_Ex_Dividend',
    # 低頻基本面 (5)
    'Monthly_Revenue', 'Whale_Hold_Ratio', 'Retail_Hold_Ratio', 'EPS', 'Gross_Margin',
    # 衍生技術指標 (16)
    'Gov_Bank_Sum_20d', 'TWII_Bias_60', 'MA_60', 'Bias_60',
    'Return_1d', 'TWII_Return_1d', 'Alpha_1d', 'Volatility_20d',
    'MA_20', 'BB_Width', 'Vol_MA_20', 'Vol_Ratio',
    'Foreign_Sum_20d', 'Trust_Sum_20d', 'MACD_Hist', 'Rev_YoY',
]

# V5.5 新增：情緒標量特徵 (6 維)
SENTIMENT_SCALAR_COLS = [
    'Sent_Stock_CN',       # 個股中文新聞情緒 (-1 ~ +1)
    'Sent_Stock_EN',       # 個股英文新聞情緒
    'Sent_Market_TW',      # 台股大盤新聞情緒
    'Sent_Market_US',      # 美股/國際新聞情緒
    'Sent_Geopolitical',   # 地緣政治新聞情緒
    'News_Volume_Stock',   # 個股新聞數量 (log 標準化)
]

# V5.5 新增：FinBERT Embedding 投影 (2 × 16 = 32 維)
SENTIMENT_EMBED_EN_COLS = [f'Sent_Embed_EN_{i}' for i in range(16)]
SENTIMENT_EMBED_CN_COLS = [f'Sent_Embed_CN_{i}' for i in range(16)]

# 全部情緒特徵 (38 維)
SENTIMENT_COLS = SENTIMENT_SCALAR_COLS + SENTIMENT_EMBED_EN_COLS + SENTIMENT_EMBED_CN_COLS

# V5.5 完整特徵列表 (46 + 38 = 84 維)
FEATURE_COLS = FEATURE_COLS_V5 + SENTIMENT_COLS


# ==========================================
# 模型超參數
# ==========================================
MODEL_CONFIG = {
    'input_dim':         len(FEATURE_COLS),       # 84 (V5.5)
    'input_dim_v5':      len(FEATURE_COLS_V5),    # 46 (V5.0 向下相容)
    'seq_len':           180,
    'd_model':           256,       # 128 → 256 (A100 升級)
    'pred_days':         30,
    'num_mamba_layers':  6,         # 4 → 6 (更深時序理解)
    'd_state':           32,        # 16 → 32 (更大狀態空間)
    'd_conv':            4,
    'expand':            2,
    'dropout_rate':      0.4,
    'k_neighbors':       10,
}

# 訓練超參數預設值
TRAIN_CONFIG = {
    'epochs':            50,
    'batch_size':        512,
    'learning_rate':     1e-4,
    'weight_decay':      1e-5,
    'early_stop_patience': 7,
    'val_ratio':         0.15,
    'min_stock_days':    120,  # 最少需要 120 天資料才納入訓練
    'seed':              42,
}


# ==========================================
# FinBERT 設定
# ==========================================
FINBERT_EN_MODEL = "ProsusAI/finbert"
FINBERT_CN_MODEL = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
FINBERT_EMBED_DIM = 16       # 768 → 16 維投影
SENTIMENT_HALF_LIFE = 3      # 情緒衰減半衰期 (天)


# ==========================================
# 新聞爬蟲設定
# ==========================================
NEWS_LOOKBACK_DAYS = 3

# 高品質來源白名單 (Google News RSS 過濾用)
NEWS_SOURCES_WHITELIST_CN = [
    'cnyes.com', 'money.udn.com', 'ctee.com.tw', 'moneydj.com',
    'wealth.com.tw', 'cna.com.tw',
]
NEWS_SOURCES_WHITELIST_EN = [
    'reuters.com', 'cnbc.com', 'bloomberg.com', 'finance.yahoo.com',
    'wsj.com', 'ft.com', 'marketwatch.com',
]

# 地緣政治關鍵字 (用於分類)
GEOPOLITICAL_KEYWORDS_EN = [
    'war', 'sanction', 'tariff', 'military', 'missile', 'invasion',
    'conflict', 'troops', 'nuclear', 'embargo', 'geopolitical',
    'nato', 'ceasefire', 'escalation', 'retaliation',
]
GEOPOLITICAL_KEYWORDS_CN = [
    '戰爭', '制裁', '關稅', '軍事', '飛彈', '入侵',
    '衝突', '核武', '禁運', '地緣政治', '報復',
    '封鎖', '動武', '軍演', '台海',
]

# 台股大盤/總經關鍵字
MARKET_KEYWORDS_TW = [
    '台股', '加權指數', '央行', '利率', '降息', '升息',
    '台積電', '外資', '投信', '融資', '融券',
]
MARKET_KEYWORDS_US = [
    'Fed', 'Federal Reserve', 'interest rate', 'S&P 500', 'Nasdaq',
    'inflation', 'CPI', 'GDP', 'unemployment', 'recession',
    'Wall Street', 'Treasury', 'bond yield',
]


# ==========================================
# GitHub 部署設定
# ==========================================
GITHUB_USER = "FrankChen0930"
GITHUB_EMAIL = "a0966469964@gmail.com"
GITHUB_REPO_URL_TEMPLATE = "https://{token}@github.com/{user}/MarketMamba.git"

# 需要推送到 GitHub 的檔案清單
GITHUB_PUSH_FILES = [
    'df_kelly.csv',
    'df_traj.csv',
    'pattern_scan_results.csv',
    'robot_ledger.json',
    'sentiment_summary.json',
    'update_time.txt',
]


# ==========================================
# 總經欄位 (補班日修復用)
# ==========================================
MACRO_FILL_COLS = [
    'TWII_Close', 'US_SOX', 'US_QQQ', 'US_VIX', 'US_TNX',
    'Gold', 'Oil', 'TWII_Bias_60', 'TWII_Return_1d',
]
