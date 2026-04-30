# ==========================================
# 🚀 V5.0 終極管線 (Cell 1/5): 環境與底層算子建置
# ==========================================
import os
import time

print("🚀 啟動 V5.0 終極端到端 (End-to-End) 自動化管線...")
!pip install -q yfinance FinMind pandas numpy requests torch-geometric pyarrow fastparquet tqdm

# 1. 掛載 Google Drive
from google.colab import drive
if not os.path.exists('/content/drive/MyDrive'):
    drive.mount('/content/drive')

# 2. 建立 Mamba 暫存環境
os.makedirs("mamba_core", exist_ok=True)
os.chdir("mamba_core")

print("📥 下載並安裝 Mamba 底層算子 (Causal_Conv1d & Mamba_SSM)...")
!wget -q "https://github.com/FrankChen0930/MarketMamba/releases/download/whl-for-mamba/causal_conv1d-1.6.0-cp312-cp312-linux_x86_64.whl"
!wget -q "https://github.com/FrankChen0930/MarketMamba/releases/download/whl-for-mamba/mamba_ssm-2.3.0-cp312-cp312-linux_x86_64.whl"

!pip install -q causal_conv1d-*.whl
!pip install -q mamba_ssm-*.whl

os.chdir("/content")
print("✅ 環境載入完成！準備進入資料同步階段。")

# ==========================================
# 🏭 V5.0 Data Pipeline (Cell 1/4): 全自動補洞與高頻資料更新
# ==========================================
!pip install -q yfinance FinMind pandas numpy tqdm pyarrow fastparquet

import os
import time
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from tqdm.auto import tqdm
from datetime import datetime, timedelta
import pytz
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from google.colab import drive

print("🔗 正在掛載 Google Drive...")
drive.mount('/content/drive')

# 🎯 聰明策略：Raw Data 共用 V4 倉庫 (省去重抓 7 年的時間)，Processed 存入 V5 新目錄
RAW_DIR = '/content/drive/MyDrive/MarketMamba_V4/Raw'
PROCESSED_DIR = '/content/drive/MyDrive/MarketMamba_V5/Processed_Features'

RAW_SUBDIRS = {
    'Daily_Macro': os.path.join(RAW_DIR, 'Daily_Macro'),
    'Daily_Market': os.path.join(RAW_DIR, 'Daily_Market'),
    'Weekly_Holdings': os.path.join(RAW_DIR, 'Weekly_Holdings'),
    'Monthly_Revenue': os.path.join(RAW_DIR, 'Monthly_Revenue'),
    'Quarterly_Financials': os.path.join(RAW_DIR, 'Quarterly_Financials'),
    'Daily_Price': os.path.join(RAW_DIR, 'Daily_Price')
}

for path in RAW_SUBDIRS.values(): os.makedirs(path, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# 🔑 請確認你的 FinMind Token
FINMIND_TOKEN = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJkYXRlIjoiMjAyNi0wMy0xMiAyMzozMToyMSIsInVzZXJfaWQiOiJGcmFua0NoZW4iLCJlbWFpbCI6ImEwOTY2NDY5OTY0QGdtYWlsLmNvbSIsImlwIjoiNDkuMjEzLjEzNy4yNyJ9.mXKyYSG2Gi-1FT8ODsRElrfnhEIKtEDLeOxz1GvCEXY'
START_DATE = "2019-01-01"
END_DATE = datetime.now(pytz.timezone('Asia/Taipei')).strftime("%Y-%m-%d")

session = requests.Session()
adapter = HTTPAdapter(max_retries=Retry(total=3, backoff_factor=0.5))
session.mount('https://', adapter)
API_URL = "https://api.finmindtrade.com/api/v4/data"

print(f"🚀 啟動 V5.0 自動補洞雷達 (掃描至 {END_DATE})...")

# ==========================================
# 🌍 1. 更新大盤與總經 (Macro)
# ==========================================
twii = yf.Ticker("^TWII").history(start=START_DATE, end=END_DATE, auto_adjust=False)
twii.index = pd.to_datetime(twii.index).tz_localize(None)
df_macro = pd.DataFrame({'Date': twii.index, 'TWII_Close': twii['Close'].values})

us_tickers = {'^SOX': 'US_SOX', 'QQQ': 'US_QQQ', '^VIX': 'US_VIX', '^TNX': 'US_TNX', 'GC=F': 'Gold', 'CL=F': 'Oil'}
df_us = pd.DataFrame(index=df_macro['Date'])
for tick, name in us_tickers.items():
    tmp = yf.Ticker(tick).history(start=START_DATE, end=END_DATE, auto_adjust=False)
    tmp.index = pd.to_datetime(tmp.index).tz_localize(None)
    df_us = pd.merge(df_us, tmp[['Close']].rename(columns={'Close': name}), left_on='Date', right_index=True, how='outer')

df_us = df_us.sort_values('Date').dropna(how='all')
df_us['US_Visible_Date'] = df_us['Date'] + pd.Timedelta(days=1)

df_macro = pd.merge_asof(df_macro, df_us.drop(columns=['Date']).sort_values('US_Visible_Date'), left_on='Date', right_on='US_Visible_Date', direction='backward')
df_macro['US_Market_Closed'] = np.where((df_macro['Date'] - df_macro['US_Visible_Date']).dt.days > 0, 1, 0)
df_macro = df_macro.drop(columns=['US_Visible_Date']).ffill().bfill()

macro_save_path = os.path.join(RAW_SUBDIRS['Daily_Macro'], 'macro_features.parquet')
df_macro.to_parquet(macro_save_path, engine='pyarrow')
print("✅ 總經與大盤已更新至最新！")

# 取得所有交易日清單，準備進行比對
trading_days = df_macro['Date'].dt.strftime('%Y-%m-%d').tolist()

# ==========================================
# 📈 2. 更新個股價量 (OHLCV)
# ==========================================
for date_str in tqdm(trading_days, desc="📈 掃描並補齊價量缺口"):
    save_path = os.path.join(RAW_SUBDIRS['Daily_Price'], f"{date_str}_price.parquet")
    if os.path.exists(save_path): continue # 🎯 關鍵：有檔案就光速跳過！
    try:
        res = session.get(API_URL, params={"dataset": "TaiwanStockPrice", "start_date": date_str, "end_date": date_str, "token": FINMIND_TOKEN}, timeout=15)
        data = res.json()
        if data.get("msg") == "success" and len(data.get("data", [])) > 0:
            df_tmp = pd.DataFrame(data["data"])
            df_tmp['stock_id'] = df_tmp['stock_id'].astype(str).str.strip()
            df_clean = df_tmp[['stock_id', 'open', 'max', 'min', 'close', 'Trading_Volume']].rename(
                columns={'open': 'Open', 'max': 'High', 'min': 'Low', 'close': 'Close', 'Trading_Volume': 'Volume'}
            )
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']: df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
            df_clean.to_parquet(save_path, engine='pyarrow')
    except: pass
    time.sleep(0.2)

# ==========================================
# 🕵️‍♂️ 3. 更新全市場籌碼 (三大法人/信用/借券)
# ==========================================
DATASETS_TO_FETCH = {
    'Inst_BuySell': 'TaiwanStockInstitutionalInvestorsBuySell',
    'Margin_Short': 'TaiwanStockMarginPurchaseShortSale',
    'Day_Trading': 'TaiwanStockDayTrading',
    'Sec_Lending': 'TaiwanStockSecuritiesLending'
}

for date_str in tqdm(trading_days[-30:], desc="📦 掃描並補齊籌碼缺口 (回看30天)"): # 只掃最近30天加速
    save_path = os.path.join(RAW_SUBDIRS['Daily_Market'], f"{date_str}_market.parquet")
    if os.path.exists(save_path): continue
    daily_merged_df = None
    for key, dataset_name in DATASETS_TO_FETCH.items():
        try:
            res = session.get(API_URL, params={"dataset": dataset_name, "start_date": date_str, "end_date": date_str, "token": FINMIND_TOKEN}, timeout=15)
            data = res.json()
            if data.get("msg") == "success" and len(data.get("data", [])) > 0:
                df_tmp = pd.DataFrame(data["data"])
                df_tmp['stock_id'] = df_tmp['stock_id'].astype(str).str.strip()
                df_clean = pd.DataFrame({'stock_id': df_tmp['stock_id'].unique()})

                if key == 'Inst_BuySell':
                    df_tmp['net_buy'] = pd.to_numeric(df_tmp['buy'], errors='coerce').fillna(0) - pd.to_numeric(df_tmp['sell'], errors='coerce').fillna(0)
                    pivot_df = df_tmp.pivot_table(index='stock_id', columns='name', values='net_buy', aggfunc='sum').reset_index()
                    pivot_df = pivot_df.rename(columns={'外資及陸資(不含外資自營商)': 'Foreign_Buy', 'Foreign_Investor': 'Foreign_Buy', '投信': 'Trust_Buy', 'Investment_Trust': 'Trust_Buy', '自營商(自行買賣)': 'Dealer_Buy', 'Dealer_self': 'Dealer_Buy'})
                    df_clean = pivot_df[['stock_id'] + [c for c in ['Foreign_Buy', 'Trust_Buy', 'Dealer_Buy'] if c in pivot_df.columns]]
                elif key == 'Margin_Short':
                    avail_cols = [c for c in ['MarginPurchaseTodayBalance', 'ShortSaleTodayBalance'] if c in df_tmp.columns]
                    if avail_cols: df_clean = df_tmp[['stock_id'] + avail_cols].rename(columns={'MarginPurchaseTodayBalance': 'Margin_Balance', 'ShortSaleTodayBalance': 'Short_Balance'})
                elif key == 'Day_Trading':
                    if 'BuyAfterSale' in df_tmp.columns and 'Volume' in df_tmp.columns:
                        df_tmp['Day_Trading_Ratio'] = pd.to_numeric(df_tmp['BuyAfterSale'], errors='coerce').fillna(0) / pd.to_numeric(df_tmp['Volume'], errors='coerce').fillna(1)
                        df_clean = df_tmp[['stock_id', 'Day_Trading_Ratio']]
                elif key == 'Sec_Lending':
                    if 'secLending' in df_tmp.columns: df_clean = df_tmp[['stock_id', 'secLending']].rename(columns={'secLending': 'Securities_Lending'})
                    elif 'volume' in df_tmp.columns: df_clean = df_tmp.groupby('stock_id')['volume'].sum().reset_index().rename(columns={'volume': 'Securities_Lending'})

                if len(df_clean.columns) > 1:
                    daily_merged_df = df_clean if daily_merged_df is None else pd.merge(daily_merged_df, df_clean, on='stock_id', how='outer')
        except: pass
        time.sleep(0.2)
    if daily_merged_df is not None and not daily_merged_df.empty:
        daily_merged_df.to_parquet(save_path, engine='pyarrow')

# ==========================================
# 📊 4. 更新低頻資料 (營收、集保、財報)
# ==========================================
print("🔄 掃描低頻財報與集保缺口...")
# 月營收
for month_str in pd.date_range(start="2024-01-01", end=END_DATE, freq='MS').strftime('%Y-%m-%d').tolist():
    save_path = os.path.join(RAW_SUBDIRS['Monthly_Revenue'], f"{month_str[:7]}_revenue.parquet")
    if not os.path.exists(save_path):
        try:
            res = session.get(API_URL, params={"dataset": "TaiwanStockMonthRevenue", "start_date": month_str, "end_date": month_str, "token": FINMIND_TOKEN}, timeout=15)
            data = res.json()
            if data.get("msg") == "success" and len(data.get("data", [])) > 0:
                df_rev = pd.DataFrame(data["data"])
                df_rev['stock_id'] = df_rev['stock_id'].astype(str).str.strip()
                df_clean = df_rev.groupby('stock_id')['revenue'].first().reset_index().rename(columns={'revenue': 'Monthly_Revenue'})
                df_clean['Monthly_Revenue'] = pd.to_numeric(df_clean['Monthly_Revenue'], errors='coerce').fillna(0)
                df_clean.to_parquet(save_path, engine='pyarrow')
        except: pass
        time.sleep(0.25)

# 執行完畢
print("🎉 V5.0 自動補洞任務完成！所有資料庫均已對齊最新進度。")

# ==========================================
# 💎 V5.0 Data Pipeline (Cell 1.5): VIP 專屬歷史資料打包機
# ==========================================
import os
import time
import requests
import pandas as pd
from tqdm.auto import tqdm
from datetime import datetime
import pytz
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from google.colab import drive

print("🔗 檢查 Google Drive 掛載狀態...")
if not os.path.exists('/content/drive/MyDrive'):
    drive.mount('/content/drive')

# 🎯 存入你原本的 V4 倉庫中，方便未來統一管理
RAW_DIR = '/content/drive/MyDrive/MarketMamba_V4/Raw'
VIP_SUBDIRS = {
    'Daily_Dividend': os.path.join(RAW_DIR, 'Daily_Dividend'),
    'Daily_GovBank': os.path.join(RAW_DIR, 'Daily_GovBank')
}

for path in VIP_SUBDIRS.values():
    os.makedirs(path, exist_ok=True)

# 🔑 請確認你的 FinMind Token (利用 Sponsor 權限一次抓滿)
FINMIND_TOKEN = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJkYXRlIjoiMjAyNi0wMy0xMiAyMzozMToyMSIsInVzZXJfaWQiOiJGcmFua0NoZW4iLCJlbWFpbCI6ImEwOTY2NDY5OTY0QGdtYWlsLmNvbSIsImlwIjoiNDkuMjEzLjEzNy4yNyJ9.mXKyYSG2Gi-1FT8ODsRElrfnhEIKtEDLeOxz1GvCEXY'
START_DATE = "2019-01-01"
END_DATE = datetime.now(pytz.timezone('Asia/Taipei')).strftime("%Y-%m-%d")

session = requests.Session()
adapter = HTTPAdapter(max_retries=Retry(total=3, backoff_factor=0.5))
session.mount('https://', adapter)
API_URL = "https://api.finmindtrade.com/api/v4/data"

print(f"💎 啟動 VIP 專屬歷史數據打包任務 ({START_DATE} to {END_DATE})...")

# 取得交易日清單 (利用之前抓好的總經日期來對齊，如果沒有就用 pandas 產生工作日)
try:
    df_macro = pd.read_parquet(os.path.join(RAW_DIR, 'Daily_Macro', 'macro_features.parquet'))
    trading_days = df_macro['Date'].dt.strftime('%Y-%m-%d').tolist()
except:
    trading_days = pd.date_range(start=START_DATE, end=END_DATE, freq='B').strftime('%Y-%m-%d').tolist()

# ==========================================
# 🛡️ 1. 抓取除權息結果表 (TaiwanStockDividendResult)
# ==========================================
for date_str in tqdm(trading_days, desc="🛡️ 掃描除權息防護網"):
    save_path = os.path.join(VIP_SUBDIRS['Daily_Dividend'], f"{date_str}_dividend.parquet")
    if os.path.exists(save_path): continue

    try:
        res = session.get(API_URL, params={"dataset": "TaiwanStockDividendResult", "start_date": date_str, "end_date": date_str, "token": FINMIND_TOKEN}, timeout=15)
        data = res.json()
        if data.get("msg") == "success" and len(data.get("data", [])) > 0:
            df_tmp = pd.DataFrame(data["data"])
            df_tmp['stock_id'] = df_tmp['stock_id'].astype(str).str.strip()
            df_tmp.to_parquet(save_path, engine='pyarrow')
    except: pass
    time.sleep(0.2) # 遵守 API 速率限制

# ==========================================
# 🏦 2. 抓取八大行庫買賣超 (TaiwanStockGovernmentBankBuySell)
# ==========================================
for date_str in tqdm(trading_days, desc="🏦 掃描國家隊護盤底線"):
    save_path = os.path.join(VIP_SUBDIRS['Daily_GovBank'], f"{date_str}_govbank.parquet")
    if os.path.exists(save_path): continue

    try:
        res = session.get(API_URL, params={"dataset": "TaiwanStockGovernmentBankBuySell", "start_date": date_str, "end_date": date_str, "token": FINMIND_TOKEN}, timeout=15)
        data = res.json()
        if data.get("msg") == "success" and len(data.get("data", [])) > 0:
            df_tmp = pd.DataFrame(data["data"])
            df_tmp['stock_id'] = df_tmp['stock_id'].astype(str).str.strip()

            # 計算淨買賣超
            df_tmp['buy'] = pd.to_numeric(df_tmp['buy'], errors='coerce').fillna(0)
            df_tmp['sell'] = pd.to_numeric(df_tmp['sell'], errors='coerce').fillna(0)
            df_tmp['Gov_Bank_Buy'] = df_tmp['buy'] - df_tmp['sell']

            df_clean = df_tmp[['stock_id', 'Gov_Bank_Buy']]
            df_clean.to_parquet(save_path, engine='pyarrow')
    except: pass
    time.sleep(0.2)

print("🎉 VIP 歷史數據打包完成！你可以帶著這批最頂級的燃料進入 V5.0 煉丹爐了！")

# ==========================================
# 🧬 V5.0 Data Pipeline (Cell 2/4): 跨頻率時序大融合 - 光速防斷線版
# ==========================================
import os
import pandas as pd
import glob
from tqdm.auto import tqdm
from google.colab import drive

# 🔌 1. 強制修復 Google Drive 連線
print("🔄 正在強制修復 Google Drive 連線...")
drive.mount('/content/drive', force_remount=True)

print("🌌 啟動跨頻率時序大融合 (包含 VIP 國家隊與除權息數據)...")

RAW_DIR = '/content/drive/MyDrive/MarketMamba_V4/Raw'
LOCAL_RAW_DIR = '/content/Local_Raw'

# 🚀 2. 終極加速：將碎小的日頻檔案複製到 Colab 本地端 (避開 GDrive 斷線)
print("⚡ 正在將資料快取到 Colab 本地硬碟中 (約需 1~3 分鐘，請稍候)...")
os.makedirs(LOCAL_RAW_DIR, exist_ok=True)
!cp -r "{RAW_DIR}/Daily_Price" "{LOCAL_RAW_DIR}/"
!cp -r "{RAW_DIR}/Daily_Market" "{LOCAL_RAW_DIR}/"

# 3. 載入日頻價量 (改從本地端讀取，速度直接起飛！)
price_files = glob.glob(os.path.join(LOCAL_RAW_DIR, 'Daily_Price', '*.parquet'))
if not price_files: raise FileNotFoundError(f"❌ 找不到價量資料！請檢查路徑")
df_master = pd.concat([pd.read_parquet(f).assign(Date=pd.to_datetime(os.path.basename(f)[:10])) for f in tqdm(price_files, desc="📈 價量讀取")], ignore_index=True)
df_master = df_master[df_master['stock_id'].astype(str).str.match(r'^([1-9]\d{3}|00\d{2,4}[A-Za-z]?)$')].copy()

# 4. 載入日頻籌碼與總經 (從本地端讀取)
market_files = glob.glob(os.path.join(LOCAL_RAW_DIR, 'Daily_Market', '*.parquet'))
df_market = pd.concat([pd.read_parquet(f).assign(Date=pd.to_datetime(os.path.basename(f)[:10])) for f in tqdm(market_files, desc="📊 籌碼讀取")], ignore_index=True)

df_macro = pd.read_parquet(os.path.join(RAW_DIR, 'Daily_Macro', 'macro_features.parquet'))
df_macro['Date'] = pd.to_datetime(df_macro['Date'])

df_master = pd.merge(df_master, df_market, on=['Date', 'stock_id'], how='left')
df_master = pd.merge(df_master, df_macro, on='Date', how='left')

# 💎 5. 載入 VIP 專屬資料 (除權息 & 八大行庫)
gov_files = glob.glob(os.path.join(RAW_DIR, 'Daily_GovBank', '*.parquet'))
if gov_files:
    df_gov = pd.concat([pd.read_parquet(f).assign(Date=pd.to_datetime(os.path.basename(f)[:10])) for f in tqdm(gov_files, desc="🏦 國家隊讀取")], ignore_index=True)
    df_master = pd.merge(df_master, df_gov[['stock_id', 'Date', 'Gov_Bank_Buy']], on=['Date', 'stock_id'], how='left')

div_files = glob.glob(os.path.join(RAW_DIR, 'Daily_Dividend', '*.parquet'))
if div_files:
    df_div = pd.concat([pd.read_parquet(f).assign(Date=pd.to_datetime(os.path.basename(f)[:10])) for f in tqdm(div_files, desc="🛡️ 除權息讀取")], ignore_index=True)
    df_div['Is_Ex_Dividend'] = 1
    df_master = pd.merge(df_master, df_div[['stock_id', 'Date', 'Is_Ex_Dividend']], on=['Date', 'stock_id'], how='left')

# 6. 低頻資料防未來函數對齊
df_master = df_master.sort_values('Date')

rev_files = glob.glob(os.path.join(RAW_DIR, 'Monthly_Revenue', '*.parquet'))
if rev_files:
    df_rev = pd.concat([pd.read_parquet(f).assign(Date=pd.to_datetime(os.path.basename(f)[:7]+'-01')) for f in rev_files])
    df_rev['Pub_Date'] = df_rev['Date'] + pd.DateOffset(months=1, days=9)
    df_master = pd.merge_asof(df_master, df_rev.sort_values('Pub_Date').drop(columns=['Date']), left_on='Date', right_on='Pub_Date', by='stock_id', direction='backward').drop(columns=['Pub_Date'])

hold_files = glob.glob(os.path.join(RAW_DIR, 'Weekly_Holdings', '*.parquet'))
if hold_files:
    df_hold = pd.concat([pd.read_parquet(f).assign(Date=pd.to_datetime(os.path.basename(f)[:10])) for f in hold_files])
    df_hold['Pub_Date'] = df_hold['Date'] + pd.Timedelta(days=3)
    df_master = pd.merge_asof(df_master, df_hold.sort_values('Pub_Date').drop(columns=['Date']), left_on='Date', right_on='Pub_Date', by='stock_id', direction='backward').drop(columns=['Pub_Date'])

fin_files = glob.glob(os.path.join(RAW_DIR, 'Quarterly_Financials', '*.parquet'))
if fin_files:
    df_fin = pd.concat([pd.read_parquet(f).assign(Date=pd.to_datetime(os.path.basename(f)[:10])) for f in fin_files])
    df_fin['Pub_Date'] = df_fin['Date'] + pd.DateOffset(days=90)
    df_master = pd.merge_asof(df_master, df_fin.sort_values('Pub_Date').drop(columns=['Date']), left_on='Date', right_on='Pub_Date', by='stock_id', direction='backward').drop(columns=['Pub_Date'])

df_master = df_master.sort_values(['stock_id', 'Date']).reset_index(drop=True)

# 🧹 7. 清理本地暫存釋放空間
!rm -rf "{LOCAL_RAW_DIR}"

print(f"✅ 大融合完畢！總筆數: {len(df_master):,}")

# ==========================================
# 🧙‍♂️ V5.0 Data Pipeline (Cell 3/4): 特徵煉金術 - VIP 升級版
# ==========================================
print("🧙‍♂️ 啟動特徵煉金：大盤平穩化、波動率、國家隊護盤與除權息防護...")

df = df_master.copy()
df = df[(df['Close'] > 0) & (df['Volume'] > 0)].copy()

if 'PER' in df.columns: df['PER'] = df['PER'].clip(0, 100)
if 'PBR' in df.columns: df['PBR'] = df['PBR'].clip(0, 10)

g = df.groupby('stock_id')
new_features = {}

# --- 💎 VIP 專屬特徵處理 ---
if 'Gov_Bank_Buy' in df.columns:
    df['Gov_Bank_Buy'] = df['Gov_Bank_Buy'].fillna(0)
    # 計算國家隊 20 日累積買賣超，抓出長線護盤底線
    new_features['Gov_Bank_Sum_20d'] = g['Gov_Bank_Buy'].transform(lambda x: x.rolling(20).sum())

if 'Is_Ex_Dividend' in df.columns:
    new_features['Is_Ex_Dividend'] = df['Is_Ex_Dividend'].fillna(0)

# --- 🌟 1. 大盤與個股的「絕對平穩化」 ---
df['TWII_MA_60'] = df['TWII_Close'].transform(lambda x: x.rolling(60).mean())
new_features['TWII_Bias_60'] = (df['TWII_Close'] - df['TWII_MA_60']) / (df['TWII_MA_60'] + 1e-8)
new_features['MA_60'] = g['Close'].transform(lambda x: x.rolling(60).mean())
new_features['Bias_60'] = (df['Close'] - new_features['MA_60']) / (new_features['MA_60'] + 1e-8)

# --- 🌟 2. 報酬率與 Alpha ---
new_features['Return_1d'] = g['Close'].pct_change(1)
new_features['TWII_Return_1d'] = df['TWII_Close'].pct_change(1)
new_features['Alpha_1d'] = new_features['Return_1d'] - new_features['TWII_Return_1d']

# --- 🌟 3. 風險認知：個股歷史波動率 ---
new_features['Volatility_20d'] = new_features['Return_1d'].groupby(df['stock_id']).transform(lambda x: x.rolling(20).std())

# --- 4. 傳統技術與籌碼指標 ---
new_features['MA_20'] = g['Close'].transform(lambda x: x.rolling(20).mean())
std_20 = g['Close'].transform(lambda x: x.rolling(20).std())
new_features['BB_Width'] = (4 * std_20) / (new_features['MA_20'] + 1e-8)
new_features['Vol_MA_20'] = g['Volume'].transform(lambda x: x.rolling(20).mean())
new_features['Vol_Ratio'] = df['Volume'] / (new_features['Vol_MA_20'] + 1e-8)
new_features['Foreign_Sum_20d'] = g['Foreign_Buy'].transform(lambda x: x.rolling(20).sum())
new_features['Trust_Sum_20d'] = g['Trust_Buy'].transform(lambda x: x.rolling(20).sum())

ema_12 = g['Close'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
ema_26 = g['Close'].transform(lambda x: x.ewm(span=26, adjust=False).mean())
macd = ema_12 - ema_26
macd_sig = macd.groupby(df['stock_id']).transform(lambda x: x.ewm(span=9, adjust=False).mean())
new_features['MACD_Hist'] = macd - macd_sig

# 併入大表 (這會自動把字典轉為欄位)
df = pd.concat([df, pd.DataFrame(new_features)], axis=1)

# === 🎯 計算 YoY (防未來函數與後綴報錯) ===
rev_lookup = df[['stock_id', 'Date', 'Monthly_Revenue']].dropna().rename(columns={'Date': 'Lookup', 'Monthly_Revenue': 'Rev_1Y_ago'})
rev_lookup = rev_lookup.sort_values('Lookup')

df['Date_1Y'] = df['Date'] - pd.DateOffset(years=1)
df = df.sort_values('Date_1Y')

df = pd.merge_asof(
    df, rev_lookup,
    left_on='Date_1Y', right_on='Lookup',
    by='stock_id', direction='backward', tolerance=pd.Timedelta(days=20)
)
df['Rev_YoY'] = (df['Monthly_Revenue'] - df['Rev_1Y_ago']) / (df['Rev_1Y_ago'] + 1e-8)

print("✅ V5.0 特徵煉金完成！無報錯通關！")

# ==========================================
# 🛡️ V5.0 Data Pipeline (Cell 4/4): 終極清洗與存檔 - 終極防重複版
# ==========================================
import os
import numpy as np
import pandas as pd

print("🛁 啟動終極清洗 (隔離 IPO 暖機期與處理補班日)...")

PROCESSED_DIR = '/content/drive/MyDrive/MarketMamba_V5/Processed_Features'
os.makedirs(PROCESSED_DIR, exist_ok=True)

df = df.sort_values(['stock_id', 'Date']).reset_index(drop=True)
df = df.replace([np.inf, -np.inf], np.nan)

# 🚨 關鍵修復：自動掃描並剔除重複的欄位 (解決 Is_Ex_Dividend 重複問題)
df = df.loc[:, ~df.columns.duplicated()].copy()

# 1. 補班日 Macro 斷層修復
macro_cols = ['TWII_Close', 'USD_TWD', 'US_SOX', 'US_VIX', 'US_TNX', 'Gold', 'Oil', 'TWII_Bias_60', 'TWII_Return_1d']
for col in macro_cols:
    if col in df.columns:
        df[col] = df.groupby('stock_id')[col].ffill().fillna(0)

# 2. 剔除無效資料 (只要算不出 Alpha 或 MA_60 就代表活不夠久，直接砍)
initial_len = len(df)
df = df.dropna(subset=['MA_60', 'Alpha_1d', 'Volatility_20d', 'Return_1d'])

# 3. 填補允許缺失的基本面/籌碼
df = df.fillna(0)

# 清理不再需要的輔助欄位
df = df.drop(columns=['Date_1Y', 'Lookup', 'Rev_1Y_ago', 'TWII_MA_60'], errors='ignore')

# 確保 0 NaN
assert df.isna().sum().sum() == 0, "❌ 警告：矩陣中仍有 NaN 殘留！"

final_path = os.path.join(PROCESSED_DIR, 'V5_Mamba_Matrix.parquet')
df.to_parquet(final_path, engine='pyarrow')

print(f"🎉 V5.0 終極資料庫建置完畢！")
print(f"📊 最終實戰維度: {df.shape[0]:,} 列 x {df.shape[1]} 欄")
print(f"👉 下一步：前往訓練用的 Colab，DataLoader 將會動態幫你算出未來 30D 的 Alpha 軌跡！")

# ==========================================
# 🚀 V5.0 終極管線 (Cell 3/5): Mamba 動態圖推論 (強制防毒版)
# ==========================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import numpy as np
import pandas as pd
from mamba_ssm import Mamba
from torch_geometric.nn import GATv2Conv

print("🧠 喚醒 MarketMamba V5.0 完全體大腦並組裝張量...")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
set_seed(42)

DATA_PATH = '/content/drive/MyDrive/MarketMamba_V5/Processed_Features/V5_Mamba_Matrix.parquet'
df = pd.read_parquet(DATA_PATH)

feature_cols = [
    'Open', 'High', 'Low', 'Close', 'Volume', 'Foreign_Buy', 'Trust_Buy', 'Dealer_Buy',
    'Margin_Balance', 'Short_Balance', 'Day_Trading_Ratio', 'Securities_Lending',
    'PER', 'PBR', 'DY', 'TWII_Close', 'US_SOX', 'US_QQQ', 'US_VIX', 'US_TNX',
    'Gold', 'Oil', 'US_Market_Closed', 'Gov_Bank_Buy', 'Is_Ex_Dividend', 'Monthly_Revenue',
    'Whale_Hold_Ratio', 'Retail_Hold_Ratio', 'EPS', 'Gross_Margin', 'Gov_Bank_Sum_20d',
    'TWII_Bias_60', 'MA_60', 'Bias_60', 'Return_1d', 'TWII_Return_1d', 'Alpha_1d',
    'Volatility_20d', 'MA_20', 'BB_Width', 'Vol_MA_20', 'Vol_Ratio', 'Foreign_Sum_20d',
    'Trust_Sum_20d', 'MACD_Hist', 'Rev_YoY'
]

# 🛡️ 防毒防線 1：確保所有需要的欄位都在，且強制清洗 DataFrame 裡面的 Inf (無限大) 和 NaN
for col in feature_cols:
    if col not in df.columns:
        df[col] = 0.0
df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

df_sorted = df.sort_values(['stock_id', 'Date'])
grouped = df_sorted.groupby('stock_id')

all_stocks_data, final_tickers, latest_volatility = [], [], []
for stock_id, group in grouped:
    if len(group) >= 120:
        # 取出最後 120 天的特徵
        group_features = group.tail(120)[feature_cols].copy()

        # 🌟 找回失落的 Z-score 標準化！(這是拯救複製人的唯一解藥)
        for col in feature_cols:
            mean = group_features[col].mean()
            std = group_features[col].std()
            # 轉換為均值 0，標準差 1 的常態分佈
            group_features[col] = (group_features[col] - mean) / (std + 1e-8)

        all_stocks_data.append(group_features.values)
        final_tickers.append(str(stock_id))
        vol = group.iloc[-1].get('Volatility_20d', 0.02)
        latest_volatility.append(vol if (not pd.isna(vol) and vol > 0) else 0.02)

test_x = torch.tensor(np.array(all_stocks_data), dtype=torch.float32).cuda()

# 🛡️ 防毒防線 2：強制清洗 Tensor，把所有神經網路最怕的 NaN 和 Inf 全部歸零！
test_x = torch.nan_to_num(test_x, nan=0.0, posinf=0.0, neginf=0.0)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class MarketMambaV5_DynamicGAT(nn.Module):
    def __init__(self, input_dim=46, seq_len=120, d_model=128, pred_days=30, num_mamba_layers=4, d_state=16, dropout_rate=0.4, k_neighbors=10):
        super().__init__()
        self.pred_days = pred_days
        self.d_model = d_model
        self.k_neighbors = k_neighbors
        self.embedding = nn.Sequential(nn.Linear(input_dim, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout_rate))
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=seq_len)
        self.mamba_layers, self.mamba_norms = nn.ModuleList(), nn.ModuleList()
        for _ in range(num_mamba_layers):
            self.mamba_layers.append(Mamba(d_model=d_model, d_state=d_state, d_conv=4, expand=2))
            self.mamba_norms.append(nn.LayerNorm(d_model))
        self.dropout = nn.Dropout(dropout_rate)
        self.gat = GATv2Conv(d_model, d_model // 4, heads=4, concat=True, dropout=dropout_rate)
        self.gating_linear = nn.Linear(d_model * 2, d_model)
        self.trajectory_head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout_rate), nn.Linear(d_model // 2, pred_days))

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        for mamba, norm in zip(self.mamba_layers, self.mamba_norms):
            x = x + self.dropout(mamba(norm(x)))
        x_temporal = x[:, -1, :]

        # 🛡️ 防毒防線 3：保護 GAT 注意力矩陣不崩潰
        x_temporal = torch.nan_to_num(x_temporal, nan=0.0)
        x_norm = F.normalize(x_temporal, p=2, dim=1)
        sim_matrix = torch.mm(x_norm, x_norm.t())

        N = x_temporal.size(0)
        k = min(self.k_neighbors + 1, N)
        _, topk_indices = torch.topk(sim_matrix, k=k, dim=1)
        source_nodes = torch.arange(N, device=x.device).repeat_interleave(k)
        target_nodes = topk_indices.view(-1)
        edge_index = torch.stack([source_nodes, target_nodes], dim=0)
        x_graph = self.gat(x_temporal, edge_index)
        combined = torch.cat([x_temporal, x_graph], dim=1)
        gate = torch.sigmoid(self.gating_linear(combined))
        x_fused = gate * x_temporal + (1 - gate) * x_graph
        return self.trajectory_head(x_fused)

MODEL_PATH = '/content/drive/MyDrive/MarketMamba_V5/Models/V5_DynamicGAT_Production.pth'
model = MarketMambaV5_DynamicGAT(input_dim=46).cuda()
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

print("⚡ 執行 One-Forward-Pass 軌跡預測...")
with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
    pred_alpha = model(test_x).cpu().numpy()

    # 🛡️ 防毒防線 4：如果模型最後還是吐出奇怪的東西，強制洗掉！
    pred_alpha = np.nan_to_num(pred_alpha, nan=0.0, posinf=0.0, neginf=0.0)

# --- 5. 輸出報告與凱利資金盤 ---
target_idx = 14 # 第 15 天的預期 Alpha
raw_volatility = np.array(latest_volatility)

# 🛡️ 濾網 1：正規化夏普值 (Regularized Sharpe)
# 加上 0.015 (1.5%) 的常數懲罰，防止微小波動率的股票分數異常膨脹
sharpe_score = pred_alpha[:, target_idx] / (raw_volatility + 0.015)

# 🛡️ 濾網 2：無情狙擊殭屍股
# 如果真實波動率極低 (< 0.005)，代表它是死魚股或暫停交易，直接給予極端負分踢出榜單
zombie_mask = raw_volatility < 0.005
sharpe_score[zombie_mask] = -999.0

# 網頁 UI 顯示用的波動率 (維持最低 0.50% 的顯示下限)
display_volatility = np.clip(raw_volatility, a_min=0.005, a_max=None)

df_export = pd.DataFrame({
    'Ticker': final_tickers,
    'Exp_Return_15D': pred_alpha[:, target_idx],
    'Volatility_Risk': display_volatility,
    'Sharpe_Score': sharpe_score
})

# 凱利公式計算 (只買正報酬，且絕對不買殭屍股)
df_export['Kelly_Raw'] = np.where(
    (df_export['Exp_Return_15D'] > 0) & (df_export['Sharpe_Score'] > -100),
    df_export['Exp_Return_15D'] / (df_export['Volatility_Risk'] ** 2),
    0
)
df_export['Suggested_Weight'] = np.clip(df_export['Kelly_Raw'] * 0.5, a_min=0.0, a_max=0.20)

# 排序：真正的 Alpha 強勢股將會浮出水面！
df_export = df_export.sort_values('Sharpe_Score', ascending=False)
df_export.to_csv("df_kelly.csv", index=False)

df_trajectory = pd.DataFrame(pred_alpha, columns=[f'Day_{i+1}' for i in range(30)])
df_trajectory.insert(0, 'Ticker', final_tickers)
df_trajectory.to_csv("df_traj.csv", index=False)

del model, test_x
torch.cuda.empty_cache()
print("✅ 預測結果生成完畢！殭屍股已全數肅清，準備交接給實盤機器人...")

# ==========================================
# 📊 MarketMamba V5.0: 傳統型態學雷達 (傳家寶最終版)
# 支援：傾斜頸線、1/2~3/4時間密碼、等幅測距、上飄旗形
# ==========================================
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from google.colab import drive

# 1. 讀取資料
drive.mount('/content/drive')
DATA_PATH = '/content/drive/MyDrive/MarketMamba_V5/Processed_Features/V5_Mamba_Matrix.parquet'

print("📥 正在讀取全市場資料...")
df_master = pd.read_parquet(DATA_PATH)
df_master['Date'] = pd.to_datetime(df_master['Date'])
latest_date = df_master['Date'].max()
print(f"📅 掃描基準日: {latest_date.strftime('%Y-%m-%d')}")

# ==========================================
# 📐 幾何與趨勢線運算核心
# ==========================================
def get_extrema(prices, order):
    mins = argrelextrema(prices, np.less, order=order)[0]
    maxs = argrelextrema(prices, np.greater, order=order)[0]
    return mins, maxs

def get_trendline_val(x1, y1, x2, y2, target_x):
    """計算兩點形成的趨勢線，在 target_x 時的 Y 值 (動態頸線)"""
    if x2 == x1: return y1
    slope = (y2 - y1) / (x2 - x1)
    return y1 + slope * (target_x - x1)

# ==========================================
# 🟡 模式 A: 標準 W底 / 多重底 (支援傾斜頸線)
# ==========================================
def detect_standard_w(prices, mins, maxs):
    if len(mins) < 2 or len(maxs) < 2: return None
    l1_idx, l2_idx = mins[-2], mins[-1]
    l1, l2 = prices[l1_idx], prices[l2_idx]

    # 取最後兩個波峰作為動態頸線基準
    h1_idx, h2_idx = maxs[-2], maxs[-1]
    h1, h2 = prices[h1_idx], prices[h2_idx]

    # 確保留下的高點在低點附近
    if h2_idx < l1_idx: return None

    current_idx = len(prices) - 1
    current = prices[-1]

    # 左右腳等高判定
    if not (0.97 <= l2 / l1 <= 1.03): return None

    # 計算今天的「動態頸線」位置
    dynamic_neckline = get_trendline_val(h1_idx, h1, h2_idx, h2, current_idx)
    if dynamic_neckline <= 0: return None

    ratio = current / dynamic_neckline
    if 0.98 <= ratio <= 1.05:
        score = 100 - abs(1 - (l2/l1))*500 - abs(1 - ratio)*300
        target = dynamic_neckline + (dynamic_neckline - min(l1, l2))
        return '🟡 標準W底', score, dynamic_neckline, target
    return None

# ==========================================
# 🟢 模式 B: 破底翻 (假跌破強勢站回)
# ==========================================
def detect_spring_w(prices, mins, maxs):
    if len(mins) < 2 or len(maxs) < 1: return None
    l1_idx, l2_idx = mins[-2], mins[-1]
    l1, l2 = prices[l1_idx], prices[l2_idx]

    m_max = [x for x in maxs if l1_idx < x < l2_idx]
    if not m_max: return None
    neckline = prices[m_max[np.argmax(prices[m_max])]]
    current = prices[-1]

    # 破底定義：跌破左腳 2% ~ 10%
    if not (0.90 <= l2 / l1 <= 0.98): return None
    if current < l1: return None # 必須站回左腳之上

    ratio = current / neckline
    if 0.98 <= ratio <= 1.05:
        score = 100 - abs(1 - ratio)*400
        target = neckline + (neckline - l2)
        return '🟢 破底翻', score, neckline, target
    return None

# ==========================================
# 🔴 模式 C: M頭 (容許極端假突破誘多)
# ==========================================
def detect_m_top(prices, mins, maxs):
    if len(maxs) < 2 or len(mins) < 1: return None
    h1_idx, h2_idx = maxs[-2], maxs[-1]
    h1, h2 = prices[h1_idx], prices[h2_idx]

    m_min = [x for x in mins if h1_idx < x < h2_idx]
    if not m_min: return None
    neckline = prices[m_min[np.argmin(prices[m_min])]]
    current = prices[-1]

    # 假突破定義：右頭可以高出左頭高達 15%！
    if not (0.95 <= h2 / h1 <= 1.15): return None

    ratio = current / neckline
    if 0.95 <= ratio <= 1.02:
        score = 100 - abs(1 - ratio)*300
        target = max(neckline - (max(h1, h2) - neckline), 0.1)
        return '🔴 M頭(偏空)', score, neckline, target
    return None

# ==========================================
# 🟣 模式 D: 頭肩底 (支援傾斜頸線)
# ==========================================
def detect_hns_bottom(prices, mins, maxs):
    if len(mins) < 3 or len(maxs) < 2: return None
    l1_idx, l2_idx, l3_idx = mins[-3], mins[-2], mins[-1]
    l1, l2, l3 = prices[l1_idx], prices[l2_idx], prices[l3_idx]

    m_max1 = [x for x in maxs if l1_idx < x < l2_idx]
    m_max2 = [x for x in maxs if l2_idx < x < l3_idx]
    if not m_max1 or not m_max2: return None

    h1_idx = m_max1[np.argmax(prices[m_max1])]
    h2_idx = m_max2[np.argmax(prices[m_max2])]
    h1, h2 = prices[h1_idx], prices[h2_idx]

    current_idx = len(prices) - 1
    current = prices[-1]

    if not (l2 < l1 * 0.98 and l2 < l3 * 0.98): return None
    if not (0.92 <= l1 / l3 <= 1.08): return None

    # 動態傾斜頸線
    dynamic_neckline = get_trendline_val(h1_idx, h1, h2_idx, h2, current_idx)

    ratio = current / dynamic_neckline
    if 0.98 <= ratio <= 1.05:
        score = 100 - abs(1 - (l1/l3))*300 - abs(1 - ratio)*300
        target = dynamic_neckline + (dynamic_neckline - l2)
        return '🟣 頭肩底', score, dynamic_neckline, target
    return None

# ==========================================
# 🔵 模式 E: 收斂三角形 (1/2 ~ 3/4 時間密碼)
# ==========================================
def detect_triangle(prices, mins, maxs):
    if len(mins) < 2 or len(maxs) < 2: return None

    h1_idx, h2_idx = maxs[-2], maxs[-1]
    l1_idx, l2_idx = mins[-2], mins[-1]
    h1, h2 = prices[h1_idx], prices[h2_idx]
    l1, l2 = prices[l1_idx], prices[l2_idx]

    current_idx = len(prices) - 1
    current = prices[-1]

    if not (h1 > h2 * 1.01 and l1 < l2 * 0.99): return None

    # 計算時間密碼 (Apex 交點)
    m_up = (h2 - h1) / (h2_idx - h1_idx) if h2_idx != h1_idx else 0
    m_low = (l2 - l1) / (l2_idx - l1_idx) if l2_idx != l1_idx else 0

    if m_up >= m_low: return None # 確保線條收斂

    b_up = h1 - m_up * h1_idx
    b_low = l1 - m_low * l1_idx
    apex_x = (b_low - b_up) / (m_up - m_low)

    start_x = min(h1_idx, l1_idx)
    total_len = apex_x - start_x
    if total_len <= 0: return None

    # 驗證是否在 50% ~ 80% 之間突破 (稍微放寬 75% 到 80% 增加實戰彈性)
    progress = (current_idx - start_x) / total_len
    if not (0.50 <= progress <= 0.80): return None

    # 動態壓力線
    resistance = m_up * current_idx + b_up

    if 0.99 <= current / resistance <= 1.05:
        score = 100 - abs(1 - current/resistance)*400
        target = resistance + (h1 - l1)
        return '🔵 收斂三角', score, resistance, target
    return None

# ==========================================
# 💀 模式 F: 上飄旗形跌破 (波段跌幅測距)
# ==========================================
def detect_bear_flag(prices, mins, maxs):
    # 尋找：大高點 -> 大低點(跌幅段) -> 小高點(反彈旗形)
    if len(maxs) < 2 or len(mins) < 1: return None

    h1_idx, h2_idx = maxs[-2], maxs[-1]
    h1, h2 = prices[h1_idx], prices[h2_idx]

    # 抓取大跌幅段的最低點
    m_min = [x for x in mins if h1_idx < x < h2_idx]
    if not m_min: return None
    l1_idx = m_min[np.argmin(prices[m_min])]
    l1 = prices[l1_idx]

    current = prices[-1]

    # 條件1: 主跌段很深 (跌幅大於 10%)
    if l1 > h1 * 0.90: return None
    # 條件2: 反彈形成上飄旗形 (反彈不超過主跌段的 2/3)
    if not (l1 < h2 < l1 + (h1 - l1) * 0.66): return None

    # 跌破旗形底部
    if current <= l1 * 1.02:
        score = 80 # 強力看空訊號
        # 目標價：旗形高點 - 主跌段跌幅 (H2 - (H1 - L1))
        target = max(h2 - (h1 - l1), 0.1)
        return '💀 上飄旗跌破', score, l1, target
    return None

# ==========================================
# 🚀 執行主程序掃描
# ==========================================
print("🕵️‍♂️ 啟動終極型態雷達 (六大經典結構、等幅測距、時間密碼解析)...")
grouped = df_master.sort_values('Date').groupby('stock_id')
scales = {'1_短線 (1~2週)': 3, '2_中線 (約1個月)': 8, '3_長線 (2~3個月)': 15, '4_半年大底': 30}
results = []

for stock_id, group in grouped:
    stock_str = str(stock_id).strip()

    if not (len(stock_str) == 4 and stock_str.isdigit()): continue
    if group['Date'].max() < latest_date - pd.Timedelta(days=5): continue
    df_recent = group.tail(180).copy()
    if len(df_recent) < 100: continue
    if df_recent['Volume'].tail(20).mean() < 500000: continue
    if df_recent['Close'].iloc[-1] < 10.0: continue

    prices = df_recent['Close'].values

    for scale_name, order in scales.items():
        mins, maxs = get_extrema(prices, order)

        if len(mins) > 0 and (len(prices) - 1 - mins[-1]) > (order * 2.5): continue
        if len(maxs) > 0 and (len(prices) - 1 - maxs[-1]) > (order * 2.5): continue

        patterns = [
            detect_standard_w(prices, mins, maxs),
            detect_spring_w(prices, mins, maxs),
            detect_m_top(prices, mins, maxs),
            detect_hns_bottom(prices, mins, maxs),
            detect_triangle(prices, mins, maxs),
            detect_bear_flag(prices, mins, maxs)
        ]

        matched_pattern = next((p for p in patterns if p and p[1] >= 60), None)

        if matched_pattern:
            p_name, score, neck, target = matched_pattern
            current_p = prices[-1]
            expected_return = ((target - current_p) / current_p) * 100

            results.append({
                'Pattern': p_name,
                'Stock_ID': stock_str,
                'Scale': scale_name,
                'Score': round(score, 1),
                'Current_Price': round(current_p, 2),
                'Neckline_Ref': round(neck, 2),
                'Target_Price': round(target, 2),
                'Exp_Return(%)': round(expected_return, 2)
            })
            break

if results:
    df_patterns = pd.DataFrame(results)
    df_patterns = df_patterns.sort_values(['Pattern', 'Scale', 'Score'], ascending=[False, True, False]).reset_index(drop=True)
    print(f"\n🎉 掃描完成！共找出 {len(df_patterns)} 檔關鍵型態股！")
    display(df_patterns)
else:
    print("\n📉 今日市場無符合結構的股票。")

# ==========================================
# 📊 輸出結果與存檔 (準備推播)
# ==========================================
csv_path = '/content/drive/MyDrive/MarketMamba_V5/pattern_scan_results.csv'

if results:
    df_patterns = pd.DataFrame(results)
    df_patterns = df_patterns.sort_values(['Pattern', 'Scale', 'Score'], ascending=[False, True, False]).reset_index(drop=True)

    # 💾 存檔！
    df_patterns.to_csv(csv_path, index=False)

    print(f"\n🎉 掃描完成！共找出 {len(df_patterns)} 檔關鍵型態股！")
    print(f"📁 檔案已成功儲存至: {csv_path}，準備交接給 GitHub 發布管線！")
    display(df_patterns)
else:
    # 如果今天沒有訊號，也要產生一個空的 CSV，這樣網頁端才不會報錯找不到檔案
    empty_df = pd.DataFrame(columns=['Pattern', 'Stock_ID', 'Scale', 'Score', 'Current_Price', 'Neckline_Ref', 'Target_Price', 'Exp_Return(%)'])
    empty_df.to_csv(csv_path, index=False)
    print("\n📉 今日市場無符合結構的股票。已產生空值報表覆蓋舊檔。")

# ==========================================
# 🚀 V5.0 終極管線 (Cell 4/5): 自動調倉機器人 (向下相容舊帳本版)
# ==========================================
import json
import requests
import datetime

print("🤖 啟動量化實盤機器人...")

LEDGER_URL = "https://raw.githubusercontent.com/FrankChen0930/MarketMamba/main/robot_ledger.json"
try:
    ledger = requests.get(LEDGER_URL).json()
except:
    ledger = {"cash": 1000000.0, "holdings": {}, "history": []}

# 直接從今日更新好的 Parquet 提取收盤價，完全不依賴 yfinance API
latest_date = df['Date'].max()
current_prices = df[df['Date'] == latest_date].set_index('stock_id')['Close'].to_dict()

total_equity = ledger["cash"]
for t, position in ledger["holdings"].items():
    # 🚨 相容性修復：舊帳本叫 "cost"，新帳本叫 "avg_cost"
    hist_cost = position.get("avg_cost", position.get("cost", 0))
    total_equity += position["shares"] * current_prices.get(t, hist_cost)

buy_list = df_export.head(10)['Ticker'].tolist()

# 賣出邏輯
for t in list(ledger["holdings"].keys()):
    if t not in buy_list:
        sell_price = current_prices.get(t, 0)
        if sell_price > 0:
            ledger["cash"] += ledger["holdings"][t]["shares"] * sell_price
            del ledger["holdings"][t]
            print(f"  🔴 賣出 {t} @ {sell_price:.2f}")

# 買入邏輯
for _, row in df_export.head(10).iterrows():
    t, weight = str(row['Ticker']), row['Suggested_Weight']
    if t not in current_prices: continue

    price = current_prices[t]
    target_value = total_equity * weight
    current_shares = ledger["holdings"].get(t, {}).get("shares", 0)
    money_to_invest = target_value - (current_shares * price)

    if money_to_invest > 0 and ledger["cash"] > money_to_invest:
        shares_to_buy = int(float(money_to_invest) // price)
        if shares_to_buy > 0:
            ledger["cash"] -= shares_to_buy * price
            old_s = ledger["holdings"].get(t, {}).get("shares", 0)

            # 🚨 相容性修復
            old_c = ledger["holdings"].get(t, {}).get("avg_cost", ledger["holdings"].get(t, {}).get("cost", price))

            new_s = old_s + shares_to_buy
            ledger["holdings"][t] = {"shares": new_s, "avg_cost": ((old_s*old_c) + (shares_to_buy*price))/new_s}
            print(f"  🟢 買入 {t}: {shares_to_buy} 股 @ {price:.2f}")

today_str = (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=8)).strftime("%Y-%m-%d")
if not ledger["history"] or ledger["history"][-1]["date"] != today_str:
    ledger["history"].append({"date": today_str, "equity": total_equity})

with open("robot_ledger.json", "w") as f:
    json.dump(ledger, f, indent=4)
print("✅ 機器人調倉完畢！")

# ==========================================
# 🚀 V5.0 終極管線 (Cell 5/5): GitHub 推送與拔除電源
# ==========================================
import os
import datetime
from google.colab import userdata, runtime
import time

print("🚀 開始執行自動化發布管線...")

GITHUB_TOKEN = userdata.get('GITHUB_TOKEN')
GITHUB_USER = "FrankChen0930"
GITHUB_EMAIL = "a0966469964@gmail.com"
REPO_URL = f"https://{GITHUB_TOKEN}@github.com/{GITHUB_USER}/MarketMamba.git"

os.system("rm -rf repo_folder")
os.system(f"git clone {REPO_URL} repo_folder")

# 🌟 修正點 1：分別從正確的路徑把檔案複製進來
# 前三個檔案在 Colab 本地暫存區
os.system("cp df_kelly.csv df_traj.csv robot_ledger.json repo_folder/")
# 型態學雷達報表在 Google Drive 裡面
os.system("cp '/content/drive/MyDrive/MarketMamba_V5/pattern_scan_results.csv' repo_folder/")

os.chdir("repo_folder")

os.system(f"git config user.name '{GITHUB_USER}'")
os.system(f"git config user.email '{GITHUB_EMAIL}'")

# 🌟 修正點 2：使用最新的 timezone-aware 寫法，消滅黃色警告
tw_time = (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S")
with open("update_time.txt", "w") as f:
    f.write(tw_time)

# 將所有更新的檔案加入 Git 追蹤
os.system("git add df_kelly.csv df_traj.csv robot_ledger.json update_time.txt pattern_scan_results.csv")

# 提交並推送
os.system(f"git commit -m '🤖 Auto-update: V5.0 End-to-End Pipeline ({tw_time})'")
push_result = os.system("git push origin main")

if push_result == 0:
    print(f"🎉 大功告成！V5.0 網頁與資料庫已同步至最新狀態！({tw_time})")
else:
    print("⚠️ 推送過程異常！請檢查 GitHub Token 權限。")

os.chdir("/content")
print("🔌 系統將在 3 秒後自動切斷執行階段電源...")
time.sleep(3)
runtime.unassign()