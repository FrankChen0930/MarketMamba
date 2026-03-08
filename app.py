import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import seaborn as sns
import yfinance as yf

# ==========================================
# 0. 網頁基本設定 & 字型處理
# ==========================================
st.set_page_config(page_title="MarketMamba 量化決策中心", page_icon="🐍", layout="wide")

# 動態載入字型檔 (確保你的資料夾裡有 NotoSansTC-Regular.ttf)
font_path = "NotoSansTC-Regular.ttf" 
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
else:
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'sans-serif'] 
plt.rcParams['axes.unicode_minus'] = False 

st.title("🐍 MarketMamba V3.1: 終極量化決策中心")
st.markdown("基於 128 維度 Mamba 與 30 天擴散軌跡的自動化資金配置系統。")

# ==========================================
# 1. 載入 V3.1 雲端資料庫 (Google Drive 直讀)
# ==========================================
# 加入 ttl=3600，讓網頁每小時自動去雲端抓取最新資料
@st.cache_data(ttl=3600, show_spinner="正在從雲端硬碟同步最新預測資料...")
def load_v3_data():
    # ⚠️ 請在這裡貼上你剛剛在 Google Drive 取得的 File ID ⚠️
    kelly_file_id = "18fbj6kS4HfvojNrdSRiGz1SXF3qmYTm1" 
    traj_file_id = "1bVd5EWp-tN8QYr9yjkPSgkmM_vp5-YF2"
    
    # Pandas 支援直接讀取 Google Drive 的下載連結
    kelly_url = f"https://drive.google.com/uc?id={kelly_file_id}"
    traj_url = f"https://drive.google.com/uc?id={traj_file_id}"
    
    try:
        df_kelly = pd.read_csv(kelly_url)
        df_traj = pd.read_csv(traj_url)
        return df_kelly, df_traj, True
    except Exception as e:
        st.error(f"讀取雲端資料失敗，請檢查 File ID 或共用權限設定。詳細錯誤：{e}")
        return None, None, False

df_kelly, df_traj, data_loaded = load_v3_data()

# ==========================================
# 2. 側邊欄導覽列
# ==========================================
with st.sidebar:
    st.header("📌 導覽選單")
    if data_loaded:
        st.success("✅ V3.1 雲端數據庫已連線")
    else:
        st.error("⚠️ 無法連線至雲端數據庫")
        
    page = st.radio("前往頁面", ["📊 今日凱利資金盤", "📈 個股軌跡透視", "🤖 持股監視與實盤 (開發中)"])
    
    st.divider()
    st.info("💡 系統運作邏輯：\n每天收盤後由後端 A100 GPU 進行離線推論，覆蓋 Google Drive 檔案。前端網頁定時自動同步，實現 0 延遲秒開體驗。")

# ==========================================
# 3. 頁面內容路由
# ==========================================
if not data_loaded:
    st.stop() # 如果資料沒抓到，就停止渲染後面的畫面

# ------------------------------------------
# 頁面 1: 凱利資金盤 (Top Picks)
# ------------------------------------------
if page == "📊 今日凱利資金盤":
    st.subheader("🎯 今日最佳防禦型飆股 (Top 10)")
    st.write("根據未來 15 天預期報酬與變異數，套用半凱利公式 (最高上限 20%) 給出的資金配置建議。")
    
    # 抓取前 10 名，並美化數字顯示
    top_picks = df_kelly.head(10).copy()
    
    # 建立一個美化版的 DataFrame 給前端展示
    display_df = top_picks[['Ticker', 'Exp_Return_15D', 'Volatility_Risk', 'Sharpe_Score', 'Suggested_Weight']].copy()
    
    # 格式化百分比
    display_df['Exp_Return_15D'] = (display_df['Exp_Return_15D'] * 100).apply(lambda x: f"{x:.2f}%")
    display_df['Volatility_Risk'] = (display_df['Volatility_Risk'] * 100).apply(lambda x: f"{x:.2f}%")
    display_df['Suggested_Weight'] = (display_df['Suggested_Weight'] * 100).apply(lambda x: f"{x:.2f}%")
    display_df['Sharpe_Score'] = display_df['Sharpe_Score'].apply(lambda x: f"{x:.4f}")
    
    # 修改欄位名稱為中文，讓使用者體驗更好
    display_df.columns = ['股票代號', '預期報酬 (15天)', '波動風險', '夏普 CP 值', '建議資金配置比例']
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)

# ------------------------------------------
# 頁面 2: 個股軌跡透視
# ------------------------------------------
elif page == "📈 個股軌跡透視":
    st.subheader("🔭 平行宇宙軌跡觀測儀")
    st.write("在這裡，我們將利用大腦預測的未來 30 天走勢，同時觀測「累積報酬率」與「實際股價」的變化。")
    
    # 建立一個選擇器，預設帶入 Top 10 的第一檔
    top_1_ticker = str(df_kelly['Ticker'].iloc[0])
    all_tickers = df_traj['Ticker'].astype(str).tolist()
    default_idx = all_tickers.index(top_1_ticker) if top_1_ticker in all_tickers else 0
    target_ticker = st.selectbox("🔍 請選擇要觀測的股票代號", all_tickers, index=default_idx)
    
    # 抓出軌跡資料
    stock_traj = df_traj[df_traj['Ticker'].astype(str) == target_ticker].iloc[0]
    days = np.arange(1, 31)
    traj_values = stock_traj[[f'Day_{i}' for i in range(1, 31)]].values * 100 
    
    # ==========================================
    # ⚡ 透過 yfinance 抓取最新股價以推算未來價格
    # ==========================================
    @st.cache_data(ttl=3600)
    def fetch_current_price(ticker):
        # 智慧判斷：台股有上市 (.TW) 跟上櫃 (.TWO) 之分，我們讓程式自動試探
        for suffix in ['.TW', '.TWO']:
            try:
                hist = yf.Ticker(f"{ticker}{suffix}").history(period="5d")
                if not hist.empty:
                    return hist['Close'].iloc[-1], suffix
            except:
                continue
        return None, None
        
    with st.spinner("正在同步市場最新即時報價..."):
        current_price, suffix = fetch_current_price(target_ticker)
        
    # ==========================================
    # 📈 繪製雙圖表 (報酬率 vs 股價)
    # ==========================================
    if current_price is None:
        st.warning(f"⚠️ 無法從 Yahoo Finance 抓取 {target_ticker} 的即時股價，僅顯示報酬率預測。")
        price_values = None
    else:
        st.success(f"✅ 成功抓取 {target_ticker}{suffix} 最新收盤價： **{current_price:.2f} TWD**")
        # 數學還原：未來股價 = 目前股價 * (1 + 累積報酬率/100)
        price_values = current_price * (1 + traj_values / 100)
        
    # 使用 Streamlit 雙欄位並排顯示 (左邊報酬率，右邊股價)
    col1, col2 = st.columns(2)
    
    with col1:
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        ax1.plot(days, traj_values, marker='o', markersize=4, linestyle='-', color='#1f77b4', linewidth=2)
        ax1.set_title(f"預期累積報酬率 (%)", fontsize=14, fontweight='bold')
        ax1.set_xlabel("未來天數 (Trading Days)", fontsize=10)
        ax1.set_ylabel("報酬率 (%)", fontsize=10)
        ax1.axhline(0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
        ax1.grid(True, linestyle=':', alpha=0.6)
        
        final_ret = traj_values[-1]
        ax1.annotate(f'{final_ret:.2f}%', xy=(30, final_ret), xytext=(28, final_ret),
                     fontsize=11, fontweight='bold', color='#1f77b4')
        st.pyplot(fig1)
        
    with col2:
        if price_values is not None:
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            ax2.plot(days, price_values, marker='o', markersize=4, linestyle='-', color='#ff7f0e', linewidth=2)
            ax2.set_title(f"預期目標股價 (TWD)", fontsize=14, fontweight='bold')
            ax2.set_xlabel("未來天數 (Trading Days)", fontsize=10)
            ax2.set_ylabel("股價 (TWD)", fontsize=10)
            # 畫一條現在股價的基準線
            ax2.axhline(current_price, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='目前股價')
            ax2.grid(True, linestyle=':', alpha=0.6)
            
            final_price = price_values[-1]
            ax2.annotate(f'{final_price:.2f}', xy=(30, final_price), xytext=(28, final_price),
                         fontsize=11, fontweight='bold', color='#ff7f0e')
            ax2.legend()
            st.pyplot(fig2)
            
    st.info("💡 解讀指南：左圖(報酬率)適合用來跨股票比較 CP 值；右圖(實際股價)適合用來設定你在看盤軟體裡的「停利/停損」掛單價位！")

# ------------------------------------------
# 頁面 3: 持股監視與實盤
# ------------------------------------------
elif page == "🤖 持股監視與實盤 (開發中)":
    st.subheader("🤖 我的虛擬量化基金")
    st.write("這是我們預計實作「持股動態健檢」與「自動化 Paper Trading 績效曲線」的地方！")


