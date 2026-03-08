import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import seaborn as sns
import yfinance as yf
import plotly.graph_objects as go

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
# 頁面 2: 個股軌跡透視 (Plotly 專業互動版)
# ------------------------------------------
elif page == "📈 個股軌跡透視":
    st.subheader("🔭 平行宇宙軌跡觀測儀 (專業互動版)")
    st.write("結合歷史真實 K 線與未來 30 天擴散機率雲，支援滑鼠懸停與拖曳縮放。")
    
    # 建立選擇器
    top_1_ticker = str(df_kelly['Ticker'].iloc[0])
    all_tickers = df_traj['Ticker'].astype(str).tolist()
    default_idx = all_tickers.index(top_1_ticker) if top_1_ticker in all_tickers else 0
    target_ticker = st.selectbox("🔍 請選擇要觀測的股票代號", all_tickers, index=default_idx)
    
    # 抓出預測軌跡與該股的波動風險
    stock_traj = df_traj[df_traj['Ticker'].astype(str) == target_ticker].iloc[0]
    volatility = df_kelly[df_kelly['Ticker'].astype(str) == target_ticker]['Volatility_Risk'].iloc[0]
    
    traj_values = stock_traj[[f'Day_{i}' for i in range(1, 31)]].values 
    
    # ==========================================
    # ⚡ 透過 yfinance 抓取「過去 20 天」的真實歷史
    # ==========================================
    @st.cache_data(ttl=3600)
    def fetch_history_data(ticker):
        for suffix in ['.TW', '.TWO']:
            try:
                hist = yf.Ticker(f"{ticker}{suffix}").history(period="1mo") # 抓一個月歷史
                if not hist.empty:
                    return hist, suffix
            except:
                continue
        return None, None
        
    with st.spinner("正在同步市場即時報價與建構機率雲..."):
        hist_df, suffix = fetch_history_data(target_ticker)
        
    if hist_df is None:
        st.warning(f"⚠️ 無法抓取 {target_ticker} 的歷史股價。")
    else:
        current_price = hist_df['Close'].iloc[-1]
        last_date = hist_df.index[-1]
        
        # 1. 計算未來的真實日期 (避開六日)
        future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=30)
        
        # 2. 計算未來的目標均價
        future_mean_prices = current_price * (1 + traj_values)
        
        # 3. 利用波動率 (Volatility) 展開機率分布漏斗
        # 隨著天數增加，不確定性 (標準差) 會以根號時間放大
        time_scale = np.sqrt(np.arange(1, 31) / 30.0) 
        upper_bound_95 = future_mean_prices * (1 + (volatility * 1.5 * time_scale))
        lower_bound_95 = future_mean_prices * (1 - (volatility * 1.5 * time_scale))
        
        upper_bound_68 = future_mean_prices * (1 + (volatility * 0.8 * time_scale))
        lower_bound_68 = future_mean_prices * (1 - (volatility * 0.8 * time_scale))
        
        # ==========================================
        # 📈 使用 Plotly 繪製專業互動圖表
        # ==========================================
        fig = go.Figure()

        # A. 畫出歷史真實股價 (左半邊)
        fig.add_trace(go.Scatter(
            x=hist_df.index, y=hist_df['Close'],
            mode='lines', name='歷史真實股價',
            line=dict(color='#00d2ff', width=3)
        ))

        # B. 畫出未來 95% 機率區間 (最外層，透明度最高)
        fig.add_trace(go.Scatter(
            x=list(future_dates) + list(future_dates)[::-1],
            y=list(upper_bound_95) + list(lower_bound_95)[::-1],
            fill='toself', fillcolor='rgba(255, 127, 14, 0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip", name='95% 機率雲'
        ))

        # C. 畫出未來 68% 機率區間 (內層，透明度較低)
        fig.add_trace(go.Scatter(
            x=list(future_dates) + list(future_dates)[::-1],
            y=list(upper_bound_68) + list(lower_bound_68)[::-1],
            fill='toself', fillcolor='rgba(255, 127, 14, 0.25)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip", name='68% 機率雲'
        ))

        # D. 畫出未來預測平均軌跡 (中心虛線)
        fig.add_trace(go.Scatter(
            x=future_dates, y=future_mean_prices,
            mode='lines+markers', name='預測平均軌跡',
            line=dict(color='#ff7f0e', width=3, dash='dot'),
            marker=dict(size=5)
        ))
        
        # E. 連接歷史與未來的橋樑
        fig.add_trace(go.Scatter(
            x=[last_date, future_dates[0]], y=[current_price, future_mean_prices[0]],
            mode='lines', line=dict(color='#ff7f0e', width=3, dash='dot'), showlegend=False
        ))

        # 圖表版面美化 (暗黑科技風)
        fig.update_layout(
            title=f"<b>{target_ticker}{suffix} 股價預測與機率分布</b>",
            yaxis_title="股價 (TWD)",
            xaxis_title="日期",
            template="plotly_dark", # 瞬間變身專業看盤軟體
            hovermode="x unified",  # 滑鼠游標會顯示整條線的資訊
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            margin=dict(l=40, r=40, t=60, b=40)
        )
        
        # 加上目前股價的水平基準線
        fig.add_hline(y=current_price, line_dash="dash", line_color="gray", annotation_text="今日收盤價")

        # 在 Streamlit 中渲染 Plotly 圖表
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("💡 操作提示：你可以用滑鼠在圖表上框選放大特定區域，或是將游標停留在未來日期上查看精準的預測價位。淺橘色區域代表未來可能發生的震盪範圍！")

# ------------------------------------------
# 頁面 3: 持股監視與實盤
# ------------------------------------------
elif page == "🤖 持股監視與實盤 (開發中)":
    st.subheader("💼 我的專屬量化基金 (Portfolio)")
    st.write("輸入你目前持有的股票，MarketMamba 將每天自動為你進行 AI 健檢與退場評估。")
    
    # 1. 初始化虛擬帳本 (Session State)
    if 'portfolio' not in st.session_state:
        st.session_state['portfolio'] = pd.DataFrame(columns=['股票代號', '持有成本', '持有股數'])

    # 2. 新增持股的表單介面
    with st.expander("➕ 新增庫存持股", expanded=True):
        col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
        with col1:
            # 為了防呆，直接讓使用者從預測清單裡面用選的
            all_tickers_clean = df_kelly['Ticker'].astype(str).tolist()
            new_ticker = st.selectbox("股票代號", all_tickers_clean)
        with col2:
            new_cost = st.number_input("持有成本 (TWD)", min_value=0.0, value=100.0, step=1.0)
        with col3:
            new_shares = st.number_input("持有股數 (股)", min_value=1, value=1000, step=100)
        with col4:
            st.markdown("<br>", unsafe_allow_html=True) # 排版對齊用
            if st.button("新增", type="primary"):
                # 將新資料加入 Session State 的 DataFrame 中
                new_row = pd.DataFrame({'股票代號': [new_ticker], '持有成本': [new_cost], '持有股數': [new_shares]})
                st.session_state['portfolio'] = pd.concat([st.session_state['portfolio'], new_row], ignore_index=True)
                st.success(f"已新增 {new_ticker}！")
                st.rerun() # 重新整理網頁以顯示最新表格

    # 3. AI 持股健檢與即時監控
    if not st.session_state['portfolio'].empty:
        st.divider()
        st.subheader("🏥 AI 庫存持股健檢報告")
        
        # 複製一份使用者的持股來做加工
        my_portfolio = st.session_state['portfolio'].copy()
        
        # 將我們的 df_kelly 預測資料與使用者的持股合併 (Merge)
        kelly_subset = df_kelly[['Ticker', 'Exp_Return_15D', 'Sharpe_Score']].copy()
        kelly_subset['Ticker'] = kelly_subset['Ticker'].astype(str)
        
        # 進行資料表關聯
        analysis_df = pd.merge(my_portfolio, kelly_subset, left_on='股票代號', right_on='Ticker', how='left')
        
        # 撰寫 AI 判斷邏輯
        def get_action_signal(row):
            if pd.isna(row['Exp_Return_15D']):
                return "⚪ 缺乏預測資料"
            elif row['Exp_Return_15D'] < 0:
                return "🔴 趨勢轉弱 (建議獲利了結/停損)"
            elif row['Sharpe_Score'] > 0.5:
                return "🟢 強勢護城河 (建議續抱)"
            else:
                return "🟡 波動加劇 (請嚴格設定停損點)"
                
        analysis_df['AI 操作建議'] = analysis_df.apply(get_action_signal, axis=1)
        
        # 格式化數字以便閱讀
        analysis_df['預期 15 天報酬'] = (analysis_df['Exp_Return_15D'] * 100).apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
        analysis_df['夏普分數'] = analysis_df['Sharpe_Score'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
        
        # 選擇要展示的欄位
        display_cols = ['股票代號', '持有成本', '持有股數', '預期 15 天報酬', '夏普分數', 'AI 操作建議']
        
        st.dataframe(
            analysis_df[display_cols].style.applymap(
                lambda x: 'color: #ff4b4b; font-weight: bold;' if '🔴' in str(x) else 
                          ('color: #00fa9a; font-weight: bold;' if '🟢' in str(x) else ''),
                subset=['AI 操作建議']
            ),
            use_container_width=True,
            hide_index=True
        )
        
        # 提供清空按鈕
        if st.button("🗑️ 清空庫存"):
            st.session_state['portfolio'] = pd.DataFrame(columns=['股票代號', '持有成本', '持有股數'])
            st.rerun()
    else:
        st.info("👆 目前庫存為空，請從上方新增持股來啟動 MarketMamba 的 AI 健檢功能。")



