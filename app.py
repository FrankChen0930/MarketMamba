import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import seaborn as sns
import yfinance as yf
import plotly.graph_objects as go
import json

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
# 2. 狀態管理 (Session State) 與 側邊欄設計
# ==========================================
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = "📊 今日凱利資金盤"
if 'target_ticker' not in st.session_state:
    st.session_state['target_ticker'] = None

# ⚡ 動態讀取外部字典檔 (加入快取，避免每次點擊都重新讀檔)
@st.cache_data
def load_ticker_mapping():
    try:
        # 未來如果上傳到 GitHub，這裡可以直接改成 requests.get("你的 GitHub Raw 網址").json()
        with open("ticker_mapping.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.warning(f"找不到 ticker_mapping.json，將直接顯示股票代號。")
        return {}

TW_STOCK_DICT = load_ticker_mapping()

def get_stock_name(ticker):
    # 從外部載入的字典中尋找，找不到就回傳原始代號
    return TW_STOCK_DICT.get(ticker, ticker)

with st.sidebar:
    st.header("📌 功能選單")
    if data_loaded:
        st.success("✅ V3.1 雲端數據庫已連線")
        
    st.markdown("---")
    
    # 使用 button 創造「方塊化」的視覺效果，點擊後更改 session_state
    if st.button("📊 今日凱利資金盤", use_container_width=True, 
                 type="primary" if st.session_state['current_page'] == "📊 今日凱利資金盤" else "secondary"):
        st.session_state['current_page'] = "📊 今日凱利資金盤"
        st.rerun()
        
    if st.button("📈 個股軌跡透視", use_container_width=True,
                 type="primary" if st.session_state['current_page'] == "📈 個股軌跡透視" else "secondary"):
        st.session_state['current_page'] = "📈 個股軌跡透視"
        st.rerun()
        
    if st.button("💼 我的持股健檢", use_container_width=True,
                 type="primary" if st.session_state['current_page'] == "💼 我的持股健檢" else "secondary"):
        st.session_state['current_page'] = "💼 我的持股健檢"
        st.rerun()
        
    if st.button("🤖 百萬實盤機器人", use_container_width=True,
                 type="primary" if st.session_state['current_page'] == "🤖 百萬實盤機器人" else "secondary"):
        st.session_state['current_page'] = "🤖 百萬實盤機器人"
        st.rerun()

# 讀取當前頁面
page = st.session_state['current_page']
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
    st.write("點擊表格中的任意一行，系統將自動為您跳轉至該股的「詳細軌跡透視」頁面。")
    
    # 抓取 Top 10 與對應的軌跡資料
    top_picks = df_kelly.head(10).copy()
    display_df = top_picks[['Ticker', 'Volatility_Risk', 'Sharpe_Score', 'Suggested_Weight']].copy()
    
    # 加上中文名稱
    display_df.insert(0, '股票名稱', display_df['Ticker'].apply(get_stock_name))
    
    # 從 df_traj 裡面挖出 5~30 天的資料
    for day in [5, 10, 15, 20, 25, 30]:
        # 把 df_traj 裡面對應天數的數值 map 過來
        day_values = df_traj.set_index('Ticker')[f'Day_{day}'].reindex(display_df['Ticker']).values
        display_df[f'預期報酬 ({day}天)'] = (day_values * 100)
        
    # 格式化數字 (保留浮點數格式給 pandas，方便 Streamlit 渲染顏色)
    display_df['Volatility_Risk'] = display_df['Volatility_Risk'] * 100
    display_df['Suggested_Weight'] = display_df['Suggested_Weight'] * 100
    
    # 重新命名欄位
    display_df = display_df.rename(columns={
        'Ticker': '代號', 'Volatility_Risk': '波動風險(%)', 
        'Sharpe_Score': '夏普CP值', 'Suggested_Weight': '資金佔比(%)'
    })

    # 使用 dataframe 的 on_select 事件來捕捉使用者的點擊 (Streamlit 1.35+ 支援)
    event = st.dataframe(
        display_df.style.background_gradient(cmap='RdYlGn', subset=[f'預期報酬 ({d}天)' for d in [5,10,15,20,25,30]]),
        use_container_width=True,
        hide_index=True,
        selection_mode="single-row",
        on_select="rerun" 
    )
    
    # 如果使用者點擊了某一行，執行跳轉動作
    if len(event.selection.rows) > 0:
        selected_idx = event.selection.rows[0]
        selected_ticker = display_df.iloc[selected_idx]['代號']
        
        # 寫入 Session State 並強制跳轉頁面
        st.session_state['target_ticker'] = selected_ticker
        st.session_state['current_page'] = "📈 個股軌跡透視"
        st.rerun()
# ------------------------------------------
# 頁面 2: 個股軌跡透視 (資訊卡片 + Plotly 雲圖融合版)
# ------------------------------------------
elif page == "📈 個股軌跡透視":
    st.subheader("🔭 平行宇宙軌跡觀測儀")
    
    all_tickers = df_traj['Ticker'].astype(str).tolist()
    
    # 如果從表格跳轉過來，帶入目標代號；否則預設第一檔
    if st.session_state.get('target_ticker') in all_tickers:
        default_idx = all_tickers.index(st.session_state['target_ticker'])
    else:
        default_idx = 0
        
    target_ticker = st.selectbox("🔍 請選擇要觀測的股票代號", all_tickers, index=default_idx)
    st.session_state['target_ticker'] = target_ticker # 更新狀態
    
    stock_name = get_stock_name(target_ticker)
    
    # ⚡ 透過 yfinance 抓取「個股基礎資訊」與「歷史股價」
    @st.cache_data(ttl=3600)
    def fetch_stock_info(ticker):
        for suffix in ['.TW', '.TWO']:
            try:
                stock = yf.Ticker(f"{ticker}{suffix}")
                hist = stock.history(period="1mo")
                info = stock.info
                if not hist.empty:
                    return hist, info, suffix
            except:
                continue
        return None, {}, None

    with st.spinner(f"正在同步 {stock_name} 的個股資訊與建構機率雲..."):
        hist_df, stock_info, suffix = fetch_stock_info(target_ticker)
        
    # ==========================================
    # 確保有抓到資料，才開始畫圖與計算 (解決 NameError 的關鍵)
    # ==========================================
    if hist_df is None:
        st.warning(f"⚠️ 無法從 Yahoo Finance 抓取 {target_ticker} 的歷史股價，請稍後再試。")
    else:
        # 1. 顯示個股資訊卡片
        st.markdown(f"### {stock_name} ({target_ticker}) 個股速覽")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("今日收盤價", f"{hist_df['Close'].iloc[-1]:.2f}")
        col2.metric("本益比 (P/E)", stock_info.get('trailingPE', 'N/A'))
        col3.metric("市值 (億)", f"{stock_info.get('marketCap', 0) / 100000000:.2f}" if stock_info.get('marketCap') else 'N/A')
        col4.metric("產業別", stock_info.get('industry', 'N/A'))
        st.divider()

        # 2. 準備大腦預測軌跡資料
        stock_traj = df_traj[df_traj['Ticker'].astype(str) == target_ticker].iloc[0]
        volatility = df_kelly[df_kelly['Ticker'].astype(str) == target_ticker]['Volatility_Risk'].iloc[0]
        traj_values = stock_traj[[f'Day_{i}' for i in range(1, 31)]].values 
        
        current_price = hist_df['Close'].iloc[-1]
        last_date = hist_df.index[-1]
        
        # 3. 計算未來的真實日期 (這裡就是剛剛消失的 future_dates！)
        future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=30)
        future_mean_prices = current_price * (1 + traj_values)
        
        # 4. 計算機率雲漏斗的邊界
        time_scale = np.sqrt(np.arange(1, 31) / 30.0) 
        upper_bound_95 = future_mean_prices * (1 + (volatility * 1.5 * time_scale))
        lower_bound_95 = future_mean_prices * (1 - (volatility * 1.5 * time_scale))
        upper_bound_68 = future_mean_prices * (1 + (volatility * 0.8 * time_scale))
        lower_bound_68 = future_mean_prices * (1 - (volatility * 0.8 * time_scale))
        
        # ==========================================
        # 📈 使用 Plotly 繪製專業互動圖表
        # ==========================================
        import plotly.graph_objects as go
        fig = go.Figure()

        # A. 歷史真實股價
        fig.add_trace(go.Scatter(
            x=hist_df.index, y=hist_df['Close'],
            mode='lines', name='歷史真實股價', line=dict(color='#00d2ff', width=3)
        ))

        # B. 95% 機率雲
        fig.add_trace(go.Scatter(
            x=list(future_dates) + list(future_dates)[::-1],
            y=list(upper_bound_95) + list(lower_bound_95)[::-1],
            fill='toself', fillcolor='rgba(255, 127, 14, 0.1)',
            line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", name='95% 機率雲'
        ))

        # C. 68% 機率雲
        fig.add_trace(go.Scatter(
            x=list(future_dates) + list(future_dates)[::-1],
            y=list(upper_bound_68) + list(lower_bound_68)[::-1],
            fill='toself', fillcolor='rgba(255, 127, 14, 0.25)',
            line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", name='68% 機率雲'
        ))

        # D. 預測平均軌跡
        fig.add_trace(go.Scatter(
            x=future_dates, y=future_mean_prices,
            mode='lines+markers', name='預測平均軌跡',
            line=dict(color='#ff7f0e', width=3, dash='dot'), marker=dict(size=5)
        ))
        
        # E. 連接點
        fig.add_trace(go.Scatter(
            x=[last_date, future_dates[0]], y=[current_price, future_mean_prices[0]],
            mode='lines', line=dict(color='#ff7f0e', width=3, dash='dot'), showlegend=False
        ))

        fig.update_layout(
            title=f"<b>{stock_name} ({target_ticker}{suffix}) 股價預測與機率分布</b>",
            yaxis_title="股價 (TWD)", xaxis_title="日期",
            template="plotly_dark", hovermode="x unified",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            margin=dict(l=40, r=40, t=60, b=40)
        )
        fig.add_hline(y=current_price, line_dash="dash", line_color="gray", annotation_text="今日收盤價")

        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------
# 頁面 3: 持股監視與實盤
# ------------------------------------------
elif page == "💼 我的持股健檢":
    st.subheader("💼 我的專屬量化基金 (Portfolio)")
    st.write("輸入你目前持有的股票，MarketMamba 將每天自動為你進行 AI 健檢與退場評估。")
    
    # 1. 初始化虛擬帳本 (Session State)
    if 'portfolio' not in st.session_state:
        st.session_state['portfolio'] = pd.DataFrame(columns=['股票代號', '持有成本', '持有股數'])

    # 定義一個輕量級的抓價函數 (加上快取避免頻繁發送 API 請求)
    @st.cache_data(ttl=3600)
    def get_latest_close(ticker):
        for suffix in ['.TW', '.TWO']:
            try:
                hist = yf.Ticker(f"{ticker}{suffix}").history(period="1d")
                if not hist.empty:
                    return float(hist['Close'].iloc[-1])
            except:
                continue
        return 100.0  # 如果真的抓不到，就預設 100 元

    # 2. 新增持股的表單介面
    with st.expander("➕ 新增庫存持股", expanded=True):
        col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
        
        with col1:
            all_tickers_clean = df_kelly['Ticker'].astype(str).tolist()
            new_ticker = st.selectbox("股票代號", all_tickers_clean)
            
        # ⚡ 核心魔法：根據選擇的代號，動態抓取最新股價
        with st.spinner("獲取最新股價中..."):
            default_price = get_latest_close(new_ticker)
            
        with col2:
            # 將抓到的最新股價 default_price 直接塞進 value 裡面
            new_cost = st.number_input("持有成本 (TWD)", min_value=0.0, value=default_price, step=1.0)
            
        with col3:
            new_shares = st.number_input("持有股數 (股)", min_value=1, value=1000, step=100)
            
        with col4:
            st.markdown("<br>", unsafe_allow_html=True) # 排版對齊用
            if st.button("新增", type="primary"):
                new_row = pd.DataFrame({'股票代號': [new_ticker], '持有成本': [new_cost], '持有股數': [new_shares]})
                st.session_state['portfolio'] = pd.concat([st.session_state['portfolio'], new_row], ignore_index=True)
                st.success(f"已新增 {new_ticker}！")
                st.rerun() # 重新整理網頁以顯示最新表格

    # 3. AI 持股健檢與即時監控
    if not st.session_state['portfolio'].empty:
        st.divider()
        st.subheader("🏥 AI 庫存持股健檢報告")
        
        my_portfolio = st.session_state['portfolio'].copy()
        kelly_subset = df_kelly[['Ticker', 'Exp_Return_15D', 'Sharpe_Score']].copy()
        kelly_subset['Ticker'] = kelly_subset['Ticker'].astype(str)
        
        analysis_df = pd.merge(my_portfolio, kelly_subset, left_on='股票代號', right_on='Ticker', how='left')
        
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
        
        analysis_df['預期 15 天報酬'] = (analysis_df['Exp_Return_15D'] * 100).apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
        analysis_df['夏普分數'] = analysis_df['Sharpe_Score'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
        
        display_cols = ['股票代號', '持有成本', '持有股數', '預期 15 天報酬', '夏普分數', 'AI 操作建議']
        
        st.dataframe(
            analysis_df[display_cols].style.applymap(
                lambda x: 'color: #ff4b4b; font-weight: bold;' if '🔴' in str(x) else 
                          ('color: #00fa9a; font-weight: bold;' if '🟢' in str(x) else ''),
                subset=['AI 操作建議']
            ).format({"持有成本": "{:.2f}"}),
            use_container_width=True,
            hide_index=True
        )
        
        if st.button("🗑️ 清空庫存"):
            st.session_state['portfolio'] = pd.DataFrame(columns=['股票代號', '持有成本', '持有股數'])
            st.rerun()
    else:
        st.info("👆 目前庫存為空，請從上方新增持股來啟動 MarketMamba 的 AI 健檢功能。")

# ------------------------------------------
# 頁面 4: 百萬實盤機器人 (Paper Trading)
# ------------------------------------------
elif page == "🤖 百萬實盤機器人":
    import json
    from datetime import datetime
    
    st.subheader("🤖 MarketMamba 全自動百萬實盤基金")
    st.write("初始資金 1,000,000 TWD。機器人將嚴格依照 V3.1 凱利資金盤的建議，每天自動進行部位的建倉與再平衡 (Rebalancing)。")
    
    LEDGER_FILE = "robot_ledger.json"
    
    # 1. 初始化或讀取機器人帳本
    def load_ledger():
        if os.path.exists(LEDGER_FILE):
            with open(LEDGER_FILE, 'r') as f:
                return json.load(f)
        else:
            # 宇宙大爆炸的第一天：給予 100 萬現金
            return {
                "start_date": datetime.now().strftime("%Y-%m-%d"),
                "cash": 1000000.0,
                "holdings": {}, # 格式: {"2330.TW": {"shares": 1000, "cost": 600}}
                "history": []   # 記錄每天的總淨值
            }
            
    def save_ledger(ledger):
        with open(LEDGER_FILE, 'w') as f:
            json.dump(ledger, f, indent=4)
            
    ledger = load_ledger()
    
    # 2. 計算當前總淨值 (現金 + 股票市值)
    @st.cache_data(ttl=3600)
    def get_current_prices(tickers):
        prices = {}
        for ticker in tickers:
            try:
                suffix = ".TW" if not ticker.endswith(".TW") and not ticker.endswith(".TWO") else ""
                # 簡單抓取最新收盤價
                hist = yf.Ticker(f"{ticker}{suffix}").history(period="1d")
                if not hist.empty:
                    prices[ticker] = float(hist['Close'].iloc[-1])
            except:
                pass
        return prices

    current_holdings = list(ledger["holdings"].keys())
    live_prices = get_current_prices(current_holdings) if current_holdings else {}
    
    stock_value = 0.0
    for t, data in ledger["holdings"].items():
        price = live_prices.get(t, data["cost"]) # 如果抓不到即時價，就用成本價估算
        stock_value += price * data["shares"]
        
    total_equity = ledger["cash"] + stock_value
    
    # 3. 儀表板展示
    st.divider()
    col1, col2, col3 = st.columns(3)
    col1.metric("💰 基金總淨值 (NAV)", f"${total_equity:,.0f}", f"{(total_equity - 1000000)/10000:,.2f}%")
    col2.metric("💵 可用現金", f"${ledger['cash']:,.0f}")
    col3.metric("📈 股票市值", f"${stock_value:,.0f}")
    
    # 4. 機器人操盤控制台
    st.subheader("⚙️ 機器人操作台")
    if st.button("🚀 執行今日建倉 / 調倉 (Rebalance)", type="primary"):
        with st.spinner("機器人正在比對最新凱利名單，並送出虛擬委託單..."):
            # 抓取 Top 10 目標配置
            top_10 = df_kelly.head(10)
            target_tickers = top_10['Ticker'].astype(str).tolist()
            target_weights = top_10['Suggested_Weight'].values
            
            # 抓取這些目標股票的最新股價
            target_prices = get_current_prices(target_tickers)
            
            # 步驟 A: 賣出不在 Top 10 裡面的股票 (獲利了結/停損)
            for t in list(ledger["holdings"].keys()):
                if t not in target_tickers:
                    shares_to_sell = ledger["holdings"][t]["shares"]
                    sell_price = live_prices.get(t, ledger["holdings"][t]["cost"])
                    ledger["cash"] += shares_to_sell * sell_price
                    del ledger["holdings"][t]
                    st.toast(f"🔴 賣出剔除名單: {t}")
            
            # 重新計算賣出後的總資金 (準備重新分配)
            current_total_funds = ledger["cash"]
            for t, data in ledger["holdings"].items():
                 current_total_funds += data["shares"] * live_prices.get(t, data["cost"])
            
            # 步驟 B: 根據凱利比例買進或加碼
            for ticker, weight in zip(target_tickers, target_weights):
                if ticker not in target_prices:
                    continue # 抓不到報價跳過
                    
                target_value = current_total_funds * weight
                current_price = target_prices[ticker]
                target_shares = int(target_value // current_price) # 計算應該持有的總股數
                
                # 目前持有股數
                current_shares = ledger["holdings"].get(ticker, {}).get("shares", 0)
                
                # 需要加碼買進的股數
                shares_to_buy = target_shares - current_shares
                
                if shares_to_buy > 0 and ledger["cash"] >= (shares_to_buy * current_price):
                    ledger["cash"] -= (shares_to_buy * current_price)
                    
                    if ticker in ledger["holdings"]:
                        # 平均成本法計算
                        old_cost = ledger["holdings"][ticker]["cost"]
                        old_shares = ledger["holdings"][ticker]["shares"]
                        new_avg_cost = ((old_cost * old_shares) + (current_price * shares_to_buy)) / target_shares
                        ledger["holdings"][ticker] = {"shares": target_shares, "cost": new_avg_cost}
                    else:
                        ledger["holdings"][ticker] = {"shares": shares_to_buy, "cost": current_price}
                    st.toast(f"🟢 買進建倉: {ticker} ({shares_to_buy} 股)")
            
            # 記錄今天的淨值到歷史中
            today_str = datetime.now().strftime("%Y-%m-%d %H:%M")
            ledger["history"].append({"date": today_str, "equity": current_total_funds})
            save_ledger(ledger)
            
            st.success("✅ 今日調倉完畢！機器人已為明天的開盤做好準備。")
            st.rerun()

    # 5. 顯示目前機器人的持股清單與淨值曲線
    if ledger["holdings"]:
        st.write("📋 **機器人目前庫存**")
        holding_list = []
        for t, d in ledger["holdings"].items():
            current_p = live_prices.get(t, d['cost'])
            profit_pct = (current_p - d['cost']) / d['cost'] * 100
            holding_list.append({
                "股票代號": t,
                "持有股數": d["shares"],
                "平均成本": d["cost"],
                "最新報價": current_p,
                "未實現損益": f"{profit_pct:.2f}%"
            })
            
        st.dataframe(pd.DataFrame(holding_list).style.applymap(
            lambda x: 'color: red;' if '-' in str(x) else 'color: green;', subset=['未實現損益']
        ), use_container_width=True)
    else:
        st.info("🤖 機器人目前空手，請點擊上方按鈕執行建倉！")
        
    # 如果有歷史紀錄，畫出淨值折線圖
    if len(ledger["history"]) > 0:
        hist_df = pd.DataFrame(ledger["history"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist_df['date'], y=hist_df['equity'], mode='lines+markers', line=dict(color='#00fa9a', width=3)))
        fig.update_layout(title="📈 基金淨值成長曲線", template="plotly_dark", yaxis_title="總淨值 (TWD)")
        st.plotly_chart(fig, use_container_width=True)






