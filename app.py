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
import requests

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
# 1. 載入核心預測數據 (加入 TTL 自動刷新機制)
# ==========================================
# ttl=3600 代表快取只存活 1 小時。1 小時後只要有人打開網頁，
# Streamlit 就會強制去硬碟裡抓最新的 CSV，完美對接 Colab 的每日更新！
@st.cache_data(ttl=3600)
def load_prediction_data():
    try:
        # 因為 CSV 檔已經被 Colab 推送到同一個 GitHub 倉庫的根目錄
        # Streamlit 雲端可以直接用相對路徑讀取它們！
        df_k = pd.read_csv("df_kelly.csv")
        df_t = pd.read_csv("df_traj.csv")
        return df_k, df_t, True
    except Exception as e:
        st.error(f"⚠️ 尚未讀取到預測資料，請確認 Colab 是否已成功推送 CSV 檔案。錯誤: {e}")
        return pd.DataFrame(), pd.DataFrame(), False

df_kelly, df_traj, data_loaded = load_prediction_data()

# ==========================================
# 2. 狀態管理 (Session State) 與 側邊欄設計
# ==========================================
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = "📊 今日凱利資金盤"
if 'target_ticker' not in st.session_state:
    st.session_state['target_ticker'] = None

# ⚡ 動態讀取外部字典檔 (使用最強的 split 切割法)
@st.cache_data(ttl=3600)
def load_ticker_mapping():
    try:
        url = "https://raw.githubusercontent.com/FrankChen0930/MarketMamba/main/ticker_mapping.json"
        res = requests.get(url)
        res.raise_for_status() 
        raw_dict = res.json()
        
        # 魔法在這裡：用 split('.')[0] 切割。
        # "6846.TWO" -> "6846", "2330.TW" -> "2330", "6846.0" -> "6846"
        clean_dict = {}
        for k, v in raw_dict.items():
            clean_key = str(k).split('.')[0].strip()
            clean_dict[clean_key] = v
            
        return clean_dict
    except Exception as e:
        st.warning(f"⚠️ 找不到 ticker_mapping.json。錯誤: {e}")
        return {}

TW_STOCK_DICT = load_ticker_mapping()

def get_stock_name(ticker):
    # 查詢時，同樣用 split 切割法洗乾淨
    clean_ticker = str(ticker).split('.')[0].strip()
    return TW_STOCK_DICT.get(clean_ticker, str(ticker))

def format_ticker(ticker):
    # 將代號與中文名稱完美組合，例如 "2330.TW 台積電"
    name = get_stock_name(ticker)
    return f"{ticker} {name}" if name else str(ticker)

# 抓取更新時間
@st.cache_data(ttl=3600)
def get_update_time():
    try:
        url = "https://raw.githubusercontent.com/FrankChen0930/MarketMamba/main/update_time.txt"
        res = requests.get(url)
        return res.text.strip()
    except:
        return "未知"

last_update = get_update_time()

# --- 側邊欄 UI ---
with st.sidebar:
    st.header("📌 功能選單")
    if data_loaded:
        st.success("✅ V3.1 雲端數據庫已連線")
        st.caption(f"🕒 最新預測時間: {last_update}")
        
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
    st.stop() 

# ------------------------------------------
# 頁面 1: 凱利資金盤 (Top Picks)
# ------------------------------------------
if page == "📊 今日凱利資金盤":
    st.subheader("🎯 今日最佳防禦型飆股 (Top 10)")
    st.write("點擊表格中的任意一行，系統將自動為您跳轉至該股的「詳細軌跡透視」頁面。")
    
    top_picks = df_kelly.head(10).copy()
    display_df = top_picks[['Ticker', 'Volatility_Risk', 'Sharpe_Score', 'Suggested_Weight']].copy()
    
    for day in [5, 10, 15, 20, 25, 30]:
        day_values = df_traj.set_index('Ticker')[f'Day_{day}'].reindex(display_df['Ticker']).values
        display_df[f'預期報酬 ({day}天)'] = (day_values * 100)
        
    display_df['Volatility_Risk'] = display_df['Volatility_Risk'] * 100
    display_df['Suggested_Weight'] = display_df['Suggested_Weight'] * 100
    
    display_df = display_df.rename(columns={
        'Ticker': '股票標的', 'Volatility_Risk': '波動風險(%)', 
        'Sharpe_Score': '夏普CP值', 'Suggested_Weight': '資金佔比(%)'
    })
    
    # 【關鍵升級】：將表格內的代號換成「代號+名稱」
    display_df['股票標的'] = display_df['股票標的'].apply(format_ticker)
    
    # 🎨 專業 UI 升級：定義台股專屬紅綠文字上色邏輯
    def color_tw_returns(val):
        if isinstance(val, (int, float)):
            if val > 0:
                return 'color: #ff4b4b; font-weight: bold;' # 漲：亮紅色
            elif val < 0:
                return 'color: #00fa9a; font-weight: bold;' # 跌：亮綠色
        return ''

    # 設定要格式化的欄位
    return_cols = [f'預期報酬 ({d}天)' for d in [5,10,15,20,25,30]]
    
    # 套用顏色，並把數字格式化為帶有 % 的字串
    styled_df = display_df.style.applymap(color_tw_returns, subset=return_cols) \
                                .format({col: "{:.2f}%" for col in return_cols}) \
                                .format({"波動風險(%)": "{:.2f}%", "資金佔比(%)": "{:.2f}%", "夏普CP值": "{:.4f}"})

    event = st.dataframe(
        styled_df, # 改用我們洗盡鉛華的極簡風格
        use_container_width=True, hide_index=True, selection_mode="single-row", on_select="rerun" 
    )
    
    if len(event.selection.rows) > 0:
        selected_idx = event.selection.rows[0]
        # 【關鍵升級】：因為現在欄位變成 "2330.TW 台積電"，我們只要空白前面的 "2330.TW"
        selected_ticker = str(display_df.iloc[selected_idx]['股票標的']).split(' ')[0]
        st.session_state['target_ticker'] = selected_ticker
        st.session_state['current_page'] = "📈 個股軌跡透視"
        st.rerun()

# ------------------------------------------
# 頁面 2: 個股軌跡透視
# ------------------------------------------
elif page == "📈 個股軌跡透視":
    st.subheader("🔭 平行宇宙軌跡觀測儀")
    
    all_tickers = df_traj['Ticker'].astype(str).tolist()
    
    if st.session_state.get('target_ticker') in all_tickers:
        default_idx = all_tickers.index(st.session_state['target_ticker'])
    else:
        default_idx = 0
        
    # 【關鍵升級】：使用 format_func 讓下拉選單顯示中文，但背後仍傳遞純代碼
    target_ticker = st.selectbox("🔍 請選擇要觀測的股票", all_tickers, index=default_idx, format_func=format_ticker)
    st.session_state['target_ticker'] = target_ticker 
    
    stock_name = get_stock_name(target_ticker)
    
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
        
    if hist_df is None:
        st.warning(f"⚠️ 無法從 Yahoo Finance 抓取 {target_ticker} 的歷史股價，請稍後再試。")
    else:
        st.markdown(f"### {stock_name} ({target_ticker}) 個股速覽")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("今日收盤價", f"{hist_df['Close'].iloc[-1]:.2f}")
        col2.metric("本益比 (P/E)", stock_info.get('trailingPE', 'N/A'))
        col3.metric("市值 (億)", f"{stock_info.get('marketCap', 0) / 100000000:.2f}" if stock_info.get('marketCap') else 'N/A')
        col4.metric("產業別", stock_info.get('industry', 'N/A'))
        
        # 準備大腦預測軌跡資料
        stock_traj = df_traj[df_traj['Ticker'].astype(str) == target_ticker].iloc[0]
        volatility = df_kelly[df_kelly['Ticker'].astype(str) == target_ticker]['Volatility_Risk'].iloc[0]
        traj_values = stock_traj[[f'Day_{i}' for i in range(1, 31)]].values 
        
        # 📅 專業 UI 升級：新增多天期預期報酬看板
        st.markdown("#### 📅 擴散模型預期報酬 (多天期解析)")
        ret_cols = st.columns(6)
        for idx, d in enumerate([5, 10, 15, 20, 25, 30]):
            val = traj_values[d-1] * 100
            # 依據正負值給予不同的箭頭與顏色
            delta_color = "normal" if val > 0 else "inverse" 
            ret_cols[idx].metric(f"{d} 天後", f"{val:.2f}%", delta_color=delta_color)

        st.divider()
        
        current_price = hist_df['Close'].iloc[-1]
        last_date = hist_df.index[-1]
        
        future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=30)
        future_mean_prices = current_price * (1 + traj_values)
        
        time_scale = np.sqrt(np.arange(1, 31) / 30.0) 
        upper_bound_95 = future_mean_prices * (1 + (volatility * 1.5 * time_scale))
        lower_bound_95 = future_mean_prices * (1 - (volatility * 1.5 * time_scale))
        upper_bound_68 = future_mean_prices * (1 + (volatility * 0.8 * time_scale))
        lower_bound_68 = future_mean_prices * (1 - (volatility * 0.8 * time_scale))
        
        import plotly.graph_objects as go
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df['Close'], mode='lines', name='歷史真實股價', line=dict(color='#00d2ff', width=3)))
        
        # 🎨 專業 UI 升級：調亮機率雲的透明度 (0.1 -> 0.25, 0.25 -> 0.55)
        fig.add_trace(go.Scatter(x=list(future_dates) + list(future_dates)[::-1], y=list(upper_bound_95) + list(lower_bound_95)[::-1], fill='toself', fillcolor='rgba(255, 127, 14, 0.25)', line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", name='95% 機率雲'))
        fig.add_trace(go.Scatter(x=list(future_dates) + list(future_dates)[::-1], y=list(upper_bound_68) + list(lower_bound_68)[::-1], fill='toself', fillcolor='rgba(255, 127, 14, 0.55)', line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", name='68% 機率雲'))
        
        fig.add_trace(go.Scatter(x=future_dates, y=future_mean_prices, mode='lines+markers', name='預測平均軌跡', line=dict(color='#ff7f0e', width=3, dash='dot'), marker=dict(size=5)))
        fig.add_trace(go.Scatter(x=[last_date, future_dates[0]], y=[current_price, future_mean_prices[0]], mode='lines', line=dict(color='#ff7f0e', width=3, dash='dot'), showlegend=False))

        fig.update_layout(
            title=f"<b>{format_ticker(target_ticker)} 股價預測與機率分布</b>",
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
    
    if 'portfolio' not in st.session_state:
        st.session_state['portfolio'] = pd.DataFrame(columns=['股票代號', '持有成本', '持有股數'])

    @st.cache_data(ttl=3600)
    def get_latest_close(ticker):
        for suffix in ['.TW', '.TWO']:
            try:
                hist = yf.Ticker(f"{ticker}{suffix}").history(period="1d")
                if not hist.empty:
                    return float(hist['Close'].iloc[-1])
            except:
                continue
        return 100.0

    with st.expander("➕ 新增庫存持股", expanded=True):
        col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
        
        with col1:
            all_tickers_clean = df_kelly['Ticker'].astype(str).tolist()
            # 【關鍵升級】：下拉選單加上中文
            new_ticker = st.selectbox("股票標的", all_tickers_clean, format_func=format_ticker)
            
        with st.spinner("獲取最新股價中..."):
            default_price = get_latest_close(new_ticker)
            
        with col2:
            new_cost = st.number_input("持有成本 (TWD)", min_value=0.0, value=default_price, step=1.0)
            
        with col3:
            new_shares = st.number_input("持有股數 (股)", min_value=1, value=1000, step=100)
            
        with col4:
            st.markdown("<br>", unsafe_allow_html=True) 
            if st.button("新增", type="primary"):
                new_row = pd.DataFrame({'股票代號': [new_ticker], '持有成本': [new_cost], '持有股數': [new_shares]})
                st.session_state['portfolio'] = pd.concat([st.session_state['portfolio'], new_row], ignore_index=True)
                st.success(f"已新增 {format_ticker(new_ticker)}！")
                st.rerun() 

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
                return "🔴 趨勢轉弱 (建議停損/了結)"
            elif row['Sharpe_Score'] > 0.5:
                return "🟢 強勢護城河 (建議續抱)"
            else:
                return "🟡 波動加劇 (嚴設停損)"
                
        analysis_df['AI 操作建議'] = analysis_df.apply(get_action_signal, axis=1)
        analysis_df['預期 15 天報酬'] = (analysis_df['Exp_Return_15D'] * 100).apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
        analysis_df['夏普分數'] = analysis_df['Sharpe_Score'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
        
        # 【關鍵升級】：將表格內的代號替換為有中文的版本
        analysis_df['股票標的'] = analysis_df['股票代號'].apply(format_ticker)
        
        display_cols = ['股票標的', '持有成本', '持有股數', '預期 15 天報酬', '夏普分數', 'AI 操作建議']
        
        st.dataframe(
            analysis_df[display_cols].style.applymap(
                lambda x: 'color: #ff4b4b; font-weight: bold;' if '🔴' in str(x) else ('color: #00fa9a; font-weight: bold;' if '🟢' in str(x) else ''),
                subset=['AI 操作建議']
            ).format({"持有成本": "{:.2f}"}),
            use_container_width=True, hide_index=True
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
    import os
    from datetime import datetime
    import plotly.graph_objects as go
    
    st.subheader("🤖 MarketMamba 全自動百萬實盤基金")
    st.write("初始資金 1,000,000 TWD。機器人將嚴格依照 V3.1 凱利資金盤的建議，每天自動進行部位的建倉與再平衡 (Rebalancing)。")
    
    LEDGER_FILE = "robot_ledger.json"
    
    def load_ledger():
        if os.path.exists(LEDGER_FILE):
            with open(LEDGER_FILE, 'r') as f:
                return json.load(f)
        else:
            return {"start_date": datetime.now().strftime("%Y-%m-%d"), "cash": 1000000.0, "holdings": {}, "history": []}
            
    def save_ledger(ledger):
        with open(LEDGER_FILE, 'w') as f:
            json.dump(ledger, f, indent=4)
            
    ledger = load_ledger()
    
    @st.cache_data(ttl=3600)
    def get_current_prices(tickers):
        prices = {}
        for ticker in tickers:
            try:
                suffix = ".TW" if not ticker.endswith(".TW") and not ticker.endswith(".TWO") else ""
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
        price = live_prices.get(t, data["cost"]) 
        stock_value += price * data["shares"]
        
    total_equity = ledger["cash"] + stock_value
    
    st.divider()
    col1, col2, col3 = st.columns(3)
    col1.metric("💰 基金總淨值 (NAV)", f"${total_equity:,.0f}", f"{(total_equity - 1000000)/10000:,.2f}%")
    col2.metric("💵 可用現金", f"${ledger['cash']:,.0f}")
    col3.metric("📈 股票市值", f"${stock_value:,.0f}")
    
    st.subheader("⚙️ 機器人操作台")
    if st.button("🚀 執行今日建倉 / 調倉 (Rebalance)", type="primary"):
        with st.spinner("機器人正在比對最新凱利名單，並送出虛擬委託單..."):
            top_10 = df_kelly.head(10)
            target_tickers = top_10['Ticker'].astype(str).tolist()
            target_weights = top_10['Suggested_Weight'].values
            target_prices = get_current_prices(target_tickers)
            
            for t in list(ledger["holdings"].keys()):
                if t not in target_tickers:
                    shares_to_sell = ledger["holdings"][t]["shares"]
                    sell_price = live_prices.get(t, ledger["holdings"][t]["cost"])
                    ledger["cash"] += shares_to_sell * sell_price
                    del ledger["holdings"][t]
                    st.toast(f"🔴 賣出剔除名單: {format_ticker(t)}")
            
            current_total_funds = ledger["cash"]
            for t, data in ledger["holdings"].items():
                 current_total_funds += data["shares"] * live_prices.get(t, data["cost"])
            
            for ticker, weight in zip(target_tickers, target_weights):
                if ticker not in target_prices: continue
                target_value = current_total_funds * weight
                current_price = target_prices[ticker]
                target_shares = int(target_value // current_price) 
                current_shares = ledger["holdings"].get(ticker, {}).get("shares", 0)
                shares_to_buy = target_shares - current_shares
                
                if shares_to_buy > 0 and ledger["cash"] >= (shares_to_buy * current_price):
                    ledger["cash"] -= (shares_to_buy * current_price)
                    if ticker in ledger["holdings"]:
                        old_cost = ledger["holdings"][ticker]["cost"]
                        old_shares = ledger["holdings"][ticker]["shares"]
                        new_avg_cost = ((old_cost * old_shares) + (current_price * shares_to_buy)) / target_shares
                        ledger["holdings"][ticker] = {"shares": target_shares, "cost": new_avg_cost}
                    else:
                        ledger["holdings"][ticker] = {"shares": shares_to_buy, "cost": current_price}
                    st.toast(f"🟢 買進建倉: {format_ticker(ticker)} ({shares_to_buy} 股)")
            
            today_str = datetime.now().strftime("%Y-%m-%d %H:%M")
            ledger["history"].append({"date": today_str, "equity": current_total_funds})
            save_ledger(ledger)
            
            st.success("✅ 今日調倉完畢！機器人已為明天的開盤做好準備。")
            st.rerun()

    if ledger["holdings"]:
        st.write("📋 **機器人目前庫存**")
        holding_list = []
        for t, d in ledger["holdings"].items():
            current_p = live_prices.get(t, d['cost'])
            profit_pct = (current_p - d['cost']) / d['cost'] * 100
            holding_list.append({
                "股票標的": format_ticker(t), # 【關鍵升級】：機器人帳本也加入中文
                "持有股數": d["shares"],
                "平均成本": d["cost"],
                "最新報價": current_p,
                "未實現損益": f"{profit_pct:.2f}%"
            })
            
        st.dataframe(pd.DataFrame(holding_list).style.applymap(
            lambda x: 'color: red;' if '-' in str(x) else 'color: green;', subset=['未實現損益']
        ), use_container_width=True, hide_index=True)
    else:
        st.info("🤖 機器人目前空手，請點擊上方按鈕執行建倉！")
        
    if len(ledger["history"]) > 0:
        hist_df = pd.DataFrame(ledger["history"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist_df['date'], y=hist_df['equity'], mode='lines+markers', line=dict(color='#00fa9a', width=3)))
        fig.update_layout(title="📈 基金淨值成長曲線", template="plotly_dark", yaxis_title="總淨值 (TWD)")
        st.plotly_chart(fig, use_container_width=True)



