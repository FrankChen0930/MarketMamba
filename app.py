import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import yfinance as yf
import plotly.graph_objects as go
import json
import requests
from datetime import datetime, timezone, timedelta

# ==========================================
# 0. 網頁基本設定 & 字型處理
# ==========================================
st.set_page_config(page_title="MarketMamba 量化決策中心", page_icon="🐍", layout="wide")

font_path = "NotoSansTC-Regular.ttf" 
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
else:
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'sans-serif'] 
plt.rcParams['axes.unicode_minus'] = False 

st.title("🐍 MarketMamba V5.0: 終極量化決策中心")
st.markdown("基於 Mamba 狀態空間與動態圖 (GAT) 架構，預測 30 天超額報酬 (Alpha) 軌跡。")
with st.expander("📖 MarketMamba V5.0 核心指標與看盤指南 (新手必看)", expanded=False):
    st.markdown("""
    ### 🎯 什麼是 Alpha (超額報酬)？
    V5.0 預測的不是「股價會漲到多少」，而是預測這檔股票**「能贏過大盤多少」**。
    * 🟢 **正 Alpha (+)**：代表模型預測該股動能強勁，表現將**超越大盤**（大盤跌它抗跌，大盤漲它漲更多）。
    * 🔴 **負 Alpha (-)**：代表模型嗅到資金撤出訊號，表現將**落後大盤**。
    *(註：Alpha 為負不代表股價一定會跌，可能只是漲幅輸給大盤。)*

    ### ⚖️ 核心評估指標
    * **波動風險 (%)**：預測未來 30 天的震盪幅度。數字越大，代表該股越活潑、風險也越高。
    * **夏普 CP 值 (Sharpe Score)**：`預期 Alpha ÷ 波動風險`。這是量化交易的聖杯！分數越高，代表你**「承擔一單位的風險，能換來越多的超額報酬」**，這也是本系統 Top 10 排名的唯一依據。
    * **資金佔比 (Kelly Weight)**：基於凱利公式 (Kelly Criterion) 計算出的建議押注比例。最高上限控管為 20%，以達到分散風險的實盤保護機制。

    ### 🕵️‍♂️ 實盤操作建議
    1. **觀察趨勢**：點擊下方表格任意行，可查看該股未來 30 天的「詳細軌跡透視」。請優先挑選**軌跡平滑向上**的標的。
    2. **動能衰退警示**：如果一檔股票前 15 天 Alpha 極高，但後 15 天開始下彎甚至轉負，代表這是一檔「短線爆發股」，機器人會在這段期間內伺機獲利了結。
    """)

# ==========================================
# 1. 載入核心預測數據
# ==========================================
@st.cache_data(ttl=3600)
def load_prediction_data():
    try:
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

@st.cache_data(ttl=3600)
def load_ticker_mapping():
    try:
        url = "https://raw.githubusercontent.com/FrankChen0930/MarketMamba/main/ticker_mapping.json"
        res = requests.get(url)
        res.raise_for_status() 
        raw_dict = res.json()
        
        clean_dict = {}
        for k, v in raw_dict.items():
            clean_key = str(k).split('.')[0].strip()
            clean_dict[clean_key] = v
        return clean_dict
    except Exception as e:
        return {}

TW_STOCK_DICT = load_ticker_mapping()

def get_stock_name(ticker):
    clean_ticker = str(ticker).split('.')[0].strip()
    return TW_STOCK_DICT.get(clean_ticker, str(ticker))

def format_ticker(ticker):
    name = get_stock_name(ticker)
    return f"{ticker} {name}" if name else str(ticker)

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
        st.success("✅ V5.0 雲端數據庫已連線")
        st.caption(f"🕒 最新預測時間: {last_update}")
        
    st.markdown("---")
    
    if st.button("📊 今日凱利資金盤", use_container_width=True, type="primary" if st.session_state['current_page'] == "📊 今日凱利資金盤" else "secondary"):
        st.session_state['current_page'] = "📊 今日凱利資金盤"
        st.rerun()
        
    if st.button("📈 個股軌跡透視", use_container_width=True, type="primary" if st.session_state['current_page'] == "📈 個股軌跡透視" else "secondary"):
        st.session_state['current_page'] = "📈 個股軌跡透視"
        st.rerun()
        
    if st.button("💼 我的持股健檢", use_container_width=True, type="primary" if st.session_state['current_page'] == "💼 我的持股健檢" else "secondary"):
        st.session_state['current_page'] = "💼 我的持股健檢"
        st.rerun()
        
    if st.button("🤖 百萬實盤機器人", use_container_width=True, type="primary" if st.session_state['current_page'] == "🤖 百萬實盤機器人" else "secondary"):
        st.session_state['current_page'] = "🤖 百萬實盤機器人"
        st.rerun()

page = st.session_state['current_page']
if not data_loaded:
    st.stop() 

# ------------------------------------------
# 頁面 1: 凱利資金盤 (Top Picks)
# ------------------------------------------
if page == "📊 今日凱利資金盤":
    st.subheader("🎯 今日最佳 Alpha 潛力股 (Top 10)")
    st.write("點擊表格中的任意一行，系統將自動為您跳轉至該股的「詳細軌跡透視」頁面。")
    
    top_picks = df_kelly.head(10).copy()
    display_df = top_picks[['Ticker', 'Volatility_Risk', 'Sharpe_Score', 'Suggested_Weight']].copy()
    
    for day in [5, 10, 15, 20, 25, 30]:
        day_values = df_traj.set_index('Ticker')[f'Day_{day}'].reindex(display_df['Ticker']).values
        display_df[f'預期 Alpha ({day}天)'] = (day_values * 100) # 轉換為百分比 Alpha
        
    display_df['Volatility_Risk'] = display_df['Volatility_Risk'] * 100
    display_df['Suggested_Weight'] = display_df['Suggested_Weight'] * 100
    
    display_df = display_df.rename(columns={
        'Ticker': '股票標的', 'Volatility_Risk': '波動風險(%)', 
        'Sharpe_Score': '夏普CP值', 'Suggested_Weight': '資金佔比(%)'
    })
    
    display_df['股票標的'] = display_df['股票標的'].apply(format_ticker)
    
    def color_tw_returns(val):
        if isinstance(val, (int, float)):
            if val > 0:
                return 'color: #ff4b4b; font-weight: bold;' 
            elif val < 0:
                return 'color: #00fa9a; font-weight: bold;' 
        return ''

    return_cols = [f'預期 Alpha ({d}天)' for d in [5,10,15,20,25,30]]
    
    styled_df = display_df.style.applymap(color_tw_returns, subset=return_cols) \
                                .format({col: "{:.2f}%" for col in return_cols}) \
                                .format({"波動風險(%)": "{:.2f}%", "資金佔比(%)": "{:.2f}%", "夏普CP值": "{:.4f}"})

    event = st.dataframe(
        styled_df, 
        use_container_width=True, hide_index=True, selection_mode="single-row", on_select="rerun" 
    )
    
    if len(event.selection.rows) > 0:
        selected_idx = event.selection.rows[0]
        selected_ticker = str(display_df.iloc[selected_idx]['股票標的']).split(' ')[0]
        st.session_state['target_ticker'] = selected_ticker
        st.session_state['current_page'] = "📈 個股軌跡透視"
        st.rerun()

# ------------------------------------------
# 頁面 2: 個股軌跡透視 (V5.0 Alpha 軌跡版)
# ------------------------------------------
elif page == "📈 個股軌跡透視":
    st.subheader("🔭 平行宇宙軌跡觀測儀")
    
    all_tickers = df_traj['Ticker'].astype(str).tolist()
    
    if st.session_state.get('target_ticker') in all_tickers:
        default_idx = all_tickers.index(st.session_state['target_ticker'])
    else:
        default_idx = 0
        
    target_ticker = st.selectbox("🔍 請選擇要觀測的股票", all_tickers, index=default_idx, format_func=format_ticker)
    st.session_state['target_ticker'] = target_ticker 
    
    stock_name = get_stock_name(target_ticker)
    
    @st.cache_data(ttl=3600)
    def fetch_stock_info(ticker):
        clean_ticker = str(ticker).split('.')[0].strip()
        for suffix in ['.TW', '.TWO']:
            try:
                stock = yf.Ticker(f"{clean_ticker}{suffix}")
                hist = stock.history(period="1mo")
                
                if not hist.empty:
                    info_data = {}
                    try:
                        info_data = stock.info
                    except:
                        pass
                        
                    return hist, info_data, suffix
            except:
                continue
        return None, {}, None

    with st.spinner(f"正在同步 {stock_name} 的個股資訊與 Mamba 運算..."):
        hist_df, stock_info, suffix = fetch_stock_info(target_ticker)
        
    if hist_df is None:
        st.warning(f"⚠️ 無法從 Yahoo Finance 抓取 {target_ticker} 的歷史股價。")
    else:
        st.markdown(f"### {stock_name} ({target_ticker}) 個股速覽")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("今日收盤價", f"{hist_df['Close'].iloc[-1]:.2f}")
        col2.metric("本益比 (P/E)", stock_info.get('trailingPE', 'N/A'))
        col3.metric("市值 (億)", f"{stock_info.get('marketCap', 0) / 100000000:.2f}" if stock_info.get('marketCap') else 'N/A')
        col4.metric("產業別", stock_info.get('industry', 'N/A'))
        
        stock_traj = df_traj[df_traj['Ticker'].astype(str) == target_ticker].iloc[0]
        volatility = df_kelly[df_kelly['Ticker'].astype(str) == target_ticker]['Volatility_Risk'].iloc[0]
        traj_values = stock_traj[[f'Day_{i}' for i in range(1, 31)]].values 
        
        # --- V5.0 Alpha 軌跡數據 ---
        st.markdown("#### 📅 V5.0 Mamba 預期超額報酬 (Alpha) 軌跡")
        ret_cols = st.columns(6)
        for idx, d in enumerate([5, 10, 15, 20, 25, 30]):
            val = traj_values[d-1] * 100
            delta_color = "normal" if val > 0 else "inverse" 
            ret_cols[idx].metric(f"{d} 天後", f"{val:.2f}%", delta_color=delta_color)

        st.divider()
        
        current_price = hist_df['Close'].iloc[-1]
        last_date = hist_df.index[-1]
        
        # 產生未來 30 個交易日的日期標籤
        future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=30)
        
        # 將預測陣列轉為百分比
        alpha_trajectory_pct = traj_values * 100
        
        # --- 繪製 V5.0 專屬 Alpha 軌跡圖 ---
        fig = go.Figure()
        
        # 1. 畫出大盤基準線 (0% 零軸)
        fig.add_trace(go.Scatter(
            x=[future_dates[0], future_dates[-1]], 
            y=[0, 0], 
            mode='lines', 
            name='TWII 大盤基準線 (0%)', 
            line=dict(color='white', width=2, dash='dash')
        ))
        
        # 2. 畫出 Mamba 預測的 Alpha 軌跡
        line_color = '#ff4b4b' if alpha_trajectory_pct[-1] > 0 else '#00fa9a' # 台股習慣：紅漲綠跌
        fig.add_trace(go.Scatter(
            x=future_dates, 
            y=alpha_trajectory_pct, 
            mode='lines+markers', 
            name=f'{target_ticker} 預測 Alpha', 
            line=dict(color=line_color, width=4), 
            marker=dict(size=6)
        ))
        
        # 3. 畫出機率雲 (根據 Volatility 計算上下界)
        upper_bound = alpha_trajectory_pct + (volatility * 100 * np.sqrt(np.arange(1, 31)/30.0))
        lower_bound = alpha_trajectory_pct - (volatility * 100 * np.sqrt(np.arange(1, 31)/30.0))
        
        fig.add_trace(go.Scatter(
            x=list(future_dates) + list(future_dates)[::-1], 
            y=list(upper_bound) + list(lower_bound)[::-1], 
            fill='toself', 
            fillcolor='rgba(255, 255, 255, 0.1)', 
            line=dict(color='rgba(255,255,255,0)'), 
            hoverinfo="skip", 
            name='模型信心區間'
        ))

        # 裝飾圖表
        fig.update_layout(
            title=f"<b>{format_ticker(target_ticker)} 未來 30 天超額報酬 (Cumulative Alpha) 預測</b>",
            yaxis_title="累積 Alpha (%)", 
            xaxis_title="未來交易日",
            template="plotly_dark", 
            hovermode="x unified",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            margin=dict(l=40, r=40, t=60, b=40)
        )

        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------
# 頁面 3: 持股監視與實盤 (記憶升級版)
# ------------------------------------------
elif page == "💼 我的持股健檢":
    st.subheader("💼 我的專屬量化基金 (Portfolio)")
    st.write("輸入持股後，請將最新的「網頁網址」加入瀏覽器我的最愛，未來點擊書籤即可自動還原持股！")
    
    import json

    if 'portfolio' not in st.session_state:
        if "saved_portfolio" in st.query_params:
            try:
                port_data = json.loads(st.query_params["saved_portfolio"])
                st.session_state['portfolio'] = pd.DataFrame(port_data)
            except:
                st.session_state['portfolio'] = pd.DataFrame(columns=['股票代號', '持有成本', '持有股數'])
        else:
            st.session_state['portfolio'] = pd.DataFrame(columns=['股票代號', '持有成本', '持有股數'])

    def save_portfolio_to_url():
        port_json = st.session_state['portfolio'].to_json(orient='records')
        st.query_params["saved_portfolio"] = port_json

    @st.cache_data(ttl=3600)
    def get_latest_close(ticker):
        clean_ticker = str(ticker).split('.')[0].strip()
        for suffix in ['.TW', '.TWO']:
            try:
                hist = yf.Ticker(f"{clean_ticker}{suffix}").history(period="1d")
                if not hist.empty:
                    return float(hist['Close'].iloc[-1])
            except:
                continue
        return 100.0

    with st.expander("➕ 新增庫存持股", expanded=True):
        col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
        
        with col1:
            all_tickers_clean = df_kelly['Ticker'].astype(str).tolist()
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
                save_portfolio_to_url()
                st.success(f"已新增！請記得更新您的瀏覽器書籤 ⭐️")
                st.rerun() 

    if not st.session_state['portfolio'].empty:
        st.divider()
        st.subheader("🏥 AI 庫存持股健檢報告")
        
        my_portfolio = st.session_state['portfolio'].copy()
        kelly_subset = df_kelly[['Ticker', 'Exp_Return_15D', 'Sharpe_Score']].copy()
        kelly_subset['Ticker'] = kelly_subset['Ticker'].astype(str)
        analysis_df = pd.merge(my_portfolio, kelly_subset, left_on='股票代號', right_on='Ticker', how='left')
        
        def get_action_signal(row):
            if pd.isna(row['Exp_Return_15D']): return "⚪ 缺乏預測資料"
            elif row['Exp_Return_15D'] < 0: return "🔴 趨勢落後大盤 (建議停損)"
            elif row['Sharpe_Score'] > 0.5: return "🟢 超額動能強勁 (建議續抱)"
            else: return "🟡 波動加劇 (嚴設停損)"
                
        analysis_df['AI 操作建議'] = analysis_df.apply(get_action_signal, axis=1)
        analysis_df['預期 15 天 Alpha'] = (analysis_df['Exp_Return_15D'] * 100).apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
        analysis_df['夏普分數'] = analysis_df['Sharpe_Score'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
        analysis_df['股票標的'] = analysis_df['股票代號'].apply(format_ticker)
        display_cols = ['股票標的', '持有成本', '持有股數', '預期 15 天 Alpha', '夏普分數', 'AI 操作建議']
        
        st.dataframe(
            analysis_df[display_cols].style.applymap(
                lambda x: 'color: #00fa9a; font-weight: bold;' if '🔴' in str(x) else ('color: #ff4b4b; font-weight: bold;' if '🟢' in str(x) else ''),
                subset=['AI 操作建議']
            ).format({"持有成本": "{:.2f}"}),
            use_container_width=True, hide_index=True
        )
        
        if st.button("🗑️ 清空庫存"):
            st.session_state['portfolio'] = pd.DataFrame(columns=['股票代號', '持有成本', '持有股數'])
            save_portfolio_to_url()
            st.rerun()
    else:
        st.info("👆 目前庫存為空，請從上方新增持股來啟動 MarketMamba 的 AI 健檢功能。")

# ------------------------------------------
# 頁面 4: 百萬實盤機器人 (全自動 Read-Only 版)
# ------------------------------------------
elif page == "🤖 百萬實盤機器人":
    st.subheader("🤖 MarketMamba V5.0 全自動百萬實盤基金")
    st.write("初始資金 1,000,000 TWD。機器人已部署於雲端伺服器，每日收盤後將自動依照 V5.0 上帝視角進行調倉與結算。")
    
    @st.cache_data(ttl=600) 
    def load_cloud_ledger():
        try:
            url = "https://raw.githubusercontent.com/FrankChen0930/MarketMamba/main/robot_ledger.json"
            res = requests.get(url)
            res.raise_for_status()
            return res.json()
        except Exception as e:
            st.warning("⚠️ 尚未讀取到雲端帳本，可能雲端機器人尚未完成首次建倉。")
            return {"start_date": "", "cash": 1000000.0, "holdings": {}, "history": []}
            
    ledger = load_cloud_ledger()
    
    current_holdings = list(ledger["holdings"].keys())
    
    # Placeholder for get_current_prices, assuming it's implemented elsewhere or not strictly needed to render layout
    # live_prices = get_current_prices(current_holdings) if current_holdings else {}
    live_prices = {} 
    
    stock_value = 0.0
    for t, data in ledger["holdings"].items():
        price = live_prices.get(t, data.get("avg_cost", data.get("cost", 0)))
        stock_value += price * data["shares"]
        
    total_equity = ledger["cash"] + stock_value
    
    st.divider()
    col1, col2, col3 = st.columns(3)
    col1.metric("💰 基金總淨值 (NAV)", f"${total_equity:,.0f}", f"{(total_equity - 1000000)/10000:,.2f}%")
    col2.metric("💵 可用現金", f"${ledger['cash']:,.0f}")
    col3.metric("📈 股票市值", f"${stock_value:,.0f}")
    
    if ledger["holdings"]:
        st.write("📋 **今日機器人最新庫存**")
        holding_list = []
        for t, d in ledger["holdings"].items():
            current_p = live_prices.get(t, d.get("avg_cost", data.get("cost", 0)))
            profit_pct = (current_p - d.get("avg_cost", data.get("cost", 0))) / d.get("avg_cost", data.get("cost", 0)) * 100
            holding_list.append({
                "股票標的": format_ticker(t), 
                "持有股數": d["shares"],
                "平均成本": f"{d.get("avg_cost", data.get("cost", 0)):.2f}",
                "最新報價": f"{current_p:.2f}",
                "未實現損益": f"{profit_pct:.2f}%"
            })
            
        st.dataframe(pd.DataFrame(holding_list).style.applymap(
            lambda x: 'color: #00fa9a; font-weight: bold;' if '-' in str(x) else 'color: #ff4b4b; font-weight: bold;', subset=['未實現損益']
        ), use_container_width=True, hide_index=True)
    else:
        st.info("🤖 機器人目前空手觀望中。")
        
    if len(ledger["history"]) > 0:
        hist_df = pd.DataFrame(ledger["history"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist_df['date'], y=hist_df['equity'], mode='lines+markers', line=dict(color='#ff4b4b', width=3))) # 換成紅色
        fig.update_layout(title="📈 基金淨值成長曲線", template="plotly_dark", yaxis_title="總淨值 (TWD)")
        st.plotly_chart(fig, use_container_width=True)
