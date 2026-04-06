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
    * 🟢 **正 Alpha (+)**：代表模型預測該股動能強勁，表現將**超越大盤**。
    * 🔴 **負 Alpha (-)**：代表模型嗅到資金撤出訊號，表現將**落後大盤**。

    ### ⚖️ 核心評估指標
    * **波動風險 (%)**：預測未來 30 天的震盪幅度。
    * **夏普 CP 值 (Sharpe Score)**：`預期 Alpha ÷ 波動風險`。分數越高，代表**「承擔一單位的風險，能換來越多的超額報酬」**。
    * **資金佔比 (Kelly Weight)**：基於凱利公式計算出的建議押注比例，最高 20%。
    """)

# ==========================================
# 1. 載入核心預測數據
# ==========================================
@st.cache_data(ttl=3600)
def load_prediction_data():
    try:
        df_k = pd.read_csv("df_kelly.csv")
        df_t = pd.read_csv("df_traj.csv")
        try:
            df_p = pd.read_csv("pattern_scan_results.csv")
        except:
            df_p = pd.DataFrame(columns=['Pattern', 'Stock_ID', 'Scale', 'Score', 'Current_Price', 'Neckline_Ref', 'Target_Price', 'Exp_Return(%)'])
        return df_k, df_t, df_p, True
    except Exception as e:
        st.error(f"⚠️ 尚未讀取到預測資料，請確認 Colab 是否已成功推送 CSV 檔案。錯誤: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), False

df_kelly, df_traj, df_patterns, data_loaded = load_prediction_data()

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
        clean_dict = {str(k).split('.')[0].strip(): v for k, v in raw_dict.items()}
        return clean_dict
    except:
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
        return requests.get(url).text.strip()
    except:
        return "未知"

last_update = get_update_time()

with st.sidebar:
    st.header("📌 功能選單")
    if data_loaded:
        st.success("✅ V5.0 雲端數據庫已連線")
        st.caption(f"🕒 最新預測時間: {last_update}")
        
    st.markdown("---")
    
    pages = ["📊 今日凱利資金盤", "📈 個股軌跡透視", "📐 傳統型態學雷達", "💼 我的持股健檢", "🤖 百萬實盤機器人"]
    for p in pages:
        if st.button(p, use_container_width=True, type="primary" if st.session_state['current_page'] == p else "secondary"):
            st.session_state['current_page'] = p
            st.rerun()

page = st.session_state['current_page']
if not data_loaded:
    st.stop() 

# ------------------------------------------
# 頁面 1: 凱利資金盤
# ------------------------------------------
if page == "📊 今日凱利資金盤":
    st.subheader("🎯 今日最佳 Alpha 潛力股 (Top 10)")
    st.write("點擊表格中的任意一行，系統將自動為您跳轉至該股的「詳細軌跡透視」頁面。")
    
    top_picks = df_kelly.head(10).copy()
    display_df = top_picks[['Ticker', 'Volatility_Risk', 'Sharpe_Score', 'Suggested_Weight']].copy()
    
    for day in [5, 10, 15, 20, 25, 30]:
        day_values = df_traj.set_index('Ticker')[f'Day_{day}'].reindex(display_df['Ticker']).values
        display_df[f'預期 Alpha ({day}天)'] = (day_values * 100)
        
    display_df['Volatility_Risk'] *= 100
    display_df['Suggested_Weight'] *= 100
    
    display_df = display_df.rename(columns={
        'Ticker': '股票標的', 'Volatility_Risk': '波動風險(%)', 
        'Sharpe_Score': '夏普CP值', 'Suggested_Weight': '資金佔比(%)'
    })
    
    display_df['股票標的'] = display_df['股票標的'].apply(format_ticker)
    
    def color_tw_returns(val):
        if isinstance(val, (int, float)):
            if val > 0: return 'color: #ff4b4b; font-weight: bold;' 
            elif val < 0: return 'color: #00fa9a; font-weight: bold;' 
        return ''

    return_cols = [f'預期 Alpha ({d}天)' for d in [5,10,15,20,25,30]]
    
    # 🌟 修復 1：使用 .map 取代 .applymap
    styled_df = display_df.style.map(color_tw_returns, subset=return_cols) \
                                .format({col: "{:.2f}%" for col in return_cols}) \
                                .format({"波動風險(%)": "{:.2f}%", "資金佔比(%)": "{:.2f}%", "夏普CP值": "{:.4f}"})

    event = st.dataframe(styled_df, use_container_width=True, hide_index=True, selection_mode="single-row", on_select="rerun")
    
    if len(event.selection.rows) > 0:
        selected_ticker = str(display_df.iloc[event.selection.rows[0]]['股票標的']).split(' ')[0]
        st.session_state['target_ticker'] = selected_ticker
        st.session_state['current_page'] = "📈 個股軌跡透視"
        st.rerun()

# ------------------------------------------
# 頁面 2: 個股軌跡透視
# ------------------------------------------
elif page == "📈 個股軌跡透視":
    st.subheader("🔭 平行宇宙軌跡觀測儀")
    
    all_tickers = df_traj['Ticker'].astype(str).tolist()
    default_idx = all_tickers.index(st.session_state['target_ticker']) if st.session_state.get('target_ticker') in all_tickers else 0
    target_ticker = st.selectbox("🔍 請選擇要觀測的股票", all_tickers, index=default_idx, format_func=format_ticker)
    st.session_state['target_ticker'] = target_ticker 
    stock_name = get_stock_name(target_ticker)
    
    # 🌟 強化版 yfinance 抓取 (防斷線、防週末空窗)
    @st.cache_data(ttl=3600)
    def fetch_stock_info(ticker):
        clean_ticker = str(ticker).split('.')[0].strip()
        for suffix in ['.TW', '.TWO']:
            try:
                stock = yf.Ticker(f"{clean_ticker}{suffix}")
                # 改抓 5 天，避免週末假日回傳空資料
                hist = stock.history(period="5d")
                if not hist.empty:
                    info_data = {}
                    try:
                        info_data = stock.info
                    except:
                        pass # 如果抓不到 info 就放過它，不要報錯
                    return hist, info_data, suffix
            except:
                continue
        return None, {}, None

    with st.spinner(f"正在同步 {stock_name} 的個股資訊..."):
        hist_df, stock_info, suffix = fetch_stock_info(target_ticker)
        
    if hist_df is None or hist_df.empty:
        st.warning(f"⚠️ Yahoo Finance 暫時無法連線，無法獲取 {target_ticker} 的最新歷史股價。請稍後再試。")
    else:
        st.markdown(f"### {stock_name} ({target_ticker}) 個股速覽")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("今日收盤價", f"{hist_df['Close'].iloc[-1]:.2f}")
        
        pe_val = stock_info.get('trailingPE', 'N/A')
        mc_val = stock_info.get('marketCap', 'N/A')
        pe_str = f"{pe_val:.2f} 倍" if isinstance(pe_val, (int, float)) else "N/A"
        mc_str = f"{mc_val / 100000000:,.2f}" if isinstance(mc_val, (int, float)) else "N/A" 

        col2.metric("本益比 (P/E)", pe_str)
        col3.metric("市值 (億)", mc_str)
        col4.metric("產業別", stock_info.get('industry', 'N/A'))
        
        stock_traj = df_traj[df_traj['Ticker'].astype(str) == target_ticker].iloc[0]
        volatility = df_kelly[df_kelly['Ticker'].astype(str) == target_ticker]['Volatility_Risk'].iloc[0]
        traj_values = stock_traj[[f'Day_{i}' for i in range(1, 31)]].values 
        
        st.markdown("#### 📅 V5.0 Mamba 預期超額報酬 (Alpha) 軌跡")
        ret_cols = st.columns(6)
        for idx, d in enumerate([5, 10, 15, 20, 25, 30]):
            val = traj_values[d-1] * 100
            ret_cols[idx].metric(f"{d} 天後", f"{val:.2f}%", delta_color="normal" if val > 0 else "inverse")

        st.divider()
        future_dates = pd.bdate_range(start=hist_df.index[-1] + pd.Timedelta(days=1), periods=30)
        alpha_trajectory_pct = traj_values * 100
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[future_dates[0], future_dates[-1]], y=[0, 0], mode='lines', name='TWII 大盤基準線', line=dict(color='white', width=2, dash='dash')))
        fig.add_trace(go.Scatter(x=future_dates, y=alpha_trajectory_pct, mode='lines+markers', name=f'{target_ticker} 預測 Alpha', line=dict(color='#ff4b4b' if alpha_trajectory_pct[-1] > 0 else '#00fa9a', width=4)))
        
        upper_bound = alpha_trajectory_pct + (volatility * 100 * np.sqrt(np.arange(1, 31)/30.0))
        lower_bound = alpha_trajectory_pct - (volatility * 100 * np.sqrt(np.arange(1, 31)/30.0))
        fig.add_trace(go.Scatter(x=list(future_dates) + list(future_dates)[::-1], y=list(upper_bound) + list(lower_bound)[::-1], fill='toself', fillcolor='rgba(255, 255, 255, 0.1)', line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", name='模型信心區間'))

        fig.update_layout(title=f"<b>{format_ticker(target_ticker)} 未來 30 天預測</b>", yaxis_title="累積 Alpha (%)", template="plotly_dark", hovermode="x unified", margin=dict(l=40, r=40, t=60, b=40))
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------
# 頁面 3: 傳統型態學雷達
# ------------------------------------------
elif page == "📐 傳統型態學雷達":
    st.subheader("📐 傳統型態學掃描雷達")
    st.write("結合多重時間框架與等幅測距，全自動掃描全市場，精準捕捉發動前夕的高勝率經典型態。")
    st.divider()

    if df_patterns.empty:
        st.success("🛡️ **今日市場結構混亂，雷達未偵測到高勝率經典型態，建議空手觀望，保護資金。**")
    else:
        display_df = df_patterns[['Pattern', 'Stock_ID', 'Scale', 'Current_Price', 'Target_Price', 'Exp_Return(%)']].copy()
        display_df['Stock_ID'] = display_df['Stock_ID'].apply(format_ticker)
        display_df.rename(columns={'Pattern': '觸發型態', 'Stock_ID': '股票標的', 'Scale': '成型時間尺度', 'Current_Price': '最新收盤價', 'Target_Price': '目標滿足價', 'Exp_Return(%)': '預期報酬空間(%)'}, inplace=True)

        def color_returns(val):
            if isinstance(val, (int, float)):
                if val > 0: return 'color: #ff4b4b; font-weight: bold;'
                elif val < 0: return 'color: #00fa9a; font-weight: bold;'
            return ''

        # 🌟 修復 2：使用 .map 取代 .applymap
        styled_df = display_df.style.map(color_returns, subset=['預期報酬空間(%)']) \
                                    .format({"最新收盤價": "{:.2f}", "目標滿足價": "{:.2f}", "預期報酬空間(%)": "{:.2f}%"})
        st.dataframe(styled_df, use_container_width=True, hide_index=True)

# ------------------------------------------
# 頁面 4: 我的持股健檢
# ------------------------------------------
elif page == "💼 我的持股健檢":
    st.subheader("💼 我的專屬量化基金 (Portfolio)")
    
    if 'portfolio' not in st.session_state:
        try:
            st.session_state['portfolio'] = pd.DataFrame(json.loads(st.query_params["saved_portfolio"]))
        except:
            st.session_state['portfolio'] = pd.DataFrame(columns=['股票代號', '持有成本', '持有股數'])

    def save_portfolio(): st.query_params["saved_portfolio"] = st.session_state['portfolio'].to_json(orient='records')

    @st.cache_data(ttl=3600)
    def get_latest_close(ticker):
        for suf in ['.TW', '.TWO']:
            try:
                # 🌟 同樣改抓 5 天防報錯
                hist = yf.Ticker(f"{str(ticker).split('.')[0].strip()}{suf}").history(period="5d")
                if not hist.empty: return float(hist['Close'].iloc[-1])
            except: pass
        return 100.0

    with st.expander("➕ 新增庫存持股", expanded=True):
        col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
        with col1: new_ticker = st.selectbox("股票標的", df_kelly['Ticker'].astype(str).tolist(), format_func=format_ticker)
        with st.spinner("獲取最新股價中..."): default_price = get_latest_close(new_ticker)
        with col2: new_cost = st.number_input("持有成本", min_value=0.0, value=default_price, step=1.0)
        with col3: new_shares = st.number_input("股數", min_value=1, value=1000, step=100)
        with col4:
            st.markdown("<br>", unsafe_allow_html=True) 
            if st.button("新增", type="primary"):
                st.session_state['portfolio'] = pd.concat([st.session_state['portfolio'], pd.DataFrame({'股票代號': [new_ticker], '持有成本': [new_cost], '持有股數': [new_shares]})], ignore_index=True)
                save_portfolio()
                st.rerun() 

    if not st.session_state['portfolio'].empty:
        st.divider()
        analysis_df = pd.merge(st.session_state['portfolio'], df_kelly[['Ticker', 'Exp_Return_15D', 'Sharpe_Score']].astype({'Ticker': str}), left_on='股票代號', right_on='Ticker', how='left')
        
        def get_action_signal(row):
            if pd.isna(row['Exp_Return_15D']): return "⚪ 缺乏預測資料"
            if row['Exp_Return_15D'] < 0: return "🔴 趨勢落後大盤 (建議停損)"
            if row['Sharpe_Score'] > 0.5: return "🟢 超額動能強勁 (建議續抱)"
            return "🟡 波動加劇 (嚴設停損)"
                
        analysis_df['AI 操作建議'] = analysis_df.apply(get_action_signal, axis=1)
        analysis_df['預期 15 天 Alpha'] = (analysis_df['Exp_Return_15D'] * 100).apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
        analysis_df['夏普分數'] = analysis_df['Sharpe_Score'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
        analysis_df['股票標的'] = analysis_df['股票代號'].apply(format_ticker)
        
        # 🌟 修復 3：使用 .map 取代 .applymap
        st.dataframe(
            analysis_df[['股票標的', '持有成本', '持有股數', '預期 15 天 Alpha', '夏普分數', 'AI 操作建議']].style.map(
                lambda x: 'color: #00fa9a; font-weight: bold;' if '🔴' in str(x) else ('color: #ff4b4b; font-weight: bold;' if '🟢' in str(x) else ''),
                subset=['AI 操作建議']
            ).format({"持有成本": "{:.2f}"}), use_container_width=True, hide_index=True
        )
        if st.button("🗑️ 清空庫存"):
            st.session_state['portfolio'] = pd.DataFrame(columns=['股票代號', '持有成本', '持有股數'])
            save_portfolio()
            st.rerun()

# ------------------------------------------
# 頁面 5: 百萬實盤機器人
# ------------------------------------------
elif page == "🤖 百萬實盤機器人":
    st.subheader("🤖 MarketMamba 全自動實盤基金")
    
    @st.cache_data(ttl=600) 
    def load_cloud_ledger():
        try: return requests.get("https://raw.githubusercontent.com/FrankChen0930/MarketMamba/main/robot_ledger.json").json()
        except: return {"start_date": "", "cash": 1000000.0, "holdings": {}, "history": []}
            
    ledger = load_cloud_ledger()
    live_prices = {} 
    
    stock_value = sum(live_prices.get(t, d.get("avg_cost", d.get("cost", 0))) * d["shares"] for t, d in ledger["holdings"].items())
    total_equity = ledger["cash"] + stock_value
    
    col1, col2, col3 = st.columns(3)
    col1.metric("💰 基金總淨值", f"${total_equity:,.0f}", f"{(total_equity - 1000000)/10000:,.2f}%")
    col2.metric("💵 可用現金", f"${ledger['cash']:,.0f}")
    col3.metric("📈 股票市值", f"${stock_value:,.0f}")
    
    if ledger["holdings"]:
        holding_list = []
        for t, d in ledger["holdings"].items():
            cost_val = d.get("avg_cost", d.get("cost", 0))
            current_p = live_prices.get(t, cost_val)
            holding_list.append({"股票標的": format_ticker(t), "持有股數": d["shares"], "平均成本": f"{cost_val:.2f}", "最新報價": f"{current_p:.2f}", "未實現損益": f"{((current_p - cost_val) / cost_val * 100) if cost_val > 0 else 0:.2f}%"})
            
        # 🌟 修復 4：使用 .map 取代 .applymap
        st.dataframe(pd.DataFrame(holding_list).style.map(lambda x: 'color: #00fa9a; font-weight: bold;' if '-' in str(x) else 'color: #ff4b4b; font-weight: bold;', subset=['未實現損益']), use_container_width=True, hide_index=True)
        
    if ledger["history"]:
        hist_df = pd.DataFrame(ledger["history"])
        fig = go.Figure(go.Scatter(x=hist_df['date'], y=hist_df['equity'], mode='lines+markers', line=dict(color='#ff4b4b', width=3)))
        fig.update_layout(title="📈 基金淨值成長曲線", template="plotly_dark", yaxis_title="總淨值 (TWD)")
        st.plotly_chart(fig, use_container_width=True)
