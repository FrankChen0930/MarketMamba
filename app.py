import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import math
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde
import yfinance as yf

# ==========================================
# 0. 網頁基本設定
# ==========================================
st.set_page_config(page_title="MarketMamba 預測終端", page_icon="📈", layout="wide")

# 動態載入我們自備的字型檔
font_path = "NotoSansTC-Regular.ttf"  # 確認檔名與你上傳的一致
if os.path.exists(font_path):
    # 告訴 Matplotlib 把這個字型加進去
    fm.fontManager.addfont(font_path)
    # 設定全域字型為這個新字型
    font_prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = font_prop.get_name()
else:
    # 如果雲端沒抓到檔案，就退回原本的設定 (本機開發時的備案)
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'sans-serif'] 
plt.rcParams['axes.unicode_minus'] = False # 解決負號 '-' 變成方塊的問題

st.title("🐍 MarketMamba: 股市機率擴散預測模型")
st.markdown("基於 **Mamba 架構** 與 **Diffusion 生成模型** 的次世代量化交易預測系統。")

# ==========================================
# 1. 模型架構定義 (必須與訓練時一模一樣)
# ==========================================
class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(in_channels=self.d_inner, out_channels=self.d_inner, bias=True, kernel_size=d_conv, groups=self.d_inner, padding=d_conv - 1)
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.log_A = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        x_and_res = self.in_proj(x)
        (x_in, res) = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)
        x_conv = self.act(self.conv1d(x_in.permute(0, 2, 1))[:, :, :seq_len].permute(0, 2, 1))
        x_dbl = self.x_proj(x_conv)
        (dt, B, C) = x_dbl.split(split_size=[self.dt_rank, 16, 16], dim=-1)
        dt = F.softplus(self.dt_proj(dt))
        A = -torch.exp(self.log_A)
        h = torch.zeros(batch_size, self.d_inner, 16, device=x.device)
        y = []
        for t in range(seq_len):
            dt_t, B_t, C_t = dt[:, t, :].unsqueeze(-1), B[:, t, :].unsqueeze(1), C[:, t, :].unsqueeze(1)
            h = torch.exp(A * dt_t) * h + dt_t * B_t * x_conv[:, t, :].unsqueeze(-1)
            y.append(torch.sum(h * C_t, dim=-1) + self.D * x_conv[:, t, :])
        return self.out_proj(torch.stack(y, dim=1) * F.silu(res))

class StockMambaModel(nn.Module):
    def __init__(self, input_dim=2, d_model=64, n_layers=2):
        super().__init__()
        self.encoder = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList([MambaBlock(d_model=d_model) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x):
        x = self.encoder(x)
        for layer in self.layers: x = layer(x)
        return self.norm(x)

class DenoiseNetwork(nn.Module):
    def __init__(self, condition_dim=64, target_dim=1, hidden_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.cond_proj = nn.Linear(condition_dim, hidden_dim)
        self.input_proj = nn.Linear(target_dim, hidden_dim)
        self.mid_layers = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
        self.final_proj = nn.Linear(hidden_dim, target_dim)
    def forward(self, x, t, condition):
        return self.final_proj(self.mid_layers(self.input_proj(x) + self.time_mlp(t) + self.cond_proj(condition)))

class DiffusionManager(nn.Module):
    def __init__(self, denoise_net, mamba_model, n_steps=100, device='cpu'):
        super().__init__()
        self.denoise_net = denoise_net
        self.mamba_model = mamba_model
        self.n_steps = n_steps
        self.device = device
        self.beta = torch.linspace(1e-4, 0.02, n_steps).to(device)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    @torch.no_grad()
    def sample(self, x_history, n_samples=100):
        batch_size = x_history.shape[0]
        condition = self.mamba_model(x_history)[:, -1, :].repeat_interleave(n_samples, dim=0)
        x = torch.randn(batch_size * n_samples, 1, device=self.device)
        
        for t in reversed(range(self.n_steps)):
            t_input = (torch.ones(batch_size * n_samples, 1, device=self.device) * t) / self.n_steps
            pred_noise = self.denoise_net(x, t_input, condition)
            alpha, alpha_bar, beta = self.alpha[t], self.alpha_bar[t], self.beta[t]
            noise = torch.randn_like(x) if t > 0 else 0
            x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_bar)) * pred_noise) + torch.sqrt(beta) * noise
        return x.view(batch_size, n_samples)

# ==========================================
# 2. 載入模型 (加入快取機制，避免每次點擊都重載)
# ==========================================
@st.cache_resource
def load_models():
    # 本機推論一律用 CPU 就夠了，不用管 CUDA
    device = torch.device('cpu') 
    mamba_path = 'models/mamba_core_phase2.pth'
    diffusion_path = 'models/diffusion_v1.pth'
    
    # 初始化空模型
    mamba_model = StockMambaModel(input_dim=2, d_model=64, n_layers=2).to(device)
    denoise_net = DenoiseNetwork(condition_dim=64, target_dim=1, hidden_dim=128).to(device)
    diffusion = DiffusionManager(denoise_net, mamba_model, n_steps=100, device=device).to(device)
    
    # 載入權重
    if os.path.exists(mamba_path) and os.path.exists(diffusion_path):
        # 處理 Mamba 權重
        mamba_ckpt = torch.load(mamba_path, map_location=device)
        if 'model_state_dict' in mamba_ckpt:
            mamba_model.load_state_dict(mamba_ckpt['model_state_dict'], strict=False)
        else:
            mamba_model.load_state_dict(mamba_ckpt)
            
        # 處理 Diffusion 權重
        diffusion.load_state_dict(torch.load(diffusion_path, map_location=device))
        
        diffusion.eval()
        return diffusion, True
    else:
        return None, False

diffusion_model, is_loaded = load_models()

# ==========================================
# 3. 網頁互動區塊
# ==========================================
with st.sidebar:
    st.header("⚙️ 預測參數設定")
    # 新增股票代號輸入框 (預設為台積電)
    ticker_input = st.text_input("🔍 輸入股票代號 (台股請加 .TW)", value="2330.TW")
    n_samples = st.slider("生成預測路徑數量 (信心區間解析度)", min_value=50, max_value=500, value=100, step=50)
    st.info("💡 採樣數量越高，生成的機率雲圖會越平滑。")

if not is_loaded:
    st.error("❌ 找不到模型權重檔！請確認 `models/mamba_core_phase2.pth` 與 `models/diffusion_v1.pth` 是否存在。")
else:
    st.success("✅ Mamba 與 Diffusion 核心已成功上線！")
    
    st.subheader("📊 模擬預測：台積電 (2330)")
    st.write("點擊下方按鈕，模型將會讀取最近 60 分鐘的市場特徵，並生成未來的可能走勢機率。")
    
    if st.button("🚀 生成未來走勢機率雲", type="primary"):
        with st.spinner(f"🧠 正在從 Yahoo Finance 抓取 {ticker_input} 最新 K 線，Diffusion 演算中..."):
            
            # ==========================================
            # 1. 透過 yfinance 抓取即時歷史資料
            # ==========================================
            try:
                # 抓取過去一年的日 K 線資料
                stock_data = yf.Ticker(ticker_input)
                df = stock_data.history(period="1y")
                
                if df.empty:
                    st.error(f"❌ 找不到代號 {ticker_input} 的資料，請確認輸入是否正確！")
                    st.stop()
                    
            except Exception as e:
                st.error(f"❌ 抓取資料失敗: {e}")
                st.stop()
            
            # 為了確保欄位名稱一致，將 yfinance 輸出的欄位轉為所需格式
            df = df[['Close', 'Volume']].copy()
            
            # 計算與訓練時一模一樣的特徵
            df['Log_Ret'] = np.log((df['Close'] + 1e-8) / (df['Close'].shift(1) + 1e-8))
            df['Log_Vol'] = np.log(df['Volume'] + 1)
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(inplace=True)
            
            # 如果資料不足 60 筆則擋下
            if len(df) < 60:
                st.error("❌ 該股票的歷史資料不足 60 筆，無法進行預測！")
                st.stop()
            
            # 取出最後 60 筆原始收盤價用來畫圖
            recent_df = df.tail(60).copy()
            historical_prices = recent_df['Close'].values
            current_price = historical_prices[-1]
            
            # -- 後面的資料標準化、推論、畫圖程式碼完全不用動！ --
            
            # ==========================================
            # 2. 推論與數值還原
            # ==========================================
            # 預測出標準化後的 Log Return
            predictions = diffusion_model.sample(real_history_tensor, n_samples=n_samples)
            preds_scaled = predictions.cpu().numpy().flatten()
            
            # 反標準化：(預測值 * 標準差) + 平均數
            preds_log_ret = (preds_scaled * scaler.scale_[0]) + scaler.mean_[0]
            
            # 轉換回真實股價：P_t = P_{t-1} * exp(Log_Ret)
            predicted_prices = current_price * np.exp(preds_log_ret)
            
            # 計算 KDE 尋找「最密集的眾數」 (你要求的最大可能性落點)
            kde = gaussian_kde(predicted_prices)
            price_range = np.linspace(min(predicted_prices), max(predicted_prices), 1000)
            kde_mode_price = price_range[np.argmax(kde(price_range))]
            
            # ==========================================
            # 📈 繪製專業走勢預測圖
            # ==========================================
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # A. 歷史走勢
            days_past = np.arange(-59, 1)
            ax.plot(days_past, historical_prices, color='#1f77b4', linewidth=2.5, label='真實歷史走勢 (最後 60 筆)')
            ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5, label='最新收盤價')
            ax.scatter(0, current_price, color='black', s=50, zorder=5) 
            
            # B. 未來預測分佈
            day_future = np.ones_like(predicted_prices) * 1 
            ax.scatter(day_future, predicted_prices, color='#ff7f0e', alpha=0.3, label='Diffusion 預測路徑')
            ax.boxplot(predicted_prices, positions=[1], widths=1.5, patch_artist=True,
                       boxprops=dict(facecolor='#ff7f0e', alpha=0.5, color='#ff7f0e'),
                       medianprops=dict(color='red', linewidth=2),
                       whis=[5, 95], showfliers=False) 
                       
            # 標示出「最密集的眾數」
            ax.scatter(1, kde_mode_price, color='purple', s=100, marker='*', zorder=10, label=f'最高機率落點 ({kde_mode_price:.2f})')
            
            # C. 圖表美化
            ax.set_title(f"台積電 (2330) 走勢與未來機率預測\n最新收盤價: {current_price:.2f}", fontsize=16, fontweight='bold')
            ax.set_xlabel("時間步數 (分鐘/日)", fontsize=12)
            ax.set_ylabel("股價 (TWD)", fontsize=12)
            ax.set_xticks([-60, -40, -20, 0, 1])
            ax.set_xticklabels(['T-60', 'T-40', 'T-20', 'T=0', 'T+1'])
            ax.legend(loc='upper left')
            ax.grid(True, linestyle=':', alpha=0.6)
            
            st.pyplot(fig)
            
            # ==========================================
            # 📊 顯示數據統計
            # ==========================================
            mean_pred_price = np.mean(predicted_prices)
            p05_price = np.percentile(predicted_prices, 5)
            p95_price = np.percentile(predicted_prices, 95)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("當前收盤價", f"{current_price:.2f}")
            col2.metric("⭐ 最高機率目標價", f"{kde_mode_price:.2f}", f"{(kde_mode_price - current_price):.2f}")
            col3.metric("保守情境 (5%)", f"{p05_price:.2f}", f"{(p05_price - current_price):.2f}")

            col4.metric("樂觀情境 (95%)", f"{p95_price:.2f}", f"{(p95_price - current_price):.2f}")


