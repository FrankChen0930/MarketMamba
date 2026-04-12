"""
MarketMamba V5.5 — 訓練模組
負責：DataLoader 建構、訓練迴圈、Early Stopping、模型儲存
所有進度條與 Loss 曲線會在 Colab Cell 中即時顯示
"""

import os
import random
import logging
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from marketmamba.config import (
    FEATURE_COLS, FEATURE_COLS_V5, MODEL_CONFIG, TRAIN_CONFIG, MODEL_DIR,
)

logger = logging.getLogger('MarketMamba.trainer')


# ==========================================
# 自定義 Dataset
# ==========================================
class AlphaTrajectoryDataset(Dataset):
    """
    Alpha 軌跡訓練資料集

    對每支股票的時序切片成 (seq_len) 視窗，
    標籤為未來 pred_days 天的累積 Alpha 軌跡
    """

    def __init__(self, df: pd.DataFrame,
                 feature_cols: list[str],
                 seq_len: int = 120,
                 pred_days: int = 30):
        self.seq_len = seq_len
        self.pred_days = pred_days
        self.samples = []  # list of (features_array, labels_array)

        grouped = df.sort_values(['stock_id', 'Date']).groupby('stock_id')
        total_window = seq_len + pred_days

        for stock_id, group in grouped:
            if len(group) < total_window:
                continue

            features = group[feature_cols].values
            returns = group['Return_1d'].values
            twii_returns = group['TWII_Return_1d'].values
            alpha_daily = returns - twii_returns

            # 滑動視窗
            for i in range(len(group) - total_window + 1):
                x = features[i: i + seq_len].copy()

                # 逐視窗 Z-Score 標準化
                x_mean = x.mean(axis=0)
                x_std = x.std(axis=0) + 1e-8
                x = (x - x_mean) / x_std

                # 未來 pred_days 天的累積 Alpha
                future_alpha = alpha_daily[i + seq_len: i + total_window]
                y = np.cumsum(future_alpha)

                self.samples.append((
                    x.astype(np.float32),
                    y.astype(np.float32),
                ))

        logger.info(f"📦 Dataset 建構完成: {len(self.samples)} 個樣本")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.from_numpy(x), torch.from_numpy(y)


# ==========================================
# 訓練主函數
# ==========================================
def train_model(
    df: pd.DataFrame,
    feature_cols: list[str] = None,
    epochs: int = None,
    batch_size: int = None,
    lr: float = None,
    weight_decay: float = None,
    early_stop_patience: int = None,
    val_ratio: float = None,
    save_path: str = None,
    seed: int = None,
    use_v5_compat: bool = False,
) -> tuple:
    """
    完整訓練管線

    所有 tqdm 進度條和 print 輸出會直接顯示在 Colab Cell 中

    Args:
        df: 完整特徵矩陣 DataFrame
        feature_cols: 特徵欄位 (None = 自動選擇)
        epochs: 訓練 epoch 數
        batch_size: 批次大小
        lr: 學習率
        weight_decay: L2 正則化
        early_stop_patience: Early Stopping 耐心
        val_ratio: 驗證集比例
        save_path: 模型儲存路徑
        seed: 隨機種子
        use_v5_compat: 是否使用 V5.0 相容模式

    Returns:
        (model, history_dict)
    """
    # === 參數初始化 ===
    cfg = TRAIN_CONFIG
    epochs = epochs or cfg['epochs']
    batch_size = batch_size or cfg['batch_size']
    lr = lr or cfg['learning_rate']
    weight_decay = weight_decay or cfg['weight_decay']
    early_stop_patience = early_stop_patience or cfg['early_stop_patience']
    val_ratio = val_ratio or cfg['val_ratio']
    seed = seed or cfg['seed']

    if use_v5_compat:
        feature_cols = FEATURE_COLS_V5
    else:
        feature_cols = feature_cols or FEATURE_COLS

    _set_seed(seed)

    print(f"🔧 訓練設定:")
    print(f"   特徵維度: {len(feature_cols)}")
    print(f"   Epochs: {epochs}, Batch: {batch_size}, LR: {lr}")
    print(f"   Early Stop: {early_stop_patience} epochs")

    # === 確保欄位存在 ===
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # === 建構 Dataset ===
    print("📦 建構訓練資料集...")
    full_dataset = AlphaTrajectoryDataset(
        df, feature_cols,
        seq_len=MODEL_CONFIG['seq_len'],
        pred_days=MODEL_CONFIG['pred_days'],
    )

    if len(full_dataset) == 0:
        raise ValueError("❌ 資料集為空！請檢查資料是否足夠")

    # Train/Val 切分
    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size
    train_set, val_set = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    print(f"   訓練集: {train_size} 樣本, 驗證集: {val_size} 樣本")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    # === 建構模型 ===
    input_dim = len(feature_cols)
    if input_dim == MODEL_CONFIG['input_dim_v5']:
        from marketmamba.models.architecture import MarketMambaV5
        model = MarketMambaV5(input_dim=input_dim)
    else:
        from marketmamba.models.architecture import MarketMambaV55
        model = MarketMambaV55(input_dim=input_dim)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🧠 模型參數量: {param_count:,}")

    # === 優化器與損失函數 ===
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    # === 訓練迴圈 ===
    history = {'train_loss': [], 'val_loss': [], 'lr': []}
    best_val_loss = float('inf')
    patience_counter = 0

    if save_path is None:
        os.makedirs(MODEL_DIR, exist_ok=True)
        if input_dim == MODEL_CONFIG['input_dim_v5']:
            save_path = os.path.join(MODEL_DIR, 'V5_DynamicGAT_Production.pth')
        else:
            save_path = os.path.join(MODEL_DIR, 'V5_5_Production.pth')

    print(f"\n🚀 開始訓練！ (設備: {device})")
    print("=" * 60)

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        train_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]",
                     leave=False)

        for x_batch, y_batch in pbar:
            x_batch = torch.nan_to_num(x_batch.to(device), nan=0.0)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_losses.append(loss.item())
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})

        avg_train = np.mean(train_losses)

        # --- Validate ---
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = torch.nan_to_num(x_batch.to(device), nan=0.0)
                y_batch = y_batch.to(device)
                pred = model(x_batch)
                loss = criterion(pred, y_batch)
                val_losses.append(loss.item())

        avg_val = np.mean(val_losses)
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        history['train_loss'].append(avg_train)
        history['val_loss'].append(avg_val)
        history['lr'].append(current_lr)

        # 進度輸出
        print(
            f"  Epoch {epoch+1:3d}/{epochs} │ "
            f"Train: {avg_train:.6f} │ Val: {avg_val:.6f} │ "
            f"LR: {current_lr:.2e} │ "
            f"{'🟢 Best!' if avg_val < best_val_loss else ''}"
        )

        # Early Stopping
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            logger.info(f"  💾 最佳模型已儲存: {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"\n⏹️ Early Stopping! 連續 {early_stop_patience} epochs 無改善")
                break

    print("=" * 60)
    print(f"🎉 訓練完成！ Best Val Loss: {best_val_loss:.6f}")
    print(f"💾 模型路徑: {save_path}")

    # GPU 記憶體報告
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"🖥️ GPU 記憶體: {allocated:.2f} GB allocated / {reserved:.2f} GB reserved")

    # 載入最佳模型
    model.load_state_dict(torch.load(save_path, map_location=device))

    return model, history


def _set_seed(seed: int) -> None:
    """設定隨機種子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
