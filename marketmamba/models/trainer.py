"""
MarketMamba V5.5 — 訓練模組 (A100 極速版)
負責：DataLoader 建構、訓練迴圈、Early Stopping、模型儲存

核心最佳化：
  1. Z-Score 預標準化 per-stock (不再 per-window)，建構 <10 秒
  2. Zero-copy Dataset：__getitem__ 只做 numpy slice，無額外計算
  3. AMP 混合精度 (FP16) + Tensor Core 加速
  4. cuDNN benchmark + non_blocking transfer
  5. 預建構張量快取到 Drive (重啟秒讀)
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
from torch.amp import autocast, GradScaler
from tqdm.auto import tqdm

from marketmamba.config import (
    FEATURE_COLS, FEATURE_COLS_V5, MODEL_CONFIG, TRAIN_CONFIG,
    MODEL_DIR, PROCESSED_DIR,
)

logger = logging.getLogger('MarketMamba.trainer')


# ==========================================
# 高速 Dataset (Zero-Copy Sliding Window)
# ==========================================
class FastAlphaDataset(Dataset):
    """
    極速滑動視窗 Dataset

    核心：Z-Score 在 __init__ 時 per-stock 一次算完
    __getitem__ 只做 numpy slice + cumsum(30)，幾乎零開銷
    """

    def __init__(self, stock_arrays: dict, seq_len: int, pred_days: int,
                 valid_indices: list):
        self.stock_arrays = stock_arrays
        self.seq_len = seq_len
        self.pred_days = pred_days
        self.valid_indices = valid_indices

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        stock_id, start = self.valid_indices[idx]
        start = int(start)  # 確保 int (快取載入可能變 str)
        features, alpha = self.stock_arrays[stock_id]

        x = features[start : start + self.seq_len].copy()
        y = np.cumsum(alpha[start + self.seq_len : start + self.seq_len + self.pred_days]).copy()

        return torch.from_numpy(x), torch.from_numpy(y)


def _prepare_stock_arrays(df: pd.DataFrame,
                          feature_cols: list[str],
                          seq_len: int,
                          pred_days: int) -> tuple[dict, list]:
    """
    預計算每支股票的標準化特徵 + Alpha 序列

    Z-Score 標準化在這裡 per-stock 一次完成，不再 per-window
    (與原版 V5.0 一致的標準化方式)

    Returns:
        stock_arrays: {stock_id: (features_normed, alpha_daily)}
        valid_indices: [(stock_id, start_idx), ...]
    """
    total_window = seq_len + pred_days
    stock_arrays = {}
    valid_indices = []

    grouped = df.sort_values(['stock_id', 'Date']).groupby('stock_id')
    stock_count = 0

    for stock_id, group in tqdm(grouped, desc="📦 預處理股票", unit="stock"):
        if len(group) < total_window:
            continue

        features = group[feature_cols].values.astype(np.float32)

        # Per-stock Z-Score (向量化，一次算完整支股票)
        f_mean = features.mean(axis=0)
        f_std = features.std(axis=0) + 1e-8
        features = (features - f_mean) / f_std

        returns = group['Return_1d'].values.astype(np.float32)
        twii_returns = group['TWII_Return_1d'].values.astype(np.float32)
        alpha_daily = returns - twii_returns

        stock_arrays[stock_id] = (features, alpha_daily)

        n_windows = len(group) - total_window + 1
        for i in range(n_windows):
            valid_indices.append((stock_id, i))

        stock_count += 1

    print(f"   ✅ {stock_count} 支股票, {len(valid_indices):,} 個樣本")
    return stock_arrays, valid_indices


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
    完整訓練管線 (A100 極速版)

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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = (device.type == 'cuda')
    n_workers = min(4, os.cpu_count() or 2)

    print(f"🔧 訓練設定:")
    print(f"   特徵維度: {len(feature_cols)}")
    print(f"   Epochs: {epochs}, Batch: {batch_size}, LR: {lr}")
    print(f"   Early Stop: {early_stop_patience} epochs")
    print(f"   AMP: {'✅' if use_amp else '❌'}, Workers: {n_workers}")

    # === 確保欄位存在 ===
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # === 建構 Dataset (快取機制) ===
    import pickle
    cache_path = os.path.join(PROCESSED_DIR, 'V55_train_cache.pkl')

    stock_arrays = None

    # 嘗試讀快取
    if os.path.exists(cache_path):
        try:
            print("⚡ 偵測到訓練快取，直接載入...")
            with open(cache_path, 'rb') as f:
                cached = pickle.load(f)
            stock_arrays = cached['stock_arrays']
            valid_indices = cached['valid_indices']
            print(f"   ✅ {len(stock_arrays)} 支股票, {len(valid_indices):,} 個樣本")
        except Exception as e:
            print(f"   ⚠️ 快取載入失敗 ({e})，重新建構...")
            stock_arrays = None

    if stock_arrays is None:
        print("📦 建構訓練資料集...")
        stock_arrays, valid_indices = _prepare_stock_arrays(
            df, feature_cols,
            seq_len=MODEL_CONFIG['seq_len'],
            pred_days=MODEL_CONFIG['pred_days'],
        )

        # 儲存快取 (pickle 保證型態正確)
        try:
            os.makedirs(PROCESSED_DIR, exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump({'stock_arrays': stock_arrays, 'valid_indices': valid_indices}, f)
            cache_mb = os.path.getsize(cache_path) / 1024**2
            print(f"   💾 快取已儲存: {cache_mb:.0f} MB (下次秒讀)")
        except Exception as e:
            print(f"   ⚠️ 快取儲存失敗: {e}")

    if len(valid_indices) == 0:
        raise ValueError("❌ 資料集為空！請檢查資料是否足夠")

    # Train/Val 切分
    n_total = len(valid_indices)
    val_size = int(n_total * val_ratio)
    train_size = n_total - val_size

    rng = random.Random(seed)
    shuffled = list(range(n_total))
    rng.shuffle(shuffled)

    train_indices = [valid_indices[i] for i in shuffled[:train_size]]
    val_indices = [valid_indices[i] for i in shuffled[train_size:]]

    print(f"   訓練集: {train_size:,} 樣本, 驗證集: {val_size:,} 樣本")

    train_set = FastAlphaDataset(
        stock_arrays, MODEL_CONFIG['seq_len'], MODEL_CONFIG['pred_days'], train_indices
    )
    val_set = FastAlphaDataset(
        stock_arrays, MODEL_CONFIG['seq_len'], MODEL_CONFIG['pred_days'], val_indices
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=n_workers, pin_memory=True,
        persistent_workers=(n_workers > 0),
        prefetch_factor=4 if n_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size * 2, shuffle=False,
        num_workers=n_workers, pin_memory=True,
        persistent_workers=(n_workers > 0),
        prefetch_factor=4 if n_workers > 0 else None,
    )

    # === 建構模型 ===
    input_dim = len(feature_cols)
    if input_dim == MODEL_CONFIG['input_dim_v5']:
        from marketmamba.models.architecture import MarketMambaV5
        model = MarketMambaV5(input_dim=input_dim)
    else:
        from marketmamba.models.architecture import MarketMambaV55
        model = MarketMambaV55(input_dim=input_dim)

    model = model.to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🧠 模型參數量: {param_count:,}")

    # === 優化器 ===
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    scaler = GradScaler('cuda') if use_amp else None

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
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"   VRAM: {vram:.1f} GB")
    print("=" * 60)

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        train_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]",
                     leave=False)

        for x_batch, y_batch in pbar:
            x_batch = torch.nan_to_num(x_batch.to(device, non_blocking=True), nan=0.0)
            y_batch = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with autocast('cuda'):
                    pred = model(x_batch)
                    loss = criterion(pred, y_batch)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(x_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
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
                x_batch = torch.nan_to_num(x_batch.to(device, non_blocking=True), nan=0.0)
                y_batch = y_batch.to(device, non_blocking=True)
                if use_amp:
                    with autocast('cuda'):
                        pred = model(x_batch)
                        loss = criterion(pred, y_batch)
                else:
                    pred = model(x_batch)
                    loss = criterion(pred, y_batch)
                val_losses.append(loss.item())

        avg_val = np.mean(val_losses)
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        history['train_loss'].append(avg_train)
        history['val_loss'].append(avg_val)
        history['lr'].append(current_lr)

        # GPU 即時用量
        gpu_info = ""
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1024**3
            gpu_info = f" │ GPU: {alloc:.1f}GB"

        print(
            f"  Epoch {epoch+1:3d}/{epochs} │ "
            f"Train: {avg_train:.6f} │ Val: {avg_val:.6f} │ "
            f"LR: {current_lr:.2e}{gpu_info} │ "
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

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"🖥️ GPU 記憶體: {allocated:.2f} GB allocated / {reserved:.2f} GB reserved")

    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))

    return model, history


def _set_seed(seed: int) -> None:
    """設定隨機種子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
