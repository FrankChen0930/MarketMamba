"""
MarketMamba V5.5 — 訓練模組 (記憶體最佳化版)
負責：DataLoader 建構、訓練迴圈、Early Stopping、模型儲存

記憶體最佳化：
  - DataFrame 提前轉 float32 (省 50% RAM)
  - valid_indices 用 numpy int32 array 存 (省 95% RAM)
  - 建構完立即釋放 df
  - DataLoader num_workers=2 避免 fork 記憶體翻倍
  - AMP FP16 訓練
"""

import gc
import os
import random
import logging
import pickle
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
# 高速 + 低記憶體 Dataset
# ==========================================
class FastAlphaDataset(Dataset):
    """
    極速滑動視窗 Dataset

    記憶體：stock_arrays 在所有 workers 間共享 (不複製)
    __getitem__: numpy slice + cumsum(30)，零開銷
    """

    def __init__(self, stock_ids: list, stock_arrays: dict,
                 seq_len: int, pred_days: int,
                 idx_stock: np.ndarray, idx_start: np.ndarray):
        """
        Args:
            stock_ids: 所有股票 ID 清單 (用 int index 對應)
            stock_arrays: {stock_id: (features, alpha)}
            idx_stock: (N,) int16 — 指向 stock_ids 的 index
            idx_start: (N,) int16 — 窗口起始位置
        """
        self.stock_ids = stock_ids
        self.stock_arrays = stock_arrays
        self.seq_len = seq_len
        self.pred_days = pred_days
        self.idx_stock = idx_stock
        self.idx_start = idx_start

    def __len__(self):
        return len(self.idx_stock)

    def __getitem__(self, idx):
        sid = self.stock_ids[self.idx_stock[idx]]
        start = int(self.idx_start[idx])
        features, alpha = self.stock_arrays[sid]

        x = features[start : start + self.seq_len].copy()
        y = np.cumsum(alpha[start + self.seq_len : start + self.seq_len + self.pred_days]).copy()

        return torch.from_numpy(x), torch.from_numpy(y)


def _prepare_stock_arrays(df: pd.DataFrame,
                          feature_cols: list[str],
                          seq_len: int,
                          pred_days: int,
                          stride: int = 1) -> tuple:
    """
    預計算每支股票的標準化特徵 + Alpha 序列 (記憶體最佳化版)

    Args:
        stride: 窗口步幅。stride=10 表示每 10 天取一個樣本 (減少 90% 重複)

    Returns:
        stock_ids, stock_arrays, idx_stock, idx_start
    """
    total_window = seq_len + pred_days
    stock_arrays = {}
    stock_ids = []

    # 暫存 indices，最後轉 numpy
    tmp_stock_idx = []
    tmp_start_idx = []

    # 排序一次就好 (inplace 省記憶體)
    df.sort_values(['stock_id', 'Date'], inplace=True)
    grouped = df.groupby('stock_id')

    for stock_id, group in tqdm(grouped, desc="📦 預處理股票", unit="stock"):
        if len(group) < total_window:
            continue

        features = group[feature_cols].values  # 已經是 float32 (從外面轉好的)

        # Per-stock Z-Score
        f_mean = features.mean(axis=0)
        f_std = features.std(axis=0) + 1e-8
        features = ((features - f_mean) / f_std).astype(np.float32)

        returns = group['Return_1d'].values.astype(np.float32)
        twii_returns = group['TWII_Return_1d'].values.astype(np.float32)
        alpha_daily = returns - twii_returns

        sid_idx = len(stock_ids)
        stock_ids.append(str(stock_id))
        stock_arrays[str(stock_id)] = (features, alpha_daily)

        n_windows = len(group) - total_window + 1
        window_starts = list(range(0, n_windows, stride))  # 步幅跳窗
        tmp_stock_idx.extend([sid_idx] * len(window_starts))
        tmp_start_idx.extend(window_starts)

    # 轉為 numpy (比 list of tuples 省 95% 記憶體)
    idx_stock = np.array(tmp_stock_idx, dtype=np.int16)
    idx_start = np.array(tmp_start_idx, dtype=np.int16)

    del tmp_stock_idx, tmp_start_idx
    gc.collect()

    print(f"   ✅ {len(stock_ids)} 支股票, {len(idx_stock):,} 個樣本 (stride={stride})")
    print(f"   📏 indices 記憶體: {(idx_stock.nbytes + idx_start.nbytes) / 1024**2:.1f} MB")

    return stock_ids, stock_arrays, idx_stock, idx_start


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
    window_stride: int = None,
) -> tuple:
    """
    完整訓練管線 (記憶體最佳化版)

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
    window_stride = window_stride or cfg.get('window_stride', 10)

    if use_v5_compat:
        feature_cols = FEATURE_COLS_V5
    else:
        feature_cols = feature_cols or FEATURE_COLS

    _set_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = (device.type == 'cuda')
    n_workers = min(2, os.cpu_count() or 1)  # T4 用 2 就夠，避免 fork 爆 RAM

    print(f"🔧 訓練設定:")
    print(f"   特徵維度: {len(feature_cols)}")
    print(f"   Epochs: {epochs}, Batch: {batch_size}, LR: {lr}")
    print(f"   Early Stop: {early_stop_patience} epochs")
    print(f"   AMP: {'✅' if use_amp else '❌'}, Workers: {n_workers}, Stride: {window_stride}")

    # === 確保欄位存在 + 轉 float32 省 RAM ===
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    # 提前轉 float32 (省 50% RAM)
    float_cols = [c for c in df.columns if df[c].dtype == np.float64]
    if float_cols:
        df[float_cols] = df[float_cols].astype(np.float32)
        print(f"   📉 已將 {len(float_cols)} 欄轉為 float32 (省 RAM)")

    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # === 建構 Dataset (快取機制) ===
    cache_path = os.path.join(PROCESSED_DIR, 'V55_train_cache.pkl')

    stock_ids = None

    if os.path.exists(cache_path):
        try:
            print("⚡ 偵測到訓練快取，直接載入...")
            with open(cache_path, 'rb') as f:
                cached = pickle.load(f)
            stock_ids = cached['stock_ids']
            stock_arrays = cached['stock_arrays']
            idx_stock = cached['idx_stock']
            idx_start = cached['idx_start']
            print(f"   ✅ {len(stock_ids)} 支股票, {len(idx_stock):,} 個樣本")
        except Exception as e:
            print(f"   ⚠️ 快取載入失敗 ({e})，重新建構...")
            stock_ids = None

    if stock_ids is None:
        print("📦 建構訓練資料集...")
        stock_ids, stock_arrays, idx_stock, idx_start = _prepare_stock_arrays(
            df, feature_cols,
            seq_len=MODEL_CONFIG['seq_len'],
            pred_days=MODEL_CONFIG['pred_days'],
            stride=window_stride,
        )

        # 儲存快取
        try:
            os.makedirs(PROCESSED_DIR, exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'stock_ids': stock_ids,
                    'stock_arrays': stock_arrays,
                    'idx_stock': idx_stock,
                    'idx_start': idx_start,
                }, f)
            cache_mb = os.path.getsize(cache_path) / 1024**2
            print(f"   💾 快取已儲存: {cache_mb:.0f} MB (下次秒讀)")
        except Exception as e:
            print(f"   ⚠️ 快取儲存失敗: {e}")

    # 釋放 df 記憶體 (不再需要)
    del df
    gc.collect()

    if len(idx_stock) == 0:
        raise ValueError("❌ 資料集為空！請檢查資料是否足夠")

    # Train/Val 切分 (用 numpy 操作，不建新 list)
    n_total = len(idx_stock)
    val_size = int(n_total * val_ratio)
    train_size = n_total - val_size

    perm = np.random.RandomState(seed).permutation(n_total)
    train_perm = perm[:train_size]
    val_perm = perm[train_size:]

    print(f"   訓練集: {train_size:,} 樣本, 驗證集: {val_size:,} 樣本")

    train_set = FastAlphaDataset(
        stock_ids, stock_arrays,
        MODEL_CONFIG['seq_len'], MODEL_CONFIG['pred_days'],
        idx_stock[train_perm], idx_start[train_perm],
    )
    val_set = FastAlphaDataset(
        stock_ids, stock_arrays,
        MODEL_CONFIG['seq_len'], MODEL_CONFIG['pred_days'],
        idx_stock[val_perm], idx_start[val_perm],
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=n_workers, pin_memory=True,
        persistent_workers=(n_workers > 0),
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size * 2, shuffle=False,
        num_workers=n_workers, pin_memory=True,
        persistent_workers=(n_workers > 0),
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
