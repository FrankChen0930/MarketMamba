"""
MarketMamba V5.5 — 訓練模組 (A100 優化版)
負責：DataLoader 建構、訓練迴圈、Early Stopping、模型儲存

優化：
  - AMP 混合精度訓練 (FP16)：A100 Tensor Core 加速 ~2-3x
  - Z-Score 預計算：不再每次 __getitem__ 重算
  - cuDNN benchmark：自動尋找最快的卷積算法
  - DataLoader num_workers 自動偵測
  - 預分配 GPU Tensor
"""

import os
import random
import logging
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.amp import autocast, GradScaler
from tqdm.auto import tqdm

from marketmamba.config import (
    FEATURE_COLS, FEATURE_COLS_V5, MODEL_CONFIG, TRAIN_CONFIG, MODEL_DIR,
)

logger = logging.getLogger('MarketMamba.trainer')


# ==========================================
# 建構預計算 Tensor (所有標準化在這裡一次做完)
# ==========================================
def _build_tensors(df: pd.DataFrame,
                   feature_cols: list[str],
                   seq_len: int,
                   pred_days: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    預計算所有 (x, y) 張量，Z-Score 在此一次完成
    回傳放在 CPU 的 float32 Tensor，訓練時直接 .to(device)

    Returns:
        (all_x, all_y): shapes (N, seq_len, features), (N, pred_days)
    """
    total_window = seq_len + pred_days
    all_x = []
    all_y = []

    grouped = df.sort_values(['stock_id', 'Date']).groupby('stock_id')
    stock_count = 0

    for stock_id, group in tqdm(grouped, desc="📦 建構張量", unit="stock"):
        if len(group) < total_window:
            continue

        features = group[feature_cols].values.astype(np.float32)
        returns = group['Return_1d'].values
        twii_returns = group['TWII_Return_1d'].values
        alpha_daily = (returns - twii_returns).astype(np.float32)

        n_windows = len(group) - total_window + 1
        stock_count += 1

        for i in range(n_windows):
            x = features[i : i + seq_len].copy()

            # Z-Score 標準化 (預計算，不在每次 __getitem__)
            x_mean = x.mean(axis=0)
            x_std = x.std(axis=0) + 1e-8
            x = (x - x_mean) / x_std

            y = np.cumsum(alpha_daily[i + seq_len : i + seq_len + pred_days])

            all_x.append(x)
            all_y.append(y)

    print(f"   股票數: {stock_count}, 樣本數: {len(all_x):,}")

    # 轉為連續 tensor
    x_tensor = torch.from_numpy(np.stack(all_x)).float()
    y_tensor = torch.from_numpy(np.stack(all_y)).float()

    # 清除 NaN/Inf
    x_tensor = torch.nan_to_num(x_tensor, nan=0.0, posinf=0.0, neginf=0.0)
    y_tensor = torch.nan_to_num(y_tensor, nan=0.0, posinf=0.0, neginf=0.0)

    return x_tensor, y_tensor


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
    完整訓練管線 (A100 優化版)

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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = (device.type == 'cuda')

    print(f"🔧 訓練設定:")
    print(f"   特徵維度: {len(feature_cols)}")
    print(f"   Epochs: {epochs}, Batch: {batch_size}, LR: {lr}")
    print(f"   Early Stop: {early_stop_patience} epochs")
    print(f"   AMP: {'✅ 啟用' if use_amp else '❌ 停用'}")

    # === 確保欄位存在 ===
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # === 建構預計算張量 ===
    print("📦 建構訓練資料集...")
    x_all, y_all = _build_tensors(
        df, feature_cols,
        seq_len=MODEL_CONFIG['seq_len'],
        pred_days=MODEL_CONFIG['pred_days'],
    )

    if len(x_all) == 0:
        raise ValueError("❌ 資料集為空！請檢查資料是否足夠")

    # Train/Val 切分
    n_total = len(x_all)
    val_size = int(n_total * val_ratio)
    train_size = n_total - val_size

    # 隨機打亂 index
    perm = torch.randperm(n_total, generator=torch.Generator().manual_seed(seed))
    train_idx = perm[:train_size]
    val_idx = perm[train_size:]

    x_train, y_train = x_all[train_idx], y_all[train_idx]
    x_val, y_val = x_all[val_idx], y_all[val_idx]

    # 釋放完整資料
    del x_all, y_all

    print(f"   訓練集: {train_size:,} 樣本, 驗證集: {val_size:,} 樣本")

    # DataLoader (num_workers=0 因為資料已經是 Tensor，不需要額外 CPU 處理)
    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        TensorDataset(x_val, y_val),
        batch_size=batch_size * 2,  # 驗證可以用更大 batch
        shuffle=False,
        num_workers=0, pin_memory=True,
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

    # === 優化器與排程器 ===
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    # AMP Scaler
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
    print("=" * 60)

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        train_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]",
                     leave=False)

        for x_batch, y_batch in pbar:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)  # 比 zero_grad() 更快

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
                x_batch = x_batch.to(device, non_blocking=True)
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
        torch.backends.cudnn.deterministic = False  # 不鎖定 cuDNN (A100 加速)
        torch.backends.cudnn.benchmark = True        # 自動尋找最快的卷積算法
