"""
MarketMamba V6 — Daily Inference Pipeline
==========================================
Main entry point for local daily inference on RTX 3060 + WSL2.

Execution flow:
  [17:00] Step 1 — Hybrid data update (yfinance + TWSE direct)
  [17:05] Step 2 — Feature engineering for today's cross-section
  [17:10] Step 3 — Mamba+GAT inference → df_kelly.csv, df_traj.csv
  [17:15] Step 4 — LLM report generation → market_summary.json
  [17:20] Step 5 — Push results to GitHub → Streamlit auto-refresh

Run via Windows Task Scheduler:
  wsl -d Ubuntu -e bash -c "cd /mnt/d/Desktop/work/MarketMamba && conda run -n mamba_env python V6/run_daily_inference.py"
"""

from __future__ import annotations

import argparse
import json
import logging
import queue as _queue_mod
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import ttk
    from tkinter import font as tkfont
    _TK_AVAILABLE = True
except ImportError:
    _TK_AVAILABLE = False

# Make package importable when run as __main__
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import torch

from marketmamba.config import (
    FEATURE_COLS,
    KG_CACHE_PATH,
    LLM_REPORT_PATH,
    MODELS_DIR,
    PROCESSED_DIR,
    RESULTS_DIR,
    SEQ_LEN,
)
from marketmamba.data.fetcher import run_daily_update, load_ticker_universe
from marketmamba.data.feature_engineer import build_features, clean_and_scale
from marketmamba.llm.report_generator import generate_market_report, build_market_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("V6.Inference")


# ============================================================
# 進度視窗（tkinter）
# ============================================================

_BG      = "#F4F6F8"   # 淺灰背景
_BG_CARD = "#FFFFFF"   # 白色卡片
_FG      = "#1C1C2E"   # 深色文字
_FG_DIM  = "#8A8FA8"   # 灰色輔助文字
_ACCENT  = "#3B6FE8"   # 藍色強調
_SEP     = "#E8EAF0"   # 分隔線
_COLORS  = {
    "pending": "#B0B8C8",
    "running": "#3B6FE8",
    "done":    "#22A96A",
    "failed":  "#E53935",
    "skipped": "#F59E0B",
}
_ICONS = {
    "pending": "○", "running": "◉",
    "done":    "✓", "failed":  "✗", "skipped": "—",
}
_STEPS_ZH = [
    "資料更新",
    "特徵矩陣建構",
    "模型推論 (Mamba+GAT)",
    "LLM 市場報告",
    "歸檔",
    "信號掃描",
    "模擬回測",
    "推送 GitHub",
]


def _fmt_time(secs: float) -> str:
    m, s = divmod(int(secs), 60)
    return f"{m}m {s:02d}s" if m else f"{s}s"


_ui_queue: "_queue_mod.Queue | None" = None


def _step_update(idx: int, status: str, note: str = "") -> None:
    """Push a step state change to the UI queue (no-op when UI is not running)."""
    if _ui_queue is not None:
        _ui_queue.put(("step", idx, status, note))


def _ui_set_info(date_str: str, device_str: str) -> None:
    if _ui_queue is not None:
        _ui_queue.put(("info", date_str, device_str))


class _UiLogHandler(logging.Handler):
    """將 log 導向 UI queue（僅在 UI 啟動時有效）。"""
    def emit(self, record: logging.LogRecord) -> None:
        if _ui_queue is not None:
            _ui_queue.put(("log", self.format(record)))


class ProgressWindow:
    """
    每日推論進度視窗。
    - 成功：3 秒倒數後自動關閉
    - 失敗：視窗保持開啟、跳到最前、失敗步驟整行標紅
    """

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("MarketMamba V6  每日推論")
        self.root.configure(bg=_BG)
        self.root.resizable(False, False)

        global _ui_queue
        self._q: _queue_mod.Queue = _queue_mod.Queue()
        _ui_queue = self._q

        self._step_status:  list = ["pending"] * len(_STEPS_ZH)
        self._step_t_start: list = [None]      * len(_STEPS_ZH)
        self._pipe_start          = None
        self._finished: bool      = False
        self._has_error: bool     = False

        self._build()

        # 置中顯示
        self.root.update_idletasks()
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        ww = self.root.winfo_reqwidth()
        wh = self.root.winfo_reqheight()
        self.root.geometry(f"{ww}x{wh}+{(sw - ww) // 2}+{(sh - wh) // 2}")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── 介面建構 ──────────────────────────────────────────────────────────
    def _build(self) -> None:
        root  = self.root
        PAD   = 24   # 外邊距

        # 自動選字體：CJK 字型優先，確保中文顯示正常
        _CJK_FONTS = [
            "Noto Sans CJK TC",
            "Noto Sans CJK SC",
            "WenQuanYi Zen Hei",
            "WenQuanYi Micro Hei",
            "Ubuntu",
            "DejaVu Sans",
        ]
        fam = "TkDefaultFont"
        for _f in _CJK_FONTS:
            try:
                _actual = tkfont.Font(family=_f, size=10).actual()["family"]
                _key = _f.split()[0].lower()  # "noto", "wenquanyi", "ubuntu", "dejavu"
                if _key in _actual.lower():
                    fam = _f
                    break
            except Exception:
                continue

        f_h1     = tkfont.Font(family=fam, size=15, weight="bold")
        f_meta   = tkfont.Font(family=fam, size=10)
        f_step   = tkfont.Font(family=fam, size=12)
        f_icon   = tkfont.Font(family=fam, size=13, weight="bold")
        f_time   = tkfont.Font(family=fam, size=11)
        f_status = tkfont.Font(family=fam, size=11)

        # ── 頂部白色標題卡 ──
        top_card = tk.Frame(root, bg=_BG_CARD)
        top_card.pack(fill="x")
        inner = tk.Frame(top_card, bg=_BG_CARD)
        inner.pack(fill="x", padx=PAD, pady=(18, 14))
        tk.Label(inner, text="MarketMamba V6", font=f_h1,
                 bg=_BG_CARD, fg=_ACCENT).pack(anchor="w")
        self._lbl_meta = tk.Label(inner, text="", font=f_meta, bg=_BG_CARD, fg=_FG_DIM)
        self._lbl_meta.pack(anchor="w", pady=(3, 0))

        # 分隔線
        tk.Frame(root, bg=_SEP, height=1).pack(fill="x")

        # ── 步驟列表（白色卡片）──
        steps_card = tk.Frame(root, bg=_BG_CARD)
        steps_card.pack(fill="x")

        self._step_row:      list[tk.Frame] = []
        self._step_icon_lbl: list[tk.Label] = []
        self._step_name_lbl: list[tk.Label] = []
        self._step_time_lbl: list[tk.Label] = []

        for i, name in enumerate(_STEPS_ZH):
            row = tk.Frame(steps_card, bg=_BG_CARD)
            row.pack(fill="x", padx=PAD, pady=(6, 0))
            self._step_row.append(row)

            icon = tk.Label(row, text=_ICONS["pending"], font=f_icon,
                            bg=_BG_CARD, fg=_COLORS["pending"], width=2, anchor="w")
            icon.pack(side="left")
            self._step_icon_lbl.append(icon)

            name_lbl = tk.Label(row, text=name, font=f_step,
                                bg=_BG_CARD, fg=_FG, anchor="w")
            name_lbl.pack(side="left", padx=(8, 0))
            self._step_name_lbl.append(name_lbl)

            time_lbl = tk.Label(row, text="—", font=f_time,
                                bg=_BG_CARD, fg=_FG_DIM, anchor="e")
            time_lbl.pack(side="right")
            self._step_time_lbl.append(time_lbl)

        tk.Frame(steps_card, bg=_BG_CARD, height=12).pack()   # 底部留白

        # 分隔線
        tk.Frame(root, bg=_SEP, height=1).pack(fill="x")

        # ── 細進度條（Canvas，避免 ttk 主題問題）──
        self._pb_canvas = tk.Canvas(root, height=5, bg=_BG,
                                     highlightthickness=0)
        self._pb_canvas.pack(fill="x")
        self._pb_canvas.bind("<Configure>", self._redraw_pb)
        self._pb_pct = 0.0

        # ── 狀態 / 計時 ──
        foot = tk.Frame(root, bg=_BG)
        foot.pack(fill="x", padx=PAD, pady=(10, 20))
        self._lbl_status = tk.Label(foot, text="準備中…", font=f_status,
                                     bg=_BG, fg=_ACCENT, anchor="w")
        self._lbl_status.pack(side="left")
        self._lbl_total = tk.Label(foot, text="", font=f_status,
                                    bg=_BG, fg=_FG_DIM, anchor="e")
        self._lbl_total.pack(side="right")

    def _redraw_pb(self, _event=None) -> None:
        c = self._pb_canvas
        w, h = c.winfo_width(), 5
        c.delete("all")
        c.create_rectangle(0, 0, w, h, fill=_SEP, outline="")
        if self._pb_pct > 0:
            c.create_rectangle(0, 0, max(h, int(w * self._pb_pct / 100)), h,
                                fill=_ACCENT, outline="")

    # ── 執行緒入口 ────────────────────────────────────────────────────────
    def run_with(self, fn, *args, **kwargs) -> None:
        def _worker():
            try:
                fn(*args, **kwargs)
            except Exception as exc:
                self._q.put(("fatal", str(exc)))
            finally:
                self._q.put(("finished",))

        threading.Thread(target=_worker, daemon=True).start()
        self.root.after(50, self._poll)
        self.root.mainloop()

    # ── 內部更新 ──────────────────────────────────────────────────────────
    def _poll(self) -> None:
        try:
            while True:
                self._dispatch(self._q.get_nowait())
        except _queue_mod.Empty:
            pass
        self._tick()
        if not self._finished:
            self.root.after(200, self._poll)

    def _dispatch(self, msg: tuple) -> None:
        kind = msg[0]
        if kind == "info":
            _, date_str, device_str = msg
            self._lbl_meta.config(text=f"{date_str}  ·  {device_str}")
            self._pipe_start = time.monotonic()
            self._lbl_status.config(text="推論執行中…", fg=_ACCENT)
        elif kind == "step":
            _, idx, status, note = msg
            self._apply_step(idx, status)
        elif kind == "log":
            pass   # log 只寫 inference.log，不佔用視窗空間
        elif kind == "fatal":
            _, err = msg
            self._has_error = True
            self._lbl_status.config(text=f"錯誤：{err[:72]}", fg=_COLORS["failed"])
        elif kind == "finished":
            self._finished = True
            self._tick()
            n_fail = sum(1 for s in self._step_status if s == "failed")
            total  = time.monotonic() - (self._pipe_start or time.monotonic())
            if n_fail == 0 and not self._has_error:
                # ✅ 成功 → 倒數 3 秒自動關閉
                self._countdown(3, total)
            else:
                # ❌ 失敗 → 保持視窗、拉到最前
                self._lbl_status.config(
                    text="推論發生錯誤，請查閱 V6/logs/inference.log",
                    fg=_COLORS["failed"])
                self._lbl_total.config(text=f"總耗時 {_fmt_time(total)}", fg=_FG_DIM)
                self.root.title("❌ MarketMamba — 推論失敗")
                self.root.lift()
                self.root.focus_force()
                self.root.attributes("-topmost", True)

    def _countdown(self, n: int, total: float) -> None:
        if not self.root.winfo_exists():
            return
        if n <= 0:
            global _ui_queue
            _ui_queue = None
            try:
                self.root.destroy()
            except Exception:
                pass
            return
        self._lbl_status.config(
            text=f"全部完成  （{n} 秒後關閉）",
            fg=_COLORS["done"])
        self._lbl_total.config(text=f"總耗時 {_fmt_time(total)}", fg=_FG_DIM)
        self.root.after(1000, self._countdown, n - 1, total)

    def _apply_step(self, idx: int, status: str) -> None:
        self._step_status[idx] = status
        if status == "running":
            self._step_t_start[idx] = time.monotonic()
        color = _COLORS[status]
        self._step_icon_lbl[idx].config(text=_ICONS[status], fg=color)
        if status in ("done", "failed", "skipped") and self._step_t_start[idx]:
            elapsed = _fmt_time(time.monotonic() - self._step_t_start[idx])
            tc = _FG_DIM if status == "done" else color
            self._step_time_lbl[idx].config(text=elapsed, fg=tc)
        if status == "failed":
            # 整行背景變紅
            ERR_BG = "#FFF0F0"
            for w in [self._step_row[idx], self._step_icon_lbl[idx],
                      self._step_name_lbl[idx], self._step_time_lbl[idx]]:
                w.configure(bg=ERR_BG)
        # 進度條
        done = sum(1 for s in self._step_status if s in ("done", "failed", "skipped"))
        self._pb_pct = done / len(_STEPS_ZH) * 100
        self._redraw_pb()

    def _tick(self) -> None:
        now = time.monotonic()
        for i, (status, t0) in enumerate(zip(self._step_status, self._step_t_start)):
            if status == "running" and t0 is not None:
                self._step_time_lbl[i].config(
                    text=_fmt_time(now - t0) + "…", fg=_COLORS["running"])
        if self._pipe_start and not self._finished:
            self._lbl_total.config(text=f"總耗時 {_fmt_time(now - self._pipe_start)}")

    def _on_close(self) -> None:
        if not self._finished:
            self.root.iconify()   # 推論仍在執行中，最小化
        else:
            global _ui_queue
            _ui_queue = None
            self.root.destroy()


# ============================================================
# Inference Core
# ============================================================

def run_inference(
    df:        pd.DataFrame,
    model_path: Path | None = None,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run V6 cross-sectional inference on the most recent trading day.

    Args:
        df         : feature DataFrame (all history, latest day will be used)
        model_path : .pt checkpoint; defaults to MODELS_DIR/v6_best.pt
        device_str : 'cuda' or 'cpu'

    Returns:
        df_kelly   : stock ranking with Alpha, Sharpe, Kelly weights
        df_traj    : multi-horizon predicted trajectories (5d, 20d, 60d)
    """
    from marketmamba.models.architecture import MarketMambaV6
    from marketmamba.models.trainer import TemporalCrossSectionDataset, load_kg_edges

    device = torch.device(device_str)

    # -- Load checkpoint --
    if model_path is None:
        model_path = MODELS_DIR / "v6_best.pt"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}\n"
            "Please train V6 first using V6/notebooks/V6_Training.py on Colab."
        )

    from marketmamba.models.trainer import TrainingHistory
    torch.serialization.add_safe_globals([TrainingHistory])
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model = MarketMambaV6()
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    logger.info(
        f"Loaded checkpoint: {model_path.name} "
        f"(epoch={ckpt.get('epoch','?')}, val_loss={ckpt.get('val_loss', 'N/A'):.5f})"
    )

    # -- Latest date's cross-section --
    df["Date"] = pd.to_datetime(df["Date"])
    latest_date = df["Date"].max()
    latest_str  = latest_date.strftime("%Y-%m-%d")
    logger.info(f"Inference date: {latest_str}")

    # Build one-day dataset
    test_ds = TemporalCrossSectionDataset(
        df, [latest_str], mode="test", n_sample=None  # always use full cross-section
    )
    if len(test_ds) == 0:
        raise ValueError(f"No valid cross-section found for {latest_str}")

    # -- Load KG edges --
    edge_index, edge_attr = load_kg_edges(df["stock_id"].unique().tolist(), device)

    # -- Inference with MC-Dropout uncertainty (mini-batch to fit 6 GB VRAM) --
    X, _, valid_stocks = test_ds[0]   # __getitem__ returns (X, Y, stock_ids)

    N = X.shape[0]
    INFER_BATCH = 128   # stocks per GPU step (tune down to 64 if still OOM)
    N_MC        = 30    # MC-Dropout samples

    pred_mean_acc = torch.zeros(N, 3)
    pred_std_acc  = torch.zeros(N, 3)

    model.train()   # enable dropout for MC
    with torch.no_grad():
        for batch_start in range(0, N, INFER_BATCH):
            x_b = X[batch_start: batch_start + INFER_BATCH].to(device)
            mc_preds = []
            for _mc in range(N_MC):
                p = model(x_b, edge_index, edge_attr)   # (B, 3)
                mc_preds.append(p.cpu())
            mc_stack = torch.stack(mc_preds, dim=0)      # (N_MC, B, 3)
            pred_mean_acc[batch_start: batch_start + INFER_BATCH] = mc_stack.mean(0)
            pred_std_acc [batch_start: batch_start + INFER_BATCH] = mc_stack.std(0)
    model.eval()

    pred_mean = pred_mean_acc.numpy()   # (N, 3)
    pred_std  = pred_std_acc.numpy()    # (N, 3)

    # Use valid_stocks from dataset (correctly ordered to match X rows)
    stocks_today = valid_stocks[:len(pred_mean)]


    # -- Build df_kelly --
    df_kelly = pd.DataFrame({
        "Ticker":        stocks_today,
        "Date":          latest_str,
        "Exp_Alpha_5d":  pred_mean[:, 0],
        "Exp_Alpha_20d": pred_mean[:, 1],   # primary
        "Exp_Alpha_60d": pred_mean[:, 2],
        "Uncertainty":   pred_std[:, 1],    # 20d uncertainty
    })

    df_kelly["Uncertainty"]   = df_kelly["Uncertainty"].clip(0.0, None)

    # Liquidity filter — use RAW prices (Volume/Close are z-scored in feature matrix)
    mask = df["Date"] == latest_date
    try:
        _raw = pd.read_parquet(
            PROCESSED_DIR / "prices_raw.parquet",
            columns=["Date", "stock_id", "Volume", "Close"],
        )
        _raw["Date"] = pd.to_datetime(_raw["Date"])
        df_today = (_raw[_raw["Date"] == latest_date]
                    .drop_duplicates("stock_id", keep="last")[["stock_id", "Volume", "Close"]])
    except Exception:
        # Fallback: use z-scored values (threshold must be relaxed to 0)
        df_today = (df[mask][["stock_id", "Volume", "Close"]]
                    .drop_duplicates("stock_id", keep="last").copy())
    df_kelly = df_kelly.merge(
        df_today.rename(columns={"stock_id": "Ticker"}),
        on="Ticker", how="left",
    )
    df_kelly["Turnover_5D"] = df_kelly["Volume"].fillna(0) * df_kelly["Close"].fillna(0)

    # Hard liquidity filter
    MIN_TURNOVER = 1e7   # 1000萬台幣
    low_liq_mask = df_kelly["Turnover_5D"] < MIN_TURNOVER
    n_filtered = int(low_liq_mask.sum())
    if n_filtered > 0:
        logger.info(f"Liquidity filter: removed {n_filtered} illiquid stocks "
                    f"out of {len(df_kelly)}")
    df_kelly.loc[low_liq_mask, "Exp_Alpha_20d"] = -999.0

    # Slippage penalty (rough estimate: 0.4% for small-mid cap)
    df_kelly["Slippage_Est"] = 0.004
    df_kelly["Net_Alpha_20d"] = (
        df_kelly["Exp_Alpha_20d"] - df_kelly["Slippage_Est"]
    ).clip(lower=-1.0)

    # Signal Quality = Net Alpha / Uncertainty (proxy for risk-adjusted rank)
    df_kelly["Signal_Quality"] = (
        df_kelly["Net_Alpha_20d"] / (df_kelly["Uncertainty"] + 1e-6)
    ).clip(lower=-10.0, upper=10.0)

    # Confidence label
    df_kelly["Confidence"] = pd.cut(
        df_kelly["Uncertainty"],
        bins=[0, 0.02, 0.05, 1.0],
        labels=["高信心", "中信心", "低信心"],
        right=False,
    )

    # Kelly weight (simplified, proportional to Signal Quality clipped at 0)
    positive = df_kelly["Signal_Quality"].clip(lower=0)
    total = positive.sum()
    df_kelly["Suggested_Weight"] = (positive / (total + 1e-9)).round(4)

    # Sort by Sharpe — exclude penalised stocks from Top 10 display
    df_kelly = df_kelly.sort_values("Signal_Quality", ascending=False).reset_index(drop=True)


    # -- Build df_traj: multi-horizon trajectory --
    df_traj = pd.DataFrame({
        "Ticker":        stocks_today,
        "Date":          latest_str,
        "Pred_5d":       pred_mean[:, 0],
        "Pred_20d":      pred_mean[:, 1],
        "Pred_60d":      pred_mean[:, 2],
        "Uncertainty_5d":  pred_std[:, 0],
        "Uncertainty_20d": pred_std[:, 1],
        "Uncertainty_60d": pred_std[:, 2],
    })

    return df_kelly, df_traj


# ============================================================
# Main Pipeline
# ============================================================

def main(target_date: str | None = None, skip_push: bool = False, forward_fill: bool = False) -> None:
    # ── 時間感知日期選擇 ──────────────────────────────────────────────────────
    # 台股 13:30 收盤，14:00 前執行時使用昨日日期
    if target_date is not None:
        today = target_date
    else:
        from zoneinfo import ZoneInfo
        from datetime import timedelta
        now_twn = datetime.now(ZoneInfo("Asia/Taipei"))
        if now_twn.hour < 14:
            today = (now_twn - timedelta(days=1)).strftime("%Y-%m-%d")
            logger.info(
                f"⏰ 台灣時間 {now_twn.strftime('%H:%M')} — 尚未收盤，使用昨日日期：{today}"
            )
        else:
            today = now_twn.strftime("%Y-%m-%d")

    logger.info(f"\n{'='*55}")
    logger.info(f"  MarketMamba V6 — 每日推論  [{today}]")
    logger.info(f"{'='*55}")

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"  裝置：{device_str}")
    gpu_name = ""
    if device_str == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"  GPU：{gpu_name}")

    pipeline_start = time.monotonic()
    _ui_set_info(today, f"{device_str}" + (f" ({gpu_name})" if gpu_name else ""))

    _fmt = _fmt_time   # 本地別名

    try:
        # ── 步驟 1：資料更新 ─────────────────────────────────────────────────
        t0 = time.monotonic()
        logger.info(f"\n{'─'*50}")
        logger.info(f"[1/8] 資料更新  [{datetime.now().strftime('%H:%M:%S')}]")
        logger.info(f"{'─'*50}")
        _step_update(0, "running")

        _MAX_FETCH_ATTEMPTS = 2
        _RETRY_WAIT_SEC     = 15 * 60   # 15 分鐘
        freshness: dict = {"missing": [], "forward_filled": []}

        for _fetch_attempt in range(_MAX_FETCH_ATTEMPTS):
            try:
                freshness = run_daily_update(
                    target_date=today,
                    allow_forward_fill=forward_fill,
                )
            except Exception as e:
                logger.error(f"資料更新失敗：{e}")
                _step_update(0, "failed", str(e)[:60])
                raise

            _missing = freshness.get("missing", [])
            if not _missing:
                break   # 所有來源都有今日資料

            if _fetch_attempt < _MAX_FETCH_ATTEMPTS - 1:
                _wait_msg = (
                    f"缺少 {'、'.join(_missing)}，"
                    f"{_RETRY_WAIT_SEC // 60} 分後重試…"
                )
                logger.warning(_wait_msg)
                _step_update(0, "running", _wait_msg)
                time.sleep(_RETRY_WAIT_SEC)
                logger.info(f"[1/8] 重試資料取得（第 {_fetch_attempt + 2} 次）…")
            else:
                _missing_str = "、".join(_missing)
                _err = (
                    f"資料不完整（已重試 {_MAX_FETCH_ATTEMPTS} 次）：缺少 {_missing_str}\n"
                    f"建議：(1) 再等 30 分鐘後重跑  "
                    f"(2) 加 --forward-fill 使用前日資料補齊後立即推論"
                )
                logger.error(_err)
                _step_update(0, "failed", f"缺少 {_missing_str}")
                raise RuntimeError(_err)

        elapsed = time.monotonic() - t0
        _fwd = freshness.get("forward_filled", [])
        _done_note = f"Forward-Fill：{'、'.join(_fwd)}" if _fwd else ""
        logger.info(f"[1/8] ✓ 完成 ({_fmt(elapsed)})" + (f" — {_done_note}" if _done_note else ""))
        _step_update(0, "done", _done_note)

        # ── 步驟 2：特徵矩陣建構 ─────────────────────────────────────────────
        t0 = time.monotonic()
        logger.info(f"\n{'─'*50}")
        logger.info(f"[2/7] 特徵矩陣建構  [{datetime.now().strftime('%H:%M:%S')}]")
        logger.info(f"{'─'*50}")
        _step_update(1, "running")

        def _read(name: str):
            path = PROCESSED_DIR / name
            return pd.read_parquet(path) if path.exists() else None

        # V6.0 核心數據
        prices       = _read("prices_raw.parquet")
        inst         = _read("institutional_raw.parquet")
        margin       = _read("margin_raw.parquet")
        per          = _read("per_raw.parquet")
        market_value = _read("market_value_raw.parquet")
        daytrade     = _read("daytrade_raw.parquet")
        revenue      = _read("revenue_raw.parquet")
        financials   = _read("financials_raw.parquet")
        balance_sheet = _read("balance_sheet_raw.parquet")
        macro        = _read("macro_raw.parquet")

        # V6.1 新增數據
        securities   = _read("securities_raw.parquet")
        holdings     = _read("holdings_raw.parquet")
        cashflow     = _read("cashflow_raw.parquet")
        dividend     = _read("dividend_raw.parquet")
        foreign_shareholding = _read("foreign_shareholding_raw.parquet")
        fear_greed   = _read("fear_greed.parquet")
        business_indicator = _read("business_indicator.parquet")
        fed_rate     = _read("fed_rate.parquet")
        futures_inst = _read("futures_institutional_raw.parquet")
        options_inst = _read("options_institutional_raw.parquet")

        if prices is None:
            raise FileNotFoundError(f"prices_raw.parquet 不存在：{PROCESSED_DIR}")

        # 只保留近 2 年，避免 OOM（覆蓋所有滾動窗口 MA_60 / ATR_14 / SEQ_LEN=60）
        INFERENCE_LOOKBACK_DAYS = 730
        prices["Date"] = pd.to_datetime(prices["Date"])
        cutoff = prices["Date"].max() - pd.Timedelta(days=INFERENCE_LOOKBACK_DAYS)
        prices = prices[prices["Date"] >= cutoff].copy()
        logger.info(f"股價資料截取近 2 年：{len(prices):,} 筆 "
                    f"({prices['Date'].min().date()} → {prices['Date'].max().date()})")

        def _trim(df_src):
            if df_src is None: return None
            if "Date" in df_src.columns:
                df_src["Date"] = pd.to_datetime(df_src["Date"])
                return df_src[df_src["Date"] >= cutoff].copy()
            return df_src

        inst     = _trim(inst)
        margin   = _trim(margin)
        daytrade = _trim(daytrade)
        holdings = _trim(holdings)
        securities = _trim(securities)
        foreign_shareholding = _trim(foreign_shareholding)

        df = build_features(
            df_price         = prices,
            df_inst          = inst,
            df_margin        = margin,
            df_per           = per,
            df_securities    = securities,
            df_market_value  = market_value,
            df_daytrade      = daytrade,
            df_holdings      = holdings,
            df_rev           = revenue,
            df_fin           = financials,
            df_balance_sheet = balance_sheet,
            df_cashflow      = cashflow,
            df_macro         = macro,
            df_futures_inst  = futures_inst,
            df_options_inst  = options_inst,
            df_dividend      = dividend,
            df_foreign_shareholding = foreign_shareholding,
            df_fear_greed    = fear_greed,
            df_business_indicator = business_indicator,
            df_fed_rate      = fed_rate,
        )
        df = clean_and_scale(df)
        # 去重：institutional_raw 長格式（4 rows/stock/date）→ 4 倍重複
        n_before = len(df)
        df = df.drop_duplicates(subset=["Date", "stock_id"], keep="last")
        if len(df) < n_before:
            logger.info(f"去重後：{n_before:,} → {len(df):,} 筆")

        # S2: Data freshness check — warn if prices >3 calendar days stale
        from zoneinfo import ZoneInfo
        _latest_price = pd.Timestamp(df["Date"].max()).date()
        _today_twn    = datetime.now(ZoneInfo("Asia/Taipei")).date()
        _stale_days   = (_today_twn - _latest_price).days
        _freshness_note = ""
        if _stale_days > 3:
            logger.warning(
                f"⚠️ 資料可能過舊：最新股價 {_latest_price}（{_stale_days} 天前），"
                "可能遇到連假或資料延遲"
            )
            _freshness_note = f" ⚠️ {_stale_days}天前"

        elapsed = time.monotonic() - t0
        logger.info(f"特徵矩陣：{df.shape}")
        logger.info(f"[2/7] ✓ 完成 ({_fmt(elapsed)})")
        _step_update(1, "done", f"{df.shape[0]:,} × {df.shape[1]} 特徵{_freshness_note}")

        # ── 步驟 3：模型推論 ─────────────────────────────────────────────────
        t0 = time.monotonic()
        logger.info(f"\n{'─'*50}")
        logger.info(f"[3/7] 模型推論  [{datetime.now().strftime('%H:%M:%S')}]")
        logger.info(f"{'─'*50}")
        _step_update(2, "running")
        df_kelly, df_traj = run_inference(df, device_str=device_str)

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        df_kelly.to_csv(RESULTS_DIR / "df_kelly.csv", index=False, encoding="utf-8-sig")
        df_traj.to_csv(RESULTS_DIR  / "df_traj.csv",  index=False, encoding="utf-8-sig")

        elapsed      = time.monotonic() - t0
        investable   = df_kelly[df_kelly["Exp_Alpha_20d"] > -999]
        n_investable = len(investable)
        top10        = investable.head(10)

        logger.info(f"\n🎯 Top 10 Alpha 股票 [{today}]：")
        print(
            top10[["Ticker", "Exp_Alpha_20d", "Signal_Quality", "Confidence", "Suggested_Weight"]]
            .to_string(index=False)
        )
        logger.info(f"可投資股票：{n_investable} / {len(df_kelly)}")
        logger.info(f"[3/7] ✓ 完成 ({_fmt(elapsed)})")

        top_ticker = top10.iloc[0]["Ticker"]       if len(top10) > 0 else "—"
        top_alpha  = top10.iloc[0]["Exp_Alpha_20d"] if len(top10) > 0 else 0.0
        _step_update(2, "done",
                     f"可投資 {n_investable}/{len(df_kelly)} · Top：{top_ticker} α={top_alpha:+.4f}")

        # ── 步驟 4：LLM 市場報告 ─────────────────────────────────────────────
        t0 = time.monotonic()
        logger.info(f"\n{'─'*50}")
        logger.info(f"[4/7] LLM 市場報告  [{datetime.now().strftime('%H:%M:%S')}]")
        logger.info(f"{'─'*50}")
        _step_update(3, "running")
        try:
            market_data = build_market_data()
            report = generate_market_report(df_kelly, market_data, save=True)
            elapsed = time.monotonic() - t0
            logger.info(f"\n📝 報告摘要：\n{report['summary'][:300]}...")
            logger.info(f"[4/7] ✓ 完成 ({_fmt(elapsed)})")
            _step_update(3, "done")
        except Exception as e:
            elapsed = time.monotonic() - t0
            logger.warning(f"LLM 報告略過：{e} ({_fmt(elapsed)})")
            _step_update(3, "skipped", str(e)[:60])

        # ── 步驟 5：歸檔 ─────────────────────────────────────────────────────
        t0 = time.monotonic()
        logger.info(f"\n{'─'*50}")
        logger.info(f"[5/7] 歸檔  [{datetime.now().strftime('%H:%M:%S')}]")
        logger.info(f"{'─'*50}")
        _step_update(4, "running")
        _archive_results(df_kelly, today)
        elapsed = time.monotonic() - t0
        logger.info(f"[5/7] ✓ 完成 ({_fmt(elapsed)})")
        _step_update(4, "done")

        # ── 步驟 6：信號掃描 ─────────────────────────────────────────────────
        t0 = time.monotonic()
        logger.info(f"\n{'─'*50}")
        logger.info(f"[6/7] 信號掃描  [{datetime.now().strftime('%H:%M:%S')}]")
        logger.info(f"{'─'*50}")
        _step_update(5, "running")
        n_buy = n_exit = n_watch = 0
        try:
            from marketmamba.signals.scanner import run_scan
            scan_result = run_scan(df_kelly_path=RESULTS_DIR / "df_kelly.csv")
            n_buy   = len(scan_result.get("buy_signals",  []))
            n_exit  = len(scan_result.get("exit_signals", []))
            n_watch = len(scan_result.get("watch_list",   []))
            elapsed = time.monotonic() - t0
            logger.info(f"  掃描結果：{n_buy} BUY · {n_exit} EXIT · {n_watch} WATCH ({_fmt(elapsed)})")
            logger.info(f"[6/7] ✓ 完成 ({_fmt(elapsed)})")
            _step_update(5, "done", f"{n_buy} BUY · {n_exit} EXIT · {n_watch} WATCH")
        except Exception as e:
            elapsed = time.monotonic() - t0
            logger.warning(f"信號掃描失敗（非致命）：{e} ({_fmt(elapsed)})")
            _step_update(5, "failed", str(e)[:60])

        # ── 步驟 7：模擬回測 ─────────────────────────────────────────────────────
        t0 = time.monotonic()
        logger.info(f"\n{'─'*50}")
        logger.info(f"[7/8] 模擬回測  [{datetime.now().strftime('%H:%M:%S')}]")
        logger.info(f"{'─'*50}")
        _step_update(6, "running")
        try:
            # Archive scanner result (generated in step 6) before running backtests
            _archive_scanner_result(today)

            # ── Alpha Robot backtest ──
            # NOTE: alias to avoid shadowing fetcher's run_daily_update in this scope
            from marketmamba.backtest.sim_engine_v2 import (
                run_robot_backtest,
                run_daily_update as run_alpha_daily_update,
            )
            sim_out = RESULTS_DIR / "sim_backtest.json"
            if sim_out.exists():
                bt = run_alpha_daily_update(
                    date=today,
                    df_today=df_kelly,
                    prices_df=prices,
                    output_path=sim_out,
                )
            else:
                bt = run_robot_backtest(
                    results_dir=RESULTS_DIR,
                    prices_df=prices,
                    output_path=sim_out,
                )
            elapsed = time.monotonic() - t0
            s = bt.get("summary", {})
            ret_str = f"{s.get('total_return_pct', 0):+.2f}%" if bt.get("trading_days", 0) > 0 else "無資料"
            pos_str = f"{s.get('current_positions', 0)} Alpha持倉"
            logger.info(f"  Alpha 機器人：{ret_str}  {pos_str}")

            # ── Scanner Robot backtest ──
            sc_ret_str = "無資料"
            sc_pos_str = "0 Scanner持倉"
            try:
                from marketmamba.backtest.scanner_engine import (
                    run_scanner_backtest, run_scanner_daily_update
                )
                scanner_out = RESULTS_DIR / "scanner_backtest.json"
                if scanner_out.exists():
                    sc_bt = run_scanner_daily_update(
                        date=today,
                        prices_df=prices,
                        output_path=scanner_out,
                        results_dir=RESULTS_DIR,
                    )
                else:
                    sc_bt = run_scanner_backtest(
                        results_dir=RESULTS_DIR,
                        prices_df=prices,
                        output_path=scanner_out,
                    )
                sc_s = sc_bt.get("summary", {})
                sc_ret_str = f"{sc_s.get('total_return_pct', 0):+.2f}%" if sc_bt.get("trading_days", 0) > 0 else "無資料"
                sc_pos_str = f"{sc_s.get('current_positions', 0)} Scanner持倉"
                logger.info(f"  Scanner 機器人：{sc_ret_str}  {sc_pos_str}")
            except Exception as e_sc:
                logger.warning(f"Scanner 回測失敗（非致命）：{e_sc}")

            elapsed = time.monotonic() - t0
            logger.info(f"[7/8] ✓ 完成 ({_fmt(elapsed)})")
            _step_update(6, "done", f"Alpha {ret_str} {pos_str} · Scanner {sc_ret_str} {sc_pos_str}")
        except Exception as e:
            elapsed = time.monotonic() - t0
            logger.warning(f"模擬回測失敗（非致命）：{e} ({_fmt(elapsed)})")
            _step_update(6, "skipped", str(e)[:60])

        # IC analysis (non-blocking, best-effort)
        try:
            from marketmamba.backtest.ic_analyzer import run_ic_analysis
            ic_result = run_ic_analysis(
                results_dir=RESULTS_DIR,
                prices_df=prices,
                output_path=RESULTS_DIR / "ic_analysis.json",
            )
            s5 = ic_result.get("horizon_summary", {}).get("5d", {})
            ic_str = f"IC_5d={s5.get('mean_ic','N/A')}  ICIR={s5.get('icir','N/A')}"
            logger.info(f"  IC分析：{ic_str}")
        except Exception as e:
            logger.warning(f"IC分析失敗（非致命）：{e}")

        # ── 步驟 8：推送 GitHub ───────────────────────────────────────────────
        logger.info(f"\n{'─'*50}")
        logger.info(f"[8/8] 推送 GitHub  [{datetime.now().strftime('%H:%M:%S')}]")
        logger.info(f"{'─'*50}")
        pushed = False
        if not skip_push:
            _step_update(7, "running")
            pushed = _push_to_github(RESULTS_DIR, today)
            if pushed:
                logger.info("  後端將在下次快取刷新後（≤1h）提供最新數據")
                _step_update(7, "done")
            else:
                _step_update(7, "failed", "git push 失敗，請手動推送")
        else:
            logger.info("  --skip-push：略過 git push（dry run 模式）")
            _step_update(7, "skipped", "--skip-push 模式")

        total = time.monotonic() - pipeline_start
        logger.info(f"\n{'='*55}")
        logger.info(f"  ✅ 每日推論完成  [{today}]")
        logger.info(f"  結果路徑：{RESULTS_DIR}")
        logger.info(f"  總耗時：{_fmt(total)}")
        logger.info(f"{'='*55}\n")

    except Exception as exc:
        total = time.monotonic() - pipeline_start
        logger.error(f"推論失敗，已執行 {_fmt(total)}：{exc}", exc_info=True)
        raise


def _push_to_github(results_dir: Path, date_str: str) -> bool:
    """
    Git add + commit + push results to GitHub.
    Render backend will pick up the new df_kelly.csv on next cache refresh (≤1h).
    Returns True on success, False on failure (non-fatal — results saved locally).
    """
    import os
    import subprocess

    repo_root = results_dir.parent.parent   # V6/results → V6 → MarketMamba (.git is here)
    # WSL2: git can't find repo across /mnt filesystem boundary without this
    git_env = {**os.environ, "GIT_DISCOVERY_ACROSS_FILESYSTEM": "1"}
    try:
        # Add the entire results directory — this handles any new files
        # (e.g. scanner_backtest.json on first run) without failing on
        # missing paths. Individual file listing caused CalledProcessError
        # when a file didn't exist yet.
        subprocess.run(
            ["git", "add", "V6/results/"],
            cwd=repo_root, check=True, capture_output=True, env=git_env,
        )

        # Check if there's anything to commit
        diff = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=repo_root, env=git_env,
        )
        if diff.returncode == 0:
            logger.info("No changes in results — git push skipped (already up to date)")
            return True

        subprocess.run(
            ["git", "commit", "-m", f"inference: daily results {date_str}"],
            cwd=repo_root, check=True, capture_output=True, env=git_env,
        )
        subprocess.run(
            ["git", "push", "origin", "main"],
            cwd=repo_root, check=True, capture_output=True, env=git_env,
        )
        logger.info("✅ Results pushed to GitHub (branch: main)")
        # Notify Render backend to refresh its cache immediately
        _refresh_render_cache()
        return True


    except subprocess.CalledProcessError as e:
        logger.warning(
            f"⚠️  git push failed — results saved locally but NOT on GitHub.\n"
            f"   Error: {e.stderr.decode() if e.stderr else str(e)}\n"
            f"   Manual fix: git add V6/results/ && git push"
        )
        return False


def _refresh_render_cache() -> None:
    """
    POST to Render backend to invalidate the 1-hour signals cache immediately
    after a git push so the dashboard shows fresh results within seconds.
    Set RENDER_BACKEND_URL in .env, e.g. https://marketmamba-api.onrender.com
    """
    import os, urllib.request
    url = os.getenv("RENDER_BACKEND_URL", "").rstrip("/")
    if not url:
        return
    endpoint = f"{url}/api/signals/cache/refresh"
    try:
        req = urllib.request.Request(endpoint, method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            logger.info(f"📡 Render cache refreshed: {resp.status}")
    except Exception as e:
        logger.warning(f"Cache refresh failed (non-fatal): {e}")





def _archive_results(df_kelly: pd.DataFrame, date_str: str) -> None:
    """Save today's results to a dated subdirectory (rolling 90-day archive)."""
    import shutil
    from datetime import timedelta

    dated_dir = RESULTS_DIR / date_str
    dated_dir.mkdir(parents=True, exist_ok=True)
    df_kelly.to_csv(dated_dir / "df_kelly.csv", index=False, encoding="utf-8-sig")


def _archive_scanner_result(date_str: str) -> None:
    """Copy action_signals.json (generated in step 6) into the dated archive dir."""
    import shutil
    src = RESULTS_DIR / "action_signals.json"
    if not src.exists():
        logger.warning("action_signals.json not found — scanner archive skipped")
        return
    dated_dir = RESULTS_DIR / date_str
    dated_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dated_dir / "action_signals.json")
    logger.info(f"Scanner result archived → {dated_dir.name}/action_signals.json")

    # Update lightweight history index for the dashboard
    _update_history_index(df_kelly, date_str)

    # Remove entries older than 90 days
    cutoff = (datetime.today() - timedelta(days=90)).strftime("%Y-%m-%d")
    for d in sorted(RESULTS_DIR.iterdir()):
        if d.is_dir() and d.name < cutoff:
            shutil.rmtree(d)
            logger.info(f"Removed old archive: {d.name}")


def _update_history_index(df_kelly: pd.DataFrame, date_str: str) -> None:
    """
    Maintain V6/results/history_index.json — a lightweight log of each
    rebalancing day (top 50 positions). Used by the Signal Scanner for
    rank stability tracking and by the Investment Sim page.
    """
    history_path = RESULTS_DIR / "history_index.json"

    # Load existing entries
    history: list = []
    if history_path.exists():
        try:
            with open(history_path, encoding="utf-8") as f:
                history = json.load(f).get("history", [])
        except Exception:
            history = []

    # Remove today if already present (idempotent)
    history = [h for h in history if h.get("date") != date_str]

    # Build today's entry — top 50 by Sharpe for scanner rank stability
    investable = df_kelly[df_kelly["Exp_Alpha_20d"] > -999]
    top50 = investable.head(50)

    cols_needed = ["Ticker", "Exp_Alpha_20d", "Signal_Quality", "Confidence", "Suggested_Weight", "Uncertainty"]
    available   = [c for c in cols_needed if c in top50.columns]
    portfolio   = []
    for i, (_, row) in enumerate(top50[available].iterrows()):
        portfolio.append({
            "rank":       i + 1,
            "ticker":     str(row.get("Ticker", "")),
            "alpha":      round(float(row.get("Exp_Alpha_20d", 0)), 6),
            "sharpe":     round(float(row.get("Signal_Quality",  0)), 3),
            "confidence": str(row.get("Confidence", "")),
            "weight":     round(float(row.get("Suggested_Weight", 0)), 4),
            "uncertainty":round(float(row.get("Uncertainty", 0)), 6),
        })

    entry = {
        "date":             date_str,
        "total_investable": int(len(investable)),
        "portfolio":        portfolio,
    }
    history.insert(0, entry)
    history = history[:60]   # keep last 60 trading days

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump({"last_updated": date_str, "history": history}, f,
                  ensure_ascii=False, indent=2)
    logger.info(f"History index updated → {history_path} ({len(history)} entries)")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MarketMamba V6 每日推論")
    parser.add_argument("--date",         type=str,  default=None,
                        help="目標日期 YYYY-MM-DD（預設：今日）")
    parser.add_argument("--skip-push",    action="store_true",
                        help="略過 git push（dry run / 測試模式）")
    parser.add_argument("--no-gui",       action="store_true",
                        help="停用 tkinter 進度視窗（純文字模式）")
    parser.add_argument("--forward-fill", "--ff", action="store_true",
                        dest="forward_fill",
                        help="寬鬆模式：資料缺漏時以前日資料補齊後繼續推論（預設：嚴格模式，缺漏即停止）")
    args = parser.parse_args()

    _main_kwargs = dict(
        target_date  = args.date,
        skip_push    = args.skip_push,
        forward_fill = args.forward_fill,
    )

    if _TK_AVAILABLE and not args.no_gui:
        try:
            ui = ProgressWindow()
            _handler = _UiLogHandler()
            _handler.setFormatter(logging.Formatter("%(levelname)s %(name)s — %(message)s"))
            logging.getLogger().addHandler(_handler)
            ui.run_with(main, **_main_kwargs)
        except Exception as e:
            # 無 DISPLAY 環境（Task Scheduler headless、WSLg 未啟動等）→ 降級為文字模式
            logger.warning(f"進度視窗無法開啟，切換為文字模式：{e}")
            main(**_main_kwargs)
    else:
        main(**_main_kwargs)
