"""
test_availability_flags.py — V6.3 可得性旗標的驗收測試（決策1）
==================================================================
三關，全部要過才算完成：

  【1】回歸：`availability_flags=False` 必須與 **git HEAD 版本逐位元相同**。
       這是本專案凡動 `feature_engineer.py` 都要跑的驗收標準（2026-07-27 決策）——
       線上 V6.1 checkpoint 的特徵語意不能被改動，判準是「最大絕對差 0.000e+00」，
       不是「看起來差不多」。
       作法：用 `git show HEAD:...` 取出改動前的原始碼，當成獨立模組載入，
       兩邊跑同一份輸入再逐欄比對。

  【2】旗標正確性：逐年印出各旗標的可得比例，與 `report_feature_availability.py`
       量到的原始來源命中率對照。旗標若與來源對不上，代表 `_mark_avail` 的
       插入位置錯了（例如插在 fillna 之後 → 恆為 1）。

  【3】旗標不被標準化：`clean_and_scale` 後旗標必須仍只有 {0.0, 1.0} 兩個值。
       這一關防的是「旗標被逐日 z-score 後自我抵銷」——那會讓整個決策1 失效，
       而且不會有任何錯誤訊息。

用法（repo 根目錄）：
    python V6/scripts/test_availability_flags.py
"""
from __future__ import annotations

import importlib.util
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

_V6_DIR = Path(__file__).resolve().parent.parent
if str(_V6_DIR) not in sys.path:
    sys.path.insert(0, str(_V6_DIR))

logging.basicConfig(level=logging.WARNING, format="%(message)s")

from marketmamba.config import PROCESSED_DIR  # noqa: E402
from marketmamba.data.feature_spec import AVAIL_COLS, patch_config_67d  # noqa: E402

P = Path(PROCESSED_DIR)
START = "2012-01-01"        # 涵蓋 institutional 覆蓋率的台階（2012 跳 56%、2013 起 74%+）
N_STOCKS = 40
HEAD_PATH = "V6/marketmamba/data/feature_engineer.py"


def load(name: str, ids: set[str] | None = None) -> pd.DataFrame | None:
    p = P / f"{name}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    for dc in ("Date", "date", "Week"):
        if dc in df.columns:
            df[dc] = pd.to_datetime(df[dc], errors="coerce")
            df = df[df[dc] >= START]
            break
    if ids is not None and "stock_id" in df.columns:
        df = df[df["stock_id"].astype(str).isin(ids)]
    return df.reset_index(drop=True)


def _load_head_module():
    """把 git HEAD 版本的 feature_engineer.py 載成獨立模組，供逐位元比對。"""
    out = subprocess.run(["git", "show", f"HEAD:{HEAD_PATH}"],
                         capture_output=True, text=True, encoding="utf-8")
    if out.returncode != 0:
        raise RuntimeError(f"git show 失敗：{out.stderr[:300]}")
    tmp = Path(tempfile.mkdtemp()) / "feature_engineer_head.py"
    tmp.write_text(out.stdout, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("_fe_head", tmp)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _pick_universe() -> set[str]:
    pr = pd.read_parquet(P / "prices_raw.parquet", columns=["Date", "stock_id"])
    pr["Date"] = pd.to_datetime(pr["Date"])
    last = pr["Date"].max()
    uni = sorted({s for s in pr.loc[pr["Date"] == last, "stock_id"].astype(str)
                  if len(s) == 4 and s.isdigit() and not s.startswith("00")})
    del pr
    # 頭尾都取：大型股（法人/融資資料齊）與冷門股（多半缺）都要在樣本裡，
    # 否則旗標會全 1，測不出東西
    return set(uni[:N_STOCKS // 2] + uni[-N_STOCKS // 2:])


def main() -> None:
    ids = _pick_universe()
    kw = dict(
        df_price=load("prices_raw", ids),
        df_inst=load("institutional_raw", ids),
        df_margin=load("margin_raw", ids),
        df_daytrade=load("daytrade_raw", ids),
        df_holdings=load("holdings_raw", ids),
        df_securities=load("securities_raw", ids),
        df_foreign_shareholding=load("foreign_shareholding_raw", ids),
        df_per=load("per_raw", ids),
        df_market_value=load("market_value_raw", ids),
        df_rev=load("revenue_raw", ids),
        df_fin=load("financials_raw", ids),
        df_balance_sheet=load("balance_sheet_raw", ids),
        df_cashflow=load("cashflow_raw", ids),
        df_dividend=load("dividend_raw", ids),
        df_macro=load("macro_raw"),
    )
    kw = {k: v for k, v in kw.items() if v is not None and not v.empty}
    print(f"[樣本] {len(ids)} 支 × 自 {START}")
    print("[載入] " + "｜".join(f"{k.replace('df_', '')} {len(v):,}" for k, v in kw.items()))

    # ================================================================
    print("\n" + "=" * 78)
    print("■ [1] 回歸測試：availability_flags=False 必須與 git HEAD 逐位元相同")
    print("=" * 78)
    from marketmamba.data.feature_engineer import build_features as build_new

    head = _load_head_module()
    old = head.build_features(**kw, fundamentals_v2=False)
    new = build_new(**kw, fundamentals_v2=False, availability_flags=False)

    print(f"  HEAD 版：{old.shape[0]:,} 列 × {old.shape[1]} 欄")
    print(f"  新版　 ：{new.shape[0]:,} 列 × {new.shape[1]} 欄")
    assert list(old.columns) == list(new.columns), (
        f"欄位不同\n  只在 HEAD: {set(old.columns) - set(new.columns)}"
        f"\n  只在新版: {set(new.columns) - set(old.columns)}")
    assert old.shape == new.shape, f"形狀不同 {old.shape} vs {new.shape}"

    worst_col, worst = None, 0.0
    n_bad = 0
    for c in old.columns:
        a, b = old[c], new[c]
        if pd.api.types.is_numeric_dtype(a):
            d = float(np.nanmax(np.abs(a.to_numpy(dtype="float64")
                                       - b.to_numpy(dtype="float64")))) if len(a) else 0.0
            # NaN 位置也要一致，否則「兩邊都 NaN」會讓 nanmax 看不出差異
            if not a.isna().equals(b.isna()):
                n_bad += 1
                print(f"  ✗ {c}: NaN 位置不同")
            if not np.isnan(d) and d > worst:
                worst, worst_col = d, c
        else:
            if not a.equals(b):
                n_bad += 1
                print(f"  ✗ {c}: 非數值欄不同")
    print(f"  最大絕對差 {worst:.3e}（欄位 {worst_col}）｜不一致欄位 {n_bad}")
    assert worst == 0.0 and n_bad == 0, "回歸失敗——線上 V6.1 會受影響，不可繼續"
    print("  ✓ build_features 逐位元相同")

    # clean_and_scale 也要比：F3 把它的迴圈重構成
    # 「winsorize 全部 → 中性化 → z-score 全部」三段（原本是逐欄 winsorize+z-score 交錯）。
    # 兩者在數學上等價（每欄獨立），但**等價不等於逐位元相同**，必須實測。
    # 這一步必須在 config patch 成 67 維之前做，否則 HEAD 版會拿到它不認識的旗標欄。
    # 用日期子集：這一關驗的是**程式等價性**（每欄獨立 → 拆成三段不該改變結果），
    # 不是資料覆蓋。全樣本跑 clean_and_scale 要數分鐘（逐日 groupby + Python lambda），
    # 而等價性用 300 個交易日就足以暴露任何差異。
    from marketmamba.data.feature_engineer import clean_and_scale as cs_new
    _sub_days = sorted(pd.to_datetime(old["Date"]).unique())[-300:]
    _o_sub = old[pd.to_datetime(old["Date"]).isin(_sub_days)]
    _n_sub = new[pd.to_datetime(new["Date"]).isin(_sub_days)]
    print(f"  （clean_and_scale 回歸用最後 {len(_sub_days)} 個交易日 / {len(_o_sub):,} 列）")
    for mn in ("cross", "ts"):
        o = head.clean_and_scale(_o_sub.copy(), macro_norm=mn)
        n = cs_new(_n_sub.copy(), macro_norm=mn, neutralize="none")
        assert o.shape == n.shape, f"macro_norm={mn} 形狀不同 {o.shape} vs {n.shape}"
        w = max(float(np.nanmax(np.abs(o[c].to_numpy(dtype="float64")
                                       - n[c].to_numpy(dtype="float64"))))
                for c in o.columns if pd.api.types.is_numeric_dtype(o[c]))
        print(f"  clean_and_scale(macro_norm='{mn}') 最大絕對差 {w:.3e}")
        assert w == 0.0, f"clean_and_scale 重構改變了 macro_norm={mn} 的輸出"
    print("  ✓ clean_and_scale 重構後逐位元相同，V6.1 零影響")

    # ================================================================
    print("\n" + "=" * 78)
    print("■ [2] 旗標正確性：逐年可得比例（應與原始來源命中率一致）")
    print("=" * 78)
    dim = patch_config_67d()
    import importlib

    import marketmamba.data.feature_engineer as fe
    importlib.reload(fe)          # 讓模組層的 FEATURE_COLS 重新綁到 patch 後的 config
    print(f"  config 已 patch：INPUT_DIM={dim}")

    flagged = fe.build_features(**kw, fundamentals_v2=True, availability_flags=True)
    print(f"  輸出 {flagged.shape[0]:,} 列 × {flagged.shape[1]} 欄")
    missing = [c for c in AVAIL_COLS if c not in flagged.columns]
    assert not missing, f"旗標欄未產生：{missing}"

    t = flagged.copy()
    t["_y"] = pd.to_datetime(t["Date"]).dt.year
    yearly = t.groupby("_y")[AVAIL_COLS].mean() * 100
    print()
    print(yearly.round(0).astype(int).to_string())

    print()
    for c in AVAIL_COLS:
        vals = set(np.unique(flagged[c].to_numpy()))
        assert vals <= {0.0, 1.0}, f"{c} 不是 0/1：{sorted(vals)[:5]}"
    print("  ✓ 全部旗標皆為 0/1")

    # 死旗標偵測：在**訓練窗內**恆為同一個值的旗標是浪費的維度。
    # 一個常數欄不只沒資訊，還會佔掉 FactorGroupedEmbedding 的投影容量，
    # 只能貢獻噪音。與其留著，不如從規格裡拿掉。
    print("\n  ── 死旗標偵測（訓練窗 2013 起）──")
    tw = flagged[pd.to_datetime(flagged["Date"]) >= "2013-01-01"]
    dead = []
    for c in AVAIL_COLS:
        m, sd = float(tw[c].mean()), float(tw[c].std())
        status = "常數" if sd < 1e-6 else ("幾乎常數" if m > 0.995 or m < 0.005 else "OK")
        if status != "OK":
            dead.append(c)
        print(f"    {c:22s} mean={m:.4f} std={sd:.4f}  {status}")
    if dead:
        print(f"  ⚠ {len(dead)} 個旗標在訓練窗內幾乎無變異：{dead}")
        print("    （本測試僅 40 支樣本、偏向老牌大型股，"
              "最終取捨請以 F5 全量建構的結果為準）")

    # ================================================================
    print("\n" + "=" * 78)
    print("■ [3] clean_and_scale 後旗標必須維持 0/1（不可被 z-score）")
    print("=" * 78)
    _fl_sub3 = flagged[pd.to_datetime(flagged["Date"]).isin(_sub_days)].copy()
    scaled = fe.clean_and_scale(_fl_sub3, macro_norm="ts")
    for c in AVAIL_COLS:
        vals = set(np.unique(scaled[c].to_numpy()))
        assert vals <= {0.0, 1.0}, (
            f"{c} 在 clean_and_scale 後變成 {sorted(vals)[:5]}——"
            f"旗標被標準化了，決策1 會完全失效")
    print(f"  ✓ {len(AVAIL_COLS)} 個旗標在標準化後仍為 0/1")

    # 對照組：一般特徵應該有被標準化（若沒有，代表跳過清單寫太寬）
    ref = "Return_5d"
    if ref in scaled.columns:
        print(f"  對照組 {ref}: mean={scaled[ref].mean():+.4f} "
              f"std={scaled[ref].std():.4f}（應 ≈ 0 / ≈ 1，證明標準化確實有跑）")

    # ================================================================
    print("\n" + "=" * 78)
    print("■ [4] 中性化（F3）：neutralize='none' 必須與現況相同，開啟後殘差要正確")
    print("=" * 78)
    from marketmamba.data.feature_spec import NEUTRALIZE_EXCLUDE, resolve_sector
    from marketmamba.data.hygiene import load_stock_info

    # 同樣用日期子集：中性化的正確性是**逐日獨立**的性質
    # （每一天各自跑一次橫斷面迴歸），用 300 天驗證與用 3,400 天等價，
    # 但 clean_and_scale 的逐欄 groupby-lambda 成本差一個數量級。
    fl_sub = flagged[pd.to_datetime(flagged["Date"]).isin(_sub_days)].copy()
    print(f"  （用最後 {len(_sub_days)} 個交易日 / {len(fl_sub):,} 列）")

    ref_none = fe.clean_and_scale(fl_sub.copy(), macro_norm="ts", neutralize="none")
    base = fe.clean_and_scale(fl_sub.copy(), macro_norm="ts")
    d0 = float(np.nanmax(np.abs(ref_none["Return_5d"].to_numpy(dtype="float64")
                                - base["Return_5d"].to_numpy(dtype="float64"))))
    print(f"  neutralize='none' vs 預設：Return_5d 最大絕對差 {d0:.3e}")
    assert d0 == 0.0, "neutralize='none' 改變了輸出——預設路徑不可受影響"

    sec = resolve_sector(load_stock_info(latest_only=False))
    sec_map = dict(zip(sec["stock_id"], sec["sector"]))
    for mode in ("industry", "industry_mktcap"):
        out = fe.clean_and_scale(fl_sub.copy(), macro_norm="ts", neutralize=mode)
        s = out["stock_id"].astype(str).map(sec_map).fillna("Unknown")
        col = "Return_5d"
        # 驗收一：同產業內殘差均值應 ≈ 0
        gm = out.groupby([out["Date"], s])[col].mean()
        # 驗收二：殘差與 log 市值的相關應 ≈ 0（只有 industry_mktcap 保證）
        mc = pd.to_numeric(fl_sub.loc[out.index, "Market_Cap_Log"], errors="coerce")
        ok = mc.notna() & out[col].notna()
        corr = float(np.corrcoef(out.loc[ok, col], mc[ok])[0, 1]) if ok.sum() > 100 else float("nan")
        print(f"  {mode:18s} 產業內殘差均值 |max|={gm.abs().max():.2e}"
              f"｜殘差 vs log市值 corr={corr:+.4f}")
        for c in AVAIL_COLS:
            assert set(np.unique(out[c].to_numpy())) <= {0.0, 1.0}, \
                f"{c} 被中性化了——旗標應在 NEUTRALIZE_EXCLUDE 內"
        print(f"                     ✓ 旗標未被中性化"
              f"（排除清單 {len(NEUTRALIZE_EXCLUDE)} 欄）")

    print("\n" + "=" * 78)
    print("全部通過")
    print("=" * 78)


if __name__ == "__main__":
    main()
