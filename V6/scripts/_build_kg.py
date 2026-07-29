import sys; sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "V6")
import pandas as pd
from marketmamba.knowledge.graph_builder import build_knowledge_graph
from marketmamba.config import PROCESSED_DIR, KG_CACHE_PATH

print("Loading prices...", flush=True)
df_prices = pd.read_parquet(PROCESSED_DIR / "prices_raw.parquet")
df_universe = df_prices[["stock_id"]].drop_duplicates()

stock_info = PROCESSED_DIR / "stock_info.parquet"
if stock_info.exists():
    # 一律走 load_stock_info（B-5）：stock_info 是多次快照的累積，
    # 直接 merge 會讓 universe 列膨脹 36.2%（846 支被複製），
    # 且可能取到過期的產業別。
    from marketmamba.data.hygiene import load_stock_info
    df_info = load_stock_info(latest_only=True)
    df_universe = df_universe.merge(df_info[["stock_id", "industry_category"]], on="stock_id", how="left")
    df_universe["industry_category"] = df_universe["industry_category"].fillna("Unknown")
    n = df_universe["industry_category"].nunique()
    print(f"Sectors: {n}", flush=True)

print(f"Building KG for {len(df_universe)} stocks...", flush=True)
build_knowledge_graph(df_universe, df_prices, force_rebuild=True)
print(f"KG saved: {KG_CACHE_PATH}", flush=True)
sz = KG_CACHE_PATH.stat().st_size / 1e6
print(f"Size: {sz:.1f} MB", flush=True)
