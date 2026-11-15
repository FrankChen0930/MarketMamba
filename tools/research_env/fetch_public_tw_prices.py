"""Fetch a public Taiwan daily-price sample (yfinance + TWSE open-data listing) into .local-data/public_tw.
Research-only, NOT trusted investment data: survivorship-biased universe (current top-turnover listing),
adjusted close from Yahoo. Re-runnable; no credentials."""
import json, sys, time, datetime as dt
from pathlib import Path
import requests, pandas as pd, yfinance as yf
OUT = Path(__file__).resolve().parents[2] / ".local-data" / "public_tw"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 150
rows = requests.get("https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL", timeout=30).json()
df = pd.DataFrame(rows)
df = df[df["Code"].str.fullmatch(r"\d{4}")].copy()
df["val"] = pd.to_numeric(df["TradeValue"], errors="coerce")
uni = df.sort_values("val", ascending=False).head(N)[["Code", "Name", "val"]]
tickers = [c + ".TW" for c in uni.Code]
frames = []
for i in range(0, len(tickers), 30):
    d = yf.download(tickers[i:i+30], start="2016-01-01", end=dt.date.today().isoformat(), auto_adjust=False,
                    progress=False, group_by="ticker", threads=True)
    for t in tickers[i:i+30]:
        if t in d.columns.get_level_values(0):
            x = d[t].dropna(how="all").reset_index(); x["code"] = t[:4]; frames.append(x)
    time.sleep(1)
px = pd.concat(frames, ignore_index=True).rename(columns=str.lower).rename(columns={"adj close": "adj_close"})
OUT.mkdir(parents=True, exist_ok=True)
px.to_parquet(OUT / "prices_daily.parquet")
uni.to_csv(OUT / "universe.csv", index=False)
meta = {"fetched": dt.datetime.now().isoformat(), "n_codes": int(px.code.nunique()), "rows": len(px),
        "min_date": str(px.date.min()), "max_date": str(px.date.max()),
        "caveats": ["survivorship-biased universe (current top-turnover)", "Yahoo adjusted close", "research-only"]}
(OUT / "meta.json").write_text(json.dumps(meta, indent=2)); print(meta)
