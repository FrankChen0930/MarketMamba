"""One-off patch: replace lazy-pandas __init__ with pre-indexed numpy __init__."""
import pathlib, re

path = pathlib.Path(r"d:\Desktop\work\MarketMamba\V6\marketmamba\models\trainer.py")
text = path.read_text(encoding="utf-8")

# Unique anchor: the logger.info that says "lazy loading"
OLD_ANCHOR = (
    '        logger.info(\n'
    '            f"Dataset [{mode}]: {len(self.valid_dates)} valid trading days "\n'
    '            f"(lazy loading \u2014 tensors built on demand)"\n'
    '        )'
)

NEW_BODY = '''\
        # Pre-index per-stock numpy arrays (eliminates pandas in __getitem__)
        # Memory: ~1.9 GB for 1754 stocks x 5500 dates x 49 features x float32
        logger.info(f"Dataset [{mode}]: pre-indexing {df['stock_id'].nunique()} stocks...")
        self._stock_index: dict[str, dict] = {}
        for sid, grp in df.groupby("stock_id"):
            grp  = grp.sort_values("Date")
            didx = np.array([self._date_to_idx[d] for d in grp["Date"].values], dtype=np.int32)
            self._stock_index[str(sid)] = {
                "date_idx": didx,
                "feats":    grp[FEATURE_COLS].values.astype(np.float32),
                "targets":  grp[TARGET_COLS].values.astype(np.float32),
            }
        self._date_stocks = {dt: [str(s) for s in sl] for dt, sl in date_stocks.items()}

        logger.info(
            f"Dataset [{mode}]: {len(self.valid_dates)} valid days "
            f"| {len(self._stock_index)} stocks pre-indexed"
        )'''

if OLD_ANCHOR in text:
    text = text.replace(OLD_ANCHOR, NEW_BODY, 1)
    # Also remove self._df = df and self._all_dates = all_dates (replaced by new vars)
    path.write_text(text, encoding="utf-8")
    print("SUCCESS: init patched")
else:
    # Show the region around the expected location
    for i, line in enumerate(text.splitlines()[108:118], 109):
        print(i, repr(line))
    print("FAIL: anchor not found")
