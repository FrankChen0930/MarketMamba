"""
Patch trainer.py:
1. Disable KG by default (get_batch_edges Python loop is the new bottleneck)
2. Replace logger.info diagnostics with print() so they show in Colab notebook
3. Add NaN gradient detection warning
"""
import pathlib, re

path = pathlib.Path(r"d:\Desktop\work\MarketMamba\V6\marketmamba\models\trainer.py")
text = path.read_text(encoding="utf-8")

# --- Fix 1: disable KG ---
OLD1 = "    # KG adjacency dict (pre-built once, O(1) per-stock lookup per batch)\n    kg_adj = build_kg_adjacency()"
NEW1 = (
    "    # KG disabled during training: get_batch_edges Python loop is a bottleneck\n"
    "    # when KG has many edges (e.g. full correlation graph). Re-enable once pipeline\n"
    "    # is verified stable by setting use_kg=True in train_model call.\n"
    "    kg_adj = None  # build_kg_adjacency() -- disabled for speed"
)

OLD1_CRLF = OLD1.replace("\n", "\r\n")
NEW1_CRLF = NEW1.replace("\n", "\r\n")

if OLD1 in text:
    text = text.replace(OLD1, NEW1, 1)
    print("Fix 1 (LF) applied")
elif OLD1_CRLF in text:
    text = text.replace(OLD1_CRLF, NEW1_CRLF, 1)
    print("Fix 1 (CRLF) applied")
else:
    print("Fix 1 NOT FOUND")
    for i, l in enumerate(text.splitlines()[458:465], 459):
        print(i, repr(l[:80]))

# --- Fix 2: replace logger.info diag with print ---
text = text.replace(
    'logger.info(f"  [diag] First batch: X={tuple(X.shape)} "\n'
    '                            f"stocks={len(batch_stocks)} | {time.time()-t0:.1f}s since epoch start")',
    'print(f"  [diag] First batch: X={tuple(X.shape)} stocks={len(batch_stocks)} | {time.time()-t0:.1f}s since epoch start", flush=True)'
)
text = text.replace(
    'logger.info(f"  [diag] KG edges: {edge_index.shape[1]}")',
    'print(f"  [diag] KG edges: {edge_index.shape[1]}", flush=True)'
)
text = text.replace(
    'logger.info(f"  [diag] Forward OK. loss={loss.item():.4f} | "\n'
    '                            f"batch took {time.time()-t0:.1f}s total")',
    'print(f"  [diag] Forward OK. loss={loss.item():.4f} | batch took {time.time()-t0:.1f}s total", flush=True)'
)
text = text.replace(
    'logger.info(\n'
    '                    f"  Ep {epoch:03d} [{batch_idx+1}/{total_b}] "\n'
    '                    f"loss={float(np.mean(train_losses)):.5f} | "\n'
    '                    f"{elapsed:.0f}s | ETA {eta:.0f}s"\n'
    '                )',
    'print(f"  Ep {epoch:03d} [{batch_idx+1}/{total_b}] loss={float(np.mean(train_losses)):.5f} | {elapsed:.0f}s | ETA {eta:.0f}s", flush=True)'
)
# Also replace dataset pre-index log with print
text = text.replace(
    'logger.info(f"Dataset [{mode}]: pre-indexing {df[\'stock_id\'].nunique()} stocks...")',
    'print(f"Dataset [{mode}]: pre-indexing {df[chr(39)}stock_id{chr(39)}].nunique()} stocks...", flush=True)'
)
# Simpler - just replace the specific string
text = re.sub(
    r'logger\.info\(f"Dataset \[{mode}\]: pre-indexing.*?\.\.\."',
    'print(f"[Dataset init] pre-indexing stocks...", flush=True)',
    text
)
text = re.sub(
    r'logger\.info\(\s*f"Dataset \[{mode}\]: \{len\(self\.valid_dates\)\} valid days.*?pre-indexed"\s*\)',
    'print(f"[Dataset init] {len(self.valid_dates)} valid days | {len(self._stock_index)} stocks pre-indexed", flush=True)',
    text, flags=re.DOTALL
)

# --- Fix 3: add NaN gradient print ---
OLD3 = "            scaler.scale(loss).backward()"
NEW3 = (
    "            # Check for NaN loss before backward\n"
    "            if torch.isnan(loss) or torch.isinf(loss):\n"
    "                if epoch == 1 and batch_idx < 5:\n"
    "                    print(f'  [WARN] NaN/Inf loss at batch {batch_idx}: {loss.item()}', flush=True)\n"
    "                continue\n"
    "            scaler.scale(loss).backward()"
)

if OLD3 in text:
    text = text.replace(OLD3, NEW3, 1)
    print("Fix 3 applied")
else:
    print("Fix 3 NOT FOUND")

path.write_text(text, encoding="utf-8")
print("Done writing trainer.py")
