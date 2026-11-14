"""
Patch trainer.py:
1. Add on_epoch_end callback parameter to train_model
2. Replace logger.info epoch logs with print() so Colab sees them
3. Also fix checkpoint logger.info → print
"""
import pathlib, re

path = pathlib.Path(r"d:\Desktop\work\MarketMamba\V6\marketmamba\models\trainer.py")
text = path.read_text(encoding="utf-8")

# Fix 1: add callback to function signature
OLD_SIG = (
    "def train_model(\n"
    "    df:              pd.DataFrame,\n"
    "    train_dates:     list[str],\n"
    "    val_dates:       list[str],\n"
    "    epochs:          int   = EPOCHS,\n"
    "    lr:              float = LR,\n"
    "    early_stop:      int   = EARLY_STOP,\n"
    "    checkpoint_name: str   = \"v6_best.pt\",\n"
    "    device_str:      str   = \"cuda\" if torch.cuda.is_available() else \"cpu\",\n"
    ") -> tuple[MarketMambaV6, TrainingHistory]:"
)
NEW_SIG = (
    "def train_model(\n"
    "    df:              pd.DataFrame,\n"
    "    train_dates:     list[str],\n"
    "    val_dates:       list[str],\n"
    "    epochs:          int   = EPOCHS,\n"
    "    lr:              float = LR,\n"
    "    early_stop:      int   = EARLY_STOP,\n"
    "    checkpoint_name: str   = \"v6_best.pt\",\n"
    "    device_str:      str   = \"cuda\" if torch.cuda.is_available() else \"cpu\",\n"
    "    on_epoch_end     = None,   # optional callback(history, epoch, epochs)\n"
    ") -> tuple[MarketMambaV6, TrainingHistory]:"
)

for eol in ("\r\n", "\n"):
    old = OLD_SIG.replace("\n", eol)
    new = NEW_SIG.replace("\n", eol)
    if old in text:
        text = text.replace(old, new, 1)
        print(f"Fix 1 applied ({repr(eol)})")
        break
else:
    print("Fix 1 NOT FOUND")

# Fix 2: replace epoch logger.info with print + callback call
OLD_LOG = (
    "        logger.info(\n"
    "            f\"Epoch {epoch:03d}/{epochs} | \"\n"
    "            f\"train={avg_train:.5f} val={avg_val:.5f} IC={avg_ic:+.4f} \"\n"
    "            f\"lr={cur_lr:.2e} | {time.time()-t0:.1f}s\"\n"
    "        )"
)
NEW_LOG = (
    "        elapsed_ep = time.time() - t0\n"
    "        print(\n"
    "            f\"Epoch {epoch:03d}/{epochs} | \"\n"
    "            f\"train={avg_train:.5f} val={avg_val:.5f} IC={avg_ic:+.4f} \"\n"
    "            f\"lr={cur_lr:.2e} | {elapsed_ep:.0f}s\",\n"
    "            flush=True,\n"
    "        )\n"
    "        if on_epoch_end is not None:\n"
    "            on_epoch_end(history, epoch, epochs)"
)

for eol in ("\r\n", "\n"):
    old = OLD_LOG.replace("\n", eol)
    new = NEW_LOG.replace("\n", eol)
    if old in text:
        text = text.replace(old, new, 1)
        print(f"Fix 2 applied ({repr(eol)})")
        break
else:
    print("Fix 2 NOT FOUND")

# Fix 3: checkpoint saved → print
for eol in ("\r\n", "\n"):
    old = f'            logger.info(f"  \u2705 Checkpoint saved \u2192 {{ckpt_path.name}}")'.replace("\n", eol)
    new = f'            print(f"  \u2705 Checkpoint saved \u2192 {{ckpt_path.name}}", flush=True)'.replace("\n", eol)
    if old in text:
        text = text.replace(old, new, 1)
        print(f"Fix 3 applied ({repr(eol)})")
        break

# Fix 4: early stop → print
for eol in ("\r\n", "\n"):
    old = f'                logger.info(f"Early stop at epoch {{epoch}}")'.replace("\n", eol)
    new = f'                print(f"  \U0001f6d1 Early stop at epoch {{epoch}}", flush=True)'.replace("\n", eol)
    if old in text:
        text = text.replace(old, new, 1)
        print(f"Fix 4 applied ({repr(eol)})")
        break

# Fix 5: training done → print
for eol in ("\r\n", "\n"):
    old = (
        "    logger.info(\n"
        "        f\"Training done. Best epoch={history.best_epoch}, \"\n"
        "        f\"val_loss={history.best_val_loss:.5f}\"\n"
        "    )"
    ).replace("\n", eol)
    new = (
        "    print(\n"
        "        f\"Training done. Best epoch={history.best_epoch} | \"\n"
        "        f\"val_loss={history.best_val_loss:.5f}\",\n"
        "        flush=True,\n"
        "    )"
    ).replace("\n", eol)
    if old in text:
        text = text.replace(old, new, 1)
        print(f"Fix 5 applied ({repr(eol)})")
        break

path.write_text(text, encoding="utf-8")
print("Done writing trainer.py")
