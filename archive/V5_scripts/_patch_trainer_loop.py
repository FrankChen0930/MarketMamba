"""Patch trainer.py: add per-batch progress logging."""
import pathlib, re

path = pathlib.Path(r"d:\Desktop\work\MarketMamba\V6\marketmamba\models\trainer.py")
text = path.read_text(encoding="utf-8")

# Find the training for-loop block and replace
OLD = (
        "        for X, Y, batch_stocks in train_loader:\r\n"
        "            if X.shape[0] <= 1:   # skip empty / degenerate cross-sections\r\n"
        "                continue\r\n"
        "            X, Y = X.to(device), Y.to(device)\r\n"
        "            # Build batch-local edge_index (indices in [0, N_batch)) \u2014 no out-of-bounds\r\n"
        "            edge_index, edge_attr = get_batch_edges(batch_stocks, kg_adj, device)\r\n"
        "\r\n"
        "            optimizer.zero_grad()\r\n"
        "            with autocast('cuda', enabled=AMP_ENABLED and device_str == \"cuda\"):\r\n"
        "                preds       = model(X, edge_index, edge_attr)\r\n"
        "                loss, brkdn = multi_horizon_loss(preds, Y)\r\n"
        "\r\n"
        "            scaler.scale(loss).backward()\r\n"
        "            scaler.unscale_(optimizer)\r\n"
        "            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)\r\n"
        "            scaler.step(optimizer)   # optimizer first\r\n"
        "            scaler.update()\r\n"
        "            scheduler.step()         # scheduler after \u2014 correct order per PyTorch docs\r\n"
        "\r\n"
        "            train_losses.append(brkdn[\"loss_total\"])\r\n"
        "            for k in epoch_bd:\r\n"
        "                epoch_bd[k].append(brkdn.get(k, 0.0))\r\n"
        "\r\n"
        "        if not train_losses:\r\n"
        "            logger.warning(f\"Epoch {epoch}: no valid training batches\")\r\n"
        "            continue"
)

NEW = (
        "        for batch_idx, (X, Y, batch_stocks) in enumerate(train_loader):\r\n"
        "            if X.shape[0] <= 1:   # skip empty / degenerate cross-sections\r\n"
        "                continue\r\n"
        "\r\n"
        "            # Timing diagnostic for first batch of first epoch\r\n"
        "            if epoch == 1 and batch_idx == 0:\r\n"
        "                logger.info(f\"  [diag] First batch: X={tuple(X.shape)} \"\r\n"
        "                            f\"stocks={len(batch_stocks)} | {time.time()-t0:.1f}s since epoch start\")\r\n"
        "\r\n"
        "            X, Y = X.to(device), Y.to(device)\r\n"
        "            edge_index, edge_attr = get_batch_edges(batch_stocks, kg_adj, device)\r\n"
        "\r\n"
        "            if epoch == 1 and batch_idx == 0:\r\n"
        "                logger.info(f\"  [diag] KG edges: {edge_index.shape[1]}\")\r\n"
        "\r\n"
        "            optimizer.zero_grad()\r\n"
        "            with autocast('cuda', enabled=AMP_ENABLED and device_str == \"cuda\"):\r\n"
        "                preds       = model(X, edge_index, edge_attr)\r\n"
        "                loss, brkdn = multi_horizon_loss(preds, Y)\r\n"
        "\r\n"
        "            if epoch == 1 and batch_idx == 0:\r\n"
        "                logger.info(f\"  [diag] Forward OK. loss={loss.item():.4f} | \"\r\n"
        "                            f\"batch took {time.time()-t0:.1f}s total\")\r\n"
        "\r\n"
        "            scaler.scale(loss).backward()\r\n"
        "            scaler.unscale_(optimizer)\r\n"
        "            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)\r\n"
        "            scaler.step(optimizer)\r\n"
        "            scaler.update()\r\n"
        "            scheduler.step()\r\n"
        "\r\n"
        "            train_losses.append(brkdn[\"loss_total\"])\r\n"
        "            for k in epoch_bd:\r\n"
        "                epoch_bd[k].append(brkdn.get(k, 0.0))\r\n"
        "\r\n"
        "            # Progress every 200 batches\r\n"
        "            if (batch_idx + 1) % 200 == 0:\r\n"
        "                elapsed = time.time() - t0\r\n"
        "                total_b = len(train_loader)\r\n"
        "                eta     = elapsed / (batch_idx + 1) * (total_b - batch_idx - 1)\r\n"
        "                logger.info(\r\n"
        "                    f\"  Ep {epoch:03d} [{batch_idx+1}/{total_b}] \"\r\n"
        "                    f\"loss={float(np.mean(train_losses)):.5f} | \"\r\n"
        "                    f\"{elapsed:.0f}s | ETA {eta:.0f}s\"\r\n"
        "                )\r\n"
        "\r\n"
        "        if not train_losses:\r\n"
        "            logger.warning(f\"Epoch {epoch}: no valid training batches\")\r\n"
        "            continue"
)

if OLD in text:
    text = text.replace(OLD, NEW, 1)
    path.write_text(text, encoding="utf-8")
    print("SUCCESS")
else:
    # Try LF version
    OLD_LF = OLD.replace("\r\n", "\n")
    if OLD_LF in text:
        text = text.replace(OLD_LF, NEW.replace("\r\n", "\n"), 1)
        path.write_text(text, encoding="utf-8")
        print("SUCCESS (LF)")
    else:
        print("NOT FOUND - showing relevant lines:")
        for i, line in enumerate(text.splitlines()[474:502], 475):
            print(i, repr(line[:80]))
