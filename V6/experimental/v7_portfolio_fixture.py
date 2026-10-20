"""Fixed synthetic history demo; never reads predictions or market data."""
from pathlib import Path
from V6.experimental.v7_portfolio_contract import PortfolioSpec
from V6.experimental.v7_portfolio_journal import PortfolioJournal
from V6.experimental.v7_portfolio_history import canonical, fixture_manifest, export_history, digest, HistoryError

def create_fixture(directory, source_commit):
    directory = Path(directory)
    # A new explicit directory isolates every run; no overwrite/recovery of inputs.
    import tempfile
    allowed = (Path(__file__).resolve().parents[2] / ".artifacts", Path(tempfile.gettempdir()))
    if not any(directory.resolve().is_relative_to(root.resolve()) for root in allowed):
        raise HistoryError("fixture directory must be checkout .artifacts or temporary")
    directory.mkdir(parents=True, exist_ok=False)
    spec = PortfolioSpec(holdings_count=1, buffer_multiple="1.5", rebalance_every_sessions=5, head="5d")
    journal = PortfolioJournal.create(directory / "fixture.jsonl", spec, "genesis", "2026-09-16T08:00:00+08:00")
    journal.append("no-target", "MARKET_SESSION", "2026-09-16T09:00:00+08:00", {"quotes": {}})
    journal.append("signal", "SIGNAL", "2026-09-16T17:00:00+08:00", {"head":"5d", "scores":{"A":"1"}})
    def quotes(status, fill):
        return {"quotes":{"A":{"ticker":"A", "price":"10", "status":status, "buy_fill_ratio":fill, "sell_fill_ratio":"1"}}}
    journal.append("blocked", "MARKET_SESSION", "2026-09-17T09:00:00+08:00", quotes("BUY_BLOCKED", "1"))
    journal.append("partial", "MARKET_SESSION", "2026-09-18T09:00:00+08:00", quotes("OPEN", "0.5"))
    payload = quotes("OPEN", "1")
    journal.append("filled", "MARKET_SESSION", "2026-09-19T09:00:00+08:00", payload)
    retried = journal.append("filled", "MARKET_SESSION", "2026-09-19T09:00:00+08:00", payload)
    assert not retried.appended
    journal.append("distribution", "CORPORATE_ACTION", "2026-09-20T09:00:00+08:00", {"ticker":"A", "quantity_multiplier":"2", "cash_per_old_share":"0.1", "post_action_price":"5"})
    manifest = directory / "fixture-manifest.json"
    manifest.write_bytes(canonical(fixture_manifest(journal.path, "portfolio-history-fixed-synthetic-v1")))
    output = directory / "history.json"
    export_history(journal.path, output, manifest, source_commit)
    return {"artifact": str(output.resolve()), "sha256": digest(output.read_bytes()), "event_count": len(journal.records())}

def main():
    import argparse, json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    try: result = create_fixture(args.directory, args.source_commit)
    except (HistoryError, OSError) as exc: parser.exit(2, f"{exc}\n")
    print(json.dumps(result, sort_keys=True))

if __name__ == "__main__": main()
