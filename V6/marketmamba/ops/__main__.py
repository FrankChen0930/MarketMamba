"""Read-only operational diagnostics command line."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from marketmamba.ops.diagnostics import render_current_state
from marketmamba.ops.projection import build_current_state
from marketmamba.ops.shadow import default_ledger_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m marketmamba.ops")
    commands = parser.add_subparsers(dest="command", required=True)
    current = commands.add_parser(
        "current-state",
        help="read the latest committed FETCH_ONLY operational state",
        epilog=(
            "The observer publishes current-state.json beside the ledger; "
            "this read-only CLI never rewrites it."
        ),
    )
    current.add_argument(
        "--db",
        type=Path,
        default=None,
        help="operational SQLite path (defaults to the ignored Data/ops path)",
    )
    current.add_argument(
        "--json",
        action="store_true",
        help="emit machine-readable JSON to stdout (not to a file)",
    )
    return parser


def _default_database() -> Path:
    v6_dir = Path(__file__).resolve().parents[2]
    return default_ledger_path(v6_dir)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command != "current-state":
        return 2
    database = arguments.db if arguments.db is not None else _default_database()
    try:
        state = build_current_state(database)
    except BaseException as error:
        # Database paths and exception messages can contain credentials. Keep
        # diagnostics useful without echoing either one.
        print(
            "unable to read operational ledger "
            f"({type(error).__name__}); no state was changed",
            file=sys.stderr,
        )
        return 2
    if arguments.json:
        json.dump(state, sys.stdout, ensure_ascii=False, sort_keys=True, indent=2)
        sys.stdout.write("\n")
    else:
        print(render_current_state(state))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
