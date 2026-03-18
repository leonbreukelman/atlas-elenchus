#!/usr/bin/env python3
"""CLI entry point for atlas-elenchus paper trading mode.

Usage:
    python scripts/run_paper.py --run-type morning
    python scripts/run_paper.py --run-type evening
    python scripts/run_paper.py --run-type morning --db-path /path/to/ledger.db
"""

import argparse
import sys
import traceback
from pathlib import Path

from dotenv import load_dotenv


def main():
    load_dotenv()

    import litellm
    litellm.suppress_debug_info = True

    parser = argparse.ArgumentParser(description="Atlas-Elenchus Paper Trading")
    parser.add_argument(
        "--run-type",
        choices=["morning", "evening"],
        required=True,
        help="Morning (fill at open) or evening (fill at close)",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("data/paper_ledger.db"),
        help="Path to SQLite ledger database",
    )
    parser.add_argument(
        "--model",
        default="openrouter/qwen/qwen3-235b-a22b",
        help="LLM model string",
    )
    parser.add_argument(
        "--probe-layers",
        default="3",
        help="Comma-separated layer numbers to probe (default: 3)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=10000.0,
        help="Starting capital (only used on first run)",
    )

    args = parser.parse_args()
    probe_layers = [int(x) for x in args.probe_layers.split(",")]

    # Ensure data directory exists
    args.db_path.parent.mkdir(parents=True, exist_ok=True)

    from src.paper import PaperTradingEngine

    engine = PaperTradingEngine(
        db_path=args.db_path,
        prompt_dir=Path("prompts/agents"),
        starting_capital=args.capital,
        model=args.model,
        probe_layers=probe_layers,
    )

    try:
        result = engine.run(args.run_type)

        if result.get("skipped"):
            print(f"Skipped: {result.get('reason', 'unknown')}")
            return

        print(f"Paper trading {args.run_type} complete:")
        print(f"  Date: {result['date']}")
        print(f"  Portfolio: ${result['portfolio_value']:.2f}")
        print(f"  Trades: {result['trades_executed']}")
        print(f"  Avg deutsch: {result['avg_deutsch_score']:.3f}")

        # Write success signal
        from scripts.signal_notify import write_paper_signal

        # Detect notable events (matches spec triggers)
        notable = []
        if result["avg_deutsch_score"] < 0.3:
            notable.append("low_deutsch_scores")
        if result["avg_deutsch_score"] > 0.8:
            notable.append("high_deutsch_scores")
        if abs(result.get("daily_return", 0)) > 0.03:
            notable.append("large_daily_move")
        if result["total_probed"] > 0:
            filter_rate = 1.0 - (result["trades_executed"] / max(result["total_recommendations"], 1))
            if filter_rate > 0.9:
                notable.append("high_filter_rate")

        write_paper_signal(
            run_type=args.run_type,
            date=result["date"],
            portfolio_value=result["portfolio_value"],
            daily_return=result.get("daily_return", 0),
            positions_changed=result["positions_changed"],
            avg_deutsch_score=result["avg_deutsch_score"],
            notable_events=notable,
        )

    except Exception as e:
        print(f"Paper trading {args.run_type} FAILED: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)

        from scripts.signal_notify import write_paper_error

        write_paper_error(
            run_type=args.run_type,
            error=str(e),
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
