#!/usr/bin/env python3
"""Write a signal file for MÆI to pick up on session start."""

import json
import sys
import uuid
from datetime import datetime
from pathlib import Path

SIGNAL_DIR = Path.home() / "maei" / "data" / "signals" / "inbox"


def write_signal(status: str, days_completed: int = 0, total_days: int = 250,
                 last_date: str = "", exit_code: int = 0, attempt: int = 0):
    """Write a JSON signal file that MÆI's SessionStart hook can detect."""
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().isoformat()
    signal_id = f"atlas-elenchus-{timestamp.replace(':', '-')}"

    signal = {
        "source": "atlas-elenchus",
        "type": "backtest_status",
        "status": status,
        "days_completed": days_completed,
        "total_days": total_days,
        "last_date": last_date,
        "exit_code": exit_code,
        "attempt": attempt,
        "timestamp": timestamp,
    }

    signal_file = SIGNAL_DIR / f"{signal_id}.json"
    signal_file.write_text(json.dumps(signal, indent=2))
    print(f"Signal written: {signal_file}")


def write_paper_signal(
    run_type: str,
    date: str,
    portfolio_value: float,
    daily_return: float,
    positions_changed: int,
    avg_deutsch_score: float,
    notable_events: list[str] | None = None,
) -> None:
    """Write a paper trading run signal."""
    signal = {
        "source": "atlas-elenchus",
        "type": "paper_trading_run",
        "run_type": run_type,
        "date": date,
        "portfolio_value": portfolio_value,
        "daily_return": daily_return,
        "positions_changed": positions_changed,
        "avg_deutsch_score": avg_deutsch_score,
        "notable_events": notable_events or [],
        "timestamp": datetime.utcnow().isoformat(),
    }
    signal_id = str(uuid.uuid4())
    path = SIGNAL_DIR / f"{signal_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(signal, indent=2))


def write_paper_error(
    run_type: str,
    error: str,
) -> None:
    """Write a paper trading error signal."""
    signal = {
        "source": "atlas-elenchus",
        "type": "paper_trading_error",
        "run_type": run_type,
        "error": error,
        "timestamp": datetime.utcnow().isoformat(),
    }
    signal_id = str(uuid.uuid4())
    path = SIGNAL_DIR / f"{signal_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(signal, indent=2))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: signal_notify.py <status> [days_completed] [total_days] [last_date] [exit_code] [attempt]")
        sys.exit(1)

    write_signal(
        status=sys.argv[1],
        days_completed=int(sys.argv[2]) if len(sys.argv) > 2 else 0,
        total_days=int(sys.argv[3]) if len(sys.argv) > 3 else 250,
        last_date=sys.argv[4] if len(sys.argv) > 4 else "",
        exit_code=int(sys.argv[5]) if len(sys.argv) > 5 else 0,
        attempt=int(sys.argv[6]) if len(sys.argv) > 6 else 0,
    )
