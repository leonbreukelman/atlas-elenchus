#!/usr/bin/env python3
"""Write a signal file for MÆI to pick up on session start."""

import json
import sys
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
