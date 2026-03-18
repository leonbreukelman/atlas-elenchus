import json
from pathlib import Path
from unittest.mock import patch

import pytest

# Import the module, then patch SIGNAL_DIR
import scripts.signal_notify as sn


def test_write_paper_signal(tmp_path):
    with patch.object(sn, "SIGNAL_DIR", tmp_path):
        sn.write_paper_signal(
            run_type="morning",
            date="2026-03-17",
            portfolio_value=10234.56,
            daily_return=0.012,
            positions_changed=3,
            avg_deutsch_score=0.45,
            notable_events=["high_filter_rate"],
        )

        files = list(tmp_path.glob("*.json"))
        assert len(files) == 1

        data = json.loads(files[0].read_text())
        assert data["source"] == "atlas-elenchus"
        assert data["type"] == "paper_trading_run"
        assert data["run_type"] == "morning"
        assert data["portfolio_value"] == 10234.56
        assert data["notable_events"] == ["high_filter_rate"]


def test_write_paper_error(tmp_path):
    with patch.object(sn, "SIGNAL_DIR", tmp_path):
        sn.write_paper_error(
            run_type="evening",
            error="yfinance timeout after 3 retries",
        )

        files = list(tmp_path.glob("*.json"))
        assert len(files) == 1

        data = json.loads(files[0].read_text())
        assert data["type"] == "paper_trading_error"
        assert "yfinance" in data["error"]
