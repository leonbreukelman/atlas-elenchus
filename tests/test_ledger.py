"""Tests for the SQLite paper trading ledger."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.ledger import LedgerPosition, LedgerTrade, PaperLedger


@pytest.fixture
def ledger(tmp_path: Path) -> PaperLedger:
    """Create a fresh PaperLedger in a temp directory."""
    return PaperLedger(db_path=tmp_path / "test_ledger.db")


def test_ledger_creates_tables(ledger: PaperLedger):
    """Verify all 6 tables exist after init, alphabetically sorted."""
    conn = ledger._conn()
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    table_names = [r["name"] for r in rows]
    assert table_names == [
        "daily_snapshots",
        "meta",
        "portfolio",
        "probe_results",
        "recommendations",
        "trades",
    ]


def test_ledger_initial_cash(ledger: PaperLedger):
    """Starting cash should be 10000.0."""
    assert ledger.get_cash() == 10000.0


def test_ledger_initial_positions_empty(ledger: PaperLedger):
    """No positions at start."""
    assert ledger.get_positions() == []


def test_record_trade_and_query(ledger: PaperLedger):
    """Record a trade and verify it persists."""
    trade = LedgerTrade(
        timestamp="2026-03-17T10:00:00",
        ticker="AAPL",
        direction="long",
        shares=10,
        fill_price=150.0,
        commission=1.0,
        reasoning="Test trade",
    )
    conn = ledger._conn()
    ledger.record_trade(conn, run_type="backtest", trade=trade)
    conn.commit()

    rows = conn.execute("SELECT * FROM trades").fetchall()
    assert len(rows) == 1
    assert rows[0]["ticker"] == "AAPL"
    assert rows[0]["direction"] == "long"
    assert rows[0]["shares"] == 10
    assert rows[0]["fill_price"] == 150.0
    assert rows[0]["commission"] == 1.0
    assert rows[0]["reasoning"] == "Test trade"
    assert rows[0]["run_type"] == "backtest"


def test_execute_rebalance_from_empty(ledger: PaperLedger):
    """Rebalance into 2 positions from empty portfolio."""
    new_positions = [
        LedgerPosition(
            ticker="AAPL",
            direction="long",
            shares=0,  # will be computed
            entry_price=150.0,
            entry_date="2026-03-17",
            current_value=0.0,
            source_agent="atlas",
            deutsch_score=0.8,
        ),
        LedgerPosition(
            ticker="MSFT",
            direction="long",
            shares=0,
            entry_price=400.0,
            entry_date="2026-03-17",
            current_value=0.0,
            source_agent="atlas",
            deutsch_score=0.9,
        ),
    ]
    prices = {"AAPL": 150.0, "MSFT": 400.0}

    trades = ledger.execute_rebalance(
        run_type="backtest", date="2026-03-17", new_positions=new_positions, prices=prices
    )

    # Should have produced trades (buys for each position)
    assert len(trades) >= 2

    # Cash should have decreased
    cash = ledger.get_cash()
    assert cash < 10000.0
    assert cash >= 0.0

    # 2 positions should exist
    positions = ledger.get_positions()
    assert len(positions) == 2

    # Total value (cash + positions) should be approximately starting capital minus costs
    total_value = cash + sum(p.current_value for p in positions)
    assert total_value < 10000.0  # costs incurred
    assert total_value > 9900.0  # but not dramatically less


def test_execute_rebalance_closes_old_positions(ledger: PaperLedger):
    """Rebalance into NVDA, then into AAPL — only AAPL should remain."""
    # First rebalance: buy NVDA
    ledger.execute_rebalance(
        run_type="backtest",
        date="2026-03-17",
        new_positions=[
            LedgerPosition(
                ticker="NVDA",
                direction="long",
                shares=0,
                entry_price=800.0,
                entry_date="2026-03-17",
                current_value=0.0,
                source_agent="atlas",
                deutsch_score=None,
            ),
        ],
        prices={"NVDA": 800.0},
    )

    positions = ledger.get_positions()
    assert len(positions) == 1
    assert positions[0].ticker == "NVDA"

    # Second rebalance: switch to AAPL
    ledger.execute_rebalance(
        run_type="backtest",
        date="2026-03-18",
        new_positions=[
            LedgerPosition(
                ticker="AAPL",
                direction="long",
                shares=0,
                entry_price=150.0,
                entry_date="2026-03-18",
                current_value=0.0,
                source_agent="atlas",
                deutsch_score=0.7,
            ),
        ],
        prices={"NVDA": 800.0, "AAPL": 150.0},
    )

    positions = ledger.get_positions()
    assert len(positions) == 1
    assert positions[0].ticker == "AAPL"


def test_rebalance_enforces_cash(ledger: PaperLedger):
    """Trying to buy an extremely expensive stock should not result in negative cash."""
    ledger.execute_rebalance(
        run_type="backtest",
        date="2026-03-17",
        new_positions=[
            LedgerPosition(
                ticker="BRK.A",
                direction="long",
                shares=0,
                entry_price=600000.0,
                entry_date="2026-03-17",
                current_value=0.0,
                source_agent="atlas",
                deutsch_score=None,
            ),
        ],
        prices={"BRK.A": 600000.0},
    )

    cash = ledger.get_cash()
    assert cash >= 0.0
