"""Tests for live market data methods — requires network access."""
from __future__ import annotations

import pytest

from src.market_data import MarketData, MarketSnapshot


SMALL_UNIVERSE = ["SPY", "AAPL", "NVDA"]


@pytest.fixture
def md() -> MarketData:
    return MarketData(universe=SMALL_UNIVERSE)


def test_snapshot_live_returns_market_snapshot(md: MarketData):
    snap = md.snapshot_live()
    assert isinstance(snap, MarketSnapshot)
    assert not snap.returns_1d.empty
    assert not snap.returns_5d.empty
    assert not snap.returns_20d.empty
    assert not snap.volatility_20d.empty
    assert snap.spy_regime in ("risk_on", "risk_off", "neutral")


def test_snapshot_live_has_prices(md: MarketData):
    snap = md.snapshot_live()
    assert not snap.prices.empty
    assert len(snap.prices) >= 20  # should have ~30 trading days


def test_snapshot_live_morning_returns_snapshot(md: MarketData):
    snap = md.snapshot_live(run_type="morning")
    assert isinstance(snap, MarketSnapshot)
    assert not snap.returns_1d.empty
    assert snap.spy_regime in ("risk_on", "risk_off", "neutral")


def test_snapshot_live_returns_fill_price(md: MarketData):
    snap = md.snapshot_live()
    fills = md.get_fill_prices(snap, run_type="evening")
    assert isinstance(fills, dict)
    assert len(fills) > 0
    for ticker, price in fills.items():
        assert price > 0, f"{ticker} fill price should be positive"
