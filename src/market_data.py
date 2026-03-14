"""
Market data provider. Fetches real historical data via yfinance.
Provides a day-by-day iterator for backtesting — agents never see future data.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from typing import Iterator


# Default universe — diversified enough to test macro/sector/quality signals
DEFAULT_UNIVERSE = [
    # Tech/Semi
    "NVDA", "AVGO", "TSM", "MSFT", "AAPL",
    # Energy
    "XOM", "CVX", "SLB",
    # Financials
    "JPM", "GS", "BRK-B",
    # Industrials
    "CAT", "GE", "LMT",
    # Consumer
    "AMZN", "COST", "PG",
    # Biotech
    "LLY", "AMGN", "MRNA",
    # Macro ETFs (for regime signals)
    "SPY", "QQQ", "TLT", "GLD", "UUP", "VIX",
]

# Macro series from FRED (fetched separately if FMP_API_KEY is set)
FRED_SERIES = ["DFF", "T10Y2Y", "BAMLH0A0HYM2"]


@dataclass
class MarketSnapshot:
    """What agents see on a given day. No future leakage."""
    date: datetime
    prices: pd.DataFrame          # OHLCV for universe, up to and including this date
    returns_1d: pd.Series         # single-day returns
    returns_5d: pd.Series         # trailing 5-day returns
    returns_20d: pd.Series        # trailing 20-day returns
    volatility_20d: pd.Series     # 20-day realized vol
    spy_regime: str               # "risk_on" | "risk_off" | "neutral"


class MarketData:
    """
    Fetches historical data once, then iterates day-by-day for backtesting.
    Each MarketSnapshot contains only data available on that date.
    """

    def __init__(
        self,
        universe: list[str] | None = None,
        start: str = "2024-01-01",
        end: str | None = None,
    ):
        self.universe = universe or DEFAULT_UNIVERSE
        self.start = start
        self.end = end or datetime.now().strftime("%Y-%m-%d")
        self._prices: pd.DataFrame | None = None

    def fetch(self) -> None:
        """Download all data. Call once before iterating."""
        tickers = [t for t in self.universe if t != "VIX"]
        data = yf.download(tickers, start=self.start, end=self.end, progress=False)
        self._prices = data
        # VIX via ^VIX
        try:
            vix = yf.download("^VIX", start=self.start, end=self.end, progress=False)
            if not vix.empty:
                self._prices[("Close", "VIX")] = vix["Close"]
        except Exception:
            pass

    def _classify_regime(self, spy_returns_20d: float, spy_vol_20d: float) -> str:
        if spy_returns_20d > 0.02 and spy_vol_20d < 0.20:
            return "risk_on"
        elif spy_returns_20d < -0.02 or spy_vol_20d > 0.25:
            return "risk_off"
        return "neutral"

    def iterate(self, warmup_days: int = 30) -> Iterator[MarketSnapshot]:
        """Yield one MarketSnapshot per trading day, after warmup period."""
        if self._prices is None:
            raise RuntimeError("Call fetch() first")

        close = self._prices["Close"] if "Close" in self._prices.columns.get_level_values(0) else self._prices
        dates = close.index[warmup_days:]

        for i, date in enumerate(dates):
            idx = warmup_days + i
            window = close.iloc[:idx + 1]

            returns_1d = window.pct_change().iloc[-1]
            returns_5d = (window.iloc[-1] / window.iloc[-5] - 1) if idx >= 5 else returns_1d
            returns_20d = (window.iloc[-1] / window.iloc[-20] - 1) if idx >= 20 else returns_1d
            volatility_20d = window.pct_change().iloc[-20:].std() * np.sqrt(252)

            spy_ret = returns_20d.get("SPY", 0.0) if isinstance(returns_20d, pd.Series) else 0.0
            spy_vol = volatility_20d.get("SPY", 0.15) if isinstance(volatility_20d, pd.Series) else 0.15
            regime = self._classify_regime(spy_ret, spy_vol)

            yield MarketSnapshot(
                date=date,
                prices=window,
                returns_1d=returns_1d,
                returns_5d=returns_5d,
                returns_20d=returns_20d,
                volatility_20d=volatility_20d,
                spy_regime=regime,
            )
