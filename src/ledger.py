"""SQLite-backed paper trading ledger for live/paper trading persistence."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class LedgerPosition:
    """A position in the portfolio."""

    ticker: str
    direction: str  # "long" or "short"
    shares: float
    entry_price: float
    entry_date: str
    current_value: float
    source_agent: str
    deutsch_score: float | None


@dataclass
class LedgerTrade:
    """A single trade execution record."""

    timestamp: str
    ticker: str
    direction: str
    shares: float
    fill_price: float
    commission: float
    reasoning: str


_SCHEMA = """
CREATE TABLE IF NOT EXISTS portfolio (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    direction TEXT NOT NULL,
    shares REAL NOT NULL,
    entry_price REAL NOT NULL,
    entry_date TEXT NOT NULL,
    current_value REAL NOT NULL,
    source_agent TEXT NOT NULL,
    deutsch_score REAL
);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    run_type TEXT NOT NULL,
    ticker TEXT NOT NULL,
    direction TEXT NOT NULL,
    shares REAL NOT NULL,
    fill_price REAL NOT NULL,
    commission REAL NOT NULL,
    reasoning TEXT
);

CREATE TABLE IF NOT EXISTS daily_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    run_type TEXT NOT NULL,
    date TEXT NOT NULL,
    total_value REAL NOT NULL,
    cash REAL NOT NULL,
    positions_value REAL NOT NULL,
    daily_return REAL NOT NULL,
    cumulative_return REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS recommendations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    run_type TEXT NOT NULL,
    date TEXT NOT NULL,
    agent TEXT NOT NULL,
    ticker TEXT NOT NULL,
    direction TEXT NOT NULL,
    conviction REAL NOT NULL,
    reasoning_components TEXT NOT NULL,
    conclusion TEXT,
    deutsch_score REAL,
    probed INTEGER NOT NULL DEFAULT 0,
    filtered INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS probe_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    run_type TEXT NOT NULL,
    date TEXT NOT NULL,
    agent TEXT NOT NULL,
    ticker TEXT NOT NULL,
    component_index INTEGER NOT NULL,
    component_text TEXT NOT NULL,
    perturbation TEXT NOT NULL,
    conclusion_survived INTEGER NOT NULL,
    probe_reasoning TEXT
);

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class PaperLedger:
    """SQLite-backed paper trading ledger."""

    COMMISSION_PER_TRADE: float = 1.0
    SLIPPAGE_PCT: float = 0.001

    def __init__(self, db_path: Path, starting_capital: float = 10000.0) -> None:
        self.db_path = db_path
        self._starting_capital = starting_capital
        self._init_db()

    def _init_db(self) -> None:
        """Create schema and seed meta if needed."""
        conn = self._conn()
        conn.executescript(_SCHEMA)
        # Initialize meta values if not present
        existing = conn.execute("SELECT key FROM meta").fetchall()
        existing_keys = {r["key"] for r in existing}
        if "cash" not in existing_keys:
            conn.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?)",
                ("cash", str(self._starting_capital)),
            )
        if "cumulative_return" not in existing_keys:
            conn.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?)",
                ("cumulative_return", "0.0"),
            )
        conn.commit()
        conn.close()

    def _conn(self) -> sqlite3.Connection:
        """Return a connection with Row factory."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    # ── Meta accessors ──────────────────────────────────────────────

    def get_cash(self) -> float:
        conn = self._conn()
        try:
            row = conn.execute("SELECT value FROM meta WHERE key = 'cash'").fetchone()
            return float(row["value"])
        finally:
            conn.close()

    def _set_cash(self, conn: sqlite3.Connection, amount: float) -> None:
        conn.execute("UPDATE meta SET value = ? WHERE key = 'cash'", (str(amount),))

    def get_cumulative_return(self) -> float:
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT value FROM meta WHERE key = 'cumulative_return'"
            ).fetchone()
            return float(row["value"])
        finally:
            conn.close()

    def _set_cumulative_return(self, conn: sqlite3.Connection, value: float) -> None:
        conn.execute(
            "UPDATE meta SET value = ? WHERE key = 'cumulative_return'", (str(value),)
        )

    # ── Position operations ─────────────────────────────────────────

    def get_positions(self) -> list[LedgerPosition]:
        conn = self._conn()
        try:
            rows = conn.execute("SELECT * FROM portfolio").fetchall()
            return [
                LedgerPosition(
                    ticker=r["ticker"],
                    direction=r["direction"],
                    shares=r["shares"],
                    entry_price=r["entry_price"],
                    entry_date=r["entry_date"],
                    current_value=r["current_value"],
                    source_agent=r["source_agent"],
                    deutsch_score=r["deutsch_score"],
                )
                for r in rows
            ]
        finally:
            conn.close()

    def clear_positions(self, conn: sqlite3.Connection) -> None:
        conn.execute("DELETE FROM portfolio")

    def add_position(self, conn: sqlite3.Connection, position: LedgerPosition) -> None:
        conn.execute(
            """INSERT INTO portfolio
               (ticker, direction, shares, entry_price, entry_date,
                current_value, source_agent, deutsch_score)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                position.ticker,
                position.direction,
                position.shares,
                position.entry_price,
                position.entry_date,
                position.current_value,
                position.source_agent,
                position.deutsch_score,
            ),
        )

    # ── Recording operations ────────────────────────────────────────

    def record_trade(
        self, conn: sqlite3.Connection, run_type: str, trade: LedgerTrade
    ) -> None:
        conn.execute(
            """INSERT INTO trades
               (timestamp, run_type, ticker, direction, shares,
                fill_price, commission, reasoning)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                trade.timestamp,
                run_type,
                trade.ticker,
                trade.direction,
                trade.shares,
                trade.fill_price,
                trade.commission,
                trade.reasoning,
            ),
        )

    def record_snapshot(
        self,
        conn: sqlite3.Connection,
        run_type: str,
        date: str,
        total_value: float,
        cash: float,
        positions_value: float,
        daily_return: float,
        cumulative_return: float,
    ) -> None:
        conn.execute(
            """INSERT INTO daily_snapshots
               (timestamp, run_type, date, total_value, cash,
                positions_value, daily_return, cumulative_return)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now(timezone.utc).isoformat(),
                run_type,
                date,
                total_value,
                cash,
                positions_value,
                daily_return,
                cumulative_return,
            ),
        )

    def record_recommendation(
        self,
        conn: sqlite3.Connection,
        run_type: str,
        date: str,
        agent: str,
        ticker: str,
        direction: str,
        conviction: float,
        reasoning_components: list[str],
        conclusion: str,
        deutsch_score: float | None,
        probed: bool,
        filtered: bool,
    ) -> None:
        # If conclusion is a dict (from JSON responses), serialize it
        if isinstance(conclusion, dict):
            conclusion = json.dumps(conclusion)
        conn.execute(
            """INSERT INTO recommendations
               (timestamp, run_type, date, agent, ticker, direction, conviction,
                reasoning_components, conclusion, deutsch_score, probed, filtered)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now(timezone.utc).isoformat(),
                run_type,
                date,
                agent,
                ticker,
                direction,
                conviction,
                json.dumps(reasoning_components),
                conclusion,
                deutsch_score,
                int(probed),
                int(filtered),
            ),
        )

    def record_probe_result(
        self,
        conn: sqlite3.Connection,
        run_type: str,
        date: str,
        agent: str,
        ticker: str,
        component_index: int,
        component_text: str,
        perturbation: str,
        conclusion_survived: bool,
        probe_reasoning: str,
    ) -> None:
        # If fields are dicts (from JSON responses), serialize them
        if isinstance(probe_reasoning, dict):
            probe_reasoning = json.dumps(probe_reasoning)
        if isinstance(perturbation, dict):
            perturbation = json.dumps(perturbation)
        if isinstance(component_text, dict):
            component_text = json.dumps(component_text)
        conn.execute(
            """INSERT INTO probe_results
               (timestamp, run_type, date, agent, ticker, component_index,
                component_text, perturbation, conclusion_survived, probe_reasoning)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now(timezone.utc).isoformat(),
                run_type,
                date,
                agent,
                ticker,
                component_index,
                component_text,
                perturbation,
                int(conclusion_survived),
                probe_reasoning,
            ),
        )

    # ── Rebalance engine ────────────────────────────────────────────

    def execute_rebalance(
        self,
        run_type: str,
        date: str,
        new_positions: list[LedgerPosition],
        prices: dict[str, float],
    ) -> list[LedgerTrade]:
        """Full portfolio rebalance in a single transaction.

        1. Close all existing positions
        2. Open new positions with equal weight
        3. Record trades and snapshot
        """
        conn = self._conn()
        try:
            trades: list[LedgerTrade] = []
            now = datetime.now(timezone.utc).isoformat()

            # Read current state
            cash = float(
                conn.execute("SELECT value FROM meta WHERE key = 'cash'").fetchone()[
                    "value"
                ]
            )
            prev_cumulative = float(
                conn.execute(
                    "SELECT value FROM meta WHERE key = 'cumulative_return'"
                ).fetchone()["value"]
            )
            # ── Step 1: Close existing positions ────────────────────────
            existing = conn.execute("SELECT * FROM portfolio").fetchall()

            # Mark to market: value existing positions at current prices,
            # not stale entry-time current_value.
            prev_total = cash + sum(
                row["shares"] * prices.get(row["ticker"], row["entry_price"])
                for row in existing
            )
            for row in existing:
                ticker = row["ticker"]
                direction = row["direction"]
                shares = row["shares"]
                price = prices.get(ticker, row["entry_price"])
                slippage_amount = shares * price * self.SLIPPAGE_PCT

                if direction == "long":
                    # Sell: proceeds = shares * price * (1 - slippage) - commission
                    proceeds = (
                        shares * price * (1 - self.SLIPPAGE_PCT)
                        - self.COMMISSION_PER_TRADE
                    )
                else:
                    # Cover short: cost = shares * price * (1 + slippage) + commission
                    # Original short proceeds were shares * entry_price
                    # Net: entry_proceeds - cover_cost
                    entry_proceeds = shares * row["entry_price"]
                    cover_cost = (
                        shares * price * (1 + self.SLIPPAGE_PCT)
                        + self.COMMISSION_PER_TRADE
                    )
                    proceeds = entry_proceeds - cover_cost

                cash += proceeds
                close_direction = "sell" if direction == "long" else "cover"
                trade = LedgerTrade(
                    timestamp=now,
                    ticker=ticker,
                    direction=close_direction,
                    shares=shares,
                    fill_price=price,
                    commission=self.COMMISSION_PER_TRADE + slippage_amount,
                    reasoning=f"Closing {direction} position in {ticker}",
                )
                trades.append(trade)
                self.record_trade(conn, run_type, trade)

            # ── Step 2: Clear portfolio ─────────────────────────────────
            self.clear_positions(conn)

            # ── Step 3: Open new positions with equal weight ────────────
            if new_positions and cash > 0:
                n = len(new_positions)
                allocation_per = cash / n

                for pos in new_positions:
                    price = prices.get(pos.ticker)
                    if price is None:
                        continue

                    # Effective price includes slippage
                    if pos.direction == "long":
                        effective_price = price * (1 + self.SLIPPAGE_PCT)
                    else:
                        effective_price = price * (1 - self.SLIPPAGE_PCT)

                    # Budget after commission
                    budget = allocation_per - self.COMMISSION_PER_TRADE
                    if budget <= 0:
                        continue

                    shares = budget / effective_price
                    if shares <= 0:
                        continue

                    cost = shares * effective_price + self.COMMISSION_PER_TRADE
                    if cost > cash:
                        # Reduce shares to fit
                        shares = (
                            cash - self.COMMISSION_PER_TRADE
                        ) / effective_price
                        if shares <= 0:
                            continue
                        cost = shares * effective_price + self.COMMISSION_PER_TRADE

                    cash -= cost
                    current_value = shares * price
                    slippage_amount = shares * price * self.SLIPPAGE_PCT

                    filled_pos = LedgerPosition(
                        ticker=pos.ticker,
                        direction=pos.direction,
                        shares=shares,
                        entry_price=price,
                        entry_date=pos.entry_date,
                        current_value=current_value,
                        source_agent=pos.source_agent,
                        deutsch_score=pos.deutsch_score,
                    )
                    self.add_position(conn, filled_pos)

                    buy_direction = "buy" if pos.direction == "long" else "short"
                    trade = LedgerTrade(
                        timestamp=now,
                        ticker=pos.ticker,
                        direction=buy_direction,
                        shares=shares,
                        fill_price=price,
                        commission=self.COMMISSION_PER_TRADE + slippage_amount,
                        reasoning=f"Opening {pos.direction} position in {pos.ticker}",
                    )
                    trades.append(trade)
                    self.record_trade(conn, run_type, trade)

            # ── Step 4: Compute returns and snapshot ────────────────────
            positions_value = sum(
                r["current_value"]
                for r in conn.execute(
                    "SELECT current_value FROM portfolio"
                ).fetchall()
            )
            total_value = cash + positions_value

            if prev_total > 0:
                daily_return = (total_value - prev_total) / prev_total
            else:
                daily_return = 0.0

            cumulative_return = (1 + prev_cumulative) * (1 + daily_return) - 1

            self.record_snapshot(
                conn,
                run_type,
                date,
                total_value,
                cash,
                positions_value,
                daily_return,
                cumulative_return,
            )

            # ── Step 5: Update meta ─────────────────────────────────────
            self._set_cash(conn, cash)
            self._set_cumulative_return(conn, cumulative_return)

            conn.commit()

            return trades
        finally:
            conn.close()
