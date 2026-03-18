"""Paper trading engine for atlas-elenchus.

Orchestrates: live market data -> pipeline -> ledger -> signals.

Note: Autoresearch prompt mutations are intentionally omitted from
paper trading mode. The first 30 days establish a stable baseline
corpus. After 30 days, AutoresearchLoop can be integrated with a
configurable mutation interval.
"""

import traceback
from datetime import datetime
from pathlib import Path

from src.agent import Recommendation
from src.elenchus import ElenchusResult
from src.ledger import LedgerPosition, PaperLedger
from src.market_data import MarketData
from src.pipeline import Pipeline

NON_TRADEABLE = {"VIX", "^VIX"}


class PaperTradingEngine:
    def __init__(
        self,
        db_path: Path,
        prompt_dir: Path,
        starting_capital: float = 10000.0,
        model: str = "openrouter/qwen/qwen3-235b-a22b",
        probe_layers: list[int] | None = None,
    ):
        self.ledger = PaperLedger(db_path=db_path, starting_capital=starting_capital)
        self.prompt_dir = prompt_dir
        self.model = model
        self.probe_layers = probe_layers or [3]
        self.pipeline = Pipeline(
            prompt_dir=prompt_dir,
            use_elenchus=True,
            model=model,
            probe_layers=self.probe_layers,
        )
        self.market = MarketData()

    def run(self, run_type: str) -> dict:
        """Execute a single paper trading run.

        Args:
            run_type: "morning" or "evening"

        Returns:
            Summary dict for signal emission.
        """
        # Check market availability
        if not self.market.is_market_open_today():
            return {"skipped": True, "reason": "Market closed"}

        # Fetch live data
        snapshot = self.market.snapshot_live(run_type=run_type)
        date_str = snapshot.date.strftime("%Y-%m-%d")

        # Run agent pipeline
        recommendations, elenchus_results = self.pipeline.run_day(snapshot)

        # Filter non-tradeable
        recommendations = self._filter_non_tradeable(recommendations)

        # Record all recommendations and probe results
        conn = self.ledger._conn()
        try:
            elenchus_map = {}
            for er in elenchus_results:
                key = f"{er.recommendation.agent_id}:{er.recommendation.ticker}"
                elenchus_map[key] = er

            for rec in recommendations:
                key = f"{rec.agent_id}:{rec.ticker}"
                er = elenchus_map.get(key)
                self.ledger.record_recommendation(
                    conn=conn,
                    run_type=run_type,
                    date=date_str,
                    agent=rec.agent_id,
                    ticker=rec.ticker,
                    direction=rec.direction,
                    conviction=rec.conviction,
                    reasoning_components=rec.reasoning_components,
                    conclusion=rec.conclusion,
                    deutsch_score=er.deutsch_score if er else None,
                    probed=er is not None,
                    filtered=er is not None and not er.is_hard_to_vary,
                )

                if er:
                    for pr in er.probe_results:
                        self.ledger.record_probe_result(
                            conn=conn,
                            run_type=run_type,
                            date=date_str,
                            agent=rec.agent_id,
                            ticker=rec.ticker,
                            component_index=pr.component_index,
                            component_text=pr.original_component,
                            perturbation=pr.replacement_component,
                            conclusion_survived=pr.conclusion_survived,
                            probe_reasoning=pr.probe_reasoning,
                        )
            conn.commit()
        finally:
            conn.close()

        # Score and select positions
        new_positions = self._select_positions(
            recommendations, elenchus_results, snapshot
        )

        # Get fill prices
        fill_prices = self.market.get_fill_prices(snapshot, run_type)

        # Execute rebalance
        old_count = len(self.ledger.get_positions())
        trades = self.ledger.execute_rebalance(
            run_type, date_str, new_positions, fill_prices
        )
        new_count = len(self.ledger.get_positions())

        # Read daily return from the snapshot the ledger just recorded
        conn = self.ledger._conn()
        try:
            last_snap = conn.execute(
                "SELECT daily_return FROM daily_snapshots ORDER BY id DESC LIMIT 1"
            ).fetchone()
        finally:
            conn.close()
        daily_ret = float(last_snap["daily_return"]) if last_snap else 0.0

        portfolio_value = self.ledger.get_cash() + sum(
            p.shares * fill_prices.get(p.ticker, p.entry_price)
            for p in self.ledger.get_positions()
        )

        # Build summary
        avg_deutsch = 0.0
        if elenchus_results:
            avg_deutsch = sum(
                er.deutsch_score for er in elenchus_results
            ) / len(elenchus_results)

        return {
            "skipped": False,
            "date": date_str,
            "run_type": run_type,
            "portfolio_value": portfolio_value,
            "daily_return": daily_ret,
            "positions_changed": abs(new_count - old_count) + len(trades),
            "avg_deutsch_score": round(avg_deutsch, 3),
            "total_recommendations": len(recommendations),
            "total_probed": len(elenchus_results),
            "trades_executed": len(trades),
        }

    def _filter_non_tradeable(
        self, recommendations: list[Recommendation]
    ) -> list[Recommendation]:
        return [r for r in recommendations if r.ticker not in NON_TRADEABLE]

    def _select_positions(
        self,
        recommendations: list[Recommendation],
        elenchus_results: list[ElenchusResult],
        snapshot,
    ) -> list[LedgerPosition]:
        """Score, rank, and select top positions.

        Uses existing scoring logic:
        score = conviction * darwinian_weight * deutsch_score
        Select top 10, equal-weight.
        """
        elenchus_map = {}
        for er in elenchus_results:
            key = f"{er.recommendation.agent_id}:{er.recommendation.ticker}"
            elenchus_map[key] = er

        scored = []
        for rec in recommendations:
            if rec.direction == "avoid":
                continue

            agent = self.pipeline.agents.get(rec.agent_id)
            if agent is None:
                continue

            key = f"{rec.agent_id}:{rec.ticker}"
            er = elenchus_map.get(key)
            deutsch = er.deutsch_score if er else 0.5

            score = rec.conviction * agent.darwinian_weight * deutsch
            scored.append((score, rec, deutsch, agent.agent_id))

        # Sort by score descending, take top 10
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:10]

        date_str = snapshot.date.strftime("%Y-%m-%d")
        return [
            LedgerPosition(
                ticker=rec.ticker,
                direction=rec.direction,
                shares=0,  # computed by ledger
                entry_price=0,  # filled by ledger
                entry_date=date_str,
                current_value=0,
                source_agent=agent_id,
                deutsch_score=deutsch,
            )
            for _, rec, deutsch, agent_id in top
        ]
