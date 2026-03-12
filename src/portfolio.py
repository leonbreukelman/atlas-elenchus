"""
Portfolio tracker and scoring engine.
Converts agent recommendations into sized positions, tracks P&L, computes Sharpe.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime

from .agent import Agent, Recommendation
from .elenchus import ElenchusResult


@dataclass
class Position:
    ticker: str
    direction: str       # "long" | "short"
    entry_price: float
    entry_date: str
    size: float          # portfolio weight [0.0, 1.0]
    source_agent: str
    deutsch_score: float | None = None  # None if Elenchus not active


@dataclass
class DailyPnL:
    date: str
    portfolio_return: float
    cumulative_return: float
    positions: list[Position]


class Portfolio:
    """
    Converts recommendations to positions, tracks returns.
    Two modes: vanilla (conviction-weighted) and elenchus (deutsch-score-weighted).
    """

    def __init__(self, use_elenchus: bool = False, max_positions: int = 10):
        self.use_elenchus = use_elenchus
        self.max_positions = max_positions
        self.positions: list[Position] = []
        self.daily_returns: list[float] = []
        self.daily_log: list[DailyPnL] = []
        self.cumulative = 1.0

    def rebalance(
        self,
        recommendations: list[Recommendation],
        agents: dict[str, Agent],
        snapshot,
        elenchus_results: list[ElenchusResult] | None = None,
    ) -> None:
        """
        Score and rank recommendations, build portfolio.
        
        Scoring:
          vanilla:   score = conviction * darwinian_weight
          elenchus:  score = conviction * darwinian_weight * deutsch_score
        
        The deutsch_score multiplier is the only difference between branches.
        """
        elenchus_map: dict[str, float] = {}
        if elenchus_results:
            for er in elenchus_results:
                key = f"{er.recommendation.agent_id}:{er.recommendation.ticker}"
                elenchus_map[key] = er.deutsch_score

        scored = []
        for rec in recommendations:
            agent = agents.get(rec.agent_id)
            if not agent or rec.direction == "avoid":
                continue

            weight = agent.darwinian_weight
            score = rec.conviction * weight

            deutsch = None
            if self.use_elenchus:
                key = f"{rec.agent_id}:{rec.ticker}"
                deutsch = elenchus_map.get(key, 0.5)  # default 0.5 if not probed
                score *= deutsch

            scored.append((score, rec, deutsch))

        # Rank by score, take top N
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:self.max_positions]

        # Equal-weight among selected positions (simplification)
        if not top:
            self.positions = []
            return

        pos_weight = 1.0 / len(top)
        self.positions = []
        for _, rec, deutsch in top:
            price = snapshot.prices["Close"].iloc[-1].get(rec.ticker, 0) if hasattr(snapshot.prices["Close"].iloc[-1], 'get') else 0
            self.positions.append(Position(
                ticker=rec.ticker,
                direction=rec.direction,
                entry_price=price,
                entry_date=rec.date,
                size=pos_weight,
                source_agent=rec.agent_id,
                deutsch_score=deutsch,
            ))

    def mark_to_market(self, snapshot) -> float:
        """Calculate daily return from current positions."""
        if not self.positions:
            self.daily_returns.append(0.0)
            return 0.0

        daily_ret = 0.0
        for pos in self.positions:
            ticker_ret = snapshot.returns_1d.get(pos.ticker, 0.0) if hasattr(snapshot.returns_1d, 'get') else 0.0
            if pos.direction == "short":
                ticker_ret = -ticker_ret
            daily_ret += ticker_ret * pos.size

        self.daily_returns.append(daily_ret)
        self.cumulative *= (1 + daily_ret)

        self.daily_log.append(DailyPnL(
            date=snapshot.date.strftime("%Y-%m-%d"),
            portfolio_return=daily_ret,
            cumulative_return=self.cumulative - 1.0,
            positions=list(self.positions),
        ))

        return daily_ret

    @property
    def sharpe(self) -> float:
        """Annualized Sharpe ratio. The scalar metric."""
        if len(self.daily_returns) < 5:
            return 0.0
        arr = np.array(self.daily_returns)
        if arr.std() == 0:
            return 0.0
        return float((arr.mean() / arr.std()) * np.sqrt(252))

    def rolling_sharpe(self, window: int = 20) -> float:
        """Rolling Sharpe over last N days."""
        if len(self.daily_returns) < window:
            return self.sharpe
        arr = np.array(self.daily_returns[-window:])
        if arr.std() == 0:
            return 0.0
        return float((arr.mean() / arr.std()) * np.sqrt(252))

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"date": d.date, "return": d.portfolio_return, "cumulative": d.cumulative_return}
            for d in self.daily_log
        ])


def score_agents(
    agents: list[Agent],
    recommendations_by_agent: dict[str, list[tuple[Recommendation, float]]],
) -> None:
    """
    Score each agent on recommendation accuracy (was direction correct?).
    Update darwinian weights based on quartile ranking.
    """
    sharpes = {}
    for agent in agents:
        recs = recommendations_by_agent.get(agent.agent_id, [])
        if not recs:
            sharpes[agent.agent_id] = 0.0
            continue

        returns = []
        for rec, actual_return in recs:
            if rec.direction == "long":
                returns.append(actual_return)
            elif rec.direction == "short":
                returns.append(-actual_return)

        arr = np.array(returns) if returns else np.array([0.0])
        sharpes[agent.agent_id] = float(arr.mean() / arr.std() * np.sqrt(252)) if arr.std() > 0 else 0.0

    # Quartile ranking
    sorted_agents = sorted(agents, key=lambda a: sharpes.get(a.agent_id, 0), reverse=True)
    top_quartile_cutoff = len(sorted_agents) // 4 or 1

    for i, agent in enumerate(sorted_agents):
        agent.rolling_sharpe = sharpes.get(agent.agent_id, 0.0)
        agent.update_weight(is_top_quartile=(i < top_quartile_cutoff))
