"""
Trading agent. Wraps a litellm completion call with a system prompt (the mutable artifact).
Every recommendation includes structured reasoning — Elenchus probes this.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from .llm import completion_with_retry
from .market_data import MarketSnapshot


@dataclass
class Recommendation:
    """Agent output. Reasoning components are individually addressable for probing."""
    agent_id: str
    date: str
    ticker: str
    direction: str                    # "long" | "short" | "avoid"
    conviction: float                 # 0.0 - 1.0
    reasoning_components: list[str]   # each component is independently probeable
    conclusion: str                   # the synthesized rationale
    raw_response: str = ""


@dataclass
class Agent:
    """
    A single trading agent. Its prompt is the mutable artifact (like train.py in autoresearch).
    The darwinian weight determines signal strength in the pipeline.
    """
    agent_id: str
    layer: int                        # 1=macro, 2=sector, 3=decision
    prompt_path: Path                 # path to the .md prompt file
    darwinian_weight: float = 1.0     # range [0.3, 2.5]
    rolling_sharpe: float = 0.0
    total_recommendations: int = 0
    correct_recommendations: int = 0

    # Darwinian bounds
    WEIGHT_MIN: float = 0.3
    WEIGHT_MAX: float = 2.5
    WEIGHT_BOOST: float = 1.05
    WEIGHT_DECAY: float = 0.95

    @property
    def prompt(self) -> str:
        return self.prompt_path.read_text()

    @prompt.setter
    def prompt(self, value: str):
        self.prompt_path.write_text(value)

    def recommend(
        self,
        snapshot: MarketSnapshot,
        upstream_signals: list[Recommendation] | None = None,
        model: str = "openrouter/qwen/qwen3-235b-a22b",
    ) -> list[Recommendation]:
        """
        Produce recommendations for current market snapshot.
        Upstream signals come from prior layers (L1→L2→L3).

        Response contract — agent must return JSON:
        {
          "recommendations": [
            {
              "ticker": "NVDA",
              "direction": "long",
              "conviction": 0.85,
              "reasoning_components": [
                "Semiconductor capex cycle accelerating per TSMC guidance",
                "Relative strength vs SOX index over trailing 20d",
                "Data center revenue mix shift reduces cyclicality"
              ],
              "conclusion": "..."
            }
          ]
        }
        """
        context = self._build_context(snapshot, upstream_signals)

        response = completion_with_retry(
            model=model,
            max_tokens=2000,
            timeout=180,
            messages=[
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": context},
            ],
        )

        raw = response.choices[0].message.content
        return self._parse_response(raw, snapshot)

    def _build_context(
        self,
        snapshot: MarketSnapshot,
        upstream: list[Recommendation] | None,
    ) -> str:
        parts = [
            f"Date: {snapshot.date.strftime('%Y-%m-%d')}",
            f"Regime: {snapshot.spy_regime}",
            "",
            "Trailing returns (1d / 5d / 20d) and 20d vol:",
        ]

        for ticker in snapshot.returns_1d.index:
            r1 = snapshot.returns_1d.get(ticker, 0)
            r5 = snapshot.returns_5d.get(ticker, 0) if hasattr(snapshot.returns_5d, 'get') else 0
            r20 = snapshot.returns_20d.get(ticker, 0) if hasattr(snapshot.returns_20d, 'get') else 0
            vol = snapshot.volatility_20d.get(ticker, 0) if hasattr(snapshot.volatility_20d, 'get') else 0
            parts.append(f"  {ticker}: {r1:+.2%} / {r5:+.2%} / {r20:+.2%}  vol={vol:.1%}")

        if upstream:
            parts.append("")
            parts.append("Upstream signals from prior layers:")
            for rec in upstream:
                parts.append(
                    f"  [{rec.agent_id}] {rec.ticker} {rec.direction} "
                    f"conviction={rec.conviction:.2f}: {rec.conclusion[:120]}"
                )

        parts.append("")
        parts.append(
            "Respond with JSON only. Each recommendation MUST include "
            "'reasoning_components' as a list of 3-5 independent reasoning components. "
            "Each component must be a distinct, falsifiable claim — not a restatement of the conclusion."
        )
        return "\n".join(parts)

    def _parse_response(self, raw: str, snapshot: MarketSnapshot) -> list[Recommendation]:
        try:
            # Strip markdown fences if present
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1]
            if cleaned.endswith("```"):
                cleaned = cleaned.rsplit("```", 1)[0]
            cleaned = cleaned.strip()

            data = json.loads(cleaned)
            # Handle both {"recommendations": [...]} and bare [...]
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict):
                items = data.get("recommendations", [])
            else:
                return []
            recs = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                ticker = item.get("ticker")
                if not ticker:
                    continue
                recs.append(Recommendation(
                    agent_id=self.agent_id,
                    date=snapshot.date.strftime("%Y-%m-%d"),
                    ticker=ticker,
                    direction=item.get("direction", "long"),
                    conviction=self._parse_conviction(item.get("conviction", 0.5)),
                    reasoning_components=item.get("reasoning_components", []),
                    conclusion=item.get("conclusion", ""),
                    raw_response=raw,
                ))
            return recs
        except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
            # Agent produced unparseable output — counts as a failed recommendation
            return []

    @staticmethod
    def _parse_conviction(value) -> float:
        """Parse conviction from numeric or text values."""
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                mapping = {
                    "very high": 0.9, "high": 0.8,
                    "moderate-high": 0.7, "medium-high": 0.7,
                    "moderate": 0.6, "medium": 0.6,
                    "moderate-low": 0.4, "medium-low": 0.4,
                    "low": 0.3, "very low": 0.2,
                }
                return mapping.get(value.lower().strip(), 0.5)
        return 0.5

    def update_weight(self, is_top_quartile: bool) -> None:
        """Darwinian weight adjustment. Called after each scoring period."""
        if is_top_quartile:
            self.darwinian_weight = min(self.darwinian_weight * self.WEIGHT_BOOST, self.WEIGHT_MAX)
        else:
            self.darwinian_weight = max(self.darwinian_weight * self.WEIGHT_DECAY, self.WEIGHT_MIN)
