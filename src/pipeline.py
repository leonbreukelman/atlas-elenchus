"""
Multi-layer agent pipeline. Simplified from ATLAS's 4-layer / 25-agent structure
to a 3-layer / 8-agent structure for tractable experimentation.

Layer 1 (Macro):     regime, rates, volatility
Layer 2 (Sector):    tech, energy, quality
Layer 3 (Decision):  risk officer, CIO

Each layer feeds into the next. Elenchus probes every recommendation
before it passes upstream.
"""

from pathlib import Path

from .agent import Agent, Recommendation
from .elenchus import ElenchusProbe, ElenchusResult
from .market_data import MarketSnapshot


# Default agent configuration — 8 agents across 3 layers
DEFAULT_AGENTS = [
    # Layer 1: Macro regime
    {"agent_id": "macro_regime",   "layer": 1, "prompt_file": "macro_regime.md"},
    {"agent_id": "rates_yield",    "layer": 1, "prompt_file": "rates_yield.md"},
    {"agent_id": "volatility",     "layer": 1, "prompt_file": "volatility.md"},
    # Layer 2: Sector
    {"agent_id": "tech_semi",      "layer": 2, "prompt_file": "tech_semi.md"},
    {"agent_id": "energy",         "layer": 2, "prompt_file": "energy.md"},
    {"agent_id": "quality",        "layer": 2, "prompt_file": "quality.md"},
    # Layer 3: Decision
    {"agent_id": "risk_officer",   "layer": 3, "prompt_file": "risk_officer.md"},
    {"agent_id": "cio",            "layer": 3, "prompt_file": "cio.md"},
]


class Pipeline:
    """
    Runs the agent layers sequentially. Each layer's output feeds into the next.
    Optionally runs Elenchus probes between layers.
    """

    def __init__(
        self,
        prompt_dir: Path,
        use_elenchus: bool = False, random_mode: bool = False,
        model: str = "openrouter/qwen/qwen3.5-plus-02-15",
        probe_model: str | None = None,
        probe_layers: list[int] | None = None,
    ):
        self.use_elenchus = use_elenchus
        self.model = model
        self.probe_layers = probe_layers or [1, 2, 3]
        self.probe = ElenchusProbe(model=model, probe_model=probe_model, random_mode=random_mode) if use_elenchus else None

        # Initialize agents
        self.agents: dict[str, Agent] = {}
        for cfg in DEFAULT_AGENTS:
            agent = Agent(
                agent_id=cfg["agent_id"],
                layer=cfg["layer"],
                prompt_path=prompt_dir / cfg["prompt_file"],
            )
            self.agents[agent.agent_id] = agent

        # Group by layer
        self.layers: dict[int, list[Agent]] = {}
        for agent in self.agents.values():
            self.layers.setdefault(agent.layer, []).append(agent)

    def run_day(
        self,
        snapshot: MarketSnapshot,
    ) -> tuple[list[Recommendation], list[ElenchusResult]]:
        """
        Run all layers for a single day. Returns final recommendations
        and all Elenchus results (empty list if Elenchus disabled).
        """
        all_recommendations: list[Recommendation] = []
        all_elenchus: list[ElenchusResult] = []
        upstream: list[Recommendation] = []

        for layer_num in sorted(self.layers.keys()):
            layer_agents = self.layers[layer_num]
            layer_recs: list[Recommendation] = []

            for agent in layer_agents:
                try:
                    recs = agent.recommend(
                        snapshot=snapshot,
                        upstream_signals=upstream if upstream else None,
                        model=self.model,
                    )
                    layer_recs.extend(recs)
                except Exception as e:
                    # Agent failure — log and continue
                    print(f"  [{agent.agent_id}] FAILED: {e}")
                    continue

            # Elenchus probe on this layer's output
            if self.probe and layer_recs and layer_num in self.probe_layers:
                # Pre-probe pruning: only probe recommendations that could enter the portfolio
                def base_score(r):
                    agent = self.agents.get(r.agent_id)
                    return r.conviction * (agent.darwinian_weight if agent else 1.0)

                layer_recs.sort(key=base_score, reverse=True)
                recs_to_probe = layer_recs[:20]

                elenchus_results = self.probe.probe_batch(recs_to_probe)
                all_elenchus.extend(elenchus_results)

                # Filter: only pass hard-to-vary recommendations upstream
                hard_recs = [er.recommendation for er in elenchus_results if er.is_hard_to_vary]
                for er in elenchus_results:
                    if not er.is_hard_to_vary:
                        print(
                            f"  [{er.recommendation.agent_id}] {er.recommendation.ticker} "
                            f"FILTERED by Elenchus (deutsch={er.deutsch_score:.2f})"
                        )

                layer_recs = hard_recs

            all_recommendations.extend(layer_recs)
            upstream = layer_recs  # feed to next layer

        return all_recommendations, all_elenchus

    def worst_agent(self) -> Agent:
        """Find the agent with lowest rolling Sharpe — target for autoresearch mutation."""
        return min(self.agents.values(), key=lambda a: a.rolling_sharpe)
