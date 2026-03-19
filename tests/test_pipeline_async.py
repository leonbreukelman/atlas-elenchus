"""Tests for async agent recommendation calls."""
import asyncio
import inspect
import time
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from src.agent import Agent, Recommendation
from src.market_data import MarketSnapshot
from src.pipeline import Pipeline


@pytest.fixture
def mock_snapshot():
    """Minimal MarketSnapshot for testing."""
    import pandas as pd
    from datetime import datetime
    return MarketSnapshot(
        date=datetime(2026, 3, 17),
        prices=pd.DataFrame(),
        returns_1d=pd.Series(dtype=float),
        returns_5d=pd.Series(dtype=float),
        returns_20d=pd.Series(dtype=float),
        volatility_20d=pd.Series(dtype=float),
        spy_regime="neutral",
    )


@pytest.fixture
def agent(tmp_path):
    """Agent with a minimal prompt file."""
    prompt_file = tmp_path / "test_agent.md"
    prompt_file.write_text("You are a test agent. Return JSON recommendations.")
    return Agent(agent_id="test_agent", layer=1, prompt_path=prompt_file)


def _mock_llm_response(ticker="AAPL"):
    """Build a mock LLM response returning a single recommendation."""
    import json
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = json.dumps({
        "recommendations": [{
            "ticker": ticker, "direction": "long", "conviction": 0.8,
            "reasoning_components": ["test reason"], "conclusion": f"buy {ticker}",
        }]
    })
    return resp


def test_arecommend_returns_recommendations(agent, mock_snapshot):
    """arecommend() must be async and return list[Recommendation]."""
    assert inspect.iscoroutinefunction(agent.arecommend)

    with patch("src.agent.acompletion_with_retry", new_callable=AsyncMock, return_value=_mock_llm_response()):
        recs = asyncio.run(agent.arecommend(snapshot=mock_snapshot))

    assert len(recs) == 1
    assert recs[0].ticker == "AAPL"
    assert isinstance(recs[0], Recommendation)


def test_sync_recommend_still_works(agent, mock_snapshot):
    """Original recommend() must remain functional."""
    with patch("src.agent.completion_with_retry", return_value=_mock_llm_response("MSFT")):
        recs = agent.recommend(snapshot=mock_snapshot)

    assert len(recs) == 1
    assert recs[0].ticker == "MSFT"


@pytest.fixture
def pipeline(tmp_path):
    """Pipeline with minimal prompt files."""
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    for name in [
        "macro_regime", "rates_yield", "volatility",
        "tech_semi", "energy", "quality",
        "risk_officer", "cio",
    ]:
        (prompt_dir / f"{name}.md").write_text(
            f"You are the {name} agent. Produce JSON recommendations."
        )
    return Pipeline(prompt_dir=prompt_dir, use_elenchus=False)


def test_layer_agents_run_concurrently(pipeline, mock_snapshot):
    """Agents within a layer must run concurrently, not sequentially."""

    async def slow_arecommend(self, snapshot, upstream_signals=None, model=None):
        await asyncio.sleep(0.1)
        return []

    with patch.object(Agent, "arecommend", slow_arecommend):
        start = time.monotonic()
        pipeline.run_day(mock_snapshot)
        elapsed = time.monotonic() - start

    # 3 layers with 3, 3, 2 agents. Sequential: 8 * 0.1s = 0.8s
    # Concurrent within layer: ~0.1s per layer * 3 layers = ~0.3s
    assert elapsed < 0.6, f"Took {elapsed:.2f}s — agents likely running sequentially"


def test_agent_failure_isolated(pipeline, mock_snapshot):
    """One agent failing must not prevent others from returning recs."""

    async def maybe_fail(self, snapshot, upstream_signals=None, model=None):
        if self.agent_id == "macro_regime":
            raise RuntimeError("Simulated failure")
        return [Recommendation(
            agent_id=self.agent_id, date="2026-03-17", ticker="AAPL",
            direction="long", conviction=0.8,
            reasoning_components=["test"], conclusion="buy",
        )]

    with patch.object(Agent, "arecommend", maybe_fail):
        recs, _ = pipeline.run_day(mock_snapshot)

    # macro_regime failed but 7 other agents should produce recs
    assert len(recs) >= 2
