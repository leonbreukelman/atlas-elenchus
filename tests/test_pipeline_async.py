"""Tests for async agent recommendation calls."""
import asyncio
import inspect
from unittest.mock import MagicMock, AsyncMock, patch
from pathlib import Path

import pytest

from src.agent import Agent, Recommendation
from src.market_data import MarketSnapshot


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
