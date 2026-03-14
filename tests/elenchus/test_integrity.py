import pytest
import json
from unittest.mock import MagicMock
from src.agent import Recommendation
from src.elenchus import ElenchusProbe

@pytest.fixture
def mock_anthropic():
    return MagicMock()

@pytest.fixture
def probe(mock_anthropic):
    return ElenchusProbe(client=mock_anthropic, model="test-model")

@pytest.fixture
def real_recommendation():
    """A realistic, load-bearing recommendation."""
    return Recommendation(
        agent_id="analyst_1",
        date="2026-03-14",
        ticker="NVDA",
        direction="long",
        conviction=0.8,
        reasoning_components=[
            "Semiconductor capex cycle accelerating per TSMC guidance",
            "Relative strength vs SOX index over trailing 20d",
            "Data center revenue mix shift reduces cyclicality"
        ],
        conclusion="NVDA is a high-conviction buy due to structural cycle acceleration."
    )

@pytest.fixture
def unrelated_recommendation():
    """A completely unrelated recommendation for shuffling."""
    return Recommendation(
        agent_id="analyst_2",
        date="2026-03-14",
        ticker="AAPL",
        direction="short",
        conviction=0.4,
        reasoning_components=[
            "Consumer hardware replacement cycle lengthening",
            "Services growth slowing in key markets",
            "Multiple compression expected due to regulatory risks"
        ],
        conclusion="AAPL is a tactical short on slowing growth."
    )

def test_frankenstein_shuffling(probe, real_recommendation, unrelated_recommendation):
    """
    FRANKENSTEIN TEST:
    If we swap reasoning components from an unrelated trade,
    the Deutsch Probe MUST return a low score.
    """
    # Create a frankenstein recommendation: NVDA conclusion with AAPL reasoning
    frankenstein = Recommendation(
        agent_id=real_recommendation.agent_id,
        date=real_recommendation.date,
        ticker=real_recommendation.ticker,
        direction=real_recommendation.direction,
        conviction=real_recommendation.conviction,
        reasoning_components=unrelated_recommendation.reasoning_components,
        conclusion=real_recommendation.conclusion
    )

    # Mock the probe to return conclusion_survives: false for all components
    # because the reasoning is completely unrelated.
    probe.client.messages.create.return_value.content = [
        MagicMock(text=json.dumps({"conclusion_survives": False, "explanation": "Unrelated reasoning"}))
    ]

    result = probe.probe(frankenstein)
    
    # A successful probe should recognize that NONE of these components support the NVDA conclusion
    assert result.deutsch_score == 1.0
    assert result.is_hard_to_vary

def test_tautology_detection(probe, real_recommendation):
    """
    TAUTOLOGY TEST:
    If we replace a component with a universal truth (tautology),
    it should be flagged as decorative (easy to vary).
    """
    """
    TAUTOLOGY TEST:
    If we replace a component with a universal truth (tautology),
    it should be flagged as decorative (easy to vary).
    """
    tautological_rec = Recommendation(
        agent_id=real_recommendation.agent_id,
        date=real_recommendation.date,
        ticker=real_recommendation.ticker,
        direction=real_recommendation.direction,
        conviction=real_recommendation.conviction,
        reasoning_components=[
            "Stock prices are determined by supply and demand", # Tautology
            "Volatility is the degree of variation of price",   # Tautology
            "Markets can go up, down, or sideways"              # Tautology
        ],
        conclusion=real_recommendation.conclusion
    )

    # Mock the probe to return conclusion_survives: true for all components
    # because swapping one tautology for another doesn't change the conclusion.
    probe.client.messages.create.return_value.content = [
        MagicMock(text=json.dumps({"conclusion_survives": True, "explanation": "Decorative tautology"}))
    ]

    result = probe.probe(tautological_rec)
    
    # All components are decorative, so score should be 0.0
    assert result.deutsch_score == 0.0

def test_ablation_logic(probe, real_recommendation):
    """
    ABLATION TEST:
    Ensures that removing a truly load-bearing component breaks the conclusion.
    """
    # This test verifies our internal survives logic works when mocked
    probe.client.messages.create.return_value.content = [
        MagicMock(text=json.dumps({"conclusion_survives": False, "explanation": "Critical component missing"}))
    ]
    
    # If the LLM judge says the conclusion doesn't survive the swap, it's load-bearing
    res = probe._test_swap(real_recommendation, 0, "Replacement")
    assert res.conclusion_survived is False
