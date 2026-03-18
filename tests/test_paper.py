"""Tests for the paper trading engine."""

from pathlib import Path

import pytest

from src.paper import PaperTradingEngine, NON_TRADEABLE


@pytest.fixture
def engine(tmp_path):
    # Create minimal prompt files for Pipeline initialization
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

    return PaperTradingEngine(
        db_path=tmp_path / "test.db",
        prompt_dir=prompt_dir,
        starting_capital=10000.0,
        model="openrouter/qwen/qwen3-235b-a22b",
        probe_layers=[3],
    )


def test_engine_initializes(engine):
    assert engine.ledger.get_cash() == 10000.0


def test_vix_filtered_from_positions(engine):
    """VIX recommendations should not produce positions."""
    from src.agent import Recommendation

    recs = [
        Recommendation(
            agent_id="volatility",
            date="2026-03-17",
            ticker="VIX",
            direction="long",
            conviction=0.9,
            reasoning_components=["vol rising"],
            conclusion="long vol",
        ),
        Recommendation(
            agent_id="tech_semi",
            date="2026-03-17",
            ticker="NVDA",
            direction="long",
            conviction=0.8,
            reasoning_components=["strong earnings"],
            conclusion="long nvda",
        ),
    ]
    filtered = engine._filter_non_tradeable(recs)
    tickers = [r.ticker for r in filtered]
    assert "VIX" not in tickers
    assert "NVDA" in tickers


def test_caret_vix_also_filtered(engine):
    """^VIX should also be filtered as non-tradeable."""
    from src.agent import Recommendation

    recs = [
        Recommendation(
            agent_id="volatility",
            date="2026-03-17",
            ticker="^VIX",
            direction="long",
            conviction=0.9,
            reasoning_components=["vol rising"],
            conclusion="long vol",
        ),
    ]
    filtered = engine._filter_non_tradeable(recs)
    assert len(filtered) == 0


def test_non_tradeable_set():
    """The NON_TRADEABLE set contains expected tickers."""
    assert "VIX" in NON_TRADEABLE
    assert "^VIX" in NON_TRADEABLE


def test_select_positions_filters_avoid(engine):
    """Recommendations with direction='avoid' should not produce positions."""
    from src.agent import Recommendation
    from src.elenchus import ElenchusResult

    recs = [
        Recommendation(
            agent_id="tech_semi",
            date="2026-03-17",
            ticker="NVDA",
            direction="avoid",
            conviction=0.9,
            reasoning_components=["weak outlook"],
            conclusion="avoid nvda",
        ),
        Recommendation(
            agent_id="energy",
            date="2026-03-17",
            ticker="XOM",
            direction="long",
            conviction=0.7,
            reasoning_components=["strong cash flows"],
            conclusion="long xom",
        ),
    ]

    er = ElenchusResult(
        recommendation=recs[1],
        probe_results=[],
        deutsch_score=0.8,
        total_components=1,
        load_bearing_count=1,
    )

    # Create a minimal snapshot with a date attribute
    import types
    from datetime import datetime
    snapshot = types.SimpleNamespace(date=datetime(2026, 3, 17))

    positions = engine._select_positions(recs, [er], snapshot)
    tickers = [p.ticker for p in positions]
    assert "NVDA" not in tickers
    assert "XOM" in tickers


def test_select_positions_scores_and_ranks(engine):
    """Positions should be ranked by score = conviction * darwinian_weight * deutsch_score."""
    from src.agent import Recommendation
    from src.elenchus import ElenchusResult

    recs = [
        Recommendation(
            agent_id="tech_semi",
            date="2026-03-17",
            ticker="NVDA",
            direction="long",
            conviction=0.9,
            reasoning_components=["strong earnings"],
            conclusion="long nvda",
        ),
        Recommendation(
            agent_id="energy",
            date="2026-03-17",
            ticker="XOM",
            direction="long",
            conviction=0.5,
            reasoning_components=["moderate outlook"],
            conclusion="long xom",
        ),
    ]

    elenchus_results = [
        ElenchusResult(
            recommendation=recs[0],
            probe_results=[],
            deutsch_score=0.9,
            total_components=1,
            load_bearing_count=1,
        ),
        ElenchusResult(
            recommendation=recs[1],
            probe_results=[],
            deutsch_score=0.6,
            total_components=1,
            load_bearing_count=1,
        ),
    ]

    import types
    from datetime import datetime
    snapshot = types.SimpleNamespace(date=datetime(2026, 3, 17))

    positions = engine._select_positions(recs, elenchus_results, snapshot)
    # NVDA should rank higher: 0.9 * 1.0 * 0.9 = 0.81 vs XOM: 0.5 * 1.0 * 0.6 = 0.30
    assert len(positions) == 2
    assert positions[0].ticker == "NVDA"
    assert positions[1].ticker == "XOM"


def test_select_positions_limits_to_10(engine):
    """At most 10 positions should be selected."""
    from src.agent import Recommendation
    from src.elenchus import ElenchusResult

    # Create 15 recommendations from the same agent
    recs = []
    elenchus_results = []
    tickers = [
        "NVDA", "AVGO", "TSM", "MSFT", "AAPL",
        "XOM", "CVX", "SLB", "JPM", "GS",
        "BRK-B", "CAT", "GE", "LMT", "AMZN",
    ]
    for i, ticker in enumerate(tickers):
        rec = Recommendation(
            agent_id="tech_semi",
            date="2026-03-17",
            ticker=ticker,
            direction="long",
            conviction=0.8 - i * 0.01,
            reasoning_components=["reason"],
            conclusion=f"long {ticker}",
        )
        recs.append(rec)
        elenchus_results.append(
            ElenchusResult(
                recommendation=rec,
                probe_results=[],
                deutsch_score=0.8,
                total_components=1,
                load_bearing_count=1,
            )
        )

    import types
    from datetime import datetime
    snapshot = types.SimpleNamespace(date=datetime(2026, 3, 17))

    positions = engine._select_positions(recs, elenchus_results, snapshot)
    assert len(positions) == 10


def test_market_closed_returns_skipped(engine, monkeypatch):
    """When market is closed, run() should return skipped=True."""
    monkeypatch.setattr(engine.market, "is_market_open_today", lambda: False)

    result = engine.run("morning")
    assert result["skipped"] is True
    assert result["reason"] == "Market closed"


def test_select_positions_unknown_agent_skipped(engine):
    """Recommendations from agents not in pipeline should be skipped."""
    from src.agent import Recommendation

    recs = [
        Recommendation(
            agent_id="nonexistent_agent",
            date="2026-03-17",
            ticker="AAPL",
            direction="long",
            conviction=0.9,
            reasoning_components=["strong"],
            conclusion="long aapl",
        ),
    ]

    import types
    from datetime import datetime
    snapshot = types.SimpleNamespace(date=datetime(2026, 3, 17))

    positions = engine._select_positions(recs, [], snapshot)
    assert len(positions) == 0


def test_select_positions_no_elenchus_uses_default_deutsch(engine):
    """When no Elenchus result exists for a rec, deutsch_score defaults to 0.5."""
    from src.agent import Recommendation

    recs = [
        Recommendation(
            agent_id="tech_semi",
            date="2026-03-17",
            ticker="NVDA",
            direction="long",
            conviction=0.8,
            reasoning_components=["strong earnings"],
            conclusion="long nvda",
        ),
    ]

    import types
    from datetime import datetime
    snapshot = types.SimpleNamespace(date=datetime(2026, 3, 17))

    # No elenchus results
    positions = engine._select_positions(recs, [], snapshot)
    assert len(positions) == 1
    assert positions[0].deutsch_score == 0.5
