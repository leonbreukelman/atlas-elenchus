#!/usr/bin/env python3
"""
atlas-elenchus runner.

Runs two parallel backtests:
  Branch A (vanilla):   agents → darwinian weights → portfolio
  Branch B (elenchus):  agents → elenchus probe → darwinian weights → portfolio

Same market data, same starting prompts, same agent structure.
The only difference is whether Elenchus filters recommendations
by explanation quality before they reach the portfolio.

Usage:
    uv run python -m scripts.run --start 2024-09-01 --end 2025-09-01
    uv run python -m scripts.run --mode elenchus --start 2024-09-01
    uv run python -m scripts.run --mode ab --start 2024-09-01 --mutation-interval 20
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import signal
import sys

_shutdown_requested = False

import litellm
litellm.suppress_debug_info = True
import pandas as pd
from dotenv import load_dotenv

from src.market_data import MarketData
from src.pipeline import Pipeline
from src.portfolio import Portfolio, score_agents
from src.autoresearch import AutoresearchLoop


def _checkpoint_path(output_dir: Path, mode: str) -> Path:
    """Path for the checkpoint file."""
    return output_dir / f"{mode}_checkpoint.json"


def _save_checkpoint(
    checkpoint_path: Path,
    mode: str,
    daily_log: list[dict],
    agent_weights: dict[str, dict],
    mutation_log: list[dict],
    elenchus_log: list[dict],
    cumulative: float,
) -> None:
    """Save backtest state so we can resume after a crash."""
    data = {
        "mode": mode,
        "daily_returns": daily_log,
        "agent_weights": agent_weights,
        "mutation_log": mutation_log,
        "elenchus_log": elenchus_log,
        "cumulative": cumulative,
    }
    # Write to temp file then rename for atomicity
    tmp = checkpoint_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.rename(checkpoint_path)


def _load_checkpoint(checkpoint_path: Path) -> dict | None:
    """Load checkpoint if it exists. Returns None if no checkpoint."""
    if not checkpoint_path.exists():
        return None
    try:
        data = json.loads(checkpoint_path.read_text())
        print(f"  Checkpoint found: {len(data.get('daily_returns', []))} days completed")
        return data
    except (json.JSONDecodeError, KeyError) as e:
        print(f"  Checkpoint corrupted, starting fresh: {e}")
        return None


def _handle_shutdown(signum, frame):
    """Handle SIGTERM/SIGINT — set flag so main loop saves checkpoint and exits cleanly."""
    global _shutdown_requested
    signal_name = signal.Signals(signum).name
    print(f"\n  Received {signal_name} — will save checkpoint and exit after current day completes")
    _shutdown_requested = True


def run_backtest(
    mode: str,
    market: MarketData,
    prompt_dir: Path,
    repo_dir: Path,
    mutation_interval: int = 20,
    model: str = "openrouter/qwen/qwen3-235b-a22b",
    output_dir: Path | None = None,
    probe_layers: list[int] | None = None,
) -> dict:
    """
    Run a single backtest.
    mode: "vanilla" | "elenchus"
    mutation_interval: run autoresearch mutation every N trading days
    """
    use_elenchus = mode == "elenchus" or mode == "random_elenchus"
    random_mode = mode == "random_elenchus"
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    pipeline = Pipeline(prompt_dir, client=client, use_elenchus=use_elenchus, random_mode=random_mode, model=model, probe_layers=probe_layers)
    portfolio = Portfolio(use_elenchus=use_elenchus)
    autoresearch = AutoresearchLoop(repo_dir, model=model)

    # Resolve output dir for checkpoints
    if output_dir is None:
        output_dir = repo_dir / "results"
    output_dir.mkdir(exist_ok=True)
    checkpoint_path = _checkpoint_path(output_dir, mode)

    day_count = 0
    recommendations_by_agent: dict[str, list] = {}
    mutation_log: list[dict] = []
    elenchus_log: list[dict] = []

    # Check for existing checkpoint to resume from
    completed_dates: set[str] = set()
    checkpoint = _load_checkpoint(checkpoint_path)
    if checkpoint:
        # Restore agent weights from checkpoint
        saved_weights = checkpoint.get("agent_weights", {})
        for agent_id, weight_info in saved_weights.items():
            if agent_id in pipeline.agents:
                pipeline.agents[agent_id].darwinian_weight = weight_info["weight"]
                pipeline.agents[agent_id].rolling_sharpe = weight_info["sharpe"]

        # Restore portfolio state from checkpoint daily returns
        for entry in checkpoint.get("daily_returns", []):
            completed_dates.add(entry["date"])
            portfolio.daily_returns.append(entry["return"])
            portfolio.cumulative = 1.0 + entry["cumulative"]
            from src.portfolio import DailyPnL
            portfolio.daily_log.append(DailyPnL(
                date=entry["date"],
                portfolio_return=entry["return"],
                cumulative_return=entry["cumulative"],
                positions=[],
            ))

        mutation_log = checkpoint.get("mutation_log", [])
        elenchus_log = checkpoint.get("elenchus_log", [])
        day_count = len(completed_dates)
        print(f"  Resuming from day {day_count + 1} (skipping {len(completed_dates)} completed days)")

    print(f"\n{'='*60}")
    print(f"  BACKTEST: {mode.upper()}")
    print(f"{'='*60}")

    import time as _time
    for snapshot in market.iterate():
        date_str = snapshot.date.strftime("%Y-%m-%d")

        # Skip days already completed in checkpoint
        if date_str in completed_dates:
            continue

        day_count += 1

        # Run pipeline
        _day_start = _time.time()
        recommendations, elenchus_results = pipeline.run_day(snapshot)
        _day_elapsed = _time.time() - _day_start
        print(f"  Day {day_count} ({date_str}): {len(recommendations)} recs in {_day_elapsed:.0f}s")

        # Log elenchus results
        if elenchus_results:
            for er in elenchus_results:
                elenchus_log.append({
                    "date": date_str,
                    "agent": er.recommendation.agent_id,
                    "ticker": er.recommendation.ticker,
                    "conclusion": er.recommendation.conclusion,
                    "reasoning_components": er.recommendation.reasoning_components,
                    "deutsch_score": er.deutsch_score,
                    "load_bearing": er.load_bearing_count,
                    "total_components": er.total_components,
                    "hard_to_vary": er.is_hard_to_vary,
                    "probes": [
                        {
                            "component": pr.original_component,
                            "replacement": pr.replacement_component,
                            "survived": pr.conclusion_survived,
                            "reasoning": pr.probe_reasoning,
                        }
                        for pr in er.probe_results
                    ],
                })

        # Rebalance portfolio
        portfolio.rebalance(
            recommendations,
            pipeline.agents,
            snapshot,
            elenchus_results if use_elenchus else None,
        )

        # Mark to market (next day's return attributed to today's positions)
        portfolio.mark_to_market(snapshot)

        # Track agent recommendations vs outcomes
        for rec in recommendations:
            if rec.agent_id not in recommendations_by_agent:
                recommendations_by_agent[rec.agent_id] = []
            actual_ret = snapshot.returns_1d.get(rec.ticker, 0.0) if hasattr(snapshot.returns_1d, 'get') else 0.0
            recommendations_by_agent[rec.agent_id].append((rec, actual_ret))

        # Periodic scoring and autoresearch
        if day_count % mutation_interval == 0:
            # Score agents
            score_agents(list(pipeline.agents.values()), recommendations_by_agent)

            # Autoresearch: mutate worst agent
            worst = pipeline.worst_agent()
            sharpe_before = worst.rolling_sharpe
            new_prompt = autoresearch.mutate_agent(worst)

            if new_prompt:
                autoresearch.apply_mutation(worst, new_prompt)
                # Run for eval_window more days before judging
                # (simplified: we judge at next mutation_interval)
                mutation_log.append({
                    "day": day_count,
                    "date": date_str,
                    "agent": worst.agent_id,
                    "sharpe_before": sharpe_before,
                    "status": "APPLIED",
                })

            # Print progress
            print(f"\n  Day {day_count} ({date_str})")
            print(f"  Portfolio Sharpe: {portfolio.sharpe:.3f}")
            print(f"  Cumulative return: {portfolio.cumulative - 1:.2%}")
            print("  Agent weights:")
            for agent in sorted(pipeline.agents.values(), key=lambda a: a.darwinian_weight, reverse=True):
                print(f"    {agent.agent_id:20s}  w={agent.darwinian_weight:.3f}  sharpe={agent.rolling_sharpe:.3f}")

        # Save checkpoint after each completed day
        daily_log = [
            {"date": d.date, "return": d.portfolio_return, "cumulative": d.cumulative_return}
            for d in portfolio.daily_log
        ]
        current_agent_weights = {
            a.agent_id: {"weight": a.darwinian_weight, "sharpe": a.rolling_sharpe}
            for a in pipeline.agents.values()
        }
        _save_checkpoint(
            checkpoint_path, mode, daily_log,
            current_agent_weights, mutation_log, elenchus_log,
            portfolio.cumulative,
        )
        if _shutdown_requested:
            print(f"  Shutdown requested — checkpoint saved at day {day_count} ({date_str})")
            sys.exit(0)

    # Backtest complete — remove checkpoint, save final results
    final_results_path = output_dir / f"{mode}_returns.csv"
    daily_df = portfolio.to_dataframe()
    daily_df.to_csv(final_results_path, index=False)
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print("  Checkpoint removed (backtest complete)")

    # Final results
    results = {
        "mode": mode,
        "trading_days": day_count,
        "final_sharpe": portfolio.sharpe,
        "cumulative_return": portfolio.cumulative - 1.0,
        "mutations": len(mutation_log),
        "daily_returns": daily_df,
        "agent_weights": {
            a.agent_id: {"weight": a.darwinian_weight, "sharpe": a.rolling_sharpe}
            for a in pipeline.agents.values()
        },
        "mutation_log": mutation_log,
        "elenchus_log": elenchus_log,
    }

    return results


def main():
    load_dotenv()

    signal.signal(signal.SIGTERM, _handle_shutdown)
    signal.signal(signal.SIGINT, _handle_shutdown)

    parser = argparse.ArgumentParser(description="atlas-elenchus backtest")
    parser.add_argument("--mode", choices=["vanilla", "elenchus", "ab", "random_elenchus"], default="ab")
    parser.add_argument("--start", default="2024-09-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--mutation-interval", type=int, default=20,
                        help="Run autoresearch mutation every N trading days")
    parser.add_argument("--model", default="openrouter/qwen/qwen3-235b-a22b")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--probe-layers", default="1,2,3",
                        help="Comma-separated layer numbers to probe with Elenchus (default: 1,2,3)")
    args = parser.parse_args()
    probe_layers = [int(x.strip()) for x in args.probe_layers.split(",")]

    # litellm reads API keys from environment automatically:
    #   OPENROUTER_API_KEY for openrouter/* models
    #   ANTHROPIC_API_KEY for anthropic/* models
    model_prefix = args.model.split("/")[0] if "/" in args.model else ""
    key_map = {
        "openrouter": "OPENROUTER_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }
    required_key = key_map.get(model_prefix)
    if required_key and not os.environ.get(required_key):
        print(f"ERROR: Set {required_key} in .env for model {args.model}")
        return

    repo_dir = Path(__file__).parent.parent
    prompt_dir = repo_dir / "prompts" / "agents"
    output_dir = repo_dir / args.output_dir
    output_dir.mkdir(exist_ok=True)

    # Fetch market data (once, shared across branches)
    print("Fetching market data...")
    market = MarketData(start=args.start, end=args.end)
    market.fetch()
    print("Market data ready.")

    results = {}

    if args.mode in ("vanilla", "ab"):
        # Copy prompts to a working dir for vanilla branch
        vanilla_prompts = repo_dir / "prompts" / "agents_vanilla"
        if vanilla_prompts.exists():
            shutil.rmtree(vanilla_prompts)
        shutil.copytree(prompt_dir, vanilla_prompts)

        vanilla_market = MarketData(start=args.start, end=args.end)
        vanilla_market.fetch()

        results["vanilla"] = run_backtest(
            mode="vanilla",
            market=vanilla_market,
            prompt_dir=vanilla_prompts,
            repo_dir=repo_dir,
            mutation_interval=args.mutation_interval,
            model=args.model,
            output_dir=output_dir,
            probe_layers=probe_layers,
        )
        # Re-fetch for second branch (iterator is consumed)
        results["vanilla"]["market_data"] = None
        mkt2 = MarketData(start=args.start, end=args.end)
        mkt2.fetch()

    if args.mode in ("elenchus", "ab"):
        # Copy prompts to a working dir for elenchus branch
        elenchus_prompts = repo_dir / "prompts" / "agents_elenchus"
        if elenchus_prompts.exists():
            shutil.rmtree(elenchus_prompts)
        shutil.copytree(prompt_dir, elenchus_prompts)

        elenchus_market = MarketData(start=args.start, end=args.end)
        elenchus_market.fetch()

        results["elenchus"] = run_backtest(
            mode="elenchus",
            market=elenchus_market,
            prompt_dir=elenchus_prompts,
            repo_dir=repo_dir,
            mutation_interval=args.mutation_interval,
            model=args.model,
            output_dir=output_dir,
            probe_layers=probe_layers,
        )

    # Comparison report
    if "vanilla" in results and "elenchus" in results:
        print(f"\n{'='*60}")
        print("  A/B COMPARISON")
        print(f"{'='*60}")
        print(f"  {'Metric':<30s} {'Vanilla':>12s} {'Elenchus':>12s}")
        print(f"  {'-'*54}")
        print(f"  {'Final Sharpe':<30s} {results['vanilla']['final_sharpe']:>12.3f} {results['elenchus']['final_sharpe']:>12.3f}")
        print(f"  {'Cumulative Return':<30s} {results['vanilla']['cumulative_return']:>11.2%} {results['elenchus']['cumulative_return']:>11.2%}")
        print(f"  {'Mutations Applied':<30s} {results['vanilla']['mutations']:>12d} {results['elenchus']['mutations']:>12d}")
        print(f"  {'Trading Days':<30s} {results['vanilla']['trading_days']:>12d} {results['elenchus']['trading_days']:>12d}")

    # Save results and analysis
    for branch_name, branch_results in results.items():
        if not branch_results:
            continue

        if "daily_returns" in branch_results:
            returns_path = output_dir / f"{branch_name}_returns.csv"
            branch_results["daily_returns"].to_csv(returns_path, index=False)
            
        # Save elenchus analysis if this is the elenchus branch
        if branch_name == "elenchus" and branch_results.get("elenchus_log"):
            elenchus_df = pd.DataFrame(branch_results["elenchus_log"])
            analysis_path = output_dir / "elenchus_analysis.csv"
            elenchus_df.to_csv(analysis_path, index=False)
            print(f"\n  Elenchus analysis saved to {analysis_path}")
            print(f"    Total probes: {len(elenchus_df)}")
            mean_ds = elenchus_df["deutsch_score"].mean()
            htv_rate = elenchus_df["hard_to_vary"].mean()
            filtered_count = (~elenchus_df["hard_to_vary"]).sum()
            print(f"    Mean deutsch_score: {mean_ds:.3f}")
            print(f"    Hard-to-vary rate: {htv_rate:.1%}")
            print(f"    Filtered (easy-to-vary): {filtered_count}")

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
