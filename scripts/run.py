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
from datetime import datetime
from pathlib import Path

import anthropic
import pandas as pd
from dotenv import load_dotenv
from tabulate import tabulate

from src.market_data import MarketData
from src.pipeline import Pipeline
from src.portfolio import Portfolio, score_agents
from src.autoresearch import AutoresearchLoop


def run_backtest(
    mode: str,
    client: anthropic.Anthropic,
    market: MarketData,
    prompt_dir: Path,
    repo_dir: Path,
    mutation_interval: int = 20,
    model: str = "claude-sonnet-4-20250514",
) -> dict:
    """
    Run a single backtest.
    mode: "vanilla" | "elenchus"
    mutation_interval: run autoresearch mutation every N trading days
    """
    use_elenchus = mode == "elenchus"
    pipeline = Pipeline(prompt_dir, client, use_elenchus=use_elenchus, model=model)
    portfolio = Portfolio(use_elenchus=use_elenchus)
    autoresearch = AutoresearchLoop(client, repo_dir, model=model)

    day_count = 0
    recommendations_by_agent: dict[str, list] = {}
    mutation_log: list[dict] = []
    elenchus_log: list[dict] = []

    print(f"\n{'='*60}")
    print(f"  BACKTEST: {mode.upper()}")
    print(f"{'='*60}")

    for snapshot in market.iterate():
        day_count += 1
        date_str = snapshot.date.strftime("%Y-%m-%d")

        # Run pipeline
        recommendations, elenchus_results = pipeline.run_day(snapshot)

        # Log elenchus results
        if elenchus_results:
            for er in elenchus_results:
                elenchus_log.append({
                    "date": date_str,
                    "agent": er.recommendation.agent_id,
                    "ticker": er.recommendation.ticker,
                    "deutsch_score": er.deutsch_score,
                    "load_bearing": er.load_bearing_count,
                    "total_components": er.total_components,
                    "hard_to_vary": er.is_hard_to_vary,
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
            print(f"  Agent weights:")
            for agent in sorted(pipeline.agents.values(), key=lambda a: a.darwinian_weight, reverse=True):
                print(f"    {agent.agent_id:20s}  w={agent.darwinian_weight:.3f}  sharpe={agent.rolling_sharpe:.3f}")

    # Final results
    results = {
        "mode": mode,
        "trading_days": day_count,
        "final_sharpe": portfolio.sharpe,
        "cumulative_return": portfolio.cumulative - 1.0,
        "mutations": len(mutation_log),
        "daily_returns": portfolio.to_dataframe(),
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

    parser = argparse.ArgumentParser(description="atlas-elenchus backtest")
    parser.add_argument("--mode", choices=["vanilla", "elenchus", "ab"], default="ab")
    parser.add_argument("--start", default="2024-09-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--mutation-interval", type=int, default=20,
                        help="Run autoresearch mutation every N trading days")
    parser.add_argument("--model", default="claude-sonnet-4-20250514")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY in .env")
        return

    client = anthropic.Anthropic(api_key=api_key)
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

        results["vanilla"] = run_backtest(
            mode="vanilla",
            client=client,
            market=MarketData(start=args.start, end=args.end),
            prompt_dir=vanilla_prompts,
            repo_dir=repo_dir,
            mutation_interval=args.mutation_interval,
            model=args.model,
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
            client=client,
            market=elenchus_market,
            prompt_dir=elenchus_prompts,
            repo_dir=repo_dir,
            mutation_interval=args.mutation_interval,
            model=args.model,
        )

    # Comparison report
    if args.mode == "ab" and "vanilla" in results and "elenchus" in results:
        print(f"\n{'='*60}")
        print(f"  A/B COMPARISON")
        print(f"{'='*60}")
        print(f"  {'Metric':<30s} {'Vanilla':>12s} {'Elenchus':>12s}")
        print(f"  {'-'*54}")
        print(f"  {'Final Sharpe':<30s} {results['vanilla']['final_sharpe']:>12.3f} {results['elenchus']['final_sharpe']:>12.3f}")
        print(f"  {'Cumulative Return':<30s} {results['vanilla']['cumulative_return']:>11.2%} {results['elenchus']['cumulative_return']:>11.2%}")
        print(f"  {'Mutations Applied':<30s} {results['vanilla']['mutations']:>12d} {results['elenchus']['mutations']:>12d}")
        print(f"  {'Trading Days':<30s} {results['vanilla']['trading_days']:>12d} {results['elenchus']['trading_days']:>12d}")

        # Save elenchus analysis
        if results["elenchus"]["elenchus_log"]:
            elenchus_df = pd.DataFrame(results["elenchus"]["elenchus_log"])
            elenchus_df.to_csv(output_dir / "elenchus_analysis.csv", index=False)
            print(f"\n  Elenchus stats:")
            print(f"    Total probes: {len(elenchus_df)}")
            print(f"    Mean deutsch_score: {elenchus_df['deutsch_score'].mean():.3f}")
            print(f"    Hard-to-vary rate: {elenchus_df['hard_to_vary'].mean():.1%}")
            print(f"    Filtered (easy-to-vary): {(~elenchus_df['hard_to_vary']).sum()}")

    # Save results
    for branch_name, branch_results in results.items():
        if branch_results and "daily_returns" in branch_results:
            branch_results["daily_returns"].to_csv(
                output_dir / f"{branch_name}_returns.csv", index=False
            )

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
