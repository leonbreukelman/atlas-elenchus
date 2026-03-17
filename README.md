# atlas-elenchus

A controlled experiment testing whether filtering AI agent outputs by explanation quality improves investment outcomes in a multi-agent trading system.

Combines three ideas:

1. **Autoresearch** (Karpathy) — autonomous AI experiment loops: mutate code, measure metric, keep or discard
2. **ATLAS** (General Intelligence Capital) — autoresearch applied to trading agent prompts instead of training code
3. **Elenchus** (this project) — a verification layer based on David Deutsch's epistemology that tests whether an agent's reasoning is load-bearing or decorative

---

## Table of Contents

- [Background: The Three Ideas](#background-the-three-ideas)
- [What This Project Does](#what-this-project-does)
- [The Deutsch Probe: How It Works](#the-deutsch-probe-how-it-works)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Experiment](#running-the-experiment)
- [Resilience](#resilience)
- [Understanding the Output](#understanding-the-output)
- [Interpreting Results](#interpreting-results)
- [Project Structure](#project-structure)
- [Modifying the System](#modifying-the-system)
- [Cost Estimates](#cost-estimates)
- [Known Limitations](#known-limitations)
- [Theoretical Context](#theoretical-context)
- [A/B Results](#ab-results)

---

## Background: The Three Ideas

### 1. Autoresearch (Karpathy, March 2026)

[Repository](https://github.com/karpathy/autoresearch)

Andrej Karpathy released a minimal autonomous ML research loop. The structure:

- **One mutable file** (`train.py`) — contains a GPT model, optimizer, and training loop
- **One fixed metric** (`val_bpb` — validation bits per byte, lower is better)
- **One fixed time budget** (5 minutes wall clock per experiment)
- **One loop**: mutate the file → train → measure → keep if improved, discard if not → repeat

An AI agent (Claude Code, Codex, etc.) runs this loop autonomously. Approximately 12 experiments per hour, 100 overnight. The agent accumulates git commits on a feature branch — each commit represents a code change that improved the metric.

The key design insight: the human writes a `program.md` file that instructs the agent on what to explore. The human programs the *research organization*. The agent programs the *training code*.

### 2. ATLAS (General Intelligence Capital)

[Repository](https://github.com/chrisworsey55/atlas-gic)

ATLAS took the autoresearch pattern and applied it to a different domain: financial markets. Instead of mutating training code, the system mutates agent *prompts*. Instead of measuring validation loss, it measures Sharpe ratio.

Architecture: 25 AI agents organized in 4 layers (Macro → Sector → Superinvestor → Decision). Each agent has a system prompt that defines its analytical framework. Every trading day, agents produce recommendations that flow up through the layers. A CIO agent synthesizes everything into portfolio decisions.

The autoresearch loop operates on the prompts:

- Identify the worst-performing agent by rolling Sharpe ratio
- Generate a targeted prompt modification via LLM
- Run for 5 trading days
- If the agent's Sharpe improved → `git commit` (keep the prompt change)
- If not → `git reset` (revert)

Additionally, a **Darwinian weighting** system adjusts each agent's influence. Top-quartile agents by Sharpe get their weight multiplied by 1.05 each day. Bottom-quartile agents get 0.95. Weights are bounded between 0.3 and 2.5. Over time, good agents get louder; bad agents get quieter.

Key findings from ATLAS's 18-month backtest:

- 70% of prompt mutations were reverted (the system mostly flails in prompt space)
- The system downweighted its own CIO (orchestration layer) to minimum — it discovered the synthesis bottleneck before the humans did
- Individual agent Sharpe improvements were measurable but portfolio returns depended heavily on the orchestration layer

### 3. Elenchus (this project)

Named after the Socratic *elenchus* (cross-examination), this project adds a verification layer based on David Deutsch's epistemological criterion: **good explanations are hard to vary**.

**What Elenchus is NOT:** It is not prediction calibration. It does not ask "did you predict the outcome accurately?" An agent could be a perfect forecaster through pure pattern matching without having any explanation for its predictions.

**What Elenchus IS:** It tests the internal structure of reasoning. For each component of an agent's explanation, it generates a plausible alternative and checks whether the conclusion survives the swap. Components that can be replaced without changing the conclusion are *decorative*. Components that, when replaced, break the conclusion are *load-bearing*.

---

## What This Project Does

This project runs a controlled A/B experiment:

- **Branch A (Vanilla):** 8 trading agents in 3 layers produce daily recommendations. Recommendations are weighted by agent conviction and Darwinian weight. The autoresearch loop mutates the worst agent's prompt every 20 trading days. This is a simplified recreation of the ATLAS architecture.

- **Branch B (Elenchus):** Identical to Branch A, except every recommendation passes through the Deutsch Probe before reaching the portfolio. Recommendations with hard-to-vary reasoning (high `deutsch_score`) receive higher signal weight. Recommendations with easy-to-vary reasoning are discounted or filtered out.

Both branches see the same historical market data, start with the same agent prompts, and use the same Darwinian weight evolution. The **only** difference is whether the Elenchus layer is active.

The experiment measures whether filtering by explanation quality — as defined by the Deutsch criterion — improves the Sharpe ratio of the resulting portfolio.

---

## The Deutsch Probe: How It Works

This is the core of the project. Every recommendation from every agent includes structured reasoning in the form of 3-5 independent *reasoning components*. Each component is a distinct, falsifiable claim.

**Example recommendation from the tech_semi agent:**

```
Ticker: NVDA
Direction: long
Conviction: 0.85
Reasoning components:
  1. "Semiconductor capex cycle is accelerating — TSMC's latest guidance raised wafer starts 15%"
  2. "NVDA shows relative strength vs SOX index over trailing 20 days"
  3. "Data center revenue mix is now 65%, reducing exposure to consumer GPU cycles"
  4. "Current risk-on regime favors high-beta tech names"
Conclusion: "NVDA is the highest-conviction long in the AI compute buildout..."
```

**The probe tests each component independently:**

**Probe Component 1:**
- Generate a plausible alternative: "Semiconductor capex cycle is decelerating — hyperscaler orders are plateauing as digestion phase begins"
- Swap it in, keep components 2-4 unchanged
- Ask: does the conclusion (NVDA long, 0.85 conviction) still logically follow?
- If **NO** → component 1 was load-bearing (the capex cycle claim was necessary for the conclusion)
- If **YES** → component 1 was decorative (the conclusion holds even with opposite capex conditions — suspicious)

**Probe Component 4:**
- Generate alternative: "Current risk-off regime favors defensive and low-volatility names"
- Swap it in, keep components 1-3 unchanged
- Ask: does the conclusion still follow?
- If the remaining three components (capex acceleration, relative strength, revenue mix) are strong enough to support NVDA long regardless of regime, then component 4 was decorative — it didn't add anything
- If the regime call was actually necessary for the trade to make sense, then component 4 was load-bearing

**Scoring:**

```
deutsch_score = load_bearing_components / total_components
```

- `deutsch_score = 1.0` → every component is necessary. The explanation is maximally hard to vary. Strong signal.
- `deutsch_score = 0.0` → every component can be swapped out. The agent is producing post-hoc rationalization. Weak signal.
- Threshold: `deutsch_score > 0.6` is classified as "hard to vary" and passes the filter.

**The probe consists of two LLM calls per component:**
1. A *perturbation agent* generates the plausible replacement (same domain, different analytical direction)
2. A *judge agent* evaluates whether the conclusion survives the swap

---

## Architecture

### Agent Layers

```
LAYER 1 — MACRO (3 agents)
  macro_regime    Classifies risk-on / risk-off / neutral using cross-asset signals
  rates_yield     Interprets rate environment, identifies rate-sensitive opportunities
  volatility      Assesses vol regime, identifies vol-mispriced names

LAYER 2 — SECTOR (3 agents)
  tech_semi       Technology and semiconductor positioning
  energy          Energy sector opportunities
  quality         Quality compounders: pricing power, FCF, competitive moats

LAYER 3 — DECISION (2 agents)
  risk_officer    Adversarial — attacks upstream ideas, flags correlated risks
  cio             Final synthesis — combines all signals into portfolio view
```

Each layer feeds into the next. Layer 2 agents see Layer 1's recommendations as upstream signals. Layer 3 sees Layer 2's filtered output.

### Data Flow

**Vanilla pipeline:**
```
Market Data (yfinance)
    → Layer 1 agents produce recommendations with reasoning
    → Layer 2 agents receive L1 signals, produce own recommendations
    → Layer 3 agents receive L2 signals, produce final recommendations
    → Portfolio: rank by (conviction × darwinian_weight), take top N
    → Mark to market daily
    → Every 20 days: score agents, adjust Darwinian weights, mutate worst prompt
```

**Elenchus pipeline (identical except for probe insertion):**
```
Market Data (yfinance)
    → Layer 1 agents produce recommendations with reasoning
    → ELENCHUS PROBE: test each L1 recommendation's reasoning components  [if L1 in --probe-layers]
    → Filter: discard recommendations with deutsch_score < 0.6
    → Layer 2 agents receive filtered L1 signals
    → ELENCHUS PROBE: test each L2 recommendation                         [if L2 in --probe-layers]
    → Filter again
    → Layer 3 agents receive filtered L2 signals
    → ELENCHUS PROBE: test each L3 recommendation                         [if L3 in --probe-layers]
    → Portfolio: rank by (conviction × darwinian_weight × deutsch_score), take top N
    → Mark to market daily
    → Every 20 days: score agents, adjust Darwinian weights, mutate worst prompt
```

Probing is configurable via `--probe-layers`. The production 250-day run used L3-only probing (`--probe-layers 3`), which applies the probe only to the CIO and risk_officer outputs, reducing probe API calls by ~75%.

### Autoresearch Loop (shared by both branches)

Every `--mutation-interval` trading days (default: 20):

1. Score all agents by rolling Sharpe ratio
2. Rank into quartiles
3. Top quartile: `darwinian_weight *= 1.05` (capped at 2.5)
4. Bottom quartile: `darwinian_weight *= 0.95` (floored at 0.3)
5. Identify the single worst agent by rolling Sharpe
6. Generate a targeted prompt modification via LLM (a "mutator" agent reads the current prompt, the agent's performance stats, and produces one focused change)
7. Write the modified prompt to disk
8. Evaluate over the next period
9. If Sharpe improved: `git commit` (keep the mutation)
10. If Sharpe degraded: `git checkout` (revert to previous prompt)

---

## A/B Results

### Full 219-day run

219 trading days (production run). Model: Qwen3-235B via OpenRouter. Elenchus used L3-only probing (`--probe-layers 3`).

| Metric | Vanilla | Elenchus (L3-only) |
|---|---|---|
| Cumulative Return | +512.96% | +402.19% |
| Trading Days | 219 | 219 |

**Caveats:** No transaction costs, survivorship bias in the ticker universe, frictionless simulation. Absolute numbers are inflated relative to real-world trading. The A/B comparison is valid — both branches share the same biases — but these figures should not be read as performance claims.

Additional findings from the full run:
- 90% filter rate — roughly 90% of recommendations that reach the probe are discarded
- Mean deutsch score of 0.243 across all probed recommendations
- The Elenchus branch consistently makes more concentrated bets, which amplifies both upside and downside relative to vanilla

### Early pilot (19-day)

19-day pilot, Dec 16 2024 -- Jan 14 2025.

| Metric | Vanilla | Elenchus |
|---|---|---|
| Cumulative Return | +13.65% | +15.05% |
| Annualized Sharpe | 7.718 | 4.619 |
| Max Drawdown | -1.61% | -8.47% |
| Win Rate | 73.7% | 73.7% |
| Daily Volatility | 1.45% | 2.74% |

Key findings from pilot:
- The Deutsch Probe produces a genuinely different portfolio (0.107 correlation with vanilla)
- 31% of filtered recommendations scored 0.00 — completely decorative reasoning
- The filter is aggressive (threshold 0.6), leaving ~3.3 recommendations/day and creating concentration risk
- The mechanism works; calibration needs refinement

---

## Prerequisites

- **Python 3.10+**
- **[uv](https://docs.astral.sh/uv/)** -- Python project manager
- **An LLM API key** -- any provider supported by [litellm](https://docs.litellm.ai/) (OpenRouter, Anthropic, local via Ollama, etc.)
- **Git** -- autoresearch uses git branches for keep/discard checkpointing
- **Internet access** -- yfinance downloads historical market data

No GPU required. This runs on any machine with API access.

---

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd atlas-elenchus

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Python dependencies
uv sync

# Create environment configuration
cp .env.example .env
# Edit .env and add your API key. OPENROUTER_API_KEY is the primary requirement.
# ANTHROPIC_API_KEY is optional — only needed if using Claude models via litellm directly.

# Initialize git repository (required for autoresearch keep/discard)
git init
git add -A
git commit -m "initial"
```

---

## Configuration

### Environment variables (`.env`)

```
# Primary requirement:
OPENROUTER_API_KEY=sk-or-v1-...
# Optional — only needed if using Claude models via litellm directly:
ANTHROPIC_API_KEY=sk-ant-...
FMP_API_KEY=                     # Optional: for fundamental data
BRANCH_PREFIX=autoresearch       # Git branch prefix
MODE=ab                          # Default run mode
```

### Command-line flags

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `ab` | `vanilla` (no probe), `elenchus` (with probe), `ab` (runs both, compares), or `random_elenchus` (random baseline — randomly accepts/rejects recommendations via coin flip instead of probing reasoning; control condition for validating whether the probe measures anything real) |
| `--start` | `2024-09-01` | Backtest start date (YYYY-MM-DD) |
| `--end` | today | Backtest end date |
| `--mutation-interval` | `20` | Run autoresearch prompt mutation every N trading days |
| `--model` | `openrouter/qwen/qwen3-235b-a22b` | LLM for all agents and probes (any litellm-supported model) |
| `--probe-layers` | `1,2,3` | Comma-separated layer numbers to probe. Use `--probe-layers 3` for L3-only probing (CIO + risk_officer only), which reduces probe API calls by ~75%. |
| `--output-dir` | `results` | Directory for output CSVs |

---

## Running the Experiment

### Full A/B comparison (recommended first run)

```bash
uv run python -m scripts.run --mode ab --start 2024-09-01 --model openrouter/qwen/qwen3-235b-a22b
```

This runs both branches sequentially over the same date range. Takes several hours depending on the number of trading days. Watch the terminal for progress updates every 20 days.

### Single branch

```bash
# Vanilla only (no Elenchus probe)
uv run python -m scripts.run --mode vanilla --start 2024-09-01 --model openrouter/qwen/qwen3-235b-a22b

# Elenchus only
uv run python -m scripts.run --mode elenchus --start 2024-09-01 --model openrouter/qwen/qwen3-235b-a22b
```

### Development / cheap iteration

Use a smaller or cheaper model during development:

```bash
uv run python -m scripts.run --mode elenchus --start 2025-01-01 --end 2025-03-01 \
    --model openrouter/deepseek/deepseek-chat-v3
```

### Short test run

```bash
uv run python -m scripts.run --mode elenchus --start 2025-06-01 --end 2025-07-01 \
    --mutation-interval 5
```

---

## Resilience

Long backtests (250+ days) take many hours. The system is built to survive interruptions and restarts without losing progress.

### Checkpointing

The backtest saves state atomically after each completed trading day. On restart, it detects the checkpoint and resumes from the last completed day. Checkpoint files are written to `results/backtest_status.json`.

### Graceful Shutdown

SIGTERM and SIGINT (Ctrl+C) are handled. The running day completes, state is saved, and the process exits cleanly. No partial-day data is written.

### Auto-Restart

`scripts/run_monitored.sh` wraps the backtest with up to 10 restart attempts, with a 2-minute gap between attempts. Each attempt rotates to a new log file to prevent unbounded log growth.

### Signal Notifications

`scripts/signal_notify.py` writes JSON signal files to `~/maei/data/signals/inbox/` on crash, completion, or failure. This integrates with external monitoring without requiring a persistent daemon.

### Log Rotation

Each restart attempt in `run_monitored.sh` writes to a numbered log file (`run_1.log`, `run_2.log`, etc.), preventing any single log file from growing unbounded across restarts.

---

## Model Support

Uses [litellm](https://docs.litellm.ai/) -- any model supported by litellm works. Examples:

- **OpenRouter:** `openrouter/qwen/qwen3-235b-a22b`, `openrouter/deepseek/deepseek-chat-v3`
- **Anthropic:** `anthropic/claude-sonnet-4-20250514`
- **Local (Ollama):** `ollama/qwen3:32b`

Set the appropriate API key in `.env` for your chosen provider.

---

## Understanding the Output

### Terminal output

Every `--mutation-interval` days, the system prints:

```
  Day 20 (2024-10-01)
  Portfolio Sharpe: 0.847
  Cumulative return: +3.21%
  Agent weights:
    macro_regime          w=1.050  sharpe=0.412
    tech_semi             w=1.050  sharpe=0.381
    quality               w=1.000  sharpe=0.220
    rates_yield           w=0.950  sharpe=0.102
    energy                w=0.950  sharpe=-0.044
    risk_officer          w=0.950  sharpe=-0.130
    cio                   w=0.950  sharpe=-0.218
    volatility            w=0.950  sharpe=-0.340
```

For Elenchus mode, you'll also see filtering messages:

```
  [volatility] GLD FILTERED by Elenchus (deutsch=0.25)
  [energy] SLB FILTERED by Elenchus (deutsch=0.33)
```

### A/B comparison (printed at end of `--mode ab`)

```
  A/B COMPARISON
  Metric                         Vanilla     Elenchus
  ------------------------------------------------------
  Final Sharpe                     0.847        1.132
  Cumulative Return               +3.21%       +5.44%
  Mutations Applied                    6            6
  Trading Days                       250          250
```

### Output files

| File | Content |
|------|---------|
| `results/vanilla_returns.csv` | Daily portfolio returns for vanilla branch |
| `results/elenchus_returns.csv` | Daily portfolio returns for Elenchus branch |
| `results/elenchus_analysis.csv` | Per-recommendation deutsch_score data |

### elenchus_analysis.csv columns

| Column | Description |
|--------|-------------|
| `date` | Trading date |
| `agent` | Which agent produced the recommendation |
| `ticker` | Recommended ticker |
| `deutsch_score` | Ratio of load-bearing reasoning components (0.0–1.0) |
| `load_bearing` | Count of components that were necessary |
| `total_components` | Total reasoning components probed |
| `hard_to_vary` | Boolean: did this pass the 0.6 threshold? |
| `conclusion` | The synthesized rationale from the agent |
| `reasoning_components` | Full list of reasoning components as submitted to the probe |
| `probes` | Nested list of per-probe results — each entry contains `component` (original text), `replacement` (generated alternative), `survived` (boolean: did the conclusion hold after replacement?), and `reasoning` (judge agent's explanation) |

---

## Interpreting Results

There are three possible outcomes. Each is meaningful.

### Outcome 1: Elenchus Sharpe > Vanilla Sharpe

The explanation-quality filter improved portfolio performance. Recommendations backed by hard-to-vary reasoning were better trading signals than those backed by decorative reasoning. This is evidence that the Deutsch criterion has instrumental value — it's not just epistemically satisfying, it's economically useful.

**What to look for:** Does the advantage appear immediately or only after the easy wins are exhausted? A crossover point would suggest that explanation-filtered search degrades slower than random search as the search space becomes harder.

### Outcome 2: Elenchus Sharpe < Vanilla Sharpe

The filter is too restrictive. Some profitable recommendations had reasoning that looked decorative to the probe but was actually capturing a real pattern the probe couldn't verify. Or: the LLM agents' reasoning is so thoroughly post-hoc that filtering by reasoning quality is filtering out real signal.

**What to look for:** Is the filtering rate very high (>60% of recommendations filtered)? If so, the agents may be fundamentally incapable of producing hard-to-vary explanations, and the filter is just destroying signal.

### Outcome 3: No significant difference

The Elenchus layer is neutral — it neither helps nor hurts. This would mean the deutsch_score is uncorrelated with recommendation quality. The agents produce reasoning that *looks* structured but has no relationship to whether the trade works.

**What to look for:** Is deutsch_score approximately constant across agents and over time? If it doesn't vary, it can't discriminate.

### Secondary metrics to examine

1. **deutsch_score trend over time**: If agents are being selected (via Darwinian weights and autoresearch mutations) in the Elenchus branch, does mean deutsch_score increase? That would mean the evolutionary pressure is indirectly selecting for better reasoners, not just better traders.

2. **Agent weight divergence**: Do the same agents get upweighted in both branches? Or does Elenchus change which agents the system trusts? If the Darwinian rankings differ, Elenchus is surfacing a different signal about agent quality.

3. **Filtering rate by agent**: Which agents produce the most decorative reasoning? This identifies which roles are most susceptible to post-hoc rationalization.

---

## Project Structure

```
atlas-elenchus/
├── .env.example              # Environment variable template
├── .gitignore
├── pyproject.toml            # Python dependencies (uv)
├── README.md
│
├── architecture/
│   └── ARCHITECTURE.md       # Visual diagrams of data flow
│
├── prompts/
│   └── agents/               # Agent system prompts — THE MUTABLE ARTIFACT
│       ├── macro_regime.md
│       ├── rates_yield.md
│       ├── volatility.md
│       ├── tech_semi.md
│       ├── energy.md
│       ├── quality.md
│       ├── risk_officer.md
│       └── cio.md
│
├── scripts/
│   ├── run.py                # Main entry point
│   ├── run_monitored.sh      # Auto-restart wrapper with up to 10 attempts, per-attempt log rotation, status tracking (results/backtest_status.json), and crash/completion signal notifications
│   └── signal_notify.py      # Writes JSON signal files to ~/maei/data/signals/inbox/ for external monitoring integration
│
├── src/
│   ├── __init__.py
│   ├── llm.py                # litellm wrapper for multi-provider support
│   ├── market_data.py        # yfinance data provider, MarketSnapshot iterator
│   ├── agent.py              # Agent class, Recommendation dataclass
│   ├── elenchus.py           # Deutsch Probe implementation
│   ├── pipeline.py           # Multi-layer agent orchestration
│   ├── portfolio.py          # Position tracking, Sharpe calculation, Darwinian scoring
│   └── autoresearch.py       # Prompt mutation engine with git keep/discard
│
└── results/                  # Output CSVs (generated at runtime)
```

### Module Responsibilities

**`src/market_data.py`** — Downloads historical OHLCV data from yfinance for a configurable stock universe. Provides a `MarketSnapshot` iterator that yields one day at a time with no future data leakage. Each snapshot includes trailing returns (1d, 5d, 20d), realized volatility, and a simple regime classification (risk-on / risk-off / neutral).

**`src/llm.py`** — litellm wrapper providing multi-provider LLM support (OpenRouter, Anthropic, Ollama, etc.).

**`src/agent.py`** — Defines the `Agent` class. Each agent wraps an LLM call (via litellm) with a system prompt loaded from a `.md` file. The response contract requires JSON with `reasoning_components` — a list of 3-5 independent, falsifiable claims. Also handles Darwinian weight bounds (0.3-2.5) and weight adjustment.

**`src/elenchus.py`** — The Deutsch Probe. For each recommendation's reasoning components: (1) a perturbation agent generates a plausible alternative for one component, (2) a judge agent evaluates whether the conclusion survives the swap. Produces a `deutsch_score` (ratio of load-bearing components). Threshold of 0.6 classifies as "hard to vary."

**`src/pipeline.py`** — Orchestrates agents across layers. Each layer's output feeds into the next as upstream signals. Accepts a `probe_layers` parameter (list of layer numbers, default `[1, 2, 3]`) controlling which layers are probed. When Elenchus is active, only the configured layers have their recommendations probed; other layers pass through without probing. The production 250-day run used L3-only probing.

**`src/portfolio.py`** — Converts recommendations into sized positions. Vanilla scoring: `conviction × darwinian_weight`. Elenchus scoring: `conviction × darwinian_weight × deutsch_score`. Tracks daily P&L, calculates annualized Sharpe, and handles Darwinian quartile-based weight updates.

**`src/autoresearch.py`** — The prompt mutation engine. Identifies the worst agent by rolling Sharpe. A mutator LLM reads the agent's current prompt and performance stats, then generates one focused modification. The mutation is applied to disk. On the next scoring interval, if Sharpe improved → `git commit`, if not → `git checkout` to revert.

---

## Modifying the System

### Adding a new agent

1. Create a new prompt file in `prompts/agents/` (e.g., `biotech.md`)
2. Add an entry to `DEFAULT_AGENTS` in `src/pipeline.py`:
   ```python
   {"agent_id": "biotech", "layer": 2, "prompt_file": "biotech.md"},
   ```
3. Add relevant tickers to `DEFAULT_UNIVERSE` in `src/market_data.py`

### Changing the Deutsch Probe threshold

In `src/elenchus.py`, modify the `is_hard_to_vary` property on `ElenchusResult`:

```python
@property
def is_hard_to_vary(self) -> bool:
    return self.deutsch_score > 0.6  # ← change this threshold
```

### Changing position sizing

The current implementation uses equal-weight among selected positions. To add conviction-weighted sizing, modify `Portfolio.rebalance()` in `src/portfolio.py`.

### Adding fundamental data

Extend `MarketSnapshot` in `src/market_data.py` and update `Agent._build_context()` in `src/agent.py` to include the new data in the prompt context.

### Using a different model for probes vs agents

In `src/pipeline.py`, the `Pipeline` constructor passes the same model to both the agent calls and the Elenchus probe. To separate them, pass a cheaper model for the probe:

```python
self.probe = ElenchusProbe(model="openrouter/deepseek/deepseek-chat-v3")  # cheaper probe
```

---

## Cost Estimates

Costs vary significantly by model. Estimates below are per 250 trading days (approximate).

**Token volume per trading day:**
- Vanilla: 8 agents x ~2,000 output tokens = ~16,000 tokens
- Elenchus: same + ~4 components/rec x ~10 recs x 2 LLM calls x ~500 tokens = ~56,000 tokens
- Autoresearch mutation (every 20 days): ~3,000 tokens (negligible)

**Full backtest cost estimates (250 trading days, A/B mode):**

| Model | Vanilla | Elenchus (all layers) | Elenchus (L3-only) | A/B (both) |
|---|---|---|---|---|
| `openrouter/qwen/qwen3-235b-a22b` | ~$15–30 | ~$50–100 | ~$4–6 | ~$65–130 |
| `openrouter/deepseek/deepseek-chat-v3` | ~$10–20 | ~$35–70 | ~$3–5 | ~$45–90 |
| `anthropic/claude-sonnet-4-20250514` | ~$125–250 | ~$500–1,000 | ~$125–200 | ~$625–1,250 |
| `ollama/qwen3:32b` (local) | Free | Free | Free | Free |

L3-only probing (`--probe-layers 3`) applies the Deutsch Probe only to the CIO and risk_officer outputs. This cuts probe API calls by ~75% relative to all-layer probing.

**Cost reduction strategies:**
- Use `--probe-layers 3` for L3-only probing — the 250-day production run used this setting
- Use a cheap model (OpenRouter Qwen/DeepSeek, or local Ollama) during development
- Reduce `--mutation-interval` to get faster feedback on fewer days
- Shorten the date range for initial validation before running full backtest

---

## Known Limitations

### Statistical limitations
- The 19-day initial test period is too short for statistical significance. Results are directional, not conclusive.

### Data limitations
- Agents see only OHLCV price data from yfinance. No fundamentals, earnings, news, order flow, or analyst estimates. Real trading systems have much richer information.
- Survivorship bias in the ticker universe -- delisted or acquired companies may not appear correctly in historical data.
- No transaction costs, slippage, or market impact modeled. Real returns would be lower.
- Look-ahead bias in regime classification.

### Experimental design limitations
- **Self-evaluation bias:** The Elenchus probe uses the same model family (Claude) as the agents whose reasoning it evaluates. The probe may be systematically lenient toward reasoning patterns that Claude tends to produce. This doesn't invalidate the A/B comparison (the bias is constant across both branches) but limits generalizability claims.
- **Equal-weight position sizing** is a simplification. Real portfolio construction would use vol-targeting, Kelly criterion, or conviction-weighted sizing.
- **No regime-aware rebalancing.** Positions are rebalanced daily based on new recommendations, regardless of holding period effects.

### Elenchus-specific limitations
- The perturbation agent may not generate sufficiently adversarial replacements. If replacements are too similar to the original, the probe underestimates how many components are decorative.
- The judge agent may have difficulty with nuanced logical dependencies, especially in multi-factor arguments where components interact.
- The 0.6 threshold for "hard to vary" is arbitrary and, based on initial results, too aggressive -- it leaves ~3.3 recommendations/day, creating concentration risk. The optimal threshold is an empirical question this experiment can help answer.

### Infrastructure limitations
- The backtest runs sequentially (one day at a time, one branch at a time). For long date ranges, this takes hours. Parallelization across days is possible but not implemented.
- Git operations for keep/discard assume a clean working tree. Running multiple experiments simultaneously in the same repo will conflict.

---

## Theoretical Context

### David Deutsch's Criterion

In *The Beginning of Infinity* (2011), physicist David Deutsch argues that good explanations are *hard to vary* — you cannot change their details and still have them account for the phenomena they explain. This distinguishes genuine understanding from ad-hoc stories that happen to fit the data.

Example: "The seasons happen because Earth's axis is tilted" is hard to vary — if you change the tilt angle, the explanation makes a different prediction. "The seasons happen because the gods will it" is easy to vary — you can substitute any supernatural mechanism and the explanation still "works" because it never constrained anything.

### Application to AI Agents

When an LLM agent produces a trading recommendation with reasoning, we can ask: are the reasoning components actually constraining the conclusion, or are they decoration? An agent that says "buy NVDA because of capex acceleration, relative strength, and revenue mix shift" might have arrived at the same conclusion from any three plausible-sounding tech narratives. If so, its reasoning is easy to vary — epistemically empty.

The Deutsch Probe operationalizes this test. By swapping each component with a plausible alternative and checking if the conclusion survives, we measure how much logical work each component is doing.

### Relationship to Existing Work

**Chain-of-thought prompting** requires models to show reasoning steps but provides no verification that the reasoning is genuine. The Deutsch Probe adds that verification layer.

**Constitutional AI** defines rules that outputs must follow. Elenchus makes no rules about content — it only tests whether the stated reasoning actually supports the stated conclusion.

**Debate protocols** (Irving et al.) pit two agents against each other to find flaws. The Deutsch Probe is simpler: it perturbs one agent's reasoning and checks structural integrity, with no adversarial agent needed.

### Why This Matters

If explanation quality (as measured by the Deutsch Probe) correlates with recommendation quality (as measured by Sharpe), then:

1. LLM agents are doing something more than pattern matching — their reasoning has instrumental value
2. Filtering by explanation quality is a general-purpose technique for improving multi-agent systems, applicable beyond finance
3. The Deutsch criterion provides a computable proxy for "understanding" that doesn't require ground-truth labels

If it doesn't correlate, that's equally important — it would mean current LLM reasoning is fundamentally decorative, and the community should stop treating chain-of-thought as evidence of understanding.

---

## License

[PolyForm Noncommercial 1.0.0](LICENSE.md) — free for research, academic, and personal use. Commercial use requires permission from the author.
