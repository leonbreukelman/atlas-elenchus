# Resilient 250-Day A/B Backtest — Design Spec

## Goal

Run a 250 trading day A/B backtest comparing vanilla (no probe) vs L3-only Deutsch Probe, with automatic crash recovery, signal notifications, and robust logging. Target runtime: ~3.5 days.

## Key Design Decision: L3-Only Probing

Instead of probing all 8 agents across all layers, probe only Layer 3 (CIO, risk_officer). Rationale:
- Layer 3 makes the highest-stakes decisions with the most room for decorative reasoning
- CIO had the worst deutsch scores in the 19-day test (mean 0.125, 0% hard-to-vary)
- Reduces probe API calls by ~75%, cutting elenchus day time from ~60 min to ~20 min
- 250-day run becomes ~3.5 days instead of ~12 days

All agents still run and produce recommendations. The only change is which recommendations get probed before portfolio construction.

## Components

### 1. L3-Only Probe Mode (src/pipeline.py, scripts/run.py)

New `--probe-layers` CLI flag (default: "1,2,3" for backward compatibility). When set to "3", Pipeline only calls ElenchusProbe.probe_batch() on Layer 3 output.

Changes:
- `scripts/run.py`: Add `--probe-layers` argument, parse to list of ints, pass to run_backtest()
- `src/pipeline.py`: Pipeline.__init__() accepts `probe_layers: list[int]` parameter. In run_day(), only probe when `layer_num in self.probe_layers`.

### 2. Auto-Restart Wrapper (scripts/run_monitored.sh)

Rewrite as restart loop:

```
MAX_ATTEMPTS=10
RESTART_DELAY=120  # 2 minutes between attempts
attempt=0

while attempt < MAX_ATTEMPTS:
    run the backtest with PYTHONUNBUFFERED=1
    if exit code 0:
        write "completed" signal
        break
    else:
        attempt++
        write "crashed, attempt N/MAX" to log
        if attempt < MAX_ATTEMPTS:
            sleep RESTART_DELAY
        else:
            write "failed_permanently" signal
```

Checkpointing means each restart resumes from last completed day. 10 attempts with 2-minute gaps gives 20 minutes of tolerance for transient outages.

Log file: `results/backtest_YYYY-MM-DD_HHMMSS.log` (new file per restart attempt).

### 3. MÆI Signal Notification (scripts/signal_notify.py)

Small Python script called by the wrapper on crash/completion. Writes a JSON signal file to ~/maei/data/signals/inbox/ (or a known location the MÆI SessionStart hook checks).

Signal payload:
```json
{
    "source": "atlas-elenchus",
    "type": "backtest_status",
    "status": "completed" | "crashed" | "failed_permanently",
    "days_completed": N,
    "total_days": 250,
    "last_date": "YYYY-MM-DD",
    "exit_code": N,
    "attempt": N,
    "timestamp": "ISO8601"
}
```

### 4. Graceful Shutdown (scripts/run.py)

Signal handlers for SIGTERM and SIGINT:
- Catch signal
- Set a flag that the main loop checks after each completed day
- Save checkpoint
- Exit with code 0 (so wrapper doesn't count as crash)

This handles system reboot, kill command, Ctrl+C.

### 5. Unbuffered Output

Wrapper sets `PYTHONUNBUFFERED=1` before launching Python. Real-time log visibility, no lost output on crash.

### 6. Run Configuration

```
--mode ab
--start 2024-09-01
--end 2025-09-01
--mutation-interval 20
--probe-layers 3
--model openrouter/qwen/qwen3-235b-a22b
```

- 250 trading days
- Mutations every 20 days (~12 total, prompts evolve under selection pressure)
- L3-only probing
- Qwen-235B via OpenRouter

Produces:
- results/vanilla_returns.csv
- results/elenchus_returns.csv
- results/elenchus_analysis.csv (deutsch scores for L3 recommendations only)

### 7. Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| scripts/run_monitored.sh | Rewrite | Restart loop, log rotation, signal calls, unbuffered |
| scripts/run.py | Modify | Add --probe-layers flag, SIGTERM/SIGINT handler |
| src/pipeline.py | Modify | Accept probe_layers param, conditional probing |
| scripts/signal_notify.py | Create | Writes MÆI signal on crash/completion |

### 8. Failure Recovery Matrix

| Failure | Detection | Recovery |
|---------|-----------|----------|
| API timeout | litellm.Timeout | Retry 3x with exponential backoff (5s/15s/45s) |
| Parse error | Exception in _parse_response | Defensive parsing returns fallback |
| Process crash (unhandled exception) | Non-zero exit code | Auto-restart from checkpoint (max 10 attempts) |
| Extended API outage (>20 min) | 10 failed restart attempts | "failed_permanently" signal to MÆI |
| System reboot / SIGTERM | Signal handler | Checkpoint saved, clean exit |
| Disk full | Unlikely (~2MB total output) | No mitigation needed |
| OOM kill | Unlikely (286MB RSS) | No mitigation needed |

### 9. Estimated Costs

| Component | Per Day | 250 Days |
|-----------|---------|----------|
| 8 agent calls (both branches) | ~18K tokens | ~4.5M tokens |
| L3 probes (2 agents, ~4 components each, 2 calls per component) | ~12K tokens | ~3M tokens |
| Autoresearch mutations (every 20 days) | ~2K tokens | ~25K tokens |
| **Total** | ~32K tokens | ~7.5M tokens |
| **Cost at Qwen-235B pricing** | | **~$4-6** |

## Out of Scope (Future Phases)

- Async parallelization of agent calls
- Full-stack probing (all layers)
- Dollar-denominated portfolio simulation
- Transaction cost modeling
- Walk-forward validation framework
- Multiple independent runs for statistical confidence
