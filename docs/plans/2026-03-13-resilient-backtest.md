# Resilient 250-Day A/B Backtest Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox syntax for tracking.

**Goal:** Make the atlas-elenchus A/B backtest resilient for a 250-day run with L3-only probing, auto-restart, graceful shutdown, and MÆI signal notifications.

**Architecture:** Four targeted changes to existing code — add probe_layers parameter to pipeline, add signal handlers and CLI flag to run.py, rewrite wrapper with restart loop, create signal notification script.

**Tech Stack:** Python 3.10+, litellm, bash, JSON signals

**Spec:** docs/specs/2026-03-13-resilient-backtest-design.md

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| src/pipeline.py | Modify | Add probe_layers parameter for selective layer probing |
| scripts/run.py | Modify | Add --probe-layers CLI flag, SIGTERM/SIGINT handlers |
| scripts/run_monitored.sh | Rewrite | Auto-restart loop, unbuffered output, log rotation, signal calls |
| scripts/signal_notify.py | Create | Write MÆI signal on crash/completion |

---

## Task 1: Add probe_layers parameter to Pipeline

**Files:**
- Modify: `src/pipeline.py`

- [ ] **Step 1: Add probe_layers parameter to Pipeline.__init__()**

In `src/pipeline.py`, modify `Pipeline.__init__()` to accept `probe_layers`:

Change the signature from:
```python
def __init__(
    self,
    prompt_dir: Path,
    use_elenchus: bool = False,
    model: str = "openrouter/qwen/qwen3-235b-a22b",
):
```

To:
```python
def __init__(
    self,
    prompt_dir: Path,
    use_elenchus: bool = False,
    model: str = "openrouter/qwen/qwen3-235b-a22b",
    probe_layers: list[int] | None = None,
):
```

Add after `self.model = model`:
```python
self.probe_layers = probe_layers or [1, 2, 3]
```

- [ ] **Step 2: Make probing conditional on layer number**

In `run_day()`, change line 97 from:
```python
if self.probe and layer_recs:
```
To:
```python
if self.probe and layer_recs and layer_num in self.probe_layers:
```

This is a one-line change. All other probing logic (filtering, fallback) stays exactly the same.

- [ ] **Step 3: Verify no other references need updating**

The only caller of Pipeline() is in `scripts/run.py`, which will be updated in Task 2.

- [ ] **Step 4: Commit**

```bash
cd ~/projects/atlas-elenchus
git add src/pipeline.py
git commit -m "feat: add probe_layers parameter for selective layer probing"
```

---

## Task 2: Add --probe-layers flag and signal handlers to run.py

**Files:**
- Modify: `scripts/run.py`

- [ ] **Step 1: Add signal handler imports and shutdown flag**

At the top of `scripts/run.py`, after the existing imports (after line 25 `from pathlib import Path`), add:

```python
import signal
import sys

_shutdown_requested = False
```

- [ ] **Step 2: Add signal handler function**

After the `_load_checkpoint` function (after line 77), add:

```python
def _handle_shutdown(signum, frame):
    """Handle SIGTERM/SIGINT — set flag so main loop saves checkpoint and exits cleanly."""
    global _shutdown_requested
    signal_name = signal.Signals(signum).name
    print(f"\n  Received {signal_name} — will save checkpoint and exit after current day completes")
    _shutdown_requested = True
```

- [ ] **Step 3: Register signal handlers in main()**

In `main()`, after `load_dotenv()` (line 263), add:

```python
signal.signal(signal.SIGTERM, _handle_shutdown)
signal.signal(signal.SIGINT, _handle_shutdown)
```

- [ ] **Step 4: Add --probe-layers CLI argument**

In `main()`, after the `--output-dir` argument (line 272), add:

```python
parser.add_argument("--probe-layers", default="1,2,3",
                    help="Comma-separated layer numbers to probe with Elenchus (default: 1,2,3)")
```

- [ ] **Step 5: Parse probe_layers and pass to run_backtest**

After `args = parser.parse_args()` (line 273), add:

```python
probe_layers = [int(x.strip()) for x in args.probe_layers.split(",")]
```

- [ ] **Step 6: Add probe_layers parameter to run_backtest function**

Change the `run_backtest` signature to include `probe_layers`:

```python
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
```

- [ ] **Step 7: Pass probe_layers to Pipeline constructor**

In `run_backtest`, change line 95 from:
```python
pipeline = Pipeline(prompt_dir, use_elenchus=use_elenchus, model=model)
```
To:
```python
pipeline = Pipeline(prompt_dir, use_elenchus=use_elenchus, model=model, probe_layers=probe_layers)
```

- [ ] **Step 8: Add shutdown check in the main loop**

In `run_backtest`, after the checkpoint save (after line 233, after `_save_checkpoint(...)`), add:

```python
if _shutdown_requested:
    print(f"  Shutdown requested — checkpoint saved at day {day_count} ({date_str})")
    sys.exit(0)
```

- [ ] **Step 9: Pass probe_layers in all run_backtest calls in main()**

Update both `run_backtest()` calls in `main()` to include `probe_layers=probe_layers`:

The vanilla call (around line 311):
```python
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
```

The elenchus call (around line 335):
```python
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
```

Note: For vanilla mode, probe_layers doesn't matter (use_elenchus=False so probe is None), but passing it keeps the interface consistent.

- [ ] **Step 10: Commit**

```bash
cd ~/projects/atlas-elenchus
git add scripts/run.py
git commit -m "feat: add --probe-layers flag and SIGTERM/SIGINT graceful shutdown"
```

---

## Task 3: Create signal notification script

**Files:**
- Create: `scripts/signal_notify.py`

- [ ] **Step 1: Create the script**

Create `scripts/signal_notify.py`:

```python
#!/usr/bin/env python3
"""Write a signal file for MÆI to pick up on session start."""

import json
import sys
from datetime import datetime
from pathlib import Path

SIGNAL_DIR = Path.home() / "maei" / "data" / "signals" / "inbox"


def write_signal(status: str, days_completed: int = 0, total_days: int = 250,
                 last_date: str = "", exit_code: int = 0, attempt: int = 0):
    """Write a JSON signal file that MÆI's SessionStart hook can detect."""
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().isoformat()
    signal_id = f"atlas-elenchus-{timestamp.replace(':', '-')}"

    signal = {
        "source": "atlas-elenchus",
        "type": "backtest_status",
        "status": status,
        "days_completed": days_completed,
        "total_days": total_days,
        "last_date": last_date,
        "exit_code": exit_code,
        "attempt": attempt,
        "timestamp": timestamp,
    }

    signal_file = SIGNAL_DIR / f"{signal_id}.json"
    signal_file.write_text(json.dumps(signal, indent=2))
    print(f"Signal written: {signal_file}")


if __name__ == "__main__":
    # Called from run_monitored.sh with args: status days_completed total_days last_date exit_code attempt
    if len(sys.argv) < 2:
        print("Usage: signal_notify.py <status> [days_completed] [total_days] [last_date] [exit_code] [attempt]")
        sys.exit(1)

    write_signal(
        status=sys.argv[1],
        days_completed=int(sys.argv[2]) if len(sys.argv) > 2 else 0,
        total_days=int(sys.argv[3]) if len(sys.argv) > 3 else 250,
        last_date=sys.argv[4] if len(sys.argv) > 4 else "",
        exit_code=int(sys.argv[5]) if len(sys.argv) > 5 else 0,
        attempt=int(sys.argv[6]) if len(sys.argv) > 6 else 0,
    )
```

- [ ] **Step 2: Make executable**

```bash
chmod +x ~/projects/atlas-elenchus/scripts/signal_notify.py
```

- [ ] **Step 3: Commit**

```bash
cd ~/projects/atlas-elenchus
git add scripts/signal_notify.py
git commit -m "feat: add MÆI signal notification script for backtest status"
```

---

## Task 4: Rewrite run_monitored.sh with auto-restart

**Files:**
- Rewrite: `scripts/run_monitored.sh`

- [ ] **Step 1: Rewrite the wrapper script**

Replace the entire contents of `scripts/run_monitored.sh` with:

```bash
#!/bin/bash
# Resilient backtest wrapper with auto-restart, log rotation, and MÆI signals.
# Usage: nohup bash scripts/run_monitored.sh [backtest args...] &
#
# Features:
#   - Auto-restarts on crash (max 10 attempts, 2-min gap)
#   - Checkpointing means each restart resumes from last completed day
#   - Per-attempt log files (no monolithic log)
#   - Unbuffered Python output for real-time visibility
#   - MÆI signal on crash/completion/permanent failure

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$PROJECT_DIR/results"
STATUS_FILE="$RESULTS_DIR/backtest_status.json"

MAX_ATTEMPTS=10
RESTART_DELAY=120  # seconds between restart attempts

mkdir -p "$RESULTS_DIR"
cd "$PROJECT_DIR"

# Helper: extract days completed from checkpoint
get_days_completed() {
    local mode="$1"
    local checkpoint="$RESULTS_DIR/${mode}_checkpoint.json"
    if [ -f "$checkpoint" ]; then
        python3 -c "import json; d=json.load(open('$checkpoint')); print(len(d.get('daily_returns',[])))" 2>/dev/null || echo "0"
    else
        echo "0"
    fi
}

# Helper: get last date from checkpoint
get_last_date() {
    local mode="$1"
    local checkpoint="$RESULTS_DIR/${mode}_checkpoint.json"
    if [ -f "$checkpoint" ]; then
        python3 -c "import json; d=json.load(open('$checkpoint')); dr=d.get('daily_returns',[]); print(dr[-1]['date'] if dr else 'none')" 2>/dev/null || echo "none"
    else
        echo "none"
    fi
}

# Determine mode from args (for checkpoint queries)
MODE="ab"
for arg in "$@"; do
    if [ "$prev_arg" = "--mode" ]; then
        MODE="$arg"
        break
    fi
    prev_arg="$arg"
done

# Main restart loop
ATTEMPT=0
START_TIME=$(date -Iseconds)

echo "{\"status\": \"running\", \"pid\": $$, \"started\": \"$START_TIME\", \"attempt\": 0, \"max_attempts\": $MAX_ATTEMPTS, \"args\": \"$*\"}" > "$STATUS_FILE"

while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    ATTEMPT=$((ATTEMPT + 1))
    LOG_FILE="$RESULTS_DIR/backtest_$(date +%Y%m%d_%H%M%S)_attempt${ATTEMPT}.log"

    echo "=== Attempt $ATTEMPT/$MAX_ATTEMPTS ($(date -Iseconds)) ===" | tee "$LOG_FILE"

    # Update status file
    echo "{\"status\": \"running\", \"pid\": $$, \"started\": \"$START_TIME\", \"attempt\": $ATTEMPT, \"max_attempts\": $MAX_ATTEMPTS, \"args\": \"$*\"}" > "$STATUS_FILE"

    # Run with unbuffered output
    PYTHONUNBUFFERED=1 uv run python -m scripts.run "$@" >> "$LOG_FILE" 2>&1
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        # Success
        END_TIME=$(date -Iseconds)
        DAYS=$(get_days_completed "$MODE")
        LAST_DATE=$(get_last_date "$MODE")

        cat > "$STATUS_FILE" << EOF
{
    "status": "completed",
    "exit_code": 0,
    "started": "$START_TIME",
    "ended": "$END_TIME",
    "attempt": $ATTEMPT,
    "days_completed": $DAYS,
    "args": "$*"
}
EOF
        # Signal MÆI
        python3 "$SCRIPT_DIR/signal_notify.py" "completed" "$DAYS" "250" "$LAST_DATE" "0" "$ATTEMPT" 2>/dev/null || true

        echo "=== Backtest completed successfully on attempt $ATTEMPT ===" | tee -a "$LOG_FILE"
        exit 0
    fi

    # Crash
    DAYS=$(get_days_completed "$MODE")
    LAST_DATE=$(get_last_date "$MODE")
    LAST_LOG=$(tail -3 "$LOG_FILE" | tr '\n' ' | ' | sed 's/"/\\"/g')

    echo "=== Crashed with exit code $EXIT_CODE on attempt $ATTEMPT (days=$DAYS, last=$LAST_DATE) ===" | tee -a "$LOG_FILE"

    # Signal MÆI about crash
    python3 "$SCRIPT_DIR/signal_notify.py" "crashed" "$DAYS" "250" "$LAST_DATE" "$EXIT_CODE" "$ATTEMPT" 2>/dev/null || true

    if [ $ATTEMPT -lt $MAX_ATTEMPTS ]; then
        echo "  Restarting in ${RESTART_DELAY}s (checkpoint will resume from day $((DAYS + 1)))..." | tee -a "$LOG_FILE"
        sleep $RESTART_DELAY
    fi
done

# All attempts exhausted
END_TIME=$(date -Iseconds)
DAYS=$(get_days_completed "$MODE")

cat > "$STATUS_FILE" << EOF
{
    "status": "failed_permanently",
    "exit_code": $EXIT_CODE,
    "started": "$START_TIME",
    "ended": "$END_TIME",
    "attempt": $ATTEMPT,
    "max_attempts": $MAX_ATTEMPTS,
    "days_completed": $DAYS,
    "args": "$*"
}
EOF

# Signal MÆI about permanent failure
python3 "$SCRIPT_DIR/signal_notify.py" "failed_permanently" "$DAYS" "250" "$LAST_DATE" "$EXIT_CODE" "$ATTEMPT" 2>/dev/null || true

echo "=== All $MAX_ATTEMPTS attempts exhausted. Backtest failed permanently. ===" | tee -a "$LOG_FILE"
exit 1
```

- [ ] **Step 2: Make executable**

```bash
chmod +x ~/projects/atlas-elenchus/scripts/run_monitored.sh
```

- [ ] **Step 3: Commit**

```bash
cd ~/projects/atlas-elenchus
git add scripts/run_monitored.sh
git commit -m "feat: rewrite wrapper with auto-restart loop, log rotation, MÆI signals"
```

---

## Task 5: Smoke test and launch

**Files:**
- None (testing only)

- [ ] **Step 1: Run a 2-day smoke test with L3-only probing**

```bash
cd ~/projects/atlas-elenchus
PYTHONUNBUFFERED=1 uv run python -m scripts.run --mode elenchus --start 2024-11-01 --end 2025-01-05 --mutation-interval 50 --probe-layers 3 --model openrouter/qwen/qwen3-235b-a22b
```

This should produce ~2 trading days after warmup. Verify:
- Only Layer 3 agents (CIO, risk_officer) get probed
- Layer 1 and 2 agents run but their output is NOT probed
- Checkpoint file is created
- Expected output: faster per-day time (~15-20 min vs ~60 min for full probing)

- [ ] **Step 2: Test graceful shutdown**

Run the smoke test again, and while it's processing day 1, send SIGTERM from another terminal:
```bash
kill -TERM <PID>
```

Verify:
- "Received SIGTERM" message appears
- Checkpoint is saved
- Process exits with code 0

- [ ] **Step 3: Test signal notification**

```bash
cd ~/projects/atlas-elenchus
python3 scripts/signal_notify.py "test" 5 250 "2024-12-20" 0 1
```

Verify: signal file created at ~/maei/data/signals/inbox/

- [ ] **Step 4: Test wrapper restart loop**

Create a deliberately broken .env to force a crash, run the wrapper, verify it restarts and logs each attempt. Then restore .env.

- [ ] **Step 5: Launch the full 250-day A/B backtest**

```bash
cd ~/projects/atlas-elenchus
nohup bash scripts/run_monitored.sh --mode ab --start 2024-09-01 --end 2025-09-01 --mutation-interval 20 --probe-layers 3 --model openrouter/qwen/qwen3-235b-a22b &
echo "PID: $!"
```

Verify wrapper PID is running and status file shows "running".

- [ ] **Step 6: Commit any test fixes**

If smoke testing revealed issues, fix and commit.

- [ ] **Step 7: Push all changes**

```bash
cd ~/projects/atlas-elenchus
git push
```
