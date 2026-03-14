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
prev_arg=""
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
    set +e
    PYTHONUNBUFFERED=1 uv run python -m scripts.run "$@" >> "$LOG_FILE" 2>&1
    EXIT_CODE=$?
    set -e

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
LAST_DATE=$(get_last_date "$MODE")

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
