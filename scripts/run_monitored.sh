#!/bin/bash
# Wrapper that runs the backtest and writes a status file when done (success or crash)
# Usage: nohup bash scripts/run_monitored.sh [args...] &

RESULTS_DIR="$(dirname "$0")/../results"
STATUS_FILE="$RESULTS_DIR/backtest_status.json"
LOG_FILE="$RESULTS_DIR/elenchus_run.log"

# Record start
START_TIME=$(date -Iseconds)
echo "{\"status\": \"running\", \"pid\": $$, \"started\": \"$START_TIME\", \"args\": \"$*\"}" > "$STATUS_FILE"

# Run the backtest, capturing output
cd "$(dirname "$0")/.."
uv run python -m scripts.run "$@" > "$LOG_FILE" 2>&1
EXIT_CODE=$?

# Record result
END_TIME=$(date -Iseconds)
LAST_LOG=$(tail -5 "$LOG_FILE" | tr '\n' '|' | sed 's/"/\\"/g')

if [ $EXIT_CODE -eq 0 ]; then
    STATUS="completed"
else
    STATUS="crashed"
fi

cat > "$STATUS_FILE" << STATUSEOF
{
    "status": "$STATUS",
    "exit_code": $EXIT_CODE,
    "started": "$START_TIME",
    "ended": "$END_TIME",
    "last_log": "$LAST_LOG",
    "args": "$*"
}
STATUSEOF
