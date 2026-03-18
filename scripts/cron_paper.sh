#!/usr/bin/env bash
# Cron wrapper for atlas-elenchus paper trading.
# Runs from the repo root on the 3070.
#
# Crontab entries (AEST):
#   30 23 * * 1-5  /home/leonb/projects/atlas-elenchus/scripts/cron_paper.sh morning
#   30 21 * * 1-5  /home/leonb/projects/atlas-elenchus/scripts/cron_paper.sh evening

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$REPO_DIR"
RUN_TYPE="${1:?Usage: cron_paper.sh <morning|evening>}"
LOG_DIR="${REPO_DIR}/logs"
DB_PATH="${REPO_DIR}/data/paper_ledger.db"

mkdir -p "$LOG_DIR" "$(dirname "$DB_PATH")"

LOGFILE="${LOG_DIR}/paper_${RUN_TYPE}_$(date +%Y-%m-%d).log"

cd "$REPO_DIR"

echo "=== Paper trading ${RUN_TYPE} started at $(date -Iseconds) ===" >> "$LOGFILE"

uv run python scripts/run_paper.py \
    --run-type "$RUN_TYPE" \
    --db-path "$DB_PATH" \
    >> "$LOGFILE" 2>&1

echo "=== Paper trading ${RUN_TYPE} finished at $(date -Iseconds) ===" >> "$LOGFILE"
