#!/usr/bin/env bash
# Cron wrapper for atlas-elenchus paper trading.
#
# Usage:
#   cron_paper.sh morning           # elenchus mode (default)
#   cron_paper.sh evening           # elenchus mode (default)
#   cron_paper.sh morning vanilla   # vanilla mode
#   cron_paper.sh evening vanilla   # vanilla mode

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="${REPO_DIR:=$(cd "$(dirname "$0")/.." && pwd)}"

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RUN_TYPE="${1:?Usage: cron_paper.sh <morning|evening> [vanilla]}"
MODE="${2:-elenchus}"
LOG_DIR="${REPO_DIR}/logs"

if [ "$MODE" = "vanilla" ]; then
    DB_PATH="${REPO_DIR}/data/paper_vanilla.db"
    VANILLA_FLAG="--vanilla"
else
    DB_PATH="${REPO_DIR}/data/paper_elenchus.db"
    VANILLA_FLAG=""
fi

mkdir -p "$LOG_DIR" "$(dirname "$DB_PATH")"

LOGFILE="${LOG_DIR}/paper_${MODE}_${RUN_TYPE}_$(date +%Y-%m-%d).log"

cd "$REPO_DIR"

echo "=== Paper trading ${MODE} ${RUN_TYPE} started at $(date -Iseconds) ===" >> "$LOGFILE"

uv run python scripts/run_paper.py \
    --run-type "$RUN_TYPE" \
    --db-path "$DB_PATH" \
    $VANILLA_FLAG \
    >> "$LOGFILE" 2>&1

echo "=== Paper trading ${MODE} ${RUN_TYPE} finished at $(date -Iseconds) ===" >> "$LOGFILE"
