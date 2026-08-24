#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
PYTHON_BIN="$PROJECT_DIR/.venv/bin/python"
LOG_DIR="$PROJECT_DIR/logs"
LOG_FILE="$LOG_DIR/pregame_context_refresh_$(date +%F).log"

mkdir -p "$LOG_DIR"
exec >> "$LOG_FILE" 2>&1

echo "[$(date --iso-8601=seconds)] pregame context refresh started"
cd "$PROJECT_DIR"
"$PYTHON_BIN" scripts/refresh_pregame_context.py
curl -fsS "http://localhost:8501/latest.html" >/dev/null
echo "[$(date --iso-8601=seconds)] pregame context refresh completed"
