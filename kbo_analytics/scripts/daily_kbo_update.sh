#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/tera/1.project/1.sports_analytics/kbo_analytics"
PYTHON_BIN="$PROJECT_DIR/.venv/bin/python"
LOG_DIR="$PROJECT_DIR/logs"
LOG_FILE="$LOG_DIR/daily_update_$(date +%F).log"

mkdir -p "$LOG_DIR"
exec >> "$LOG_FILE" 2>&1

echo "[$(date --iso-8601=seconds)] daily KBO update started"

cd "$PROJECT_DIR"

if [ -f "$PROJECT_DIR/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$PROJECT_DIR/.env"
  set +a
fi

docker compose up -d kbo-db kbo-api dashboard
"$PYTHON_BIN" official_kbo_dashboard.py --training-start-year 2016

curl -fsS "http://localhost:8501/latest.html" >/dev/null
curl -fsS "http://localhost:8501/kt.html" >/dev/null

if ! git diff --quiet -- dashboard data/official modeling/results ../docs; then
  git add dashboard data/official modeling/results ../docs
  git commit -m "Update KBO daily analytics outputs $(date +%F)"
  git push origin main
fi

echo "[$(date --iso-8601=seconds)] daily KBO update completed"
