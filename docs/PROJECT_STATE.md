# KBO Analytics Project State

Last verified: 2026-07-29

## Locations
- Git repository on VM: `/home/wsr/1.project/1.sports_analytics`
- KBO project on VM: `/home/wsr/1.project/1.sports_analytics/kbo_analytics`
- Local process-document worktree: `sports_analytics_process_architecture`
- Production branch on VM: `main`
- Production remote: `git@github-sports-analytics:SaRangWOO/sports_analytics.git`

The old `/home/tera/...` path in earlier documentation is not the current VM path.

## Runtime
- Python: `3.10.12`
- Virtual environment: `/home/wsr/1.project/1.sports_analytics/kbo_analytics/.venv`
- Python dependencies: plain `requirements.txt` with requests, pandas, SQLAlchemy, psycopg2, and scikit-learn
- Services: Docker Compose runs PostgreSQL, legacy mock API, Metabase, and the static dashboard server
- Tests: one focused unittest module at `modeling/test_feature_engineering.py`
- Runtime logs: `kbo_analytics/logs/`; ignored by Git

## Implemented
- KBO standings, schedules/results, hitter/pitcher records, and registered roster collection
- Team and player CSV export with optional PostgreSQL loading
- Team style, tactical/situational proxy, player impact, and dependency reports
- Leakage-safe rolling team features and one-game home/away feature frames
- Candidate win-model training and chronological holdout validation
- Accuracy, Brier Score, Log Loss, calibration, confidence-bucket, ablation, bootstrap, and gate reports
- Today win probability generation using completed games through the previous day
- Independent expected-runs model with validation/today output separation
- Confirmed/estimated starter and lineup context display
- Pitching snapshot accumulation and quality gate
- Team pages, integrated tabbed HTML dashboard, and `docs/` static output
- Daily and pregame cron scripts with pipeline health reporting

## Current Production State
- Operational win model: `핵심 수치 GradientBoosting 보수 시간가중 모델`
- Prediction unit: team rows normalized within each game
- Server validation accuracy: `0.552`
- Validation cutoff: `2026-07-26`
- Prediction training cutoff: `2026-07-28`
- Candidate count in latest server payload: `41`
- `safe_to_replace_model`: `false`
- Failed production gates: accuracy delta, bootstrap stability, current-season evidence

The selected model is re-fit with completed games through the day before prediction after candidate selection. The JSON stores metadata and results; the full sklearn model is not persisted as a standalone production artifact.

## Current Production Feature Group
- Home/away and rest: `is_home`, `rest_days`, `games_last_7_days`, `back_to_back`
- Recent form: recent 5/10 win rates and run differentials
- Season strength: prior season win rate and run-differential gaps
- Opponent comparison: opponent recent form and run differentials
- Context: venue win-rate gap and head-to-head prior gap
- Rating: `elo_diff`

The current server payload contains 21 selected columns. Current lineup and pitching snapshots are not production model features.

## Current Data Sources
- KBO team rank and head-to-head pages
- KBO schedule web service
- KBO player record pages
- KBO registered-player page
- KBO GameCenter game list, starter, and lineup analysis responses
- Manual confirmed starter override: `data/manual/confirmed_starters.csv`
- PostgreSQL is an optional copy of exported tables, not the only source for HTML generation

## Validation Completed
- Current-game exclusion in team priors and previous-game features
- Chronological train/test split
- Accuracy, Brier Score, Log Loss, and calibration tables
- Confidence-bucket performance
- Feature group and segment diagnostics
- Bootstrap and production replacement gates
- Current-season challenger degradation review
- Player-snapshot leakage warning and exclusion from production selection
- Pitching snapshot duplicate/freshness/leakage checks
- Run-model validation vs today prediction output separation
- Daily pipeline health checks

## Not Yet Validated or Ready
- Pitching snapshots as model features: only 10 accumulated game dates on the VM
- Historical as-of starter recent-three-game form
- Historical actual bullpen pitch counts and reliever availability
- Historical confirmed lineup features
- Persisted production-model loading without candidate retraining
- Incremental official schedule collection
- Dashboard rendering by changed team only
- End-to-end automated tests for the full official collector

## Known Problems
- `official_kbo_dashboard.py` is about 250 KB and owns collection, DB loading, reports, prediction, rendering, and publishing.
- Every daily/pregame run fetches schedules from the training start year and evaluates many model candidates.
- Daily operations and model development are not yet separated in code.
- Generated data, metrics, dashboard, and `docs/` files are Git-tracked; frequent pregame commits grow history rapidly.
- The live VM worktree contains generated modifications by design, so code deployment and output commits need explicit separation.
- `collector.py` and `weekly_update.py` are legacy mock-API/PostgreSQL flows, not the official production collector.
- `modeling/README.md` describes the older weekly/mock path and needs a later focused refresh.

## Main Outputs
- Source exports: `data/official/`
- Model features and diagnostics: `modeling/results/`
- Independent run model: `run_model/results/`
- Local dashboard: `dashboard/latest.html`
- Published static files: repository `docs/`
- Runtime logs: `kbo_analytics/logs/` on the VM, ignored by Git

## Resource Snapshot
- VM logical CPUs: 8
- VM memory: 62 GiB total, about 61 GiB available when checked
- Disk: 476 GiB total, 445 GiB available
- Repository: 1.3 GiB
- `data/official`: 2.3 MiB
- `modeling/results`: 28 MiB
- `run_model/results`: 7.4 MiB

## Production Artifact Path
- Explicit model development can create a candidate artifact containing the D-1 refit, preprocessing state, feature order, class order, metrics, and integrity manifest.
- Candidate validation, production promotion, and rollback are separate commands.
- Predict-only loads only `modeling/artifacts/production/current` after checksum, schema, compatibility, approval, and smoke checks.
- Missing or invalid production artifacts fail closed and never trigger automatic training.
- Actual generated model binaries remain local operating artifacts and are not committed to normal Git.

## Next Recommended Work
Run an explicit operating-VM model-development build, validate serialization identity against the live selected model, promote only after human review, and compare full versus predict-only outputs before changing cron.
