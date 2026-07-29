# KBO Analytics Working Rules

## Scope
- This repository is a KBO sports analytics project. It is unrelated to automated trading.
- Primary project: `kbo_analytics/`.
- Server project: `/home/wsr/1.project/1.sports_analytics/kbo_analytics`.
- Do not modify the separate `kbo_run_model/` project unless the task names it.

## Read First
- Start with this file and `docs/PROJECT_STATE.md`.
- Read only the entrypoint and modules required for the current task.
- Do not repeatedly scan the whole repository or load all generated CSV/HTML files.
- Treat code and current server artifacts as authoritative over old conversation text.

## Entrypoints
- Full official pipeline: `kbo_analytics/official_kbo_dashboard.py`.
- Feature generation: `kbo_analytics/modeling/feature_engineering.py`.
- Model evaluation: `kbo_analytics/modeling/model_training.py`.
- Independent run model: `kbo_analytics/run_model/run_prediction_model.py`.
- Run-model dashboard: `kbo_analytics/run_model/run_model_dashboard.py`.
- Daily/pregame operations: `kbo_analytics/scripts/daily_kbo_update.sh` and `pregame_kbo_update.sh`.
- Safe task wrapper: `kbo_analytics/scripts/kbo_tasks.py`.

## Pipeline Boundaries
- Daily operations collect current data, validate outputs, predict, render, and publish.
- Model development builds candidate features, runs chronological validation, calibration, ablation, bootstrap checks, and replacement gates.
- The current full entrypoint still performs both flows. Do not claim they are separated until a predict-only production path exists.
- Run full candidate training only when the task explicitly requests model development or a full integration check.

## Data Rules
- Never invent unavailable KBO records.
- Label proxy metrics as proxies; do not present them as actual tactical instructions or success rates.
- All rolling, streak, matchup, and aggregate features must exclude the current game using prior rows or `shift(1)`.
- Do not attach current player or pitching snapshots to historical games.
- Keep `pitching_daily_snapshot.csv` out of model features until its quality gate allows experimentation.
- Keep validation outputs and today-prediction outputs separate.

## Change Rules
- One goal, one Codex task, one Git branch.
- Do not mix dashboard wording/design changes with model or feature changes.
- Do not mix large collector, model, and dashboard refactors in one task.
- Preserve PostgreSQL-skip behavior: DB failure must not block CSV/JSON/HTML generation.
- Do not replace the production model unless the recorded production gate passes.
- Do not delete or rewrite tracked generated artifacts unless the task requires regeneration.

## Minimum Validation
- Syntax: `.venv/bin/python -m py_compile official_kbo_dashboard.py modeling/feature_engineering.py modeling/model_training.py`.
- Unit test: run `modeling/test_feature_engineering.py` from `kbo_analytics/modeling`.
- Wrapper smoke: `.venv/bin/python scripts/kbo_tasks.py --dry-run smoke`.
- Use a full dashboard run only when required; it re-collects multiple seasons and trains many candidates.
- Report commands, pass/fail, skipped expensive checks, and the reason for each skip.

## Git
- Inspect `git status` before editing and do not revert unrelated changes.
- Commit only files in the task scope.
- Update relevant documentation with behavior changes.
- Never use force push, `reset --hard`, `clean -fd`, or rebase for routine work.
- Push a validated feature branch when repository access is configured; never push secrets or `.env`.
