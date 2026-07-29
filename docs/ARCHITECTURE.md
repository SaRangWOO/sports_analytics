# KBO Analytics Architecture

## Scope and Source of Truth

This document covers `kbo_analytics/`. The separate top-level `kbo_run_model/` project is not part of this workflow.

The current implementation is a working monolith with extracted modeling helpers. The target is incremental separation, not a rewrite.

## Current Flow

```text
KBO official pages and web services
  -> official_kbo_dashboard.py fetch_* functions
  -> data/official/*.csv
  -> optional PostgreSQL table replacement
  -> modeling/model_training.py:evaluate_model
     -> modeling/feature_engineering.py
     -> modeling/game_level_features.py
     -> candidate models, diagnostics, gates
     -> latest completed-game refit
     -> modeling/results/win_predictor_model.json
  -> team/player report calculations
  -> pitching and lineup context snapshots
  -> run_model/run_prediction_model.py
     -> validation predictions
     -> today expected-runs predictions
  -> official_kbo_dashboard.py HTML rendering
  -> kbo_analytics/dashboard/*.html
  -> repository docs/*.html
  -> daily health status
```

## Current Stage Responsibilities

| Stage | Actual owner | Main input | Main output | Required each run | Cache potential | Failure impact |
|---|---|---|---|---|---|---|
| Standings/results collection | `official_kbo_dashboard.py` | KBO HTTP | DataFrames | Current data yes | Historical schedules high | HTTP/schema failure stops full run |
| Player/roster collection | `official_kbo_dashboard.py` | KBO HTTP | DataFrames | Reports yes | Same-day response medium | Failure stops full run |
| Source export | `export_sources` | collected frames | `data/official/*.csv` | Yes | Hash-identical writes can skip | Low if prior files are explicitly reused; current code does not |
| PostgreSQL copy | `load_official_tables_to_db` | collected frames | six DB tables | No | Replace only on change | Non-blocking by design |
| Feature build | `feature_engineering.py` | completed team-game CSV | `features.csv` | Prediction needs features | Input-hash cache high | Model/dashboard prediction unavailable |
| Game feature build | `game_level_features.py` | team features/context | game-level CSVs | Candidate evaluation yes | Input-hash cache high | Game candidates unavailable |
| Model development | `model_training.py:evaluate_model` | historical games/features | 40+ candidate reports and gate files | No for normal daily prediction | Very high | Currently blocks dashboard because coupled |
| Production refit/predict | `train_prediction_bundle` and `build_payload` | selected candidate, games through D-1 | today probabilities | Yes | Load persisted artifact | No prediction cards |
| Expected-runs model | `run_model/run_prediction_model.py` | completed and scheduled games | validation/today CSV, JSON | Only run-model tab | Input-hash cache high | Main win tab can still exist, but current call is coupled |
| Reports/rendering | `official_kbo_dashboard.py` | frames and model payload | team pages and `latest.html` | Yes when inputs change | Team-level render cache possible | No refreshed dashboard |
| Publish | `build_dashboard` | generated HTML | repository `docs/` | Yes when HTML changes | Content hash high | Local dashboard remains usable |
| Git publication | daily/pregame shell scripts | tracked outputs | commit and push to `main` | Only on changes | `git diff` already used | Push failure leaves local output generated |

## Coupling Findings

- `official_kbo_dashboard.py:main` always invokes collection, full evaluation, all reports, and all HTML rendering.
- `evaluate_model` creates production predictions and extensive research reports in the same call.
- The independent run model is invoked inside dashboard rendering.
- Official collection, validation, and processed-data boundaries are functions rather than explicit stages.
- The PostgreSQL boundary is appropriately non-blocking and should stay that way.

## File Responsibilities

| File | Verified responsibility | Status |
|---|---|---|
| `official_kbo_dashboard.py` | Official KBO HTTP collection, CSV/DB export, context snapshots, full model call, reports, dashboard, `docs/`, and health output | Production entrypoint; over-coupled |
| `collector.py` | Fetches legacy mock API `games` and `player-stats` endpoints and writes PostgreSQL date ranges | Exists; not the official production collector |
| `weekly_update.py` | Reads PostgreSQL mock/weekly tables, trains legacy candidates, and renders a weekly dashboard | Exists; legacy path |
| `modeling/feature_engineering.py` | Builds leakage-safe team-row priors, rolling/streak/context features, Elo, and targets | Active |
| `modeling/game_level_features.py` | Converts paired team rows to one game row and attaches pitching/player context | Active |
| `modeling/model_evaluation.py` | Probability normalization, confidence metrics, Brier/Log Loss, calibration, and candidate comparison helpers | Active |
| `modeling/model_training.py` | Dataset export, candidate training, diagnostics, ablation, bootstrap, gates, production refit, and today prediction | Active; primary model bottleneck |
| `modeling/train_win_predictor.py` | Matrix preparation plus the older from-scratch logistic implementation and standalone CLI | Active helper plus legacy standalone CLI |
| `docker-compose.yaml` | PostgreSQL, mock API, Metabase, and Python static dashboard server | Active operations |
| `README.md` | Repository overview and top-level run instructions | Updated by this task |
| `kbo_analytics/README.md` | Detailed KBO feature, model, schedule, and output documentation | Updated by this task |

## Target Responsibility Layout

Use existing files first. Do not create parallel modules until logic is extracted.

```text
kbo_analytics/
  config/                 # later: typed paths and runtime options
  collectors/             # later: official standings/schedule/player/GameCenter clients
  validation/             # later: source and snapshot quality checks
  features/
    team/                 # later extraction from feature_engineering.py
    player/               # later extraction from game_level_features.py/report code
    game/                 # later extraction from game_level_features.py
  modeling/
    dataset/              # later: dataset assembly only
    train/                # later: candidate fitting
    evaluate/             # reuse model_evaluation.py and gate writers
    ablation/             # later: experiment report writers
    predict/              # persisted production artifact and predict-only path
  reports/                # later: analysis text and tables
  dashboard/              # keep generated HTML directory; later add renderer module
  publish/                # later: dashboard -> docs copying by content hash
  pipeline/               # later: daily and model-development controllers
  scripts/                # current operational and safe task entrypoints
```

## Keep Now

- `modeling/feature_engineering.py` as the team feature source.
- `modeling/game_level_features.py` as the game/player context source.
- `modeling/model_evaluation.py` for probability metrics and selection helpers.
- `run_model/` as an independent expected-runs experiment.
- `dashboard/`, `docs/`, and current CSV/JSON paths for compatibility.
- PostgreSQL skip-on-failure behavior.

## Rename or Clarify Later

- Mark `collector.py` and `weekly_update.py` as legacy mock/PostgreSQL flows.
- Refresh `modeling/README.md` around the official production path.
- Name a persisted model artifact separately from `win_predictor_model.json`, which is currently primarily a report payload.

## Split Later

1. Extract official HTTP clients and schema validation.
2. Split `evaluate_model` into dataset, candidate evaluation, gate audit, production fit, and predict functions.
3. Extract team/player report builders from the HTML entrypoint.
4. Extract pure renderers and content-hash publishing.

## Do Not Split Yet

- Do not move tracked data/result paths while cron and dashboard links depend on them.
- Do not merge the independent expected-runs model with the production classifier.
- Do not promote player or pitching snapshots before as-of history and gates are ready.
- Do not rewrite the HTML templates while pipeline boundaries are being extracted.

## Target Daily Pipeline

```text
reference date
  -> collect current month and incremental completed games
  -> validate source schemas and game pairing
  -> update processed current data
  -> check production model artifact compatibility
  -> build only required current features
  -> predict today's games
  -> update reports whose input hashes changed
  -> render changed pages and latest.html
  -> copy changed HTML to docs
  -> health checks
  -> commit/push changed generated artifacts
```

The daily path must not run candidate comparison, ablation, bootstrap, or hyperparameter experiments.

## Target Model-Development Pipeline

```text
frozen as-of dataset
  -> schema and leakage audit
  -> feature dataset build
  -> chronological and rolling validation
  -> candidate model comparison
  -> calibration and confidence buckets
  -> feature-group ablation
  -> bootstrap stability
  -> current-season and segment checks
  -> production gate audit
  -> explicit human approval
  -> versioned production artifact
```

Minimum promotion gate:
- accuracy improvement greater than the configured production threshold
- Brier Score and Log Loss not worse beyond tolerances
- calibration not worse
- high-confidence segment lift
- recent-season stability
- bootstrap interval supports the improvement
- no leakage finding
- sufficient sample size

## Data-Layer Mapping

| Logical layer | Current path | Git state | Regeneration | Growth/risk |
|---|---|---|---|---|
| raw/source export | `data/official/*.csv` | tracked | mostly regenerated from KBO | moderate history growth |
| manual source | `data/manual/` | tracked | not automatically reproducible | retain permanently |
| processed | `data/official/model_training_games.csv` | tracked | regenerated | duplicated historical rows on disk |
| features | `modeling/results/features.csv`, game-level CSVs | tracked | regenerated | largest current files |
| models/metadata | `modeling/results/win_predictor_model.json` | tracked | regenerated | JSON is large; no standalone persisted estimator |
| predictions | pregame history and today CSV/JSON files | tracked | partly reproducible | retain audit history |
| metrics | `modeling/results/*report*` | tracked | experiment-generated | 90+ files |
| run-model outputs | `run_model/results/` | tracked | regenerated | validation and today roles are separate |
| local dashboard | `kbo_analytics/dashboard/` | tracked | regenerated | current local serving source |
| published dashboard | repository `docs/` | tracked | regenerated | GitHub/static publication |
| cache | none explicit | n/a | n/a | add only with input fingerprints |
| logs | `kbo_analytics/logs/` on VM | ignored | runtime only | rotate by retention policy |

## Minimal Commands

From `kbo_analytics/`:

```bash
# Inspect available safe wrappers
.venv/bin/python scripts/kbo_tasks.py --help

# Low-cost syntax and unit smoke checks
.venv/bin/python scripts/kbo_tasks.py smoke

# Full current behavior: expensive collection + model development + rendering
.venv/bin/python scripts/kbo_tasks.py full --reference-date YYYY-MM-DD --training-start-year 2016

# Feature file only
.venv/bin/python scripts/kbo_tasks.py features

# Independent expected-runs model and renderer
.venv/bin/python scripts/kbo_tasks.py run-model --reference-date YYYY-MM-DD
.venv/bin/python scripts/kbo_tasks.py run-dashboard
```

Official collection-only, validation-only, production-predict-only, main-dashboard-only, and publish-only commands do not yet exist. Adding wrappers cannot safely create those boundaries; the implementation must be extracted first.

## Incremental Transition

1. Persist and load the approved production estimator; add predict-only mode.
2. Extract collection plus schema validation with incremental schedule fetching.
3. Add input fingerprints for features and expected-runs outputs.
4. Separate report data from HTML rendering.
5. Add content-hash team-page and `docs/` publication.
6. Change cron to the predict-only daily controller.
7. Retain the current full command for explicit model-development runs.

## Follow-up Work Packages

Each package uses one Codex task and one branch. Do not run packages that edit the same files in parallel.

### 1. Separate Official Collection and Validation
- Goal: move KBO HTTP calls behind explicit collectors and validate response schemas/game pairs before export.
- Prerequisites: freeze current CSV schemas and add fixture samples.
- Expected files: `official_kbo_dashboard.py`, new `collectors/`, new `validation/`, focused tests.
- Cost: medium; no full historical fetch during development.
- Reasoning: medium.
- Done when: current-month fixture output matches existing columns and invalid pair/schema tests fail clearly.
- Branch: `codex/kbo-collector-validation`
- Next task prompt: `Read AGENTS.md and PROJECT_STATE.md. Extract official KBO collection and schema validation without changing CSV columns or dashboard/model behavior.`

### 2. Modularize Team and Player Reports
- Goal: extract pure team/player metrics and narrative builders from the HTML entrypoint.
- Prerequisites: snapshot representative team-page report data.
- Expected files: `official_kbo_dashboard.py`, new `reports/`, report tests.
- Cost: medium.
- Reasoning: medium.
- Done when: report payloads and generated team text match the baseline for all 10 teams.
- Branch: `codex/kbo-report-builders`
- Next task prompt: `Extract team and player report builders into pure modules. Preserve all generated values and HTML output.`

### 3. Add Leakage-safe Player Game Features
- Goal: use historical as-of player/lineup snapshots only after sufficient coverage.
- Prerequisites: dated snapshots, coverage report, and explicit as-of join policy.
- Expected files: `features/player/` or `game_level_features.py`, dataset tests, leakage audit reports.
- Cost: high.
- Reasoning: high.
- Done when: every join uses information available before first pitch and rolling validation shows the feature effect.
- Branch: `codex/kbo-player-game-features`
- Next task prompt: `Design and validate as-of player-derived game features. Do not use current snapshots for historical games and do not promote a model automatically.`

### 4. Formalize Feature-group Ablation
- Goal: make feature-set definitions, segment metrics, and output schema consistent.
- Prerequisites: stable baseline dataset and selected production artifact metadata.
- Expected files: `modeling/model_training.py`, later `modeling/ablation/`, result schema tests.
- Cost: high.
- Reasoning: high.
- Done when: one command compares baseline and named groups across fixed chronological windows.
- Branch: `codex/kbo-ablation-pipeline`
- Next task prompt: `Extract feature-group ablation from model_training.py into a reproducible explicit model-development command with unchanged metrics.`

### 5. Strengthen the Production Model Gate
- Goal: persist one approved artifact and separate candidate evidence from production promotion.
- Prerequisites: artifact serialization format and reproducibility metadata.
- Expected files: `modeling/train/`, `modeling/predict/`, gate audit, artifact loader tests.
- Cost: high.
- Reasoning: high.
- Done when: prediction can load the approved artifact, a failed challenger cannot replace it, and rollback is documented.
- Branch: `codex/kbo-production-model-artifact`
- Next task prompt: `Persist and load the approved KBO production model. Keep safe_to_replace_model false unless every existing gate passes.`

### 6. Add Incremental Daily Execution
- Goal: fetch only changed dates and skip unchanged feature/report stages.
- Prerequisites: collector separation and production artifact loading.
- Expected files: new `pipeline/daily.py`, cache manifest, cron scripts, health checks.
- Cost: medium to high.
- Reasoning: high.
- Done when: a no-change daily run performs no historical collection or candidate training and produces identical predictions.
- Branch: `codex/kbo-incremental-daily-pipeline`
- Next task prompt: `Build a predict-only incremental daily controller using input fingerprints. Preserve current output paths and PostgreSQL non-blocking behavior.`

### 7. Add Incremental Dashboard Rendering
- Goal: render only pages whose report payload changed and publish by content hash.
- Prerequisites: report builder extraction and stable renderer inputs.
- Expected files: renderer modules, publish module, manifest tests.
- Cost: medium.
- Reasoning: medium.
- Done when: changing one team payload rewrites only that team page, `latest.html`, and changed `docs/` files.
- Branch: `codex/kbo-incremental-dashboard-render`
- Next task prompt: `Add content-hash dashboard rendering without changing UI, prediction values, links, or current output paths.`

### 8. Improve Automation and Incident Observability
- Goal: record stage durations, cache hits, source failures, artifact versions, and publication status.
- Prerequisites: daily controller with named stages.
- Expected files: pipeline health module, cron scripts, log rotation configuration/documentation.
- Cost: medium.
- Reasoning: medium.
- Done when: one status JSON explains the failed stage, last good artifact, duration, and whether publication occurred.
- Branch: `codex/kbo-pipeline-observability`
- Next task prompt: `Add stage-level health and timing to the KBO daily controller. Do not expose credentials or make PostgreSQL failure block HTML generation.`
