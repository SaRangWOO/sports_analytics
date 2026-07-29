# KBO Analytics Resource Policy

Verified VM: 2026-07-29

## Current Capacity
- Logical CPUs: 8
- Memory: 62 GiB total, about 61 GiB available when checked
- Swap: 15 GiB, unused when checked
- Disk: 476 GiB total, 445 GiB available
- Repository: 1.3 GiB
- Model results: 28 MiB
- Independent run-model results: 7.4 MiB

Capacity is sufficient. The main inefficiency is repeated work and Git history growth, not immediate hardware pressure.

## Cache Policy
- A cache key must include input file hash, reference date, code/version identifier, and relevant options.
- Cache only reproducible intermediate data.
- Never cache manual starter overrides as derived data.
- Invalidate current schedule/player caches when the KBO source changes or the reference date changes.
- Invalidate feature caches when historical completed games or feature code changes.
- Invalidate model-development results when feature columns, split policy, candidate definitions, or gate policy changes.
- Do not introduce a cache until the stage has a clear input/output boundary.

## Collection Policy
- Current behavior re-fetches seasons from the training start year.
- Target behavior should reuse closed seasons and fetch only the current season/current month incrementally.
- Re-fetch a closed season only for explicit repair, source correction, or reproducibility checks.
- Compare schema and row keys before replacing prior exports.
- Keep manual source files permanently and back them up.

## Retraining Policy
- Daily and pregame updates should load one approved production artifact.
- Full candidate training runs only for explicit model-development work, scheduled periodic validation, or input/schema changes that invalidate the artifact.
- Ablation, bootstrap, rolling backtests, and challenger sweeps never run in the normal daily path.
- Production replacement remains manual and gate-controlled.

## Dashboard Policy
- Regenerate `latest.html` when current games, prediction output, lineup/starter status, report data, or template code changes.
- Regenerate a team page only when that team's source/report fingerprint changes.
- Copy to `docs/` only when content differs.
- Keep a full rebuild command for release validation.

## Parallelism
- Default concurrent model jobs: 2.
- Maximum recommended CPU-heavy jobs on this 8-core VM: 4 after measuring peak memory.
- Do not run the daily and pregame jobs concurrently; existing `flock` locks should remain.
- Keep sklearn `n_jobs=-1` out of multiple concurrent candidate processes.
- Parallelize only independent collectors or candidate groups with measured memory use.

## DataFrame Policy
- Avoid repeated `.copy()` of full historical frames when a column subset or view is sufficient.
- Load only required CSV columns for diagnostics.
- Write large CSVs once per changed input fingerprint.
- Prefer one canonical team-game frame and one canonical game-level frame per run.

## PostgreSQL Policy
- PostgreSQL is optional for dashboard generation.
- Preserve the current non-blocking warning when DB authentication or connectivity fails.
- Avoid writing the same unchanged DataFrame to both CSV and PostgreSQL repeatedly.
- Add table-level hashes or update timestamps before implementing incremental DB writes.

## Artifact Retention
- Keep: manual inputs, production gate audit, selected model metadata, prediction audit history, latest dashboard, and published `docs/`.
- Keep model-development reports needed to justify a production decision.
- Compress or release-archive large milestone artifacts instead of retaining every generated variant in Git.
- Candidate scratch outputs and temporary training CSVs belong in an ignored cache/temp path.
- Keep daily logs for 14 days and pregame logs for 7 days; compress only when incident review requires longer retention.
- Do not delete historical tracked artifacts as part of this process-document task.

## Git Tracking
- Currently tracked by design: `data/official`, `modeling/results`, `run_model/results`, `dashboard`, and `docs`.
- Currently ignored: `.venv`, `.env`, logs, Python caches, PostgreSQL data, and Metabase data.
- A later artifact-policy task should decide whether large reproducible features move to releases/object storage.
- Never commit credentials, tunnel tokens, deploy keys, `.env`, or database volumes.

## Temporary Files
- Use a dedicated ignored cache/temp directory with run ID and reference date.
- Remove successful-run temporary files after final outputs are atomically written.
- Retain failed-run temp data only when referenced by an incident log.
- Do not run bulk deletion against shared project paths.

## Codex Task Policy
- One Codex task equals one goal and one Git branch.
- New tasks first read `AGENTS.md` and `docs/PROJECT_STATE.md`.
- Use low reasoning for wording/doc-only changes.
- Use medium reasoning for bounded refactors, tests, and ordinary bugs.
- Use high reasoning for leakage, time-series validation, player/pitcher feature design, and promotion decisions.
- Do not run agents in parallel on the same file.
- Parallel work is appropriate only for non-conflicting source inspection and document review.
- Report paths and key errors instead of pasting long logs.

## Rebuild Cost Labels
- Low: syntax/import/unit checks, dashboard HTTP check, render from existing JSON/CSV.
- Medium: feature regeneration, independent expected-runs model.
- High: official full run from 2016, all candidate models, ablation, bootstrap, full static rebuild.
- Run high-cost work only when explicitly required and record reference date and elapsed time.
