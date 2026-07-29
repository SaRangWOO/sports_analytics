# KBO Production Artifact Parity Validation

## Result

- Status: PASS
- Reference date: 2026-07-29
- Latest completed game date used: 2026-07-28
- Scheduled games: 5
- Selected model: 핵심 수치 RandomForest 보수 시간가중 모델
- Prediction unit: team
- Candidate artifact: `2026-07-28-ea851a2e6414f8ba`

## Parity

- Stable game keys: True
- Maximum raw probability delta: 1.1102230246251565e-16
- Maximum normalized probability delta: 1.1102230246251565e-16
- Seven output fields identical: True
- Serialization identity passed: True
- Predict-only validation runs: 3
- Predict-only training calls detected: []

## Artifact

- Manifest/checksum/schema validation: True
- Isolated promotion validation: True
- Feature count: 21
- Artifact size: 1859930 bytes

## Timing

- Full model-development path: 732.481s
- Candidate creation: not separately captured
- Actual model isolated re-serialization roundtrip: 1.116s
- Candidate validation: 0.367s
- Isolated promotion: 0.726s
- Predict-only path: 11.397s
- Production artifact load during predict-only: 1.278s
- Peak process RSS: 198028 KiB

## Boundaries

- The operating repository was used only as the source of copied CSV data.
- No operating production artifact was promoted.
- No dashboard, docs, cron, systemd, database, or deployment output was changed.
- `pitching_daily_snapshot.csv` was not part of the artifact feature schema.
- Generated model artifacts and large validation outputs remain outside Git.
- The full path ran once. A validation-only schema lookup failed after candidate validation, so only low-cost follow-up checks were resumed.

## Failed Checks

- None
