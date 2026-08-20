# KBO Model Artifacts

This directory holds trusted local model artifacts used by the predict-only path.

```text
candidate/<artifact_id>/
production/current/
previous/<artifact_id>/
```

Each artifact contains `model.joblib`, `metadata.json`, `feature_schema.json`,
`metrics.json`, and `manifest.json`. Candidate creation never promotes a model.
Promotion and rollback require explicit CLI commands.

Actual model binaries and generated metadata are intentionally ignored by Git.
The operating VM should keep them on local persistent storage with a separate
backup. GitHub Release, CI artifacts, Git LFS, or another artifact store can be
evaluated later; this change does not introduce one.

Only artifacts created inside this trusted root and passing manifest checksum,
schema, compatibility, approval, and smoke-prediction checks may be loaded.
External joblib files must not be copied here and loaded without an independent
trust and integrity process.
