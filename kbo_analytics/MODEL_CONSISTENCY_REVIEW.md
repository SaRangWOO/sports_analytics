# Model consistency review (2026-10-13)

## Operational contract

Morning and pregame jobs require the same approved production artifact. A missing
artifact fails the job instead of falling back to model selection or training.
Morning collection still refreshes completed-game history and pitcher logs.
Evaluation metrics are loaded only from that artifact, with model and metric IDs.
Legacy artifacts without validation-period metadata report that omission, rather
than borrowing dates or scores from a subsequent evaluation. Artifact age over
30 days is a monitoring warning, not an automatic replacement criterion.

The daily, pregame and context-refresh shell entrypoints use the same flock.
Both publication scripts include independent run-model results in their output
paths. The active server address is 192.168.11.70.

## Offline experiments

Pitcher candidates use a chronological development/selection/final-holdout
split. Feature availability and imputation are fitted without holdout data.
Confidence intervals resample whole game dates. A refitted feature-ablation
baseline is explicitly not the deployed production model. The final holdout
must not become a repeatedly tuned selection set in subsequent experiments.

The 2026-09-04 through 2026-10-11 holdout contains 124 games. The selected
RandomForest starter candidate scored 0.645 accuracy versus 0.532 for the
refitted baseline (Brier 0.233 versus 0.247; log loss 0.659 versus 0.688).
The paired date-block accuracy delta interval was [0.0165, 0.2138]. This is an
exploratory signal, not verified production improvement: historical boxscore
starter identities still lack prediction-time provenance, and the approved
production model has not been replayed on the identical games.

The workload WHIP field remains a hits-plus-BB/HBP proxy, not official WHIP.
Rest days measure calendar days since the previous start, not all appearances.
Same-day workload is excluded until within-day completion timestamps exist.
Neither pitching snapshots nor these candidate results promote a model.

Run-model `--temporal-oof` fits win conversion on prior-date out-of-fold run
predictions. It requires a separate output directory and is not enabled by the
operational dashboard. The legacy path remains unchanged pending full comparison.

## Remaining work

- Verify starter identity availability at each prediction time and match the
  deployed baseline on identical eligible games before any promotion decision.
- Audit existing historical core features, including global-mean imputation,
  independently; this patch does not certify all historical features leak-free.
- Evaluate OOF calibration prospectively and version the independent run model
  as an approved artifact before changing the live hybrid probabilities.
- Retain separate candidate-training jobs; daily serving no longer selects a
  new winner or rewrites production gates.

## Verification

Modeling unittest discovery: 50 tests passed, including artifact round-trip,
probability normalization, metadata isolation, workload boundaries, snapshot
storage, hybrid policy, holdout separation and OOF date ordering.
The live artifact and operational model gates remain unchanged. Experiment
outputs are isolated outside the tracked production result directories.
