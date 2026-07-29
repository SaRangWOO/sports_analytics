# KT Wiz Challenger PR Summary

## 1. Overview
The KT Wiz challenger is an offline-monitoring experiment for selected KT recommendation segments.

## 2. What changed
- Hardened selected-pick precision target reporting.
- Added Wilson confidence intervals, segment-search warnings, and forward-validation requirements.
- Added a per-game precision target audit file.
- Clarified leakage-safe pregame feature roadmap and cutoff policy.

## 3. Current KT challenger performance
Current best segment: `rolling_agreement_probability_ge_52`.
Current best segment precision is 0.571 across 77 games.

## 4. 85% precision target status
The 85% target remains active but is not achieved.
The gap to target is 0.279.

## 5. Why target is not met
The current model lacks confirmed starter, bullpen usage, lineup, catcher status, and hitter availability signals. Historical segment search also requires forward validation.

## 6. Production safety
KT challenger remains offline-monitoring only. The production KBO-wide model is unchanged.

## 7. Pregame feature roadmap
The next step is leakage-safe pregame data collection and forward validation for starters, bullpen, lineup, catcher status, hitter availability, stadium, travel, and weather context.

## 8. Statistical guardrails
A target segment must have precision >= 0.85, at least 50 games, a medium or large sample bucket, and a Wilson lower bound >= 0.85 before it can be considered strongly met.

## 9. Dashboard validation
Dashboard file: `kbo_analytics/dashboard/kt_wiz_challenger.html`.
Local URL: `http://127.0.0.1:8501/kt_wiz_challenger.html`.
Server URL: `http://192.168.11.23:8501/kt_wiz_challenger.html`.

## 10. Limitations
This is not a production replacement. It does not use betting odds, handicap lines, over/under lines, post-game data, or future data as model features.

## 11. Next steps
Collect timestamped pregame data before the prediction cutoff and forward-validate selected KT segments.

## 12. Final policy
Offline monitoring only. No production replacement. No betting recommendation.
