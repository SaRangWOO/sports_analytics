from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from modeling.player_challenger import (
    CONTRIBUTION_COLUMNS,
    METRIC_COLUMNS,
    challenger_gate,
    contribution_report,
    evaluate_challengers,
    fit_full_challenger,
    write_csv_atomic,
    write_json_atomic,
)
from modeling.player_feature_pipeline import (
    PlayerFeatureConfig,
    batter_rates,
    build_snapshot_game_features,
    load_player_feature_config,
    pitcher_rates,
    player_feature_leakage_audit,
    player_feature_schema,
    select_as_of_player_rows,
    validate_player_ids,
)


PLAYER_OUTPUT_FILES = {
    "coverage": "player_feature_coverage_report.csv",
    "leakage": "player_feature_leakage_audit.json",
    "schema": "player_feature_schema.json",
    "summary": "player_feature_summary.csv",
    "pitching_comparison": "baseline_vs_pitching_challenger.csv",
    "full_comparison": "baseline_vs_full_player_challenger.csv",
    "gate": "player_challenger_gate_audit.json",
    "game_contribution": "player_contribution_by_game.csv",
    "team_contribution": "player_contribution_by_team.csv",
}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False)


def _checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_paths(project_root: Path) -> dict[str, Path]:
    data = project_root / "data"
    return {
        "games": data / "official" / "game_results.csv",
        "pitching": data / "official" / "pitching_daily_snapshot.csv",
        "lineup": data / "official" / "lineup_daily_snapshot.csv",
        "player_stats": data / "weekly" / "player_game_stats.csv",
        "roster": project_root / "mock_api" / "player_roster_mapping.csv",
    }


def load_player_sources(project_root: Path) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    paths = source_paths(project_root)
    frames = {name: _read_csv(path) for name, path in paths.items()}
    checksums = {name: _checksum(path) for name, path in paths.items()}
    return frames, checksums


def _coverage_frame(coverage: dict[str, Any], config: PlayerFeatureConfig) -> pd.DataFrame:
    thresholds = config.values["coverage_thresholds"]
    threshold_by_metric = {
        "starter_coverage": float(thresholds["starter"]),
        "bullpen_coverage": float(thresholds["bullpen"]),
        "lineup_coverage": float(thresholds["lineup"]),
        "player_id_mapping_coverage": float(thresholds["player_id_mapping"]),
    }
    rows = []
    for metric, value in coverage.items():
        threshold = threshold_by_metric.get(metric)
        status = "pass" if threshold is None or float(value) >= threshold else "blocked"
        rows.append(
            {
                "metric": metric,
                "value": value,
                "required": threshold,
                "status": status,
            }
        )
    return pd.DataFrame(rows)


def _player_summary(
    player_rows: pd.DataFrame,
    roster_mapping: pd.DataFrame,
    reference_datetime: datetime,
    config: PlayerFeatureConfig,
) -> pd.DataFrame:
    selected = select_as_of_player_rows(player_rows, reference_datetime)
    if "player_type" not in selected.columns:
        innings = pd.to_numeric(selected["innings_pitched"], errors="coerce").fillna(0.0)
        selected["player_type"] = innings.gt(0).map({True: "pitcher", False: "batter"})
    id_status = validate_player_ids(selected, roster_mapping)
    prior = config.values["shrinkage_weights"]
    rows: list[dict[str, Any]] = []
    for (team, player_id, player_name, player_type), group in selected.groupby(
        ["team", "player_id", "player_name", "player_type"], dropna=False
    ):
        rates = (
            batter_rates(group, float(prior["batter_prior_plate_appearances"]))
            if player_type == "batter"
            else pitcher_rates(group, float(prior["pitcher_prior_innings"]))
        )
        rows.append(
            {
                "reference_datetime": reference_datetime.isoformat(),
                "team": team,
                "player_id": player_id,
                "player_name": player_name,
                "player_type": player_type,
                "latest_eligible_game_date": pd.to_datetime(group["date"]).max().date(),
                "eligible_game_rows": len(group),
                **rates,
                **id_status,
            }
        )
    return pd.DataFrame(rows)


def build_player_feature_outputs(
    project_root: Path,
    output_root: Path,
    reference_datetime: datetime,
    player_config_path: Path,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    config = load_player_feature_config(player_config_path)
    frames, checksums = load_player_sources(project_root)
    selected_player_rows = select_as_of_player_rows(
        frames["player_stats"], reference_datetime
    )
    id_status = validate_player_ids(selected_player_rows, frames["roster"])
    features, coverage = build_snapshot_game_features(
        frames["games"],
        frames["pitching"],
        frames["lineup"],
        frames["roster"],
        reference_datetime,
        config,
        player_stats=selected_player_rows,
    )
    coverage.update(id_status)
    reference_timestamp = pd.Timestamp(reference_datetime).tz_localize(None)
    eligible_pitching = frames["pitching"][
        pd.to_datetime(frames["pitching"]["snapshot_time"], errors="coerce")
        <= reference_timestamp
    ].copy()
    eligible_lineup = frames["lineup"][
        pd.to_datetime(frames["lineup"]["snapshot_time"], errors="coerce")
        <= reference_timestamp
    ].copy()
    leakage = player_feature_leakage_audit(
        selected_player_rows,
        eligible_pitching,
        eligible_lineup,
        reference_datetime,
    )
    schema = player_feature_schema()
    schema.update(
        {
            "generated_at": reference_datetime.isoformat(),
            "source_checksums": checksums,
        }
    )
    summary = _player_summary(
        selected_player_rows,
        frames["roster"],
        reference_datetime,
        config,
    )
    minimum = config.values["minimum_player_sample"]
    if summary.empty:
        coverage["minimum_sample_failure_count"] = 0
    else:
        plate_appearances = pd.to_numeric(
            summary.get("plate_appearances", pd.Series(0.0, index=summary.index)),
            errors="coerce",
        ).fillna(0.0)
        innings = pd.to_numeric(
            summary.get("innings", pd.Series(0.0, index=summary.index)),
            errors="coerce",
        ).fillna(0.0)
        batter_failure = summary["player_type"].eq("batter") & plate_appearances.lt(
            float(minimum["batter_plate_appearances"])
        )
        pitcher_failure = summary["player_type"].eq("pitcher") & innings.lt(
            float(minimum["pitcher_innings"])
        )
        coverage["minimum_sample_failure_count"] = int((batter_failure | pitcher_failure).sum())
    if not dry_run:
        write_csv_atomic(output_root / PLAYER_OUTPUT_FILES["coverage"], _coverage_frame(coverage, config))
        write_json_atomic(output_root / PLAYER_OUTPUT_FILES["leakage"], leakage)
        write_json_atomic(output_root / PLAYER_OUTPUT_FILES["schema"], schema)
        write_csv_atomic(output_root / PLAYER_OUTPUT_FILES["summary"], summary)
    return {
        "status": "blocked" if leakage["status"] != "pass" else "succeeded",
        "dry_run": dry_run,
        "reference_datetime": reference_datetime.isoformat(),
        "feature_rows": len(features),
        "player_summary_rows": len(summary),
        "coverage": coverage,
        "leakage_status": leakage["status"],
        "output_root": str(output_root),
        "features": features,
        "config": config,
    }


def player_feature_quality(
    project_root: Path,
    output_root: Path,
    reference_datetime: datetime,
    player_config_path: Path,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    result = build_player_feature_outputs(
        project_root,
        output_root,
        reference_datetime,
        player_config_path,
        dry_run=dry_run,
    )
    return {key: value for key, value in result.items() if key not in {"features", "config"}}


def _comparison_rows(metrics: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame(columns=METRIC_COLUMNS)
    return metrics[metrics["feature_set"].isin(["production_baseline_proxy", feature_set])].copy()


def _team_contributions(game_report: pd.DataFrame) -> pd.DataFrame:
    if game_report.empty:
        return pd.DataFrame(
            columns=[
                "team",
                "games",
                "starter_contribution_pct_point",
                "bullpen_contribution_pct_point",
                "lineup_contribution_pct_point",
            ]
        )
    rows = []
    for side, sign in [("home_team", 1.0), ("away_team", -1.0)]:
        frame = game_report.copy()
        frame["team"] = frame[side]
        for column in [
            "starter_contribution_pct_point",
            "bullpen_contribution_pct_point",
            "lineup_contribution_pct_point",
        ]:
            frame[column] = frame[column] * sign
        rows.append(frame)
    combined = pd.concat(rows, ignore_index=True)
    return combined.groupby("team", as_index=False).agg(
        games=("official_game_id", "count"),
        starter_contribution_pct_point=("starter_contribution_pct_point", "mean"),
        bullpen_contribution_pct_point=("bullpen_contribution_pct_point", "mean"),
        lineup_contribution_pct_point=("lineup_contribution_pct_point", "mean"),
    )


def evaluate_player_challenger(
    project_root: Path,
    output_root: Path,
    reference_datetime: datetime,
    player_config_path: Path,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    built = build_player_feature_outputs(
        project_root,
        output_root,
        reference_datetime,
        player_config_path,
        dry_run=dry_run,
    )
    features: pd.DataFrame = built["features"]
    config: PlayerFeatureConfig = built["config"]
    leakage = (
        {"status": built["leakage_status"]}
        if dry_run
        else json.loads(
            (output_root / PLAYER_OUTPUT_FILES["leakage"]).read_text(encoding="utf-8")
        )
    )
    metrics, predictions = evaluate_challengers(features)
    gate = challenger_gate(
        metrics,
        built["coverage"],
        leakage,
        config,
        predictions=predictions,
        production_parity_verified=False,
    )
    game_contribution = pd.DataFrame(columns=CONTRIBUTION_COLUMNS)
    if (
        len(features) >= config.minimum_comparable_games
        and leakage.get("status") == "pass"
        and features["target_home_win"].nunique() == 2
    ):
        game_contribution = contribution_report(fit_full_challenger(features), features)
    team_contribution = _team_contributions(game_contribution)
    if not dry_run:
        write_csv_atomic(
            output_root / PLAYER_OUTPUT_FILES["pitching_comparison"],
            _comparison_rows(metrics, "baseline_plus_pitching"),
        )
        write_csv_atomic(
            output_root / PLAYER_OUTPUT_FILES["full_comparison"],
            _comparison_rows(metrics, "baseline_plus_full_player"),
        )
        write_json_atomic(output_root / PLAYER_OUTPUT_FILES["gate"], gate)
        write_csv_atomic(output_root / PLAYER_OUTPUT_FILES["game_contribution"], game_contribution)
        write_csv_atomic(output_root / PLAYER_OUTPUT_FILES["team_contribution"], team_contribution)
    return {
        "status": "eligible" if gate["passed"] else "blocked",
        "dry_run": dry_run,
        "feature_rows": len(features),
        "metric_rows": len(metrics),
        "prediction_rows": len(predictions),
        "contribution_rows": len(game_contribution),
        "gate": gate,
        "production_model_changed": False,
        "auto_promotion_enabled": False,
        "output_root": str(output_root),
    }


def build_player_contribution_report(
    project_root: Path,
    output_root: Path,
    reference_datetime: datetime,
    player_config_path: Path,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    result = evaluate_player_challenger(
        project_root,
        output_root,
        reference_datetime,
        player_config_path,
        dry_run=dry_run,
    )
    return {
        "status": result["status"],
        "dry_run": dry_run,
        "contribution_rows": result["contribution_rows"],
        "production_model_changed": False,
        "output_root": str(output_root),
    }
