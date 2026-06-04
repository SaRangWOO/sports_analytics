from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from collectors.load_games import load_completed_team_games
from collectors.load_starter_data import load_starter_inputs
from collectors.internal_pitcher_mapping import write_internal_pitcher_mapping_outputs
from collectors.pitcher_data_validation import validate_pitcher_data_pipeline
from collectors.schedule import load_schedule, select_target_games, validate_schedule_selection
from collectors.schedule_update import build_schedule_status, write_schedule_update_report
from collectors.starter_pitcher_collector import collect_and_maybe_apply, load_collection_summary
from collectors.schema import inspect_starter_schema
from collectors.search_internal_data import search_internal_pitcher_data
from dashboard.report import write_html_report
from evaluation.error_analysis import build_error_analysis
from evaluation.improvement_experiment import run_performance_improvement_experiment
from evaluation.metrics import evaluate_win_conversion, season_metrics, select_model, team_bias_metrics, to_game_predictions
from features.prediction_features import build_prediction_feature_matrix
from features.starter_features import add_starter_features
from features.team_features import build_feature_matrix
from models.recommendations import DEFAULT_HANDICAP_LINE, DEFAULT_OVER_UNDER_LINE, confidence_level, handicap_pick, moneyline_pick, over_under_pick
from models.train import chronological_split, train_run_models


PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = PROJECT_DIR.parent
DEFAULT_INPUT = REPO_DIR / "kbo_analytics" / "data" / "official" / "model_training_games.csv"
DEFAULT_SCHEDULE = REPO_DIR / "kbo_analytics" / "data" / "official" / "prediction_games.csv"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "results"
DEFAULT_STARTERS = PROJECT_DIR / "data" / "starter_pitchers.csv"
DEFAULT_PITCHER_LOGS = PROJECT_DIR / "data" / "pitcher_game_logs.csv"
DASHBOARD_PATH = "kbo_run_model/results/report.html"
KST = ZoneInfo("Asia/Seoul")


def _merge_scores(run_scores: list[dict], win_scores: list[dict]) -> list[dict]:
    combined = {row["model"]: dict(row) for row in run_scores}
    for row in win_scores:
        combined[row["model"]].update(row)
    return list(combined.values())


def _format_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    output = predictions.copy()
    output["date"] = pd.to_datetime(output["date"]).dt.strftime("%Y-%m-%d")
    output["expected_score"] = output["home_expected_runs"].round(1).astype(str) + " - " + output["away_expected_runs"].round(1).astype(str)
    for column in [
        "home_expected_runs",
        "away_expected_runs",
        "expected_run_diff",
        "expected_total_runs",
        "home_win_probability",
        "confidence",
    ]:
        output[column] = output[column].round(4)
    return output[
        [
            "date",
            "game_key",
            "home_team",
            "away_team",
            "expected_score",
            "home_expected_runs",
            "away_expected_runs",
            "expected_run_diff",
            "expected_total_runs",
            "home_win_probability",
            "confidence",
            "predicted_winner",
            "actual_winner",
            "home_actual_runs",
            "away_actual_runs",
        ]
    ]


def _train_win_converter(model: object, train_df: pd.DataFrame, feature_columns: list[str]) -> LogisticRegression:
    train_scored = train_df.copy()
    prediction_column = "predicted_runs"
    train_scored[prediction_column] = np.clip(model.predict(train_scored[feature_columns]), 0, None)
    train_games = to_game_predictions(train_scored, prediction_column)
    converter = LogisticRegression()
    converter.fit(train_games[["expected_run_diff"]], train_games["home_actual_win"])
    return converter


def _build_match_predictions(
    model: object,
    converter: LogisticRegression,
    prediction_features: pd.DataFrame,
    feature_columns: list[str],
    target_context: dict,
) -> pd.DataFrame:
    columns = [
        "target_date",
        "report_mode",
        "date",
        "game_id",
        "away_team",
        "home_team",
        "predicted_away_runs",
        "predicted_home_runs",
        "predicted_score",
        "predicted_winner",
        "away_win_probability",
        "home_win_probability",
        "expected_run_diff",
        "total_expected_runs",
        "moneyline_pick",
        "handicap_line",
        "handicap_pick",
        "over_under_line",
        "over_under_pick",
        "confidence_level",
    ]
    if prediction_features.empty:
        return pd.DataFrame(columns=columns)

    scored = prediction_features.copy()
    scored["predicted_runs"] = np.clip(model.predict(scored[feature_columns]), 0, None)
    games = to_game_predictions(scored, "predicted_runs")
    games["home_win_probability"] = converter.predict_proba(games[["expected_run_diff"]])[:, 1]
    games["away_win_probability"] = 1 - games["home_win_probability"]
    games["predicted_home_win"] = games["home_win_probability"].ge(0.5).astype(int)
    games["predicted_winner"] = np.where(games["predicted_home_win"].eq(1), games["home_team"], games["away_team"])
    games["moneyline_pick"] = [
        moneyline_pick(row.home_team, row.away_team, row.home_win_probability) for row in games.itertuples(index=False)
    ]
    games["handicap_line"] = DEFAULT_HANDICAP_LINE
    games["handicap_pick"] = [
        handicap_pick(row.home_team, row.away_team, row.expected_run_diff, DEFAULT_HANDICAP_LINE) for row in games.itertuples(index=False)
    ]
    games["over_under_line"] = DEFAULT_OVER_UNDER_LINE
    games["over_under_pick"] = games["expected_total_runs"].map(lambda value: over_under_pick(float(value), DEFAULT_OVER_UNDER_LINE))
    games["confidence_level"] = games["expected_run_diff"].map(lambda value: confidence_level(float(value)))
    games["predicted_score"] = games.apply(
        lambda row: f"{row['away_team']} {row['away_expected_runs']:.1f} - {row['home_team']} {row['home_expected_runs']:.1f}",
        axis=1,
    )
    output = pd.DataFrame(
        {
            "target_date": target_context["target_date"],
            "report_mode": target_context["report_mode"],
            "date": pd.to_datetime(games["date"]).dt.strftime("%Y-%m-%d"),
            "game_id": games["game_key"],
            "away_team": games["away_team"],
            "home_team": games["home_team"],
            "predicted_away_runs": games["away_expected_runs"].round(4),
            "predicted_home_runs": games["home_expected_runs"].round(4),
            "predicted_score": games["predicted_score"],
            "predicted_winner": games["predicted_winner"],
            "away_win_probability": games["away_win_probability"].round(4),
            "home_win_probability": games["home_win_probability"].round(4),
            "expected_run_diff": games["expected_run_diff"].round(4),
            "total_expected_runs": games["expected_total_runs"].round(4),
            "moneyline_pick": games["moneyline_pick"],
            "handicap_line": games["handicap_line"],
            "handicap_pick": games["handicap_pick"],
            "over_under_line": games["over_under_line"],
            "over_under_pick": games["over_under_pick"],
            "confidence_level": games["confidence_level"],
        }
    )
    return output[columns]


def run_pipeline(
    input_path: Path,
    schedule_path: Path,
    output_dir: Path,
    train_ratio: float,
    target_date: date | None,
    allow_past_fallback: bool,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    team_games = load_completed_team_games(input_path)
    schedule = load_schedule(schedule_path)
    current_date_kst = datetime.now(KST).date()
    schedule_update_status = build_schedule_status(schedule_path, current_date_kst)
    starter_collection_report_path = output_dir / "starter_pitcher_collection_report.csv"
    if not starter_collection_report_path.exists():
        collect_and_maybe_apply(schedule, DEFAULT_STARTERS, output_dir, apply=False, probe_external=False)
    starter_pitcher_collection = load_collection_summary(starter_collection_report_path)
    target_games, target_context = select_target_games(schedule, target_date, current_date_kst, allow_past_fallback)
    feature_df, feature_columns = build_feature_matrix(team_games)
    starters, pitcher_logs, starter_data_status = load_starter_inputs(DEFAULT_STARTERS, DEFAULT_PITCHER_LOGS)
    feature_df, feature_columns, starter_feature_status = add_starter_features(feature_df, feature_columns, starters, pitcher_logs)
    train_df, validation_df, cutoff = chronological_split(feature_df, train_ratio)

    trained_models, run_scores = train_run_models(train_df, validation_df, feature_columns)
    win_scores, prediction_map, bucket_map, abs_bucket_map = evaluate_win_conversion(trained_models, train_df, validation_df, feature_columns)
    selected_model = select_model(run_scores, win_scores)
    candidate_scores = _merge_scores(run_scores, win_scores)
    selected_name = str(selected_model["model"])
    selected_run_model = trained_models[selected_name]
    selected_scored_games = prediction_map[selected_name]
    selected_predictions = _format_predictions(prediction_map[selected_name])
    error_analysis = build_error_analysis(selected_scored_games, validation_df)
    improvement_experiment = run_performance_improvement_experiment(
        feature_df,
        feature_columns,
        selected_run_model,
        selected_model,
        error_analysis,
        train_ratio,
    )
    selected_season_metrics = season_metrics(selected_scored_games)
    selected_team_metrics, over_predicted_teams, under_predicted_teams = team_bias_metrics(selected_scored_games)
    starter_schema = inspect_starter_schema(input_path)
    internal_data_search = search_internal_pitcher_data(REPO_DIR)
    pitcher_data_validation = validate_pitcher_data_pipeline(DEFAULT_STARTERS, DEFAULT_PITCHER_LOGS, schedule)
    internal_pitcher_mapping = write_internal_pitcher_mapping_outputs(REPO_DIR, output_dir, apply=False)
    prediction_feature_df = build_prediction_feature_matrix(team_games, target_games) if not target_games.empty else pd.DataFrame()
    win_converter = _train_win_converter(selected_run_model, train_df, feature_columns)
    match_predictions = _build_match_predictions(selected_run_model, win_converter, prediction_feature_df, feature_columns, target_context)
    schedule_check = validate_schedule_selection(target_games, match_predictions, target_context)

    feature_df.to_csv(output_dir / "team_game_features.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(candidate_scores).to_csv(output_dir / "model_scores.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(bucket_map[selected_name]).to_csv(output_dir / "run_diff_bucket_accuracy.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(abs_bucket_map[selected_name]).to_csv(output_dir / "abs_run_diff_bucket_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(selected_season_metrics).to_csv(output_dir / "season_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(selected_team_metrics).to_csv(output_dir / "team_bias_metrics.csv", index=False, encoding="utf-8-sig")
    selected_predictions.to_csv(output_dir / "expected_runs_predictions.csv", index=False, encoding="utf-8-sig")
    match_predictions.to_csv(output_dir / "match_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([schedule_check]).to_csv(output_dir / "schedule_selection_check.csv", index=False, encoding="utf-8-sig")
    write_schedule_update_report(schedule_update_status, output_dir / "schedule_update_report.csv")
    error_analysis["game_errors"].to_csv(output_dir / "error_analysis_games.csv", index=False, encoding="utf-8-sig")
    error_analysis["win_probability_buckets"].to_csv(output_dir / "win_probability_bucket_metrics.csv", index=False, encoding="utf-8-sig")
    error_analysis["total_runs"].to_csv(output_dir / "total_runs_error_metrics.csv", index=False, encoding="utf-8-sig")
    error_analysis["handicap"].to_csv(output_dir / "handicap_metrics.csv", index=False, encoding="utf-8-sig")
    error_analysis["team"].to_csv(output_dir / "team_error_metrics.csv", index=False, encoding="utf-8-sig")
    error_analysis["ballpark"].to_csv(output_dir / "ballpark_error_metrics.csv", index=False, encoding="utf-8-sig")
    error_analysis["monthly"].to_csv(output_dir / "monthly_error_metrics.csv", index=False, encoding="utf-8-sig")
    improvement_experiment["comparison"].to_csv(output_dir / "performance_improvement_summary.csv", index=False, encoding="utf-8-sig")
    improvement_experiment["park_metrics"].to_csv(output_dir / "park_factor_metrics.csv", index=False, encoding="utf-8-sig")
    improvement_experiment["bias_metrics"].to_csv(output_dir / "team_bias_feature_metrics.csv", index=False, encoding="utf-8-sig")
    improvement_experiment["model_scores"].to_csv(output_dir / "improvement_model_scores.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {
                "dataset": "starter_pitchers",
                **pitcher_data_validation["starter_pitchers_validation"],
            },
            {
                "dataset": "pitcher_game_logs",
                **pitcher_data_validation["pitcher_game_logs_validation"],
            },
        ]
    ).to_csv(output_dir / "pitcher_data_validation.csv", index=False, encoding="utf-8-sig")

    summary = {
        "generated_at": datetime.now(KST).isoformat(timespec="seconds"),
        "input_file": str(input_path),
        "schedule_file": str(schedule_path),
        "target_context": {
            **target_context,
            "game_count": int(len(match_predictions)),
            "requested_target_date": target_date.isoformat() if target_date else "",
        },
        "current_date_kst": target_context["current_date_kst"],
        "schedule_latest_date": target_context["schedule_latest_date"],
        "selected_target_date": target_context["selected_target_date"],
        "schedule_is_stale": target_context["schedule_is_stale"],
        "stale_schedule_days": target_context["stale_schedule_days"],
        "has_future_schedule": target_context["has_future_schedule"],
        "allow_past_fallback": target_context["allow_past_fallback"],
        "schedule_selection_reason": target_context["schedule_selection_reason"],
        "user_prediction_available": target_context["user_prediction_available"],
        "schedule_update_check_completed": schedule_update_status["schedule_update_check_completed"],
        "schedule_file_latest_date": schedule_update_status["schedule_max_date"],
        "schedule_file_total_games": schedule_update_status["total_games"],
        "schedule_file_future_games": schedule_update_status["future_games"],
        "schedule_update_needed": schedule_update_status["schedule_update_needed"],
        "schedule_update_blocker": schedule_update_status["schedule_update_blocker"],
        "schedule_update_status": schedule_update_status,
        **starter_pitcher_collection,
        "schedule_selection_check": schedule_check,
        "train_ratio": train_ratio,
        "training_cutoff": pd.Timestamp(cutoff).strftime("%Y-%m-%d"),
        "feature_rows": int(len(feature_df)),
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(validation_df)),
        "validation_games": int(len(selected_predictions)),
        "feature_columns": feature_columns,
        "candidate_scores": candidate_scores,
        "run_diff_bucket_accuracy": bucket_map[selected_name],
        "abs_run_diff_bucket_metrics": abs_bucket_map[selected_name],
        "season_metrics": selected_season_metrics,
        "team_bias_metrics": selected_team_metrics,
        "team_bias_summary": {
            "over_predicted_teams": over_predicted_teams,
            "under_predicted_teams": under_predicted_teams,
        },
        "starter_schema_inspection": starter_schema,
        "starter_data_status": starter_data_status,
        "starter_feature_status": starter_feature_status,
        "v2_status": "data_collection_phase",
        "starter_data_available": bool(starter_data_status["starter_data_available"]),
        "pitcher_logs_available": bool(starter_data_status["pitcher_logs_available"]),
        "starter_schema_ready": bool(starter_data_status["starter_schema_ready"]),
        "pitcher_log_schema_ready": bool(starter_data_status["pitcher_log_schema_ready"]),
        "v3_bullpen_schema_reusable": True,
        "v2_data_search_completed": bool(internal_data_search["v2_data_search_completed"]),
        "internal_pitcher_data_found": bool(internal_data_search["internal_pitcher_data_found"]),
        "v2_ready_to_train": bool(internal_data_search["v2_ready_to_train"]),
        "v2_blocker": internal_data_search["v2_blocker"],
        "internal_pitcher_data_search": internal_data_search,
        **{
            key: value
            for key, value in pitcher_data_validation.items()
            if key not in {"starter_pitchers_validation", "pitcher_game_logs_validation"}
        },
        "pitcher_data_validation": {
            "starter_pitchers": pitcher_data_validation["starter_pitchers_validation"],
            "pitcher_game_logs": pitcher_data_validation["pitcher_game_logs_validation"],
        },
        **internal_pitcher_mapping,
        **error_analysis["summary"],
        **improvement_experiment["summary"],
        "dashboard": DASHBOARD_PATH,
        "selected_model": selected_model,
        "outputs": [
            "team_game_features.csv",
            "model_scores.csv",
            "run_diff_bucket_accuracy.csv",
            "abs_run_diff_bucket_metrics.csv",
            "season_metrics.csv",
            "team_bias_metrics.csv",
            "expected_runs_predictions.csv",
            "match_predictions.csv",
            "schedule_selection_check.csv",
            "schedule_update_report.csv",
            "starter_pitcher_collection_report.csv",
            "starter_pitcher_validation.csv",
            "error_analysis_games.csv",
            "win_probability_bucket_metrics.csv",
            "total_runs_error_metrics.csv",
            "handicap_metrics.csv",
            "team_error_metrics.csv",
            "ballpark_error_metrics.csv",
            "monthly_error_metrics.csv",
            "performance_improvement_summary.csv",
            "park_factor_metrics.csv",
            "team_bias_feature_metrics.csv",
            "improvement_model_scores.csv",
            "pitcher_data_validation.csv",
            "internal_pitcher_data_inventory.csv",
            "internal_pitcher_mapping_report.csv",
            "internal_pitcher_conversion_check.csv",
            "summary.json",
            "report.html",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html_report(
        output_dir / "report.html",
        summary,
        candidate_scores,
        bucket_map[selected_name],
        abs_bucket_map[selected_name],
        selected_season_metrics,
        selected_team_metrics,
        selected_predictions,
        match_predictions,
        error_analysis,
        improvement_experiment,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KBO score, win probability, handicap, and over-under prediction dashboard.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--schedule", type=Path, default=DEFAULT_SCHEDULE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--target-date", type=lambda value: datetime.strptime(value, "%Y-%m-%d").date(), default=None)
    parser.add_argument("--allow-past-fallback", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_pipeline(args.input, args.schedule, args.output_dir, args.train_ratio, args.target_date, args.allow_past_fallback)
    selected = summary["selected_model"]
    print("KBO expected-runs model completed")
    print(f"selected_model={selected['model']}")
    print(f"run_mae={selected['run_mae']}")
    print(f"run_rmse={selected['run_rmse']}")
    print(f"home_win_accuracy={selected['home_win_accuracy']}")
    print(f"brier_score={selected['brier_score']}")
    print(f"target_date={summary['target_context']['target_date']}")
    print(f"report_mode={summary['target_context']['report_mode']}")
    print(f"schedule_is_stale={summary['schedule_is_stale']}")
    print(f"user_prediction_available={summary['user_prediction_available']}")
    print(f"match_predictions={summary['target_context']['game_count']}")
    print(f"outputs={args.output_dir}")


if __name__ == "__main__":
    main()
