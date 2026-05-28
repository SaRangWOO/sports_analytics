from __future__ import annotations

import pandas as pd

from evaluation.error_analysis import build_error_analysis
from evaluation.metrics import evaluate_win_conversion, select_model
from features.improvement_features import build_improvement_feature_matrix
from models.train import chronological_split, train_run_models


COMPARISON_METRICS = [
    "run_mae",
    "run_rmse",
    "total_runs_mae",
    "home_win_accuracy",
    "brier_score",
    "over_under_accuracy_8_5",
    "handicap_accuracy_2_5",
]


def _score_row(label: str, selected_model: dict, error_summary: dict) -> dict:
    return {
        "model_version": label,
        "selected_model": selected_model["model"],
        "run_mae": selected_model["run_mae"],
        "run_rmse": selected_model["run_rmse"],
        "total_runs_mae": error_summary["total_runs_mae"],
        "home_win_accuracy": selected_model["home_win_accuracy"],
        "brier_score": selected_model["brier_score"],
        "over_under_accuracy_8_5": error_summary["over_under_accuracy_8_5"],
        "handicap_accuracy_2_5": error_summary["handicap_accuracy_2_5"],
    }


def _metric_delta(baseline: dict, improved: dict, metric: str) -> float:
    return round(float(improved[metric]) - float(baseline[metric]), 4)


def _is_improved(baseline: dict, improved: dict) -> tuple[bool, str]:
    total_delta = _metric_delta(baseline, improved, "total_runs_mae")
    brier_delta = _metric_delta(baseline, improved, "brier_score")
    run_delta = _metric_delta(baseline, improved, "run_mae")
    over_under_delta = _metric_delta(baseline, improved, "over_under_accuracy_8_5")
    handicap_delta = _metric_delta(baseline, improved, "handicap_accuracy_2_5")
    if total_delta <= -0.05 and brier_delta <= 0.005 and run_delta <= 0.05 and over_under_delta >= 0 and handicap_delta >= 0:
        return True, "총득점 MAE가 개선되고 brier score 악화가 제한적이어서 개선 모델 적용 기준을 충족했습니다."
    return False, "총득점 MAE 개선 폭이 적용 기준보다 작거나 오버/언더·핸디캡 적중률이 악화되어 기존 baseline을 유지합니다."


def _comparison_frame(baseline: dict, improved: dict) -> pd.DataFrame:
    rows = []
    for metric in COMPARISON_METRICS:
        lower_is_better = metric in {"run_mae", "run_rmse", "total_runs_mae", "brier_score"}
        delta = _metric_delta(baseline, improved, metric)
        improved_flag = delta < 0 if lower_is_better else delta > 0
        worsened_flag = delta > 0 if lower_is_better else delta < 0
        rows.append(
            {
                "metric": metric,
                "baseline": baseline[metric],
                "improved": improved[metric],
                "delta": delta,
                "direction": "개선" if improved_flag else "악화" if worsened_flag else "동일",
            }
        )
    return pd.DataFrame(rows)


def run_performance_improvement_experiment(
    feature_df: pd.DataFrame,
    baseline_feature_columns: list[str],
    baseline_model: object,
    baseline_selected_model: dict,
    baseline_error_analysis: dict,
    train_ratio: float,
) -> dict:
    improved_df, improved_columns, park_metrics, bias_metrics = build_improvement_feature_matrix(
        feature_df,
        baseline_model,
        baseline_feature_columns,
    )
    improved_train_df, improved_validation_df, _ = chronological_split(improved_df, train_ratio)
    trained_models, run_scores = train_run_models(improved_train_df, improved_validation_df, improved_columns)
    win_scores, prediction_map, _, _ = evaluate_win_conversion(trained_models, improved_train_df, improved_validation_df, improved_columns)
    selected_model = select_model(run_scores, win_scores)
    selected_name = str(selected_model["model"])
    error_analysis = build_error_analysis(prediction_map[selected_name], improved_validation_df)

    baseline_row = _score_row("baseline", baseline_selected_model, baseline_error_analysis["summary"])
    improved_row = _score_row("improved", selected_model, error_analysis["summary"])
    applied, reason = _is_improved(baseline_row, improved_row)
    comparison = _comparison_frame(baseline_row, improved_row)
    model_scores = pd.DataFrame([{**row, "model_version": "improved"} for row in run_scores]).merge(
        pd.DataFrame([{**row, "model_version": "improved"} for row in win_scores]),
        on=["model_version", "model"],
        how="left",
    )
    improved_metrics = comparison[comparison["direction"].eq("개선")]["metric"].tolist()
    worsened_metrics = comparison[comparison["direction"].eq("악화")]["metric"].tolist()
    summary = {
        "performance_improvement_experiment_completed": True,
        "baseline_total_runs_mae": baseline_row["total_runs_mae"],
        "improved_total_runs_mae": improved_row["total_runs_mae"],
        "total_runs_mae_delta": _metric_delta(baseline_row, improved_row, "total_runs_mae"),
        "baseline_brier_score": baseline_row["brier_score"],
        "improved_brier_score": improved_row["brier_score"],
        "final_model_applied": applied,
        "final_model_reason": reason,
        "next_recommended_improvement": "최근 경기 가중치 개선과 선발투수/불펜/라인업 데이터 수집",
        "improved_metrics": improved_metrics,
        "worsened_metrics": worsened_metrics,
        "improved_selected_model": selected_name,
    }
    return {
        "improved_feature_df": improved_df,
        "improved_feature_columns": improved_columns,
        "improved_model": trained_models[selected_name],
        "improved_selected_model": selected_model,
        "improved_error_analysis": error_analysis,
        "comparison": comparison,
        "park_metrics": park_metrics,
        "bias_metrics": bias_metrics,
        "model_scores": model_scores,
        "summary": summary,
    }
