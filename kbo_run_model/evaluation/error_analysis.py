from __future__ import annotations

import numpy as np
import pandas as pd

from models.recommendations import DEFAULT_HANDICAP_LINE, DEFAULT_OVER_UNDER_LINE, handicap_pick, over_under_pick


WIN_PROBABILITY_BINS = [0.5, 0.55, 0.60, 0.65, np.inf]
WIN_PROBABILITY_LABELS = ["50~55%", "55~60%", "60~65%", "65% 이상"]


def _round_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    output = frame.copy()
    for column in columns:
        if column in output.columns:
            output[column] = output[column].round(4)
    return output


def _safe_accuracy(series: pd.Series) -> float | str:
    if series.empty:
        return ""
    return round(float(series.mean()), 4)


def _attach_ballpark(scored_games: pd.DataFrame, validation_df: pd.DataFrame) -> pd.DataFrame:
    output = scored_games.copy()
    if "ballpark" not in validation_df.columns:
        output["ballpark"] = ""
        return output
    ballparks = validation_df[validation_df["is_home"].eq(1)][["game_key", "ballpark"]].drop_duplicates("game_key")
    return output.merge(ballparks, on="game_key", how="left")


def build_game_error_analysis(scored_games: pd.DataFrame, validation_df: pd.DataFrame) -> pd.DataFrame:
    games = _attach_ballpark(scored_games, validation_df)
    games["home_run_error"] = games["home_expected_runs"] - games["home_actual_runs"]
    games["away_run_error"] = games["away_expected_runs"] - games["away_actual_runs"]
    games["home_abs_error"] = games["home_run_error"].abs()
    games["away_abs_error"] = games["away_run_error"].abs()
    games["run_mae"] = games[["home_abs_error", "away_abs_error"]].mean(axis=1)
    games["home_error_direction"] = np.select(
        [games["home_run_error"].gt(0), games["home_run_error"].lt(0)],
        ["과대예측", "과소예측"],
        default="정확",
    )
    games["away_error_direction"] = np.select(
        [games["away_run_error"].gt(0), games["away_run_error"].lt(0)],
        ["과대예측", "과소예측"],
        default="정확",
    )
    games["actual_total_runs"] = games["home_actual_runs"] + games["away_actual_runs"]
    games["total_run_error"] = games["expected_total_runs"] - games["actual_total_runs"]
    games["total_abs_error"] = games["total_run_error"].abs()
    games["actual_home_win"] = games["home_actual_win"].astype(int)
    games["win_correct"] = games["predicted_home_win"].eq(games["actual_home_win"])
    games["favorite_win_probability"] = np.maximum(games["home_win_probability"], 1 - games["home_win_probability"])
    games["favorite_probability_bucket"] = pd.cut(
        games["favorite_win_probability"],
        bins=WIN_PROBABILITY_BINS,
        labels=WIN_PROBABILITY_LABELS,
        include_lowest=True,
        right=False,
    )
    games["moneyline_pick_actual_result"] = np.where(games["win_correct"], "적중", "실패")
    games["over_under_line"] = DEFAULT_OVER_UNDER_LINE
    games["over_under_pick"] = games["expected_total_runs"].map(lambda value: over_under_pick(float(value), DEFAULT_OVER_UNDER_LINE))
    games["actual_over_under_result"] = np.select(
        [games["actual_total_runs"].gt(DEFAULT_OVER_UNDER_LINE), games["actual_total_runs"].lt(DEFAULT_OVER_UNDER_LINE)],
        ["오버", "언더"],
        default="푸시",
    )
    games["over_under_correct"] = games["over_under_pick"].eq(games["actual_over_under_result"])
    games["handicap_line"] = DEFAULT_HANDICAP_LINE
    games["handicap_pick"] = [
        handicap_pick(row.home_team, row.away_team, row.expected_run_diff, DEFAULT_HANDICAP_LINE)
        for row in games.itertuples(index=False)
    ]
    home_cover = games["actual_run_diff"].gt(DEFAULT_HANDICAP_LINE)
    away_cover = games["actual_run_diff"].lt(-DEFAULT_HANDICAP_LINE)
    home_plus_cover = games["actual_run_diff"].gt(-DEFAULT_HANDICAP_LINE)
    away_plus_cover = games["actual_run_diff"].lt(DEFAULT_HANDICAP_LINE)
    games["handicap_correct"] = np.select(
        [
            games["handicap_pick"].eq("관망"),
            games["handicap_pick"].eq(games["home_team"] + f" -{DEFAULT_HANDICAP_LINE}") & home_cover,
            games["handicap_pick"].eq(games["away_team"] + f" -{DEFAULT_HANDICAP_LINE}") & away_cover,
            games["handicap_pick"].eq(games["home_team"] + f" +{DEFAULT_HANDICAP_LINE}") & home_plus_cover,
            games["handicap_pick"].eq(games["away_team"] + f" +{DEFAULT_HANDICAP_LINE}") & away_plus_cover,
        ],
        [np.nan, True, True, True, True],
        default=False,
    )
    games["season_phase"] = pd.cut(
        games.groupby("season").cumcount() / games.groupby("season")["game_key"].transform("count"),
        bins=[-0.01, 0.33, 0.66, 1],
        labels=["시즌 초반", "시즌 중반", "시즌 후반"],
    )
    columns = [
        "season",
        "date",
        "game_key",
        "ballpark",
        "away_team",
        "home_team",
        "away_actual_runs",
        "away_expected_runs",
        "away_run_error",
        "home_actual_runs",
        "home_expected_runs",
        "home_run_error",
        "run_mae",
        "actual_total_runs",
        "expected_total_runs",
        "total_run_error",
        "total_abs_error",
        "expected_run_diff",
        "actual_run_diff",
        "home_win_probability",
        "predicted_winner",
        "actual_winner",
        "win_correct",
        "favorite_probability_bucket",
        "over_under_pick",
        "actual_over_under_result",
        "over_under_correct",
        "handicap_pick",
        "handicap_correct",
        "season_phase",
    ]
    return _round_columns(games[columns], [
        "away_expected_runs",
        "away_run_error",
        "home_expected_runs",
        "home_run_error",
        "run_mae",
        "expected_total_runs",
        "total_run_error",
        "total_abs_error",
        "expected_run_diff",
        "actual_run_diff",
        "home_win_probability",
    ])


def win_probability_bucket_metrics(game_errors: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bucket, group in game_errors.groupby("favorite_probability_bucket", observed=False):
        rows.append(
            {
                "bucket": str(bucket),
                "games": int(len(group)),
                "accuracy": _safe_accuracy(group["win_correct"]) if not group.empty else "",
                "avg_expected_run_diff_abs": round(float(group["expected_run_diff"].abs().mean()), 4) if not group.empty else "",
                "avg_run_mae": round(float(group["run_mae"].mean()), 4) if not group.empty else "",
            }
        )
    close = game_errors[game_errors["expected_run_diff"].abs().lt(0.5)]
    strong = game_errors[game_errors["expected_run_diff"].abs().ge(1.0)]
    rows.extend(
        [
            {
                "bucket": "접전 예측(득실차 0.5 미만)",
                "games": int(len(close)),
                "accuracy": _safe_accuracy(close["win_correct"]),
                "avg_expected_run_diff_abs": round(float(close["expected_run_diff"].abs().mean()), 4) if not close.empty else "",
                "avg_run_mae": round(float(close["run_mae"].mean()), 4) if not close.empty else "",
            },
            {
                "bucket": "강한 우세 예측(득실차 1.0 이상)",
                "games": int(len(strong)),
                "accuracy": _safe_accuracy(strong["win_correct"]),
                "avg_expected_run_diff_abs": round(float(strong["expected_run_diff"].abs().mean()), 4) if not strong.empty else "",
                "avg_run_mae": round(float(strong["run_mae"].mean()), 4) if not strong.empty else "",
            },
        ]
    )
    return pd.DataFrame(rows)


def total_runs_error_metrics(game_errors: pd.DataFrame) -> pd.DataFrame:
    high_threshold = game_errors["actual_total_runs"].quantile(0.75)
    low_threshold = game_errors["actual_total_runs"].quantile(0.25)
    groups = {
        "전체": game_errors,
        "실제 고득점 경기": game_errors[game_errors["actual_total_runs"].ge(high_threshold)],
        "실제 저득점 경기": game_errors[game_errors["actual_total_runs"].le(low_threshold)],
        "오버/언더 추천 경기": game_errors[game_errors["over_under_pick"].ne("관망")],
    }
    rows = []
    for label, group in groups.items():
        rows.append(
            {
                "category": label,
                "games": int(len(group)),
                "actual_total_runs_avg": round(float(group["actual_total_runs"].mean()), 4) if not group.empty else "",
                "expected_total_runs_avg": round(float(group["expected_total_runs"].mean()), 4) if not group.empty else "",
                "total_runs_mae": round(float(group["total_abs_error"].mean()), 4) if not group.empty else "",
                "over_under_accuracy_8_5": _safe_accuracy(group["over_under_correct"]) if not group.empty else "",
            }
        )
    return pd.DataFrame(rows)


def handicap_metrics(game_errors: pd.DataFrame) -> pd.DataFrame:
    recommended = game_errors[game_errors["handicap_pick"].ne("관망")]
    rows = [
        {
            "handicap_pick": "전체",
            "games": int(len(game_errors)),
            "recommended_games": int(len(recommended)),
            "accuracy_2_5": _safe_accuracy(recommended["handicap_correct"].dropna()),
        }
    ]
    for pick, group in game_errors.groupby("handicap_pick", sort=True):
        valid = group[group["handicap_pick"].ne("관망")]["handicap_correct"].dropna()
        rows.append(
            {
                "handicap_pick": str(pick),
                "games": int(len(group)),
                "recommended_games": int(len(valid)),
                "accuracy_2_5": _safe_accuracy(valid),
            }
        )
    return pd.DataFrame(rows)


def team_error_metrics(game_errors: pd.DataFrame) -> pd.DataFrame:
    home = game_errors[
        ["home_team", "home_actual_runs", "home_expected_runs", "away_actual_runs", "away_expected_runs"]
    ].rename(
        columns={
            "home_team": "team",
            "home_actual_runs": "actual_runs",
            "home_expected_runs": "predicted_runs",
            "away_actual_runs": "opponent_actual_runs",
            "away_expected_runs": "opponent_predicted_runs",
        }
    )
    away = game_errors[
        ["away_team", "away_actual_runs", "away_expected_runs", "home_actual_runs", "home_expected_runs"]
    ].rename(
        columns={
            "away_team": "team",
            "away_actual_runs": "actual_runs",
            "away_expected_runs": "predicted_runs",
            "home_actual_runs": "opponent_actual_runs",
            "home_expected_runs": "opponent_predicted_runs",
        }
    )
    rows = pd.concat([home, away], ignore_index=True)
    rows["run_error"] = rows["predicted_runs"] - rows["actual_runs"]
    rows["opponent_run_error"] = rows["opponent_predicted_runs"] - rows["opponent_actual_runs"]
    rows["abs_error"] = rows["run_error"].abs()
    rows["opponent_abs_error"] = rows["opponent_run_error"].abs()
    metrics = (
        rows.groupby("team", as_index=False)
        .agg(
            games=("team", "size"),
            actual_avg_runs=("actual_runs", "mean"),
            predicted_avg_runs=("predicted_runs", "mean"),
            run_mae=("abs_error", "mean"),
            run_bias=("run_error", "mean"),
            opponent_actual_avg_runs=("opponent_actual_runs", "mean"),
            opponent_predicted_avg_runs=("opponent_predicted_runs", "mean"),
            opponent_run_mae=("opponent_abs_error", "mean"),
            opponent_run_bias=("opponent_run_error", "mean"),
        )
        .sort_values("run_mae", ascending=False)
    )
    metrics["bias_direction"] = np.select(
        [metrics["run_bias"].gt(0.25), metrics["run_bias"].lt(-0.25)],
        ["지속 과대예측", "지속 과소예측"],
        default="중립",
    )
    return _round_columns(
        metrics,
        [
            "actual_avg_runs",
            "predicted_avg_runs",
            "run_mae",
            "run_bias",
            "opponent_actual_avg_runs",
            "opponent_predicted_avg_runs",
            "opponent_run_mae",
            "opponent_run_bias",
        ],
    )


def ballpark_error_metrics(game_errors: pd.DataFrame) -> pd.DataFrame:
    if "ballpark" not in game_errors.columns or game_errors["ballpark"].fillna("").eq("").all():
        return pd.DataFrame(
            [
                {
                    "ballpark": "구장 정보 없음",
                    "games": 0,
                    "actual_total_runs_avg": "",
                    "expected_total_runs_avg": "",
                    "total_runs_mae": "",
                    "total_runs_bias": "",
                    "bias_direction": "분석 불가",
                }
            ]
        )
    metrics = (
        game_errors.groupby("ballpark", as_index=False)
        .agg(
            games=("game_key", "size"),
            actual_total_runs_avg=("actual_total_runs", "mean"),
            expected_total_runs_avg=("expected_total_runs", "mean"),
            total_runs_mae=("total_abs_error", "mean"),
            total_runs_bias=("total_run_error", "mean"),
        )
        .sort_values("total_runs_mae", ascending=False)
    )
    metrics["bias_direction"] = np.select(
        [metrics["total_runs_bias"].gt(0.5), metrics["total_runs_bias"].lt(-0.5)],
        ["총득점 과대예측", "총득점 과소예측"],
        default="중립",
    )
    return _round_columns(metrics, ["actual_total_runs_avg", "expected_total_runs_avg", "total_runs_mae", "total_runs_bias"])


def monthly_error_metrics(game_errors: pd.DataFrame) -> pd.DataFrame:
    frame = game_errors.copy()
    frame["month"] = pd.to_datetime(frame["date"]).dt.month
    monthly = (
        frame.groupby("month", as_index=False)
        .agg(
            games=("game_key", "size"),
            run_mae=("run_mae", "mean"),
            total_runs_mae=("total_abs_error", "mean"),
            win_accuracy=("win_correct", "mean"),
        )
        .sort_values("month")
    )
    phase = (
        frame.groupby("season_phase", observed=False, as_index=False)
        .agg(
            games=("game_key", "size"),
            run_mae=("run_mae", "mean"),
            total_runs_mae=("total_abs_error", "mean"),
            win_accuracy=("win_correct", "mean"),
        )
        .rename(columns={"season_phase": "month"})
    )
    phase["month"] = phase["month"].astype(str)
    combined = pd.concat([monthly, phase], ignore_index=True)
    return _round_columns(combined, ["run_mae", "total_runs_mae", "win_accuracy"])


def build_error_analysis(scored_games: pd.DataFrame, validation_df: pd.DataFrame) -> dict[str, pd.DataFrame | dict]:
    game_errors = build_game_error_analysis(scored_games, validation_df)
    win_buckets = win_probability_bucket_metrics(game_errors)
    total_metrics = total_runs_error_metrics(game_errors)
    handicap = handicap_metrics(game_errors)
    team_metrics = team_error_metrics(game_errors)
    ballpark_metrics = ballpark_error_metrics(game_errors)
    monthly_metrics = monthly_error_metrics(game_errors)
    top_errors = game_errors.sort_values("run_mae", ascending=False).head(10)

    weakest_team = team_metrics.iloc[0].to_dict() if not team_metrics.empty else {}
    valid_ballparks = ballpark_metrics[ballpark_metrics["games"].astype(str).ne("0")]
    weakest_ballpark = valid_ballparks.iloc[0].to_dict() if not valid_ballparks.empty else {}
    total_row = total_metrics[total_metrics["category"].eq("전체")].iloc[0].to_dict()
    handicap_row = handicap[handicap["handicap_pick"].eq("전체")].iloc[0].to_dict()
    category_mae = {
        "득점 예측": float(game_errors["run_mae"].mean()),
        "총득점 예측": float(game_errors["total_abs_error"].mean()),
        "승패 예측": 1 - float(game_errors["win_correct"].mean()),
    }
    biggest_error_category = max(category_mae, key=category_mae.get)
    recommendations = []
    if weakest_ballpark:
        recommendations.append("구장 득점 팩터")
    if weakest_team:
        recommendations.append("팀별 공격/실점 편향 보정")
    recommendations.append("선발투수와 불펜 로그 수집")

    summary = {
        "error_analysis_completed": True,
        "biggest_error_category": biggest_error_category,
        "weakest_team_prediction": weakest_team.get("team", ""),
        "weakest_ballpark_prediction": weakest_ballpark.get("ballpark", ""),
        "total_runs_mae": total_row.get("total_runs_mae", ""),
        "over_under_accuracy_8_5": total_row.get("over_under_accuracy_8_5", ""),
        "handicap_accuracy_2_5": handicap_row.get("accuracy_2_5", ""),
        "recommended_next_improvement": " → ".join(recommendations),
    }
    return {
        "game_errors": game_errors,
        "top_errors": top_errors,
        "win_probability_buckets": win_buckets,
        "total_runs": total_metrics,
        "handicap": handicap,
        "team": team_metrics,
        "ballpark": ballpark_metrics,
        "monthly": monthly_metrics,
        "summary": summary,
    }
