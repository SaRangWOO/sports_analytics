from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .feature_engineering import build_features
from .game_level_features import (
    align_game_level_matrix,
    attach_pitching_context,
    build_game_level_frame,
    prepare_game_level_matrix,
)
from .model_artifacts import predict_bundle_probabilities, validate_available_features
from .model_evaluation import normalize_game_probabilities
from .train_win_predictor import prepare_matrix


PREDICTION_OUTPUT_COLUMNS = ["경기일", "기준팀", "상대팀", "예측 구단", "예측승률", "예측", "예측 근거"]


def _team_matrix(
    today_features: pd.DataFrame,
    feature_order: list[str],
    prediction_bundle: dict,
    feature_schema: dict | None,
) -> pd.DataFrame:
    raw, _ = prepare_matrix(today_features)
    if feature_schema:
        validate_available_features(list(raw.columns), feature_schema)
    aligned = raw.reindex(columns=feature_order, fill_value=0)
    return (aligned - prediction_bundle["mean"]) / prediction_bundle["std"].replace(0, 1)


def _game_matrix(
    today_games: pd.DataFrame,
    feature_order: list[str],
    prediction_bundle: dict,
    feature_schema: dict | None,
) -> pd.DataFrame:
    raw, _ = prepare_game_level_matrix(today_games)
    if feature_schema:
        validate_available_features(list(raw.columns), feature_schema)
    return align_game_level_matrix(
        today_games,
        feature_order,
        prediction_bundle["mean"],
        prediction_bundle["std"],
    )


def generate_today_predictions(
    current_games: pd.DataFrame,
    prediction_date: date,
    data_dir: str | Path,
    feature_order: list[str],
    prediction_unit: str,
    prediction_bundle: dict,
    team_reason: Callable[[pd.Series, str], str],
    game_reason: Callable[[pd.Series, str], str],
    feature_schema: dict | None = None,
) -> list[dict]:
    data_dir = Path(data_dir)
    prediction_input = data_dir / "prediction_games.csv"
    current_games.to_csv(prediction_input, index=False, encoding="utf-8-sig")
    prediction_features = build_features(prediction_input, include_unlabeled=True)
    prediction_features["date_obj"] = pd.to_datetime(prediction_features["date"]).dt.date
    today_features = prediction_features[
        (prediction_features["date_obj"] == prediction_date) & (prediction_features["target_win"].isna())
    ].copy()
    if today_features.empty:
        return []

    if prediction_unit == "game":
        pitching_context_path = data_dir / "pitching_context.csv"
        pitching_context = pd.read_csv(pitching_context_path) if pitching_context_path.exists() else pd.DataFrame()
        game_frame = attach_pitching_context(build_game_level_frame(prediction_features), pitching_context)
        game_frame["date_obj"] = pd.to_datetime(game_frame["date"]).dt.date
        today_games = game_frame[
            (game_frame["date_obj"] == prediction_date) & (game_frame["target_home_win"].isna())
        ].copy()
        if today_games.empty:
            return []
        model_frame = today_games.drop(columns=["date_obj"])
        scaled = _game_matrix(model_frame, feature_order, prediction_bundle, feature_schema)
        probability = predict_bundle_probabilities(prediction_bundle, scaled, feature_schema or _runtime_schema(feature_order))
        rows = []
        for (_, row), home_probability in zip(today_games.iterrows(), probability):
            home_pick = home_probability >= 0.5
            predicted_team = row["home_team"] if home_pick else row["away_team"]
            reason = game_reason(row, predicted_team)
            rows.extend(
                [
                    {
                        "경기일": row["date"],
                        "기준팀": row["home_team"],
                        "상대팀": row["away_team"],
                        "예측 구단": predicted_team,
                        "예측승률": f"{home_probability:.1%}",
                        "예측": "승리 예측" if home_pick else "패배 예측",
                        "예측 근거": reason,
                    },
                    {
                        "경기일": row["date"],
                        "기준팀": row["away_team"],
                        "상대팀": row["home_team"],
                        "예측 구단": predicted_team,
                        "예측승률": f"{1 - home_probability:.1%}",
                        "예측": "승리 예측" if not home_pick else "패배 예측",
                        "예측 근거": reason,
                    },
                ]
            )
        return rows

    model_features = today_features.drop(columns=["date_obj"])
    scaled = _team_matrix(model_features, feature_order, prediction_bundle, feature_schema)
    raw_probability = predict_bundle_probabilities(
        prediction_bundle,
        scaled,
        feature_schema or _runtime_schema(feature_order),
    )
    probability = normalize_game_probabilities(today_features, raw_probability)
    today_features["경기일"] = pd.to_datetime(today_features["date"]).dt.strftime("%Y-%m-%d")
    today_features["기준팀"] = today_features["team"]
    today_features["상대팀"] = today_features["opponent"]
    today_features["예측승률"] = [f"{value:.1%}" for value in probability]
    today_features["예측"] = np.where(probability >= 0.5, "승리 예측", "패배 예측")
    today_features["예측 구단"] = np.where(
        probability >= 0.5,
        today_features["team"],
        today_features["opponent"],
    )
    today_features["예측 근거"] = today_features.apply(
        lambda row: team_reason(row, row["예측 구단"]),
        axis=1,
    )
    return today_features[PREDICTION_OUTPUT_COLUMNS].to_dict(orient="records")


def _runtime_schema(feature_order: list[str]) -> dict:
    return {
        "feature_order": feature_order,
        "required_features": feature_order,
        "optional_features": [],
    }
