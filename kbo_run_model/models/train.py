from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import PoissonRegressor, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error


def chronological_split(df: pd.DataFrame, train_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    dates = df["date"].drop_duplicates().sort_values().reset_index(drop=True)
    if len(dates) < 2:
        raise ValueError("At least two game dates are required for chronological validation.")
    split_pos = max(1, min(int(len(dates) * train_ratio), len(dates) - 1))
    cutoff = dates.iloc[split_pos]
    return df[df["date"] < cutoff].copy(), df[df["date"] >= cutoff].copy(), cutoff


def candidate_regressors() -> dict[str, object]:
    return {
        "PoissonRegressor": PoissonRegressor(max_iter=500, alpha=0.1),
        "Ridge": Ridge(alpha=2.0),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=8,
            random_state=42,
            n_jobs=-1,
        ),
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor(
            max_iter=220,
            learning_rate=0.04,
            max_leaf_nodes=15,
            l2_regularization=0.08,
            random_state=42,
        ),
    }


def train_run_models(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[dict[str, object], list[dict[str, float | str]]]:
    trained: dict[str, object] = {}
    scores: list[dict[str, float | str]] = []
    x_train = train_df[feature_columns]
    y_train = train_df["target_runs"]
    x_validation = validation_df[feature_columns]
    y_validation = validation_df["target_runs"]

    for name, model in candidate_regressors().items():
        model.fit(x_train, y_train)
        predictions = np.clip(model.predict(x_validation), 0, None)
        trained[name] = model
        scores.append(
            {
                "model": name,
                "run_mae": round(float(mean_absolute_error(y_validation, predictions)), 4),
                "run_rmse": round(float(np.sqrt(mean_squared_error(y_validation, predictions))), 4),
            }
        )
    return trained, scores
