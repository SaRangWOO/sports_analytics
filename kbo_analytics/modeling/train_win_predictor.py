from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .feature_engineering import build_features
except ImportError:
    from feature_engineering import build_features


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -30, 30)
    return 1 / (1 + np.exp(-values))


def prepare_matrix(features: pd.DataFrame):
    leakage_columns = [
        "date",
        "game_id",
        "target_win",
        "actual_run_margin",
        "actual_close_game",
        "actual_blowout_game",
    ]
    x = features.drop(columns=[column for column in leakage_columns if column in features.columns])
    dummy_columns = [column for column in ["team", "opponent"] if column in x.columns]
    x = pd.get_dummies(x, columns=dummy_columns, drop_first=False, dtype=float)
    y = features["target_win"].to_numpy(dtype=float)
    return x, y


def standardize_train_test(x_train: pd.DataFrame, x_test: pd.DataFrame):
    mean = x_train.mean()
    std = x_train.std().replace(0, 1)
    return (x_train - mean) / std, (x_test - mean) / std, mean, std


def train_logistic_regression(x: np.ndarray, y: np.ndarray, lr: float = 0.08, epochs: int = 3000):
    weights = np.zeros(x.shape[1])
    bias = 0.0

    for _ in range(epochs):
        pred = sigmoid(x @ weights + bias)
        error = pred - y
        weights -= lr * (x.T @ error) / len(y)
        bias -= lr * error.mean()

    return weights, bias


def evaluate(y_true: np.ndarray, probability: np.ndarray):
    y_pred = (probability >= 0.5).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    accuracy = (tp + tn) / len(y_true)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    return {
        "accuracy": round(accuracy, 3),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def main():
    parser = argparse.ArgumentParser(description="Train a simple KBO win prediction model.")
    parser.add_argument("--input", default="../data/weekly/game_results.csv")
    parser.add_argument("--model-output", default="win_predictor_model.json")
    parser.add_argument("--features-output", default="features.csv")
    args = parser.parse_args()

    features = build_features(args.input)
    features.to_csv(args.features_output, index=False, encoding="utf-8-sig")

    x, y = prepare_matrix(features)
    split_index = max(int(len(x) * 0.8), 1)
    x_train, x_test = x.iloc[:split_index], x.iloc[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    if len(x_test) == 0:
        raise SystemExit("Not enough rows for a chronological train/test split.")

    x_train_scaled, x_test_scaled, mean, std = standardize_train_test(x_train, x_test)
    weights, bias = train_logistic_regression(x_train_scaled.to_numpy(), y_train)
    test_probability = sigmoid(x_test_scaled.to_numpy() @ weights + bias)
    metrics = evaluate(y_test, test_probability)

    coefficients = sorted(
        zip(x.columns, weights),
        key=lambda item: abs(item[1]),
        reverse=True,
    )

    model_payload = {
        "model": "from_scratch_logistic_regression",
        "target": "target_win",
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "metrics": metrics,
        "feature_mean": mean.round(6).to_dict(),
        "feature_std": std.round(6).to_dict(),
        "bias": round(float(bias), 6),
        "coefficients": {name: round(float(value), 6) for name, value in coefficients},
    }

    Path(args.model_output).write_text(
        json.dumps(model_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps({"metrics": metrics, "top_features": coefficients[:5]}, ensure_ascii=False, default=str))
    print(f"Saved model: {args.model_output}")
    print(f"Saved features: {args.features_output}")


if __name__ == "__main__":
    main()
