from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_game_probabilities(features: pd.DataFrame, probability: np.ndarray):
    normalized = pd.Series(probability, index=features.index, dtype=float)
    game_keys = features["game_id"].astype(str).str.rsplit("_", n=1).str[0]
    for _, indexes in game_keys.groupby(game_keys).groups.items():
        indexes = list(indexes)
        if len(indexes) != 2:
            continue
        total = normalized.loc[indexes].sum()
        if total > 0:
            normalized.loc[indexes] = normalized.loc[indexes] / total
    return normalized.to_numpy()


def confidence_metrics(y_true: np.ndarray, probability: np.ndarray):
    pred = (probability >= 0.5).astype(int)
    correct = pred == y_true
    confidence = np.maximum(probability, 1 - probability)
    metrics = [{"구간": "전체 경기", "경기 수": int(len(y_true)), "적중률": round(float(correct.mean()), 3)}]
    for threshold in [0.55, 0.58, 0.60]:
        mask = confidence >= threshold
        metrics.append(
            {
                "구간": f"{round(threshold * 100)}% 이상 확신 경기",
                "경기 수": int(mask.sum()),
                "적중률": round(float(correct[mask].mean()), 3) if mask.any() else "-",
            }
        )
    score = probability_scores(y_true, probability)
    metrics.append({"구간": "Brier Score", "경기 수": "-", "적중률": score["Brier Score"]})
    metrics.append({"구간": "Log Loss", "경기 수": "-", "적중률": score["Log Loss"]})
    return metrics


def probability_scores(y_true: np.ndarray, probability: np.ndarray):
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    return {
        "Brier Score": round(float(np.mean((probability - y_true) ** 2)), 3),
        "Log Loss": round(float(-np.mean(y_true * np.log(clipped) + (1 - y_true) * np.log(1 - clipped))), 3),
    }


def calibration_table(y_true: np.ndarray, probability: np.ndarray):
    frame = pd.DataFrame({"actual": y_true, "probability": probability})
    bins = [0.0, 0.45, 0.50, 0.55, 0.60, 1.0]
    labels = ["45% 미만", "45~50%", "50~55%", "55~60%", "60% 이상"]
    frame["구간"] = pd.cut(frame["probability"], bins=bins, labels=labels, include_lowest=True)
    rows = []
    for label in labels:
        subset = frame[frame["구간"] == label]
        rows.append(
            {
                "예측승률 구간": label,
                "경기 수": int(len(subset)),
                "평균 예측승률": f"{subset['probability'].mean():.1%}" if len(subset) else "-",
                "실제 승률": f"{subset['actual'].mean():.1%}" if len(subset) else "-",
            }
        )
    return rows


def pick_better_model(current: dict | None, candidate: dict):
    if current is None:
        return candidate
    if candidate["accuracy"] > current["accuracy"] + 0.005:
        return candidate
    if abs(candidate["accuracy"] - current["accuracy"]) <= 0.005 and candidate["score"]["Brier Score"] < current["score"]["Brier Score"]:
        return candidate
    return current
