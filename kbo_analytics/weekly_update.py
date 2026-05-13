from __future__ import annotations

import argparse
import json
import os
from datetime import date, datetime, timedelta
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from modeling.feature_engineering import build_features
from modeling.train_win_predictor import (
    evaluate,
    prepare_matrix,
    sigmoid,
    standardize_train_test,
    train_logistic_regression,
)


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "weekly"
DASHBOARD_DIR = BASE_DIR / "dashboard"
RESULTS_DIR = BASE_DIR / "modeling" / "results"
DB_URL = os.getenv("DB_URL", "postgresql://tera:tera@localhost:5432/baseball")
MODEL_HISTORY_PATH = RESULTS_DIR / "model_history.json"

FEATURE_LABELS = {
    "is_home": "홈 경기 여부",
    "month": "경기 월",
    "series_game_no": "같은 상대 연전 내 경기 순서",
    "rest_days": "직전 경기 이후 휴식일",
    "recent_5_win_rate": "최근 5경기 승률",
    "avg_score_last_5": "최근 5경기 평균 득점",
    "avg_allowed_last_5": "최근 5경기 평균 실점",
    "avg_run_diff_last_5": "최근 5경기 평균 득실차",
}
TEAM_LABELS = {
    "Doosan": "두산",
    "Hanwha": "한화",
    "KIA": "KIA",
    "KT": "KT",
    "Lotte": "롯데",
    "SSG": "SSG",
}


def previous_week_window(reference_date: date | None = None):
    reference_date = reference_date or date.today()
    this_monday = reference_date - timedelta(days=reference_date.weekday())
    last_monday = this_monday - timedelta(days=7)
    last_sunday = this_monday - timedelta(days=1)
    return last_monday.isoformat(), last_sunday.isoformat()


def load_tables():
    engine = create_engine(DB_URL)
    games = pd.read_sql("SELECT * FROM game_results ORDER BY date, game_id", engine)
    players = pd.read_sql("SELECT * FROM player_game_stats ORDER BY date, game_id, team, player_id", engine)
    return games, players


def persist_exports(games: pd.DataFrame, players: pd.DataFrame):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    games_path = DATA_DIR / "game_results.csv"
    players_path = DATA_DIR / "player_game_stats.csv"
    games.to_csv(games_path, index=False, encoding="utf-8-sig")
    players.to_csv(players_path, index=False, encoding="utf-8-sig")
    return games_path, players_path


def feature_label(name: str):
    if name.startswith("opponent_"):
        return f"상대팀: {team_label(name.replace('opponent_', ''))}"
    return FEATURE_LABELS.get(name, name)


def team_label(name: str):
    return TEAM_LABELS.get(str(name), str(name))


def player_label(player_name: str, team: str):
    name = str(player_name)
    team_ko = team_label(team)
    if name == "KT Starter":
        return "KT 선발"
    if name.startswith("KT Reliever"):
        return name.replace("KT Reliever", "KT 불펜")
    if name.startswith("KT Batter"):
        return name.replace("KT Batter", "KT 타자")
    if " Player " in name:
        try:
            number_part = int(name.rsplit(" ", 1)[-1])
        except ValueError:
            return name
        if 1 <= number_part <= 9:
            return f"{team_ko} 타자 {number_part}"
        if number_part == 10:
            return f"{team_ko} 선발"
        return f"{team_ko} 불펜 {number_part - 10}"
    return name


def model_candidates():
    return [
        {
            "name": "전체 변수 로지스틱 회귀",
            "columns": None,
            "learning_rate": 0.08,
            "epochs": 3000,
            "thresholds": [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85],
        },
        {
            "name": "공격/실점 흐름 중심 로지스틱 회귀",
            "columns": ["recent_5_win_rate", "avg_score_last_5", "avg_allowed_last_5", "avg_run_diff_last_5"],
            "learning_rate": 0.05,
            "epochs": 4500,
            "thresholds": [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85],
        },
        {
            "name": "구장/일정 포함 로지스틱 회귀",
            "columns": ["is_home", "rest_days", "series_game_no", "recent_5_win_rate", "avg_run_diff_last_5"],
            "learning_rate": 0.05,
            "epochs": 4500,
            "thresholds": [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85],
        },
    ]


def evaluate_with_threshold(y_true: np.ndarray, probability: np.ndarray, threshold: float):
    y_pred = (probability >= threshold).astype(int)
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


def select_candidate(x: pd.DataFrame, y: np.ndarray, split_index: int):
    best = None
    all_results = []
    x_train, x_test = x.iloc[:split_index], x.iloc[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    for candidate in model_candidates():
        cols = candidate["columns"] or list(x.columns)
        current_train = x_train[cols]
        current_test = x_test[cols]
        train_scaled, test_scaled, mean, std = standardize_train_test(current_train, current_test)
        weights, bias = train_logistic_regression(
            train_scaled.to_numpy(),
            y_train,
            lr=candidate["learning_rate"],
            epochs=candidate["epochs"],
        )
        probability = sigmoid(test_scaled.to_numpy() @ weights + bias)
        threshold_results = [
            (threshold, evaluate_with_threshold(y_test, probability, threshold))
            for threshold in candidate["thresholds"]
        ]
        threshold, metrics = max(
            threshold_results,
            key=lambda item: (item[1]["accuracy"], item[1]["f1"], item[1]["recall"]),
        )
        result = {
            "candidate": candidate,
            "columns": cols,
            "weights": weights,
            "bias": bias,
            "mean": mean,
            "std": std,
            "threshold": threshold,
            "metrics": metrics,
        }
        all_results.append(
            {
                "name": candidate["name"],
                "features": [feature_label(col) for col in cols],
                "threshold": threshold,
                "metrics": metrics,
            }
        )
        if best is None or (
            metrics["accuracy"],
            metrics["f1"],
            metrics["recall"],
        ) > (
            best["metrics"]["accuracy"],
            best["metrics"]["f1"],
            best["metrics"]["recall"],
        ):
            best = result

    best["all_results"] = all_results
    return best


def train_model(games_path: Path):
    features = build_features(games_path)
    if len(features) < 4:
        raise ValueError("승패 예측 모델 학습에는 최소 4경기 이상의 Win/Loss 데이터가 필요합니다.")

    x, y = prepare_matrix(features)
    split_index = max(int(len(x) * 0.8), 1)
    split_index = min(split_index, len(x) - 1)
    selected = select_candidate(x, y, split_index)
    selected_x = x[selected["columns"]]
    x_all_scaled = (selected_x - selected["mean"]) / selected["std"]
    all_probability = sigmoid(x_all_scaled.to_numpy() @ selected["weights"] + selected["bias"])
    features = features.copy()
    features["predicted_win_probability"] = np.round(all_probability, 3)
    features["predicted_result"] = np.where(all_probability >= selected["threshold"], "승리 예측", "패배 예측")

    coefficients = sorted(zip(selected["columns"], selected["weights"]), key=lambda item: abs(item[1]), reverse=True)
    top_features = [
        {
            "feature": name,
            "label": feature_label(name),
            "coefficient": round(float(value), 6),
            "direction": "승리 확률 상승" if value > 0 else "승리 확률 하락",
        }
        for name, value in coefficients[:10]
    ]

    recent_predictions = features.tail(8)[
        ["date", "game_id", "opponent", "is_home", "target_win", "predicted_win_probability", "predicted_result"]
    ].copy()
    recent_predictions["date"] = pd.to_datetime(recent_predictions["date"]).dt.strftime("%Y-%m-%d")
    recent_predictions["opponent"] = recent_predictions["opponent"].map(team_label)
    recent_predictions["home_away"] = np.where(recent_predictions["is_home"] == 1, "홈", "원정")
    recent_predictions["actual_result"] = np.where(recent_predictions["target_win"] == 1, "승", "패")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": "from_scratch_logistic_regression",
        "model_name_ko": selected["candidate"]["name"],
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "train_rows": int(split_index),
        "test_rows": int(len(x) - split_index),
        "decision_threshold": selected["threshold"],
        "selected_features": [feature_label(col) for col in selected["columns"]],
        "candidate_results": selected["all_results"],
        "metrics": selected["metrics"],
        "top_features": top_features,
        "recent_predictions": recent_predictions[
            ["date", "opponent", "home_away", "actual_result", "predicted_win_probability", "predicted_result"]
        ].to_dict(orient="records"),
        "feature_mean": selected["mean"].round(6).to_dict(),
        "feature_std": selected["std"].round(6).to_dict(),
        "bias": round(float(selected["bias"]), 6),
    }
    (RESULTS_DIR / "win_predictor_model.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    features.to_csv(RESULTS_DIR / "features.csv", index=False, encoding="utf-8-sig")
    history = []
    if MODEL_HISTORY_PATH.exists():
        history = json.loads(MODEL_HISTORY_PATH.read_text(encoding="utf-8"))
    history.append(
        {
            "updated_at": payload["updated_at"],
            "model_name_ko": payload["model_name_ko"],
            "accuracy": payload["metrics"]["accuracy"],
            "f1": payload["metrics"]["f1"],
            "train_rows": payload["train_rows"],
            "test_rows": payload["test_rows"],
        }
    )
    MODEL_HISTORY_PATH.write_text(
        json.dumps(history[-30:], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return payload


def pct(value):
    if pd.isna(value):
        return "-"
    return f"{float(value) * 100:.1f}%"


def number(value, digits: int = 1):
    if pd.isna(value):
        return "-"
    return f"{float(value):.{digits}f}"


def render_table(df: pd.DataFrame, columns: list[str], headers: dict[str, str] | None = None, limit: int = 10):
    headers = headers or {}
    if df.empty:
        return "<p class=\"empty\">데이터가 없습니다.</p>"

    safe = df[columns].head(limit).copy()
    rows = ["<table><thead><tr>"]
    rows.extend(f"<th>{escape(headers.get(col, col))}</th>" for col in columns)
    rows.append("</tr></thead><tbody>")
    for _, row in safe.iterrows():
        rows.append("<tr>")
        rows.extend(f"<td>{escape(str(row[col]))}</td>" for col in columns)
        rows.append("</tr>")
    rows.append("</tbody></table>")
    return "".join(rows)


def build_score_trend(games: pd.DataFrame):
    trend = games[games["status"] == "Final"].copy()
    trend["경기일"] = trend["date"].dt.strftime("%Y-%m-%d")
    trend["상대"] = trend["opponent"].map(team_label)
    trend["구장"] = trend["home_away"].map({"H": "홈", "A": "원정"}).fillna(trend["home_away"])
    trend["결과"] = trend["result"].map({"Win": "승", "Loss": "패", "Draw": "무"}).fillna(trend["result"])
    trend["스코어"] = trend["score_team"].astype(int).astype(str) + " - " + trend["score_opp"].astype(int).astype(str)
    trend["득실차"] = (trend["score_team"] - trend["score_opp"]).astype(int)
    return trend.sort_values("date", ascending=False)


def build_opponent_summary(games: pd.DataFrame):
    summary = (
        games[games["status"] == "Final"]
        .groupby("opponent")
        .agg(
            경기=("game_id", "count"),
            승=("result", lambda s: int((s == "Win").sum())),
            패=("result", lambda s: int((s == "Loss").sum())),
            평균득점=("score_team", "mean"),
            평균실점=("score_opp", "mean"),
            평균득실차=("run_diff", "mean"),
        )
        .reset_index()
        .rename(columns={"opponent": "상대팀"})
    )
    if summary.empty:
        return summary
    summary["상대팀"] = summary["상대팀"].map(team_label)
    summary["승률"] = summary["승"] / summary["경기"]
    summary["평균득점"] = summary["평균득점"].map(lambda v: number(v, 2))
    summary["평균실점"] = summary["평균실점"].map(lambda v: number(v, 2))
    summary["평균득실차"] = summary["평균득실차"].map(lambda v: number(v, 2))
    summary["승률"] = summary["승률"].map(pct)
    return summary.sort_values(["승", "경기"], ascending=False)


def build_home_away_summary(games: pd.DataFrame):
    summary = (
        games[games["status"] == "Final"]
        .groupby("home_away")
        .agg(
            경기=("game_id", "count"),
            승=("result", lambda s: int((s == "Win").sum())),
            패=("result", lambda s: int((s == "Loss").sum())),
            평균득점=("score_team", "mean"),
            평균실점=("score_opp", "mean"),
            평균득실차=("run_diff", "mean"),
        )
        .reset_index()
    )
    if summary.empty:
        return summary
    summary["구분"] = summary["home_away"].map({"H": "홈", "A": "원정"}).fillna(summary["home_away"])
    summary["승률"] = (summary["승"] / summary["경기"]).map(pct)
    for col in ["평균득점", "평균실점", "평균득실차"]:
        summary[col] = summary[col].map(lambda v: number(v, 2))
    return summary[["구분", "경기", "승", "패", "승률", "평균득점", "평균실점", "평균득실차"]]


def build_monthly_summary(games: pd.DataFrame):
    monthly = games[games["status"] == "Final"].copy()
    monthly["월"] = monthly["date"].dt.strftime("%Y-%m")
    summary = (
        monthly.groupby("월")
        .agg(
            경기=("game_id", "count"),
            승=("result", lambda s: int((s == "Win").sum())),
            패=("result", lambda s: int((s == "Loss").sum())),
            평균득실차=("run_diff", "mean"),
        )
        .reset_index()
        .sort_values("월")
    )
    if summary.empty:
        return summary
    summary["승률"] = (summary["승"] / summary["경기"]).map(pct)
    summary["평균득실차"] = summary["평균득실차"].map(lambda v: number(v, 2))
    return summary[["월", "경기", "승", "패", "승률", "평균득실차"]]


def build_hitter_leaders(players: pd.DataFrame):
    hitters = players[players["plate_appearances"] > 0].copy()
    if hitters.empty:
        return hitters
    hitters["display_player_name"] = hitters.apply(lambda row: player_label(row["player_name"], row["team"]), axis=1)
    hitters["display_team"] = hitters["team"].map(team_label)
    leaders = hitters.groupby(["display_player_name", "display_team"], as_index=False).agg(
        경기=("game_id", "nunique"),
        타석=("plate_appearances", "sum"),
        타수=("at_bats", "sum"),
        안타=("hits", "sum"),
        홈런=("home_runs", "sum"),
        볼넷=("walks", "sum"),
        삼진=("strikeouts", "sum"),
        도루=("stolen_bases", "sum"),
    )
    leaders["타율"] = leaders["안타"] / leaders["타수"].replace(0, np.nan)
    leaders["출루율"] = (leaders["안타"] + leaders["볼넷"]) / leaders["타석"].replace(0, np.nan)
    leaders["장타율_proxy"] = (leaders["안타"] + leaders["홈런"] * 3) / leaders["타수"].replace(0, np.nan)
    leaders["OPS_proxy"] = leaders["출루율"] + leaders["장타율_proxy"]
    leaders = leaders.rename(columns={"display_player_name": "선수", "display_team": "팀"})
    leaders = leaders.sort_values(["OPS_proxy", "안타"], ascending=False)
    for col in ["타율", "출루율", "장타율_proxy", "OPS_proxy"]:
        leaders[col] = leaders[col].map(lambda v: number(v, 3))
    return leaders


def build_pitcher_leaders(players: pd.DataFrame):
    pitchers = players[players["innings_pitched"] > 0].copy()
    if pitchers.empty:
        return pitchers
    pitchers["display_player_name"] = pitchers.apply(lambda row: player_label(row["player_name"], row["team"]), axis=1)
    pitchers["display_team"] = pitchers["team"].map(team_label)
    leaders = pitchers.groupby(["display_player_name", "display_team"], as_index=False).agg(
        경기=("game_id", "nunique"),
        이닝=("innings_pitched", "sum"),
        투구수=("pitches", "sum"),
        자책=("earned_runs", "sum"),
        탈삼진=("strikeouts_pitched", "sum"),
        볼넷=("walks_allowed", "sum"),
        피안타=("hits_allowed", "sum"),
    )
    leaders["ERA"] = leaders["자책"] * 9 / leaders["이닝"].replace(0, np.nan)
    leaders["WHIP"] = (leaders["볼넷"] + leaders["피안타"]) / leaders["이닝"].replace(0, np.nan)
    leaders["K/9"] = leaders["탈삼진"] * 9 / leaders["이닝"].replace(0, np.nan)
    leaders = leaders.rename(columns={"display_player_name": "선수", "display_team": "팀"})
    leaders = leaders.sort_values(["ERA", "WHIP"], ascending=True)
    for col in ["이닝", "ERA", "WHIP", "K/9"]:
        leaders[col] = leaders[col].map(lambda v: number(v, 2))
    return leaders


def build_model_tables(model_payload: dict):
    top_features = pd.DataFrame(model_payload["top_features"])
    if not top_features.empty:
        top_features = top_features.rename(
            columns={"label": "변수", "coefficient": "계수", "direction": "해석"}
        )
        top_features["계수"] = top_features["계수"].map(lambda v: number(v, 3))

    predictions = pd.DataFrame(model_payload["recent_predictions"])
    if not predictions.empty:
        predictions = predictions.rename(
            columns={
                "date": "경기일",
                "opponent": "상대",
                "home_away": "구장",
                "actual_result": "실제",
                "predicted_win_probability": "승리확률",
                "predicted_result": "예측",
            }
        )
        predictions["승리확률"] = predictions["승리확률"].map(pct)
    candidates = pd.DataFrame(model_payload.get("candidate_results", []))
    if not candidates.empty:
        candidates["features"] = candidates["features"].map(lambda values: ", ".join(values[:4]) + (" ..." if len(values) > 4 else ""))
        candidates["accuracy"] = candidates["metrics"].map(lambda metrics: pct(metrics["accuracy"]))
        candidates["f1"] = candidates["metrics"].map(lambda metrics: number(metrics["f1"], 3))
        candidates["threshold"] = candidates["threshold"].map(lambda value: number(value, 2))
        candidates = candidates.rename(
            columns={"name": "후보 모델", "features": "사용 변수", "accuracy": "정확도", "f1": "F1", "threshold": "판정 기준"}
        )
    return top_features, predictions, candidates


def build_dashboard(games: pd.DataFrame, players: pd.DataFrame, model_payload: dict, start_date: str, end_date: str):
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    games = games.copy()
    games["date"] = pd.to_datetime(games["date"])
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    games["run_diff"] = games["score_team"] - games["score_opp"]

    weekly_games = games[(games["date"] >= start_ts) & (games["date"] <= end_ts)].copy()
    final_games = weekly_games[weekly_games["status"] == "Final"].copy()
    all_final_games = games[games["status"] == "Final"].copy()
    recent_five = all_final_games.tail(5)

    wins = int((final_games["result"] == "Win").sum())
    losses = int((final_games["result"] == "Loss").sum())
    draws = int((final_games["result"] == "Draw").sum())
    games_count = len(final_games)
    win_rate = wins / games_count if games_count else 0
    run_diff = int(final_games["run_diff"].sum()) if games_count else 0
    avg_score = final_games["score_team"].mean() if games_count else np.nan
    avg_allowed = final_games["score_opp"].mean() if games_count else np.nan
    recent_record = (
        f"{int((recent_five['result'] == 'Win').sum())}승 "
        f"{int((recent_five['result'] == 'Loss').sum())}패 "
        f"{int((recent_five['result'] == 'Draw').sum())}무"
    )

    score_trend = build_score_trend(games)
    opponent_summary = build_opponent_summary(games)
    home_away_summary = build_home_away_summary(games)
    monthly_summary = build_monthly_summary(games)
    hitter_leaders = build_hitter_leaders(players)
    pitcher_leaders = build_pitcher_leaders(players)
    top_features, recent_predictions, model_candidates_table = build_model_tables(model_payload)
    metrics = model_payload["metrics"]

    html = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>KBO 팀 성과 대시보드</title>
  <style>
    body {{ font-family: Arial, 'Noto Sans KR', sans-serif; margin: 0; color: #172026; background: #f6f8fa; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 30px 22px 44px; }}
    h1 {{ margin: 0 0 8px; font-size: 30px; }}
    h2 {{ margin: 30px 0 10px; font-size: 20px; }}
    p {{ line-height: 1.55; }}
    .subtitle {{ color: #52616b; margin-top: 0; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(150px, 1fr)); gap: 12px; margin: 18px 0; }}
    .metric {{ background: white; border: 1px solid #d9e2ec; padding: 14px; border-radius: 6px; }}
    .metric span {{ color: #607080; font-size: 13px; }}
    .metric strong {{ display: block; font-size: 25px; margin-top: 6px; }}
    .section {{ background: white; border: 1px solid #d9e2ec; border-radius: 6px; padding: 16px; margin-top: 16px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 10px 0 4px; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid #e6edf3; padding: 9px 8px; text-align: left; white-space: nowrap; }}
    th {{ background: #eef3f7; color: #27343d; }}
    .empty {{ color: #607080; }}
    .note {{ color: #52616b; font-size: 13px; margin-top: 8px; }}
    @media (max-width: 840px) {{ .grid {{ grid-template-columns: repeat(2, 1fr); }} table {{ font-size: 13px; }} }}
  </style>
</head>
<body>
<main>
  <h1>KBO 팀 성과 대시보드</h1>
  <p class="subtitle">최근 업데이트 기간: {escape(start_date)} ~ {escape(end_date)} / DB 누적 경기: {len(all_final_games)}경기</p>

  <div class="grid">
    <div class="metric"><span>주간 전적</span><strong>{wins}승 {losses}패 {draws}무</strong></div>
    <div class="metric"><span>주간 승률</span><strong>{pct(win_rate)}</strong></div>
    <div class="metric"><span>주간 득실차</span><strong>{run_diff:+d}</strong></div>
    <div class="metric"><span>최근 5경기</span><strong>{escape(recent_record)}</strong></div>
    <div class="metric"><span>평균 득점</span><strong>{number(avg_score, 2)}</strong></div>
    <div class="metric"><span>평균 실점</span><strong>{number(avg_allowed, 2)}</strong></div>
    <div class="metric"><span>모델 정확도</span><strong>{pct(metrics["accuracy"])}</strong></div>
    <div class="metric"><span>모델 학습/검증</span><strong>{model_payload["train_rows"]}/{model_payload["test_rows"]}</strong></div>
  </div>

  <div class="section">
    <h2>최근 경기 흐름</h2>
    {render_table(score_trend, ["경기일", "상대", "구장", "결과", "스코어", "득실차"], limit=12)}
  </div>

  <div class="section">
    <h2>상대팀별 성적</h2>
    {render_table(opponent_summary, ["상대팀", "경기", "승", "패", "승률", "평균득점", "평균실점", "평균득실차"], limit=10)}
  </div>

  <div class="section">
    <h2>홈/원정 성적</h2>
    {render_table(home_away_summary, ["구분", "경기", "승", "패", "승률", "평균득점", "평균실점", "평균득실차"], limit=10)}
  </div>

  <div class="section">
    <h2>월별 흐름</h2>
    {render_table(monthly_summary, ["월", "경기", "승", "패", "승률", "평균득실차"], limit=12)}
  </div>

  <div class="section">
    <h2>타자 주요 지표</h2>
    {render_table(hitter_leaders, ["선수", "팀", "경기", "타석", "안타", "홈런", "볼넷", "삼진", "타율", "출루율", "장타율_proxy", "OPS_proxy"], limit=15)}
    <p class="note">장타율과 OPS는 현재 box score mock 스키마에서 계산 가능한 proxy 지표입니다.</p>
  </div>

  <div class="section">
    <h2>투수 주요 지표</h2>
    {render_table(pitcher_leaders, ["선수", "팀", "경기", "이닝", "투구수", "자책", "탈삼진", "볼넷", "피안타", "ERA", "WHIP", "K/9"], limit=15)}
  </div>

  <div class="section">
    <h2>승패 예측 모델</h2>
    <p>{escape(model_payload["model_name_ko"])}로 경기 전 정보 기반 승리 확률을 추정했습니다. 여러 후보 모델을 비교해 검증 정확도가 가장 높은 설정을 자동 선택합니다. 정확도 {pct(metrics["accuracy"])}, 정밀도 {pct(metrics["precision"])}, 재현율 {pct(metrics["recall"])}, F1 {number(metrics["f1"], 3)}입니다.</p>
    <h2>후보 모델 비교</h2>
    {render_table(model_candidates_table, ["후보 모델", "정확도", "F1", "판정 기준", "사용 변수"], limit=8)}
    <h2>선택 모델 주요 변수</h2>
    {render_table(top_features, ["변수", "계수", "해석"], limit=10)}
    <h2>최근 경기 예측 결과</h2>
    {render_table(recent_predictions, ["경기일", "상대", "구장", "실제", "승리확률", "예측"], limit=8)}
  </div>
</main>
</body>
</html>
"""
    (DASHBOARD_DIR / "latest.html").write_text(html, encoding="utf-8")
    (DASHBOARD_DIR / "latest_summary.md").write_text(
        "\n".join(
            [
                "# KBO 팀 성과 대시보드",
                f"- 기간: {start_date} ~ {end_date}",
                f"- 주간 전적: {wins}승 {losses}패 {draws}무",
                f"- 주간 승률: {pct(win_rate)}",
                f"- 주간 득실차: {run_diff:+d}",
                f"- 평균 득점/실점: {number(avg_score, 2)} / {number(avg_allowed, 2)}",
                f"- 모델 정확도: {pct(metrics['accuracy'])}",
                f"- DB 누적 경기: {len(games)}",
                f"- DB 선수 경기 기록: {len(players)}",
            ]
        ),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description="Build Korean dashboard and win prediction outputs from PostgreSQL.")
    parser.add_argument("--start-date", help="YYYY-MM-DD. 생략하면 지난주 월요일")
    parser.add_argument("--end-date", help="YYYY-MM-DD. 생략하면 지난주 일요일")
    parser.add_argument("--reference-date", help="YYYY-MM-DD. 지난주 계산 기준일")
    args = parser.parse_args()

    if args.reference_date and not (args.start_date and args.end_date):
        reference_date = datetime.strptime(args.reference_date, "%Y-%m-%d").date()
        args.start_date, args.end_date = previous_week_window(reference_date)
    elif not args.start_date or not args.end_date:
        args.start_date, args.end_date = previous_week_window()

    games, players = load_tables()
    games_path, _ = persist_exports(games, players)
    model_payload = train_model(games_path)
    build_dashboard(games, players, model_payload, args.start_date, args.end_date)

    print(f"[Success] dashboard/latest.html generated from {len(games)} games and {len(players)} player rows.")


if __name__ == "__main__":
    main()
