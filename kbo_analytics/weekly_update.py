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


def train_model(games_path: Path):
    features = build_features(games_path)
    if len(features) < 4:
        raise ValueError("승패 예측 모델 학습에는 최소 4경기 이상의 Win/Loss 데이터가 필요합니다.")

    x, y = prepare_matrix(features)
    split_index = max(int(len(x) * 0.8), 1)
    split_index = min(split_index, len(x) - 1)
    x_train, x_test = x.iloc[:split_index], x.iloc[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    x_train_scaled, x_test_scaled, mean, std = standardize_train_test(x_train, x_test)
    weights, bias = train_logistic_regression(x_train_scaled.to_numpy(), y_train)
    probability = sigmoid(x_test_scaled.to_numpy() @ weights + bias)
    metrics = evaluate(y_test, probability)
    coefficients = sorted(zip(x.columns, weights), key=lambda item: abs(item[1]), reverse=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": "from_scratch_logistic_regression",
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "metrics": metrics,
        "top_features": [{"feature": name, "coefficient": round(float(value), 6)} for name, value in coefficients[:10]],
        "feature_mean": mean.round(6).to_dict(),
        "feature_std": std.round(6).to_dict(),
        "bias": round(float(bias), 6),
    }
    (RESULTS_DIR / "win_predictor_model.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    features.to_csv(RESULTS_DIR / "features.csv", index=False, encoding="utf-8-sig")
    return payload


def render_table(df: pd.DataFrame, columns: list[str], limit: int = 10):
    if df.empty:
        return "<p>No data.</p>"

    rows = ["<table><thead><tr>"]
    rows.extend(f"<th>{escape(col)}</th>" for col in columns)
    rows.append("</tr></thead><tbody>")
    for _, row in df[columns].head(limit).iterrows():
        rows.append("<tr>")
        rows.extend(f"<td>{escape(str(row[col]))}</td>" for col in columns)
        rows.append("</tr>")
    rows.append("</tbody></table>")
    return "".join(rows)


def build_dashboard(games: pd.DataFrame, players: pd.DataFrame, model_payload: dict, start_date: str, end_date: str):
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    games = games.copy()
    games["date"] = pd.to_datetime(games["date"])
    weekly_games = games[(games["date"] >= start_date) & (games["date"] <= end_date)].copy()
    final_games = weekly_games[weekly_games["status"] == "Final"].copy()

    wins = int((final_games["result"] == "Win").sum())
    losses = int((final_games["result"] == "Loss").sum())
    draws = int((final_games["result"] == "Draw").sum())
    run_diff = int((final_games["score_team"] - final_games["score_opp"]).sum()) if not final_games.empty else 0

    opponent = (
        games[games["status"] == "Final"]
        .groupby("opponent")
        .agg(games=("game_id", "count"), wins=("result", lambda s: int((s == "Win").sum())))
        .reset_index()
    )
    if not opponent.empty:
        opponent["win_rate"] = (opponent["wins"] / opponent["games"]).round(3)
        opponent = opponent.sort_values(["win_rate", "wins"], ascending=False)

    by_home = (
        games[games["status"] == "Final"]
        .groupby("home_away")
        .agg(games=("game_id", "count"), wins=("result", lambda s: int((s == "Win").sum())))
        .reset_index()
    )
    if not by_home.empty:
        by_home["win_rate"] = (by_home["wins"] / by_home["games"]).round(3)

    hitters = players[players["plate_appearances"] > 0].copy()
    if not hitters.empty:
        hitters = hitters.groupby(["player_name", "team"], as_index=False).agg(
            games=("game_id", "nunique"),
            pa=("plate_appearances", "sum"),
            ab=("at_bats", "sum"),
            hits=("hits", "sum"),
            home_runs=("home_runs", "sum"),
            walks=("walks", "sum"),
        )
        hitters["avg"] = (hitters["hits"] / hitters["ab"].replace(0, np.nan)).round(3)
        hitters = hitters.sort_values(["hits", "home_runs"], ascending=False)

    metrics = model_payload["metrics"]
    html = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>KBO Performance Dashboard</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #1f2933; }}
    h1, h2 {{ margin-bottom: 8px; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(120px, 1fr)); gap: 12px; margin: 18px 0; }}
    .metric {{ border: 1px solid #d9e2ec; padding: 14px; border-radius: 6px; }}
    .metric strong {{ display: block; font-size: 24px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 26px; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 8px; text-align: left; }}
    th {{ background: #f0f4f8; }}
  </style>
</head>
<body>
  <h1>KBO Performance Dashboard</h1>
  <p>최근 업데이트 기간: {escape(start_date)} ~ {escape(end_date)}</p>
  <div class="grid">
    <div class="metric"><span>Games</span><strong>{len(final_games)}</strong></div>
    <div class="metric"><span>Record</span><strong>{wins}-{losses}-{draws}</strong></div>
    <div class="metric"><span>Run Diff</span><strong>{run_diff:+d}</strong></div>
    <div class="metric"><span>Model Accuracy</span><strong>{metrics["accuracy"]}</strong></div>
  </div>

  <h2>Recent Games</h2>
  {render_table(final_games, ["date", "opponent", "home_away", "result", "score_team", "score_opp"], 12)}

  <h2>Opponent Performance</h2>
  {render_table(opponent, ["opponent", "games", "wins", "win_rate"], 10)}

  <h2>Home/Away Performance</h2>
  {render_table(by_home, ["home_away", "games", "wins", "win_rate"], 10)}

  <h2>Player Batting Leaders</h2>
  {render_table(hitters, ["player_name", "team", "games", "pa", "hits", "home_runs", "walks", "avg"], 12)}

  <h2>Win Prediction Model</h2>
  <p>Accuracy {metrics["accuracy"]}, Precision {metrics["precision"]}, Recall {metrics["recall"]}, F1 {metrics["f1"]}</p>
</body>
</html>
"""
    (DASHBOARD_DIR / "latest.html").write_text(html, encoding="utf-8")
    (DASHBOARD_DIR / "latest_summary.md").write_text(
        "\n".join(
            [
                "# KBO Performance Dashboard",
                f"- Period: {start_date} ~ {end_date}",
                f"- Record: {wins}-{losses}-{draws}",
                f"- Run differential: {run_diff:+d}",
                f"- Model accuracy: {metrics['accuracy']}",
                f"- Total games in DB: {len(games)}",
                f"- Player game rows in DB: {len(players)}",
            ]
        ),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description="Build dashboard and win prediction outputs from PostgreSQL.")
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
