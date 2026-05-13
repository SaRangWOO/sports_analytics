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

    x_all_scaled = (x - mean) / std
    all_probability = sigmoid(x_all_scaled.to_numpy() @ weights + bias)
    features = features.copy()
    features["predicted_win_probability"] = np.round(all_probability, 3)
    features["predicted_result"] = np.where(all_probability >= 0.5, "승리 예측", "패배 예측")

    coefficients = sorted(zip(x.columns, weights), key=lambda item: abs(item[1]), reverse=True)
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
        "model_name_ko": "로지스틱 회귀 승패 예측 모델",
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "metrics": metrics,
        "top_features": top_features,
        "recent_predictions": recent_predictions[
            ["date", "opponent", "home_away", "actual_result", "predicted_win_probability", "predicted_result"]
        ].to_dict(orient="records"),
        "feature_mean": mean.round(6).to_dict(),
        "feature_std": std.round(6).to_dict(),
        "bias": round(float(bias), 6),
    }
    (RESULTS_DIR / "win_predictor_model.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    features.to_csv(RESULTS_DIR / "features.csv", index=False, encoding="utf-8-sig")
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
    leaders = hitters.groupby(["player_name", "team"], as_index=False).agg(
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
    leaders = leaders.rename(columns={"player_name": "선수", "team": "팀"})
    for col in ["타율", "출루율", "장타율_proxy", "OPS_proxy"]:
        leaders[col] = leaders[col].map(lambda v: number(v, 3))
    return leaders.sort_values(["OPS_proxy", "안타"], ascending=False)


def build_pitcher_leaders(players: pd.DataFrame):
    pitchers = players[players["innings_pitched"] > 0].copy()
    if pitchers.empty:
        return pitchers
    leaders = pitchers.groupby(["player_name", "team"], as_index=False).agg(
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
    leaders = leaders.rename(columns={"player_name": "선수", "team": "팀"})
    for col in ["이닝", "ERA", "WHIP", "K/9"]:
        leaders[col] = leaders[col].map(lambda v: number(v, 2))
    return leaders.sort_values(["ERA", "WHIP"], ascending=True)


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
    return top_features, predictions


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
    top_features, recent_predictions = build_model_tables(model_payload)
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
    <p>로지스틱 회귀 모델로 경기 전 정보 기반 승리 확률을 추정했습니다. 정확도 {pct(metrics["accuracy"])}, 정밀도 {pct(metrics["precision"])}, 재현율 {pct(metrics["recall"])}, F1 {number(metrics["f1"], 3)}입니다.</p>
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
