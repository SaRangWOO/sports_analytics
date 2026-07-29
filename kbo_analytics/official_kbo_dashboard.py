from __future__ import annotations

import argparse
import json
import os
import re
from datetime import date, datetime, timedelta
from html import escape, unescape
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from modeling.feature_engineering import build_features
from modeling.model_training import evaluate_model as run_model_evaluation
from modeling.train_win_predictor import (
    prepare_matrix,
    sigmoid,
    standardize_train_test,
    train_logistic_regression,
)
from run_model.run_model_dashboard import DEFAULT_RESULTS as RUN_MODEL_RESULTS
from run_model.run_model_dashboard import render_prediction_board_embedded
from run_model.run_prediction_model import DEFAULT_INPUT as RUN_MODEL_INPUT
from run_model.run_prediction_model import DEFAULT_SCHEDULE_INPUT as RUN_MODEL_SCHEDULE_INPUT
from run_model.run_prediction_model import run_pipeline as run_expected_runs_pipeline


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "official"
DASHBOARD_DIR = BASE_DIR / "dashboard"
RESULTS_DIR = BASE_DIR / "modeling" / "results"
PUBLIC_DIR = BASE_DIR.parent / "docs"
DB_URL = os.getenv("DB_URL", "postgresql://tera:tera@localhost:5432/baseball")
KBO_BASE = "https://www.koreabaseball.com"
HEADERS = {"User-Agent": "Mozilla/5.0"}
TEAM_CODES = {
    "KT": "KT",
    "삼성": "SS",
    "LG": "LG",
    "SSG": "SK",
    "두산": "OB",
    "한화": "HH",
    "KIA": "HT",
    "NC": "NC",
    "롯데": "LT",
    "키움": "WO",
}
TEAM_ALIASES = {
    "Samsung": "삼성",
    "Lotte": "롯데",
    "Doosan": "두산",
    "Hanwha": "한화",
    "Kiwoom": "키움",
}
TEAM_PAGE_SLUGS = {
    "KT": "kt",
    "삼성": "samsung",
    "LG": "lg",
    "SSG": "ssg",
    "두산": "doosan",
    "한화": "hanwha",
    "KIA": "kia",
    "NC": "nc",
    "롯데": "lotte",
    "키움": "kiwoom",
}


def previous_sunday(reference_date: date):
    this_monday = reference_date - timedelta(days=reference_date.weekday())
    return this_monday - timedelta(days=1)


def clean_html(value: str):
    return unescape(re.sub(r"<.*?>", "", value or "")).strip()


def extract_cells(row_html: str):
    return [clean_html(cell) for cell in re.findall(r"<td[^>]*>(.*?)</td>", row_html, re.S)]


def fetch_team_standings():
    html = requests.get(f"{KBO_BASE}/Record/TeamRank/TeamRank.aspx", headers=HEADERS, timeout=30).text
    tables = re.findall(r"<tbody>(.*?)</tbody>", html, re.S)
    standings = []
    for row_html in re.findall(r"<tr>(.*?)</tr>", tables[0], re.S):
        cells = extract_cells(row_html)
        if len(cells) < 12:
            continue
        standings.append(
            {
                "순위": int(cells[0]),
                "팀": cells[1],
                "경기": int(cells[2]),
                "승": int(cells[3]),
                "패": int(cells[4]),
                "무": int(cells[5]),
                "승률": cells[6],
                "게임차": cells[7],
                "최근10경기": cells[8],
                "연속": cells[9],
                "홈": cells[10],
                "방문": cells[11],
            }
        )

    vs_rows = []
    if len(tables) > 1:
        for row_html in re.findall(r"<tr>(.*?)</tr>", tables[1], re.S):
            cells = extract_cells(row_html)
            if len(cells) >= 12:
                team = cells[0]
                for opponent, record in zip([row["팀"] for row in standings] + ["합계"], cells[1:]):
                    if opponent != "합계" and opponent != team and record != "■":
                        vs_rows.append({"팀": team, "상대": opponent, "전적": record})
    return pd.DataFrame(standings), pd.DataFrame(vs_rows)


def fetch_schedule_month(year: int, month: int):
    session = requests.Session()
    url = f"{KBO_BASE}/ws/Schedule.asmx/GetScheduleList"
    response = session.post(
        url,
        headers={**HEADERS, "Referer": f"{KBO_BASE}/Schedule/Schedule.aspx", "X-Requested-With": "XMLHttpRequest"},
        data={"leId": 1, "srIdList": "0,9,6", "seasonId": str(year), "gameMonth": f"{month:02d}", "teamId": ""},
        timeout=30,
    )
    response.raise_for_status()
    rows = response.json().get("rows", [])
    parsed = []
    current_day = None

    for raw_row in rows:
        cells = raw_row.get("row", [])
        if not cells:
            continue
        texts = [cell.get("Text") or "" for cell in cells]
        if cells[0].get("Class") == "day":
            current_day = clean_html(texts.pop(0))
        if not current_day or len(texts) < 2:
            continue

        date_match = re.match(r"(\d{2})\.(\d{2})", current_day)
        if not date_match:
            continue
        game_date = f"{year}-{date_match.group(1)}-{date_match.group(2)}"
        play_html = next((text for text in texts if 'class="win"' in text or "<em>" in text), "")
        teams = [clean_html(item) for item in re.findall(r"<span[^>]*>(.*?)</span>", play_html, re.S)]
        scores = [int(score) for score in re.findall(r'class="(?:win|lose)">(\d+)</span>', play_html)]
        if len(teams) < 2:
            continue
        away_team, home_team = teams[0], teams[-1]
        away_score = scores[0] if len(scores) >= 2 else np.nan
        home_score = scores[1] if len(scores) >= 2 else np.nan
        status = "Final" if len(scores) >= 2 else "Scheduled"
        game_id_match = re.search(r"gameId=([A-Za-z0-9]+)", " ".join(texts))
        game_id = game_id_match.group(1) if game_id_match else f"{game_date}_{away_team}_{home_team}"
        ballpark = clean_html(texts[-2]) if len(texts) >= 2 else ""

        for team, opponent, home_away, score_for, score_against in [
            (away_team, home_team, "A", away_score, home_score),
            (home_team, away_team, "H", home_score, away_score),
        ]:
            if status == "Final":
                if score_for > score_against:
                    result = "Win"
                elif score_for < score_against:
                    result = "Loss"
                else:
                    result = "Draw"
            else:
                result = ""
            parsed.append(
                {
                    "game_id": f"{game_id}_{team}",
                    "date": game_date,
                    "team": team,
                    "opponent": opponent,
                    "home_away": home_away,
                    "status": status,
                    "result": result,
                    "score_team": score_for,
                    "score_opp": score_against,
                    "ballpark": ballpark,
                }
            )
    return parsed


def fetch_schedule(year: int, through_month: int, start_month: int = 3):
    rows = []
    for month in range(start_month, through_month + 1):
        rows.extend(fetch_schedule_month(year, month))
    schedule = pd.DataFrame(rows)
    if schedule.empty:
        return schedule

    for game_date_text, date_rows in schedule.groupby("date", sort=False):
        if not date_rows["status"].eq("Scheduled").any():
            continue
        game_date = datetime.strptime(str(game_date_text), "%Y-%m-%d").date()
        game_list = fetch_kbo_game_list(game_date)
        game_ids = {
            (str(game.get("AWAY_NM", "")).strip(), str(game.get("HOME_NM", "")).strip()): str(game.get("G_ID", "")).strip()
            for game in game_list
            if str(game.get("G_ID", "")).strip()
        }
        for (away_team, home_team), game_id in game_ids.items():
            mask = (
                schedule.index.isin(date_rows.index)
                & (
                    ((schedule["team"] == away_team) & (schedule["opponent"] == home_team))
                    | ((schedule["team"] == home_team) & (schedule["opponent"] == away_team))
                )
            )
            schedule.loc[mask, "game_id"] = schedule.loc[mask, "team"].map(lambda team: f"{game_id}_{team}")
    return schedule


def fetch_training_schedule(start_year: int, reference_date: date):
    seasons = []
    for year in range(start_year, reference_date.year + 1):
        through_month = reference_date.month if year == reference_date.year else 11
        season_games = fetch_schedule(year, through_month)
        if not season_games.empty:
            seasons.append(season_games)
    return pd.concat(seasons, ignore_index=True) if seasons else pd.DataFrame()


def hidden_fields(html: str):
    fields = {}
    for match in re.finditer(r'<input type="hidden" name="([^"]+)"[^>]*>', html):
        value_match = re.search(r'value="([^"]*)"', match.group(0))
        fields[match.group(1)] = unescape(value_match.group(1)) if value_match else ""
    return fields


def fetch_team_player_page(session: requests.Session, path: str, team_code: str):
    url = f"{KBO_BASE}/Record/Player/{path}"
    html = session.get(url, headers=HEADERS, timeout=30).text
    payload = hidden_fields(html)
    payload.update(
        {
            "__EVENTTARGET": "ctl00$ctl00$ctl00$cphContents$cphContents$cphContents$ddlTeam$ddlTeam",
            "__EVENTARGUMENT": "",
            "ctl00$ctl00$ctl00$cphContents$cphContents$cphContents$ddlSeason$ddlSeason": "2026",
            "ctl00$ctl00$ctl00$cphContents$cphContents$cphContents$ddlSeries$ddlSeries": "0",
            "ctl00$ctl00$ctl00$cphContents$cphContents$cphContents$ddlTeam$ddlTeam": team_code,
            "ctl00$ctl00$ctl00$cphContents$cphContents$cphContents$ddlPos$ddlPos": "",
            "ctl00$ctl00$ctl00$cphContents$cphContents$cphContents$ddlSituation$ddlSituation": "",
            "ctl00$ctl00$ctl00$cphContents$cphContents$cphContents$ddlSituationDetail$ddlSituationDetail": "",
        }
    )
    return session.post(url, headers=HEADERS, data=payload, timeout=30).text


def parse_player_table(html: str, columns: list[str]):
    tbody = re.search(r"<tbody>(.*?)</tbody>", html, re.S)
    if not tbody:
        return []
    rows = []
    for row_html in re.findall(r"<tr>(.*?)</tr>", tbody.group(1), re.S):
        cells = extract_cells(row_html)
        if len(cells) == len(columns):
            rows.append(dict(zip(columns, cells)))
    return rows


def fetch_player_stats():
    hitter_basic_cols = ["순위", "선수", "팀", "타율", "경기", "타석", "타수", "득점", "안타", "2루타", "3루타", "홈런", "루타", "타점", "희생번트", "희생플라이"]
    hitter_adv_cols = ["순위", "선수", "팀", "타율", "볼넷", "고의4구", "사구", "삼진", "병살", "장타율", "출루율", "OPS", "멀티히트", "득점권타율", "대타타율"]
    pitcher_cols = ["순위", "선수", "팀", "ERA", "경기", "승", "패", "세이브", "홀드", "승률", "이닝", "피안타", "피홈런", "볼넷", "사구", "탈삼진", "실점", "자책", "WHIP"]
    session = requests.Session()
    hitters = []
    pitchers = []
    for team, code in TEAM_CODES.items():
        basic = pd.DataFrame(parse_player_table(fetch_team_player_page(session, "HitterBasic/Basic1.aspx", code), hitter_basic_cols))
        advanced = pd.DataFrame(parse_player_table(fetch_team_player_page(session, "HitterBasic/Basic2.aspx", code), hitter_adv_cols))
        if not basic.empty and not advanced.empty:
            merged = basic.merge(advanced[["선수", "팀", "볼넷", "삼진", "장타율", "출루율", "OPS", "득점권타율"]], on=["선수", "팀"], how="left")
            hitters.append(merged)
        pitcher = pd.DataFrame(parse_player_table(fetch_team_player_page(session, "PitcherBasic/Basic1.aspx", code), pitcher_cols))
        if not pitcher.empty:
            pitchers.append(pitcher)
    return pd.concat(hitters, ignore_index=True), pd.concat(pitchers, ignore_index=True)


def split_registered_people(value):
    text = str(value or "").replace("\n", "").strip()
    if not text or text == "nan":
        return ""
    people = re.findall(r"[^()]+?\(\d+\)", text)
    if people:
        return ", ".join(person.strip() for person in people)
    return text


def fetch_registered_rosters():
    html = requests.get(f"{KBO_BASE}/Player/RegisterAll.aspx", headers=HEADERS, timeout=30).text
    roster_rows = []
    for table in pd.read_html(StringIO(html)):
        if "구단" not in table.columns or table.empty:
            continue
        row = table.iloc[0]
        team = str(row.get("구단", "")).split()[0]
        if team not in TEAM_CODES:
            continue
        output = {"팀": team}
        for label in ["감독", "코치", "투수", "포수", "내야수", "외야수"]:
            matching = [column for column in table.columns if str(column).startswith(label)]
            output[label] = split_registered_people(row.get(matching[0], "")) if matching else ""
        roster_rows.append(output)
    return pd.DataFrame(roster_rows)


def export_sources(standings, vs_table, games, hitters, pitchers, rosters):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    standings.to_csv(DATA_DIR / "team_standings.csv", index=False, encoding="utf-8-sig")
    vs_table.to_csv(DATA_DIR / "team_vs_team.csv", index=False, encoding="utf-8-sig")
    games.to_csv(DATA_DIR / "game_results.csv", index=False, encoding="utf-8-sig")
    hitters.to_csv(DATA_DIR / "hitter_stats.csv", index=False, encoding="utf-8-sig")
    pitchers.to_csv(DATA_DIR / "pitcher_stats.csv", index=False, encoding="utf-8-sig")
    rosters.to_csv(DATA_DIR / "registered_rosters.csv", index=False, encoding="utf-8-sig")


def load_official_tables_to_db(standings, vs_table, games, hitters, pitchers, rosters):
    from sqlalchemy import create_engine

    engine = create_engine(DB_URL)
    tables = {
        "game_results": games,
        "official_team_standings": standings,
        "official_team_vs_team": vs_table,
        "official_hitter_stats": hitters,
        "official_pitcher_stats": pitchers,
        "official_registered_rosters": rosters,
    }
    try:
        with engine.begin() as connection:
            for table_name, dataframe in tables.items():
                dataframe.to_sql(table_name, connection, if_exists="replace", index=False)
    except Exception as exc:
        print(f"[Warn] PostgreSQL 적재를 건너뜁니다: {exc}")


def align_prediction_matrix(features: pd.DataFrame, feature_columns: list[str], mean: pd.Series, std: pd.Series):
    x, _ = prepare_matrix(features)
    x = x.reindex(columns=feature_columns, fill_value=0)
    return (x - mean) / std.replace(0, 1)


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
    metrics = [
        {"구간": "전체 경기", "경기 수": int(len(y_true)), "적중률": round(float(correct.mean()), 3)}
    ]
    for threshold in [0.55, 0.58, 0.60]:
        mask = confidence >= threshold
        metrics.append(
            {
                "구간": f"{round(threshold * 100)}% 이상 확신 경기",
                "경기 수": int(mask.sum()),
                "적중률": round(float(correct[mask].mean()), 3) if mask.any() else "-",
            }
        )
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    brier = np.mean((probability - y_true) ** 2)
    log_loss = -np.mean(y_true * np.log(clipped) + (1 - y_true) * np.log(1 - clipped))
    metrics.append({"구간": "Brier Score", "경기 수": "-", "적중률": round(float(brier), 3)})
    metrics.append({"구간": "Log Loss", "경기 수": "-", "적중률": round(float(log_loss), 3)})
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


def build_game_level_frame(features: pd.DataFrame):
    rows = []
    working = features.copy()
    working["actual_game_id"] = working["game_id"].astype(str).str.rsplit("_", n=1).str[0]
    for game_id, game_rows in working.groupby("actual_game_id", sort=False):
        if len(game_rows) != 2:
            continue
        home_rows = game_rows[game_rows["is_home"] == 1]
        away_rows = game_rows[game_rows["is_home"] == 0]
        if home_rows.empty or away_rows.empty:
            continue
        home = home_rows.iloc[0]
        away = away_rows.iloc[0]
        rows.append(
            {
                "game_id": game_id,
                "date": pd.to_datetime(home["date"]).strftime("%Y-%m-%d"),
                "home_team": home["team"],
                "away_team": away["team"],
                "target_home_win": np.nan if pd.isna(home["target_win"]) else int(home["target_win"]),
                "home_recent_10_win_rate": round(float(home["recent_10_win_rate"]), 4),
                "away_recent_10_win_rate": round(float(away["recent_10_win_rate"]), 4),
                "recent_10_win_rate_gap": round(float(home["recent_10_win_rate"] - away["recent_10_win_rate"]), 4),
                "season_win_rate_gap": round(float(home["season_win_rate_prior"] - away["season_win_rate_prior"]), 4),
                "season_avg_run_diff_gap": round(float(home["season_avg_run_diff_prior"] - away["season_avg_run_diff_prior"]), 4),
                "recent_run_diff_10_gap": round(float(home["avg_run_diff_last_10"] - away["avg_run_diff_last_10"]), 4),
                "home_venue_win_rate": round(float(home["venue_win_rate_prior"]), 4),
                "away_venue_win_rate": round(float(away["venue_win_rate_prior"]), 4),
                "venue_win_rate_gap": round(float(home["venue_win_rate_prior"] - away["venue_win_rate_prior"]), 4),
                "home_games_last_7_days": int(home["games_last_7_days"]),
                "away_games_last_7_days": int(away["games_last_7_days"]),
            }
        )
    return pd.DataFrame(rows)


def export_game_level_dataset(features: pd.DataFrame, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    build_game_level_frame(features).to_csv(output_path, index=False, encoding="utf-8-sig")


def prepare_game_level_matrix(frame: pd.DataFrame):
    x = frame.drop(columns=["date", "game_id", "target_home_win"])
    x = pd.get_dummies(x, columns=["home_team", "away_team"], drop_first=False, dtype=float)
    y = frame["target_home_win"].to_numpy(dtype=float)
    return x, y


def align_game_level_matrix(frame: pd.DataFrame, feature_columns: list[str], mean: pd.Series, std: pd.Series):
    x, _ = prepare_game_level_matrix(frame)
    x = x.reindex(columns=feature_columns, fill_value=0)
    return (x - mean) / std.replace(0, 1)


def pick_better_model(current: dict | None, candidate: dict):
    if current is None:
        return candidate
    if candidate["accuracy"] > current["accuracy"] + 0.005:
        return candidate
    if abs(candidate["accuracy"] - current["accuracy"]) <= 0.005 and candidate["score"]["Brier Score"] < current["score"]["Brier Score"]:
        return candidate
    return current


def prediction_reason(row: pd.Series, predicted_team: str | None = None):
    predicted_team = predicted_team or row.get("team")
    team_perspective = predicted_team == row.get("team")
    reasons = []
    season_gap = row.get("season_win_rate_gap", 0) if team_perspective else -row.get("season_win_rate_gap", 0)
    recent_gap = row.get("recent_5_win_rate_gap", 0) if team_perspective else -row.get("recent_5_win_rate_gap", 0)
    recent_10_gap = row.get("recent_10_win_rate_gap", 0) if team_perspective else -row.get("recent_10_win_rate_gap", 0)
    h2h_gap = row.get("head_to_head_win_rate_gap", 0) if team_perspective else -row.get("head_to_head_win_rate_gap", 0)
    venue_gap = row.get("venue_win_rate_gap", 0) if team_perspective else -row.get("venue_win_rate_gap", 0)
    season_run_gap = row.get("season_avg_run_diff_gap", 0) if team_perspective else -row.get("season_avg_run_diff_gap", 0)
    own_run_diff = row.get("avg_run_diff_last_5", 0) if team_perspective else row.get("opponent_avg_run_diff_last_5", 0)
    opponent_run_diff = row.get("opponent_avg_run_diff_last_5", 0) if team_perspective else row.get("avg_run_diff_last_5", 0)
    is_home_side = row.get("is_home", 0) == 1 if team_perspective else row.get("is_home", 0) == 0
    games_last_7 = row.get("games_last_7_days", 0)

    if season_gap > 0.03:
        reasons.append(f"{predicted_team} 시즌 누적 승률 우위")

    if recent_gap > 0.15:
        reasons.append(f"{predicted_team} 최근 5경기 흐름 우위")

    if recent_10_gap > 0.12:
        reasons.append(f"{predicted_team} 최근 10경기 흐름 우위")

    if h2h_gap > 0.2:
        reasons.append(f"{predicted_team} 시즌 상대전적 우위")

    if venue_gap > 0.15:
        reasons.append(f"{predicted_team} 홈/원정 성향 우위")

    if season_run_gap > 0.5:
        reasons.append(f"{predicted_team} 시즌 득실차 우위")

    if own_run_diff > opponent_run_diff + 0.8:
        reasons.append(f"{predicted_team} 최근 득실차 우위")

    if is_home_side:
        reasons.append(f"{predicted_team} 홈 경기")
    if team_perspective and games_last_7 <= 4:
        reasons.append(f"{predicted_team} 최근 일정 부담 낮음")

    return ", ".join(reasons[:2]) if reasons else "양 팀 지표가 비슷해 기본 전력과 최근 흐름을 종합"


def game_prediction_reason(row: pd.Series, predicted_team: str):
    home_perspective = predicted_team == row.get("home_team")
    reasons = []
    recent_gap = row.get("recent_10_win_rate_gap", 0) if home_perspective else -row.get("recent_10_win_rate_gap", 0)
    season_gap = row.get("season_win_rate_gap", 0) if home_perspective else -row.get("season_win_rate_gap", 0)
    run_gap = row.get("season_avg_run_diff_gap", 0) if home_perspective else -row.get("season_avg_run_diff_gap", 0)
    recent_run_gap = row.get("recent_run_diff_10_gap", 0) if home_perspective else -row.get("recent_run_diff_10_gap", 0)
    venue_gap = row.get("venue_win_rate_gap", 0) if home_perspective else -row.get("venue_win_rate_gap", 0)

    if recent_run_gap > 0.8:
        reasons.append(f"{predicted_team} 최근 득실차 우위")
    if recent_gap > 0.12:
        reasons.append(f"{predicted_team} 최근 10경기 흐름 우위")
    if season_gap > 0.03:
        reasons.append(f"{predicted_team} 시즌 승률 우위")
    if run_gap > 0.5:
        reasons.append(f"{predicted_team} 시즌 득실차 우위")
    if venue_gap > 0.15:
        reasons.append(f"{predicted_team} 홈/원정 성향 우위")
    if home_perspective:
        reasons.append(f"{predicted_team} 홈 경기")

    return ", ".join(reasons[:2]) if reasons else "양 팀 지표가 비슷해 기본 전력과 최근 흐름을 종합"


def prediction_tier(confidence: float):
    if confidence < 0.53:
        return {"우세": "박빙", "신뢰도": "낮음", "판단": "참고만"}
    if confidence < 0.56:
        return {"우세": "박빙 우세", "신뢰도": "낮음", "판단": "참고"}
    if confidence < 0.60:
        return {"우세": "약우세", "신뢰도": "보통", "판단": "예측 가능"}
    return {"우세": "우세", "신뢰도": "주의", "판단": "과신 주의"}


def parse_innings(value):
    if pd.isna(value):
        return 0.0
    text = str(value).strip()
    if not text or text == "-":
        return 0.0
    total = 0.0
    for part in text.split():
        if "/" in part:
            numerator, denominator = part.split("/", 1)
            total += float(numerator) / float(denominator)
        else:
            total += float(part)
    return total


def to_float(value, default=0.0):
    try:
        text = str(value).replace(",", "").strip()
        if text in {"", "-"}:
            return default
        return float(text)
    except (TypeError, ValueError):
        return default


def fetch_confirmed_pitcher_ids(game_date: date, game_id: str):
    url = f"{KBO_BASE}/Schedule/GameCenter/Main.aspx"
    params = {
        "gameDate": game_date.strftime("%Y%m%d"),
        "gameId": game_id,
        "section": "START_PIT",
    }
    try:
        html = requests.get(url, params=params, headers=HEADERS, timeout=20).text
    except requests.RequestException:
        return {}
    attrs = {}
    for key in ["away_p_id", "home_p_id"]:
        match = re.search(rf'{key}=["\']([^"\']+)["\']', html)
        if match and match.group(1).strip():
            attrs[key] = match.group(1).strip()
    return attrs


def fetch_kbo_game_list(game_date: date):
    url = f"{KBO_BASE}/ws/Main.asmx/GetKboGameList"
    try:
        response = requests.post(
            url,
            headers={**HEADERS, "Referer": f"{KBO_BASE}/Schedule/GameCenter/Main.aspx", "X-Requested-With": "XMLHttpRequest"},
            data={"leId": 1, "srId": 0, "date": game_date.strftime("%Y%m%d")},
            timeout=20,
        )
        response.raise_for_status()
        return response.json().get("game", [])
    except (requests.RequestException, ValueError):
        return []


def load_manual_confirmed_starters(prediction_date: date):
    path = DATA_DIR.parent / "manual" / "confirmed_starters.csv"
    if not path.exists():
        return {}
    manual = pd.read_csv(path)
    if manual.empty or not {"date", "team", "starter_name"}.issubset(manual.columns):
        return {}
    manual["date_obj"] = pd.to_datetime(manual["date"]).dt.date
    rows = manual[manual["date_obj"] == prediction_date]
    return {
        str(row["team"]): {
            "starter_name": str(row["starter_name"]).strip(),
            "source": str(row.get("source", "manual")).strip() or "manual",
            "confirmed_at": str(row.get("confirmed_at", "")).strip(),
        }
        for _, row in rows.iterrows()
        if str(row.get("starter_name", "")).strip()
    }


def build_confirmed_starter_source(games: pd.DataFrame, prediction_date: date, reference_datetime: datetime):
    source_games = fetch_kbo_game_list(prediction_date)
    manual = load_manual_confirmed_starters(prediction_date)
    raw_games = []
    confirmed = {}

    for game in source_games:
        game_id = str(game.get("G_ID", ""))
        away_team = str(game.get("AWAY_NM", "")).strip()
        home_team = str(game.get("HOME_NM", "")).strip()
        away_starter = str(game.get("T_PIT_P_NM", "") or "").strip()
        home_starter = str(game.get("B_PIT_P_NM", "") or "").strip()
        start_pit_ck = str(game.get("START_PIT_CK", "")).strip()
        parse_status = "success" if start_pit_ck == "1" and (away_starter or home_starter) else "missing_starter"
        raw_games.append(
            {
                "game": f"{away_team} vs {home_team}",
                "game_id": game_id,
                "source": "KBO GetKboGameList",
                "raw_away_starter": away_starter,
                "raw_home_starter": home_starter,
                "start_pit_ck": start_pit_ck,
                "parse_status": parse_status,
            }
        )
        if parse_status == "success":
            if away_team and away_starter:
                confirmed[away_team] = {
                    "starter_name": away_starter,
                    "source": "confirmed",
                    "confirmed_at": reference_datetime.strftime("%Y-%m-%d %H:%M"),
                    "game_id": game_id,
                }
            if home_team and home_starter:
                confirmed[home_team] = {
                    "starter_name": home_starter,
                    "source": "confirmed",
                    "confirmed_at": reference_datetime.strftime("%Y-%m-%d %H:%M"),
                    "game_id": game_id,
                }

    for team, values in manual.items():
        confirmed[team] = {
            "starter_name": values["starter_name"],
            "source": "manual",
            "confirmed_at": values.get("confirmed_at") or reference_datetime.strftime("%Y-%m-%d %H:%M"),
            "game_id": "",
        }

    payload = {
        "reference_date": prediction_date.isoformat(),
        "fetched_at": reference_datetime.strftime("%Y-%m-%d %H:%M:%S"),
        "source": "KBO GetKboGameList",
        "manual_override_count": len(manual),
        "games": raw_games,
    }
    for path in [
        BASE_DIR / "logs" / f"starter_raw_source_{prediction_date.isoformat()}.json",
        BASE_DIR / "logs" / "starter_raw_source_latest.json",
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return confirmed


def pitcher_record_for_starter(team_pitchers: pd.DataFrame, starter_name: str):
    if not starter_name:
        return None
    normalized = team_pitchers.copy()
    normalized["선수_norm"] = normalized["선수"].astype(str).str.strip()
    matched = normalized[normalized["선수_norm"] == starter_name.strip()]
    if matched.empty:
        matched = normalized[normalized["선수_norm"].str.replace(" ", "", regex=False) == starter_name.strip().replace(" ", "")]
    return None if matched.empty else matched.iloc[0]


def starter_source_for_game(games: pd.DataFrame, team: str, prediction_date: date, reference_datetime: datetime, update_stage: str):
    if update_stage != "pregame":
        return "estimated", "", 0.5
    scheduled = games[
        (games["team"] == team)
        & (pd.to_datetime(games["date"]).dt.date == prediction_date)
        & (games["status"] == "Scheduled")
    ]
    if scheduled.empty:
        return "estimated", "", 0.5
    row = scheduled.iloc[0]
    game_base_id = str(row["game_id"]).rsplit("_", 1)[0]
    confirmed = fetch_confirmed_pitcher_ids(prediction_date, game_base_id)
    side_key = "home_p_id" if row["home_away"] == "H" else "away_p_id"
    if confirmed.get(side_key):
        return "confirmed", reference_datetime.strftime("%Y-%m-%d %H:%M"), 1.0
    return "estimated", "", 0.5


def build_pitching_context(games: pd.DataFrame, pitchers: pd.DataFrame, prediction_date: date, reference_datetime: datetime | None = None, update_stage: str = "morning"):
    reference_datetime = reference_datetime or datetime.combine(prediction_date, datetime.min.time())
    completed = games[games["status"] == "Final"].copy()
    completed["date"] = pd.to_datetime(completed["date"]).dt.date
    confirmed_starters = build_confirmed_starter_source(games, prediction_date, reference_datetime) if update_stage == "pregame" else {}
    context = {}
    for team, team_pitchers in pitchers.groupby("팀"):
        records = team_pitchers.copy()
        records["이닝_float"] = records["이닝"].apply(parse_innings)
        records["경기_float"] = records["경기"].apply(to_float).replace(0, np.nan)
        records["ERA_float"] = records["ERA"].apply(lambda value: to_float(value, 99.0))
        records["WHIP_float"] = records["WHIP"].apply(lambda value: to_float(value, 9.99))
        records["이닝_per_game"] = (records["이닝_float"] / records["경기_float"]).fillna(0)
        starters = records[
            (records["이닝_float"] >= 20) | (records["이닝_per_game"] >= 3.0)
        ].sort_values(["이닝_float", "이닝_per_game"], ascending=False).head(5)
        if starters.empty:
            starters = records.sort_values("이닝_float", ascending=False).head(5)

        team_completed = completed[completed["team"] == team].sort_values("date")
        rotation_index = int(len(team_completed) % max(len(starters), 1))
        starter = starters.iloc[rotation_index] if not starters.empty else None

        recent = team_completed[
            (team_completed["date"] < prediction_date)
            & (team_completed["date"] >= prediction_date - timedelta(days=3))
        ]
        played_yesterday = not team_completed[team_completed["date"] == prediction_date - timedelta(days=1)].empty
        recent_games = len(recent)
        if recent_games >= 3 or (recent_games >= 2 and played_yesterday):
            fatigue = "높음"
        elif recent_games >= 2 or played_yesterday:
            fatigue = "보통"
        else:
            fatigue = "낮음"

        confirmed_starter = confirmed_starters.get(team)
        if confirmed_starter:
            starter_name = confirmed_starter["starter_name"]
            official_record = pitcher_record_for_starter(records, starter_name)
            if official_record is not None:
                starter_era = official_record["ERA"]
                starter_whip = official_record["WHIP"]
                starter_text = f"{starter_name} · ERA {starter_era} · WHIP {starter_whip}"
            else:
                starter_era = ""
                starter_whip = ""
                starter_text = f"{starter_name} · 기록 매칭 대기"
            source = confirmed_starter["source"]
            confirmed_at = confirmed_starter["confirmed_at"]
            quality = 1.0
        elif starter is not None:
            starter_text = f"{starter['선수']} · ERA {starter['ERA']} · WHIP {starter['WHIP']}"
            starter_name = starter["선수"]
            starter_era = starter["ERA"]
            starter_whip = starter["WHIP"]
            source, confirmed_at, quality = "estimated", "", 0.5
        else:
            starter_text = "추정 불가"
            starter_name = "-"
            starter_era = "-"
            starter_whip = "-"
            source, confirmed_at, quality = "unknown", "", 0.0
        source_label = "확정 선발" if source in {"confirmed", "manual"} else "예상 선발"
        context[team] = {
            "투수 표시": f"{source_label}: {starter_text}",
            "선발명": starter_name,
            "starter_source": source,
            "starter_confirmed_at": confirmed_at,
            "starter_info_quality": quality,
            "ERA": starter_era,
            "WHIP": starter_whip,
            "불펜 피로": fatigue,
            "최근3일 경기": int(recent_games),
            "주의": "확정 선발은 GameCenter 기준, 미확인 시 누적 기록과 로테이션 순서로 추정",
        }
    return context


def export_pitching_context(context: dict, output_path: Path, prediction_date: date):
    rows = []
    for team, values in sorted(context.items()):
        rows.append(
            {
                "date": prediction_date.isoformat(),
                "team": team,
                "starter_name": values.get("선발명", "-"),
                "starter_source": values.get("starter_source", "estimated"),
                "starter_confirmed_at": values.get("starter_confirmed_at", ""),
                "starter_info_quality": values.get("starter_info_quality", 0.5),
                "starter_era": values.get("ERA", "-"),
                "starter_whip": values.get("WHIP", "-"),
                "bullpen_fatigue": values.get("불펜 피로", "-"),
                "recent_3day_games": values.get("최근3일 경기", 0),
                "note": values.get("주의", ""),
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")


def _pitching_snapshot_source(starter_source: str):
    if starter_source == "manual":
        return "manual"
    if starter_source == "confirmed":
        return "KBO GameCenter"
    if starter_source == "estimated":
        return "estimated rotation"
    return "unknown"


def append_pitching_daily_snapshot(games: pd.DataFrame, context: dict, output_path: Path, results_dir: Path, prediction_date: date, reference_datetime: datetime):
    scheduled = games[
        (pd.to_datetime(games["date"]).dt.date == prediction_date)
        & (games["status"] == "Scheduled")
    ].copy()
    rows = []
    for _, game in scheduled.iterrows():
        team = str(game["team"])
        values = context.get(team, {})
        starter_source = values.get("starter_source", "unknown")
        rows.append(
            {
                "snapshot_date": reference_datetime.date().isoformat(),
                "snapshot_time": reference_datetime.strftime("%Y-%m-%d %H:%M:%S"),
                "reference_date": prediction_date.isoformat(),
                "team": team,
                "starter_name": values.get("선발명", "-"),
                "starter_source": starter_source,
                "starter_info_quality": values.get("starter_info_quality", 0.0),
                "starter_era": values.get("ERA", "-"),
                "starter_whip": values.get("WHIP", "-"),
                "bullpen_fatigue_label": values.get("불펜 피로", "-"),
                "recent_3day_games": values.get("최근3일 경기", 0),
                "scheduled_game_id": str(game.get("game_id", "")),
                "opponent": game.get("opponent", ""),
                "home_away": game.get("home_away", ""),
                "data_source": _pitching_snapshot_source(starter_source),
                "note": "경기 전 수집 스냅샷",
            }
        )

    columns = [
        "snapshot_date",
        "snapshot_time",
        "reference_date",
        "team",
        "starter_name",
        "starter_source",
        "starter_info_quality",
        "starter_era",
        "starter_whip",
        "bullpen_fatigue_label",
        "recent_3day_games",
        "scheduled_game_id",
        "opponent",
        "home_away",
        "data_source",
        "note",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    new_frame = pd.DataFrame(rows, columns=columns)
    if output_path.exists():
        existing = pd.read_csv(output_path)
        frame = existing if new_frame.empty else pd.concat([existing, new_frame], ignore_index=True)
    else:
        frame = new_frame
    if not frame.empty:
        frame = frame.sort_values("snapshot_time").drop_duplicates(
            subset=["snapshot_date", "reference_date", "team", "scheduled_game_id"],
            keep="last",
        )
    frame.to_csv(output_path, index=False, encoding="utf-8-sig")

    ref_frame = frame[frame["reference_date"].astype(str).eq(prediction_date.isoformat())] if not frame.empty else frame
    source_counts = ref_frame["starter_source"].replace({"manual": "confirmed"}).value_counts().to_dict() if not ref_frame.empty else {}
    quality_status = write_pitching_snapshot_quality_reports(frame, scheduled, output_path, results_dir, prediction_date, reference_datetime)
    status_payload = {
        "generated_at": reference_datetime.strftime("%Y-%m-%d %H:%M:%S"),
        "reference_date": prediction_date.isoformat(),
        "snapshot_rows_total": int(len(frame)),
        "snapshot_rows_for_reference_date": int(len(ref_frame)),
        "confirmed_starter_count": int(source_counts.get("confirmed", 0)),
        "estimated_starter_count": int(source_counts.get("estimated", 0)),
        "unknown_starter_count": int(source_counts.get("unknown", 0)),
        "teams_with_snapshot": sorted(ref_frame["team"].dropna().astype(str).unique().tolist()) if not ref_frame.empty else [],
        "scheduled_games": int(len(scheduled) // 2) if not scheduled.empty else int(len(ref_frame) // 2),
        "snapshot_file": str(output_path.relative_to(BASE_DIR)),
        "leakage_policy_note": "pitching_daily_snapshot.csv는 예측 시점에 저장된 정보만 누적하며 현재 모델 학습 피처로 사용하지 않습니다.",
        "quality_status": quality_status.get("quality_status"),
        "safe_for_future_feature_use": quality_status.get("safe_for_future_feature_use"),
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "pitching_snapshot_status.json").write_text(json.dumps(status_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return status_payload


def _quality_result(check_name: str, failed_rows: int, checked_rows: int, severity: str, detail: str, recommended_action: str):
    failure_rate = round(float(failed_rows) / checked_rows, 4) if checked_rows else 0.0
    status = "pass" if failed_rows == 0 else ("warning" if severity == "warning" else "fail")
    return {
        "check_name": check_name,
        "status": status,
        "severity": severity,
        "checked_rows": int(checked_rows),
        "failed_rows": int(failed_rows),
        "failure_rate": failure_rate,
        "detail": detail,
        "recommended_action": recommended_action,
    }


def write_pitching_feature_experiment_plan(results_dir: Path):
    plan = {
        "trigger_condition": "pitching_daily_snapshot.csv가 최소 30일 이상 누적되고 품질 상태가 blocking issue 없이 유지될 때",
        "required_minimum_days": 30,
        "candidate_feature_sets": [
            "baseline_core",
            "baseline_plus_streak",
            "baseline_plus_pitching_snapshot",
            "baseline_plus_streak_pitching_snapshot",
        ],
        "expected_features": [
            "team_starter_info_quality",
            "opponent_starter_info_quality",
            "both_starters_confirmed",
            "team_starter_era_snapshot",
            "opponent_starter_era_snapshot",
            "starter_era_gap",
            "team_starter_whip_snapshot",
            "opponent_starter_whip_snapshot",
            "starter_whip_gap",
            "team_bullpen_fatigue_label_encoded",
            "opponent_bullpen_fatigue_label_encoded",
            "bullpen_fatigue_gap",
        ],
        "evaluation_metrics": [
            "accuracy",
            "brier_score",
            "log_loss",
            "over_55_accuracy",
            "recent_3year_accuracy",
            "winning_streak_accuracy",
            "losing_streak_accuracy",
            "close_game_accuracy",
        ],
        "replacement_policy": "새 모델은 전체 accuracy, Brier Score, Log Loss, over_55_accuracy, 최근 3년 성능이 동시에 안정적으로 개선될 때만 운영 모델로 교체한다.",
        "leakage_policy": "예측 시점 이전에 저장된 스냅샷만 사용하고, 현재 경기 결과나 경기 종료 후 정보는 학습 피처에 포함하지 않는다.",
        "planned_experiment_name": "baseline_plus_pitching_snapshot",
    }
    (results_dir / "pitching_feature_experiment_plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    return plan


def write_pitching_snapshot_accumulation_report(frame: pd.DataFrame, results_dir: Path, minimum_required_days: int):
    rows = []
    if not frame.empty:
        work = frame.copy()
        normalized_source = work["starter_source"].fillna("unknown").replace({"manual": "confirmed"})
        work["starter_source_normalized"] = normalized_source
        for (snapshot_date, reference_date), group in work.groupby(["snapshot_date", "reference_date"], sort=True):
            duplicate_key_count = int(group.duplicated(subset=["snapshot_date", "reference_date", "team", "scheduled_game_id"]).sum())
            actual_team_rows = int(len(group))
            scheduled_games = int(actual_team_rows // 2)
            expected_team_rows = scheduled_games * 2
            blocking = duplicate_key_count > 0 or actual_team_rows != expected_team_rows or group["starter_info_quality"].isna().any()
            rows.append(
                {
                    "snapshot_date": snapshot_date,
                    "reference_date": reference_date,
                    "scheduled_games": scheduled_games,
                    "expected_team_rows": expected_team_rows,
                    "actual_team_rows": actual_team_rows,
                    "confirmed_starter_count": int(group["starter_source_normalized"].eq("confirmed").sum()),
                    "estimated_starter_count": int(group["starter_source_normalized"].eq("estimated").sum()),
                    "unknown_starter_count": int(group["starter_source_normalized"].eq("unknown").sum()),
                    "duplicate_key_count": duplicate_key_count,
                    "quality_status": "fail" if blocking else "pass",
                    "safe_for_future_feature_use": False,
                    "note": "누적 기간 부족: 최소 30일 이상 필요" if len(work["snapshot_date"].dropna().astype(str).unique()) < minimum_required_days else "품질 통과 후 별도 피처 실험 필요",
                }
            )
    report = pd.DataFrame(
        rows,
        columns=[
            "snapshot_date",
            "reference_date",
            "scheduled_games",
            "expected_team_rows",
            "actual_team_rows",
            "confirmed_starter_count",
            "estimated_starter_count",
            "unknown_starter_count",
            "duplicate_key_count",
            "quality_status",
            "safe_for_future_feature_use",
            "note",
        ],
    )
    report.to_csv(results_dir / "pitching_snapshot_accumulation_report.csv", index=False, encoding="utf-8-sig")
    return report


def write_pitching_snapshot_quality_reports(frame: pd.DataFrame, scheduled: pd.DataFrame, snapshot_path: Path, results_dir: Path, prediction_date: date, reference_datetime: datetime):
    results_dir.mkdir(parents=True, exist_ok=True)
    minimum_required_days = 30
    reference_key = prediction_date.isoformat()
    ref_frame = frame[frame["reference_date"].astype(str).eq(reference_key)].copy() if not frame.empty else frame
    actual_team_rows = int(len(ref_frame))
    expected_team_rows = int(len(scheduled)) if not scheduled.empty else actual_team_rows
    key_cols = ["snapshot_date", "reference_date", "team", "scheduled_game_id"]
    duplicate_key_count = int(frame.duplicated(subset=key_cols).sum()) if not frame.empty else 0
    starter_sources = {"confirmed", "manual", "estimated"}
    missing_starter = ref_frame[
        ref_frame["starter_source"].isin(starter_sources)
        & ref_frame["starter_name"].fillna("").astype(str).str.strip().isin(["", "-"])
    ] if not ref_frame.empty else ref_frame
    unknown_count = int((ref_frame["starter_source"].fillna("unknown").replace({"manual": "confirmed"}) == "unknown").sum()) if not ref_frame.empty else 0
    confirmed_count = int((ref_frame["starter_source"].fillna("unknown").replace({"manual": "confirmed"}) == "confirmed").sum()) if not ref_frame.empty else 0
    estimated_count = int((ref_frame["starter_source"].fillna("unknown") == "estimated").sum()) if not ref_frame.empty else 0
    mapping_fail = ref_frame[ref_frame["scheduled_game_id"].fillna("").astype(str).str.strip().eq("")] if not ref_frame.empty else ref_frame
    freshness_fail = ref_frame[ref_frame["snapshot_date"].astype(str) != reference_datetime.date().isoformat()] if not ref_frame.empty else ref_frame
    snapshot_dates = pd.to_datetime(frame["snapshot_date"], errors="coerce").dt.date if not frame.empty else pd.Series(dtype=object)
    reference_dates = pd.to_datetime(frame["reference_date"], errors="coerce").dt.date if not frame.empty else pd.Series(dtype=object)
    leakage_fail = frame[snapshot_dates > reference_dates] if not frame.empty else frame
    late_snapshot = ref_frame[
        pd.to_datetime(ref_frame["snapshot_time"], errors="coerce").dt.hour.fillna(0).astype(int) >= 22
    ] if not ref_frame.empty else ref_frame
    quality_missing = ref_frame[ref_frame["starter_info_quality"].isna()] if not ref_frame.empty else ref_frame

    rows = [
        _quality_result("duplicate_key_check", duplicate_key_count, len(frame), "blocking", "snapshot_date, reference_date, team, scheduled_game_id 조합 중복 검사", "중복 키가 있으면 최신 snapshot_time 기준으로 deduplicate"),
        _quality_result("missing_starter_name_check", len(missing_starter), actual_team_rows, "blocking", "starter_source가 confirmed/manual/estimated인데 starter_name이 비어 있는 행 검사", "선발명 파싱 또는 추정 로직 점검"),
        _quality_result("unknown_starter_ratio_check", unknown_count, actual_team_rows, "warning", f"reference_date 기준 unknown 선발 {unknown_count}건", "unknown 비율이 높으면 GameCenter 또는 추정 선발 수집 점검"),
        _quality_result("confirmed_starter_ratio_check", 0 if confirmed_count else actual_team_rows, actual_team_rows, "warning", f"reference_date 기준 confirmed 선발 {confirmed_count}건", "경기 전 확정 선발 발표 시간 이후 pregame 업데이트 확인"),
        _quality_result("team_count_check", 0 if actual_team_rows == expected_team_rows else abs(actual_team_rows - expected_team_rows), expected_team_rows, "blocking", "예정 경기 팀 행 수와 스냅샷 행 수 일치 검사", "예정 경기 수와 snapshot 생성 루프 매칭 점검"),
        _quality_result("scheduled_game_mapping_check", len(mapping_fail), actual_team_rows, "blocking", "scheduled_game_id 비어 있는 행 검사", "일정 수집 game_id 생성 로직 점검"),
        _quality_result("snapshot_time_freshness_check", len(freshness_fail), actual_team_rows, "warning", "reference_date 스냅샷이 현재 실행일에 생성됐는지 검사", "자동 실행 시점과 reference_date 전달값 확인"),
        _quality_result("leakage_guard_check", len(leakage_fail), len(frame), "blocking", "snapshot_date가 reference_date보다 미래인 행 검사", "과거 기준일 재실행 산출물을 모델 피처로 사용하지 않도록 보관만 허용"),
        _quality_result("postgame_snapshot_warning_check", len(late_snapshot), actual_team_rows, "warning", "22시 이후 생성되어 경기 종료 후 수집으로 의심되는 행 검사", "경기 전 업데이트 산출물만 학습 피처 후보로 분리"),
        _quality_result("starter_info_quality_check", len(quality_missing), actual_team_rows, "blocking", "starter_info_quality 누락 검사", "선발 정보 품질 기본값 저장 로직 점검"),
    ]
    pd.DataFrame(rows).to_csv(results_dir / "pitching_snapshot_quality_report.csv", index=False, encoding="utf-8-sig")

    snapshot_day_values = sorted({d.isoformat() for d in snapshot_dates.dropna()}) if not frame.empty else []
    reference_day_values = sorted({d.isoformat() for d in reference_dates.dropna()}) if not frame.empty else []
    teams_covered = sorted(frame["team"].dropna().astype(str).unique().tolist()) if not frame.empty else []
    blocking_issues = [row["check_name"] for row in rows if row["status"] == "fail" and row["severity"] == "blocking"]
    warnings = [row["check_name"] for row in rows if row["status"] == "warning"]
    accumulated_days = len(snapshot_day_values)
    remaining_days = max(minimum_required_days - accumulated_days, 0)
    progress_pct = round(min(accumulated_days / minimum_required_days, 1.0) * 100, 1)
    enough_history = accumulated_days >= minimum_required_days
    safe = not blocking_issues and enough_history
    reason = ""
    if blocking_issues:
        reason = "blocking quality issues: " + ", ".join(blocking_issues)
    elif not enough_history:
        reason = f"누적 기간 부족: 최소 {minimum_required_days}일 이상 필요"
    feature_gate = "ready_for_experiment" if safe else "blocked_until_minimum_history"
    if blocking_issues:
        feature_gate = "blocked_by_quality_issues"
    experiment_date = ""
    if snapshot_day_values:
        experiment_date = (datetime.strptime(snapshot_day_values[0], "%Y-%m-%d").date() + timedelta(days=minimum_required_days - 1)).isoformat()
    write_pitching_snapshot_accumulation_report(frame, results_dir, minimum_required_days)
    write_pitching_feature_experiment_plan(results_dir)
    status = {
        "generated_at": reference_datetime.strftime("%Y-%m-%d %H:%M:%S"),
        "reference_date": reference_key,
        "snapshot_file": str(snapshot_path.relative_to(BASE_DIR)),
        "total_snapshot_rows": int(len(frame)),
        "reference_date_rows": actual_team_rows,
        "scheduled_games": int(len(scheduled) // 2) if not scheduled.empty else int(actual_team_rows // 2),
        "expected_team_rows": expected_team_rows,
        "actual_team_rows": actual_team_rows,
        "duplicate_key_count": duplicate_key_count,
        "unknown_starter_count": unknown_count,
        "estimated_starter_count": estimated_count,
        "confirmed_starter_count": confirmed_count,
        "quality_status": "pass" if not blocking_issues else "fail",
        "blocking_issues": blocking_issues,
        "warnings": warnings,
        "safe_for_future_feature_use": safe,
        "reason_if_not_safe": reason,
        "minimum_required_days": minimum_required_days,
        "accumulated_snapshot_days": accumulated_days,
        "remaining_days_to_feature_use": remaining_days,
        "accumulation_progress_pct": progress_pct,
        "earliest_feature_experiment_date": experiment_date,
        "feature_use_gate_status": feature_gate,
        "first_snapshot_date": snapshot_day_values[0] if snapshot_day_values else "",
        "latest_snapshot_date": snapshot_day_values[-1] if snapshot_day_values else "",
        "reference_dates_covered": reference_day_values,
        "teams_covered": teams_covered,
    }
    (results_dir / "pitching_snapshot_quality_status.json").write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    return status


def update_pitching_snapshot_diagnostics(results_dir: Path, status_payload: dict):
    starter_report = results_dir / "starter_data_availability_report.csv"
    if starter_report.exists():
        frame = pd.read_csv(starter_report)
        frame = frame[frame["data_item"] != "예측 시점 투수 스냅샷"]
        frame = pd.concat(
            [
                frame,
                pd.DataFrame(
                    [
                        {
                            "data_item": "예측 시점 투수 스냅샷",
                            "current_source": "pitching_daily_snapshot.csv",
                            "available_now": "started",
                            "collection_method": "official_kbo_dashboard.py 실행 시점의 선발/불펜 context 누적 저장",
                            "known_before_game": "yes",
                            "leakage_risk": "low",
                            "expected_effect": "high_after_accumulation",
                            "implementation_difficulty": "low",
                            "next_action": "충분한 기간 누적 후 날짜 기준 shift(1) 피처 실험",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        frame.to_csv(starter_report, index=False, encoding="utf-8-sig")

    bullpen_report = results_dir / "bullpen_data_availability_report.csv"
    if bullpen_report.exists():
        frame = pd.read_csv(bullpen_report)
        frame = frame[frame["data_item"] != "불펜 피로 proxy 스냅샷"]
        frame = pd.concat(
            [
                frame,
                pd.DataFrame(
                    [
                        {
                            "data_item": "불펜 피로 proxy 스냅샷",
                            "current_source": "pitching_daily_snapshot.csv",
                            "available_now": "proxy_snapshot_started",
                            "collection_method": "최근 3일 경기 수와 불펜 피로 라벨을 예측 시점 기준으로 누적",
                            "known_before_game": "yes",
                            "leakage_risk": "low",
                            "expected_effect": "medium_after_accumulation",
                            "implementation_difficulty": "low",
                            "next_action": "실제 불펜 투구 수 로그 확보 전까지 proxy 히스토리로만 보관",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        frame.to_csv(bullpen_report, index=False, encoding="utf-8-sig")

    summary_path = results_dir / "model_insight_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        quality_path = results_dir / "pitching_snapshot_quality_status.json"
        quality_status = json.loads(quality_path.read_text(encoding="utf-8")) if quality_path.exists() else {}
        summary["pitching_snapshot_collection_status"] = status_payload
        summary["pitching_snapshot_quality_summary"] = quality_status
        summary["pitching_snapshot_accumulation_progress"] = {
            "accumulated_snapshot_days": quality_status.get("accumulated_snapshot_days", 0),
            "first_snapshot_date": quality_status.get("first_snapshot_date", ""),
            "latest_snapshot_date": quality_status.get("latest_snapshot_date", ""),
            "minimum_days_required": quality_status.get("minimum_required_days", 30),
            "remaining_days_to_feature_use": quality_status.get("remaining_days_to_feature_use", 30),
            "accumulation_progress_pct": quality_status.get("accumulation_progress_pct", 0),
        }
        summary["safe_to_use_pitching_snapshot_as_features"] = bool(quality_status.get("safe_for_future_feature_use", False))
        summary["reason_pitching_snapshot_not_used_yet"] = quality_status.get("reason_if_not_safe") or "현재 운영 모델 피처로 연결하지 않는 정책 유지"
        summary["recommended_next_step_after_30_days"] = "30일 이상 누적 후 선발 정보 품질, 선발 ERA/WHIP 스냅샷, 불펜 피로 proxy를 별도 후보 모델에서 leakage-safe rolling 피처로 검증합니다."
        summary["pitching_snapshot_accumulation_gate"] = {
            "feature_use_gate_status": quality_status.get("feature_use_gate_status", "blocked_until_minimum_history"),
            "minimum_required_days": quality_status.get("minimum_required_days", 30),
            "accumulated_snapshot_days": quality_status.get("accumulated_snapshot_days", 0),
            "remaining_days_to_feature_use": quality_status.get("remaining_days_to_feature_use", 30),
            "earliest_feature_experiment_date": quality_status.get("earliest_feature_experiment_date", ""),
        }
        summary["pitching_snapshot_monitoring_summary"] = (
            f"투수 스냅샷은 품질 상태 {quality_status.get('quality_status', '-')}, "
            f"누적 {quality_status.get('accumulated_snapshot_days', 0)}/{quality_status.get('minimum_required_days', 30)}일입니다."
        )
        summary["earliest_pitching_feature_experiment_plan"] = {
            "planned_experiment_name": "baseline_plus_pitching_snapshot",
            "earliest_feature_experiment_date": quality_status.get("earliest_feature_experiment_date", ""),
            "plan_file": "modeling/results/pitching_feature_experiment_plan.json",
        }
        summary["reason_pitching_features_still_blocked"] = (
            f"투수 스냅샷은 품질 점검을 통과했지만 누적 기간이 {quality_status.get('accumulated_snapshot_days', 0)}일로 짧아 모델 피처 사용을 차단합니다. "
            "최소 30일 이상 누적 후 baseline_plus_pitching_snapshot 실험을 진행합니다."
        )
        summary["leakage_safe_pitching_data_policy"] = "투수 스냅샷은 예측 시점에 알고 있던 정보만 누적 저장하며, 현재 운영 모델 학습 피처로 바로 사용하지 않습니다."
        summary["next_step_after_snapshot_accumulation"] = "스냅샷이 충분히 쌓이면 선발 최근 성적, 선발 정보 품질, 불펜 피로 proxy를 날짜 기준 shift(1) 피처로 별도 후보 모델에서 검증합니다."
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def _lineup_grid_rows(raw_grid) -> list[dict]:
    if isinstance(raw_grid, str):
        try:
            raw_grid = json.loads(raw_grid)
        except json.JSONDecodeError:
            return []
    rows = []
    for item in raw_grid.get("rows", []) if isinstance(raw_grid, dict) else []:
        cells = [cell.get("Text", "") for cell in item.get("row", [])]
        if len(cells) < 4:
            continue
        rows.append(
            {
                "타순": cells[0],
                "포지션": cells[1],
                "선수": cells[2],
                "WAR": cells[3],
            }
        )
    return rows


def fetch_lineup_analysis(game_id: str, season_id: int):
    url = f"{KBO_BASE}/ws/Schedule.asmx/GetLineUpAnalysis"
    try:
        response = requests.post(
            url,
            headers={**HEADERS, "Referer": f"{KBO_BASE}/Schedule/GameCenter/Main.aspx", "X-Requested-With": "XMLHttpRequest"},
            data={"leId": 1, "srId": 0, "seasonId": season_id, "gameId": game_id},
            timeout=20,
        )
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError, IndexError, KeyError):
        return None

    lineup_confirmed = bool(data[0][0].get("LINEUP_CK")) if data and data[0] else False
    home_meta = data[1][0] if len(data) > 1 and data[1] else {}
    away_meta = data[2][0] if len(data) > 2 and data[2] else {}
    home_lineup = _lineup_grid_rows(data[3][0]) if len(data) > 3 and data[3] else []
    away_lineup = _lineup_grid_rows(data[4][0]) if len(data) > 4 and data[4] else []
    return {
        "lineup_source": "confirmed" if lineup_confirmed else "recent",
        "status_label": "금일 라인업 기준" if lineup_confirmed else "최근 라인업 기준",
        "home_team": home_meta.get("T_NM", ""),
        "away_team": away_meta.get("T_NM", ""),
        "home_lineup": home_lineup,
        "away_lineup": away_lineup,
    }


def _lineup_war_sum(rows: list[dict]) -> float:
    total = 0.0
    for row in rows:
        total += to_float(row.get("WAR"), 0.0)
    return round(total, 2)


def _lineup_preview(rows: list[dict], limit: int = 5) -> str:
    if not rows:
        return "-"
    return ", ".join(f'{row["타순"]}. {row["선수"]}({row["포지션"]})' for row in rows[:limit])


def build_lineup_context(games: pd.DataFrame, prediction_date: date):
    scheduled = games[
        (pd.to_datetime(games["date"]).dt.date == prediction_date)
        & (games["status"] == "Scheduled")
    ].copy()
    if scheduled.empty:
        return {}

    scheduled["actual_game_id"] = scheduled["game_id"].astype(str).str.rsplit("_", n=1).str[0]
    context = {}
    for game_id, game_rows in scheduled.groupby("actual_game_id", sort=False):
        analysis = fetch_lineup_analysis(str(game_id), prediction_date.year)
        if not analysis:
            continue
        home_rows = game_rows[game_rows["home_away"] == "H"]
        away_rows = game_rows[game_rows["home_away"] == "A"]
        if home_rows.empty or away_rows.empty:
            continue
        home_team = home_rows.iloc[0]["team"]
        away_team = away_rows.iloc[0]["team"]
        for team, side, rows in [
            (home_team, "home", analysis["home_lineup"]),
            (away_team, "away", analysis["away_lineup"]),
        ]:
            context[team] = {
                "game_id": str(game_id),
                "team": team,
                "side": side,
                "lineup_source": analysis["lineup_source"],
                "status_label": analysis["status_label"],
                "lineup": rows,
                "lineup_war": _lineup_war_sum(rows),
                "lineup_preview": _lineup_preview(rows),
                "lineup_count": len(rows),
            }
    return context


def export_lineup_context(context: dict, output_path: Path, prediction_date: date):
    rows = []
    for team, values in sorted(context.items()):
        for player in values.get("lineup", []):
            rows.append(
                {
                    "date": prediction_date.isoformat(),
                    "game_id": values.get("game_id", ""),
                    "team": team,
                    "home_away": "H" if values.get("side") == "home" else "A",
                    "lineup_source": values.get("lineup_source", "unknown"),
                    "status_label": values.get("status_label", "라인업 정보 미확인"),
                    "batting_order": player.get("타순", ""),
                    "position": player.get("포지션", ""),
                    "player": player.get("선수", ""),
                    "war": player.get("WAR", ""),
                }
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")


def append_lineup_daily_snapshot(
    context: dict,
    output_path: Path,
    prediction_date: date,
    snapshot_time: datetime,
):
    rows = []
    captured_at = snapshot_time.strftime("%Y-%m-%d %H:%M:%S")
    for team, values in sorted(context.items()):
        for player in values.get("lineup", []):
            rows.append(
                {
                    "snapshot_date": snapshot_time.date().isoformat(),
                    "snapshot_time": captured_at,
                    "reference_date": prediction_date.isoformat(),
                    "scheduled_game_id": values.get("game_id", ""),
                    "team": team,
                    "home_away": "H" if values.get("side") == "home" else "A",
                    "lineup_source": values.get("lineup_source", "unknown"),
                    "lineup_info_quality": 1.0 if values.get("lineup_source") == "confirmed" else 0.5,
                    "batting_order": player.get("타순", ""),
                    "position": player.get("포지션", ""),
                    "player": player.get("선수", ""),
                    "war": player.get("WAR", ""),
                    "data_source": "KBO GetLineUpAnalysis",
                }
            )
    if not rows:
        return

    current = pd.DataFrame(rows)
    if output_path.exists():
        history = pd.read_csv(output_path)
        comparison_columns = [
            "scheduled_game_id",
            "team",
            "lineup_source",
            "batting_order",
            "position",
            "player",
        ]
        latest = history[history["reference_date"].astype(str) == prediction_date.isoformat()].copy()
        if not latest.empty:
            latest_time = latest["snapshot_time"].astype(str).max()
            latest = latest[latest["snapshot_time"].astype(str) == latest_time]
            previous_signature = latest[comparison_columns].fillna("").astype(str).sort_values(comparison_columns).to_dict("records")
            current_signature = current[comparison_columns].fillna("").astype(str).sort_values(comparison_columns).to_dict("records")
            if previous_signature == current_signature:
                return
        current = pd.concat([history, current], ignore_index=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    current.to_csv(output_path, index=False, encoding="utf-8-sig")


def starter_status_label(status: str):
    return {
        "both_confirmed": "확정 선발 반영 완료",
        "partial_confirmed": "일부 확정 선발 반영",
        "estimated_only": "추정 선발 기준",
        "unknown": "선발 정보 미확인",
    }.get(status, "선발 정보 미확인")


def game_starter_status(home_source: str, away_source: str):
    sources = {"confirmed" if source == "manual" else (source or "unknown") for source in [home_source, away_source]}
    if sources == {"confirmed"}:
        return "both_confirmed"
    if "confirmed" in sources:
        return "partial_confirmed"
    if sources == {"estimated"}:
        return "estimated_only"
    return "unknown"


def build_pregame_update_status(games: pd.DataFrame, pitching_context: dict, prediction_date: date, reference_datetime: datetime, update_stage: str):
    status_path = DASHBOARD_DIR / "pregame_update_status.json"
    previous = {}
    if status_path.exists():
        try:
            previous = json.loads(status_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            previous = {}
    previous_games = {row.get("game_id"): row for row in previous.get("games", [])}

    scheduled = games[
        (pd.to_datetime(games["date"]).dt.date == prediction_date)
        & (games["status"] == "Scheduled")
    ].copy()
    if scheduled.empty:
        games_payload = []
    else:
        scheduled["actual_game_id"] = scheduled["game_id"].astype(str).str.rsplit("_", n=1).str[0]
        games_payload = []
        for game_id, game_rows in scheduled.groupby("actual_game_id", sort=False):
            home_rows = game_rows[game_rows["home_away"] == "H"]
            away_rows = game_rows[game_rows["home_away"] == "A"]
            if home_rows.empty or away_rows.empty:
                continue
            home_team = home_rows.iloc[0]["team"]
            away_team = away_rows.iloc[0]["team"]
            home_context = pitching_context.get(home_team, {})
            away_context = pitching_context.get(away_team, {})
            home_source = home_context.get("starter_source", "unknown")
            away_source = away_context.get("starter_source", "unknown")
            status = game_starter_status(home_source, away_source)
            games_payload.append(
                {
                    "game_id": str(game_id),
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_starter": home_context.get("선발명", "-"),
                    "away_starter": away_context.get("선발명", "-"),
                    "home_starter_source": home_source,
                    "away_starter_source": away_source,
                    "game_starter_status": status,
                    "status_label": starter_status_label(status),
                }
            )

    source_counts = {"confirmed": 0, "estimated": 0, "unknown": 0}
    for game in games_payload:
        for key in ["home_starter_source", "away_starter_source"]:
            source = game.get(key, "unknown")
            if source == "manual":
                source = "confirmed"
            source_counts[source if source in source_counts else "unknown"] += 1

    changes = []
    for game in games_payload:
        before = previous_games.get(game["game_id"])
        if not before:
            continue
        for field in ["home_starter_source", "away_starter_source", "home_starter", "away_starter"]:
            if before.get(field) != game.get(field):
                starter_key = "home_starter" if field.startswith("home") else "away_starter"
                changes.append(
                    {
                        "game_id": game["game_id"],
                        "game": f'{game["away_team"]} vs {game["home_team"]}',
                        "field": field,
                        "before": before.get(field, ""),
                        "after": game.get(field, ""),
                        "starter": game.get(starter_key, "-"),
                    }
                )

    payload = {
        "run_time": reference_datetime.strftime("%Y-%m-%d %H:%M:%S"),
        "reference_date": prediction_date.isoformat(),
        "update_stage": update_stage,
        "games_checked": len(games_payload),
        "teams_checked": sum(source_counts.values()),
        "starter_status_summary": source_counts,
        "dashboard_updated": True,
        "github_pushed": False,
        "games": games_payload,
        "changes": changes,
    }
    return payload


def export_pregame_update_status(status_payload: dict):
    for path in [
        DASHBOARD_DIR / "pregame_update_status.json",
        PUBLIC_DIR / "pregame_update_status.json",
        BASE_DIR / "logs" / "pregame_update_status.json",
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(status_payload, ensure_ascii=False, indent=2), encoding="utf-8")


def status_lookup_by_matchup(status_payload: dict):
    lookup = {}
    for game in status_payload.get("games", []):
        key = "|".join(sorted([game["home_team"], game["away_team"]]))
        lookup[key] = game
    return lookup


def build_prediction_cards(today_predictions: list[dict], pitching_context: dict | None = None, status_payload: dict | None = None, lineup_context: dict | None = None):
    pitching_context = pitching_context or {}
    lineup_context = lineup_context or {}
    status_lookup = status_lookup_by_matchup(status_payload or {})
    cards = {}
    for row in today_predictions:
        try:
            probability = float(str(row["예측승률"]).replace("%", "")) / 100
        except (KeyError, ValueError):
            continue
        confidence = max(probability, 1 - probability)
        key = "|".join(sorted([row["기준팀"], row["상대팀"]]))
        if key in cards and confidence <= cards[key]["confidence_value"]:
            continue
        tier = prediction_tier(confidence)
        pick_context = pitching_context.get(row["예측 구단"], {})
        pick_lineup = lineup_context.get(row["예측 구단"], {})
        game_status = status_lookup.get(key, {})
        status_label = game_status.get("status_label", "선발 정보 미확인")
        home_team = game_status.get("home_team", row["기준팀"])
        away_team = game_status.get("away_team", row["상대팀"])

        def starter_line(team):
            context = pitching_context.get(team, {})
            source = context.get("starter_source", "unknown")
            source_label = "확정" if source in {"confirmed", "manual"} else ("추정" if source == "estimated" else "미확인")
            era = context.get("ERA", "")
            whip = context.get("WHIP", "")
            era_text = f" · ERA {era}" if str(era).strip() not in {"", "-"} else ""
            whip_text = f" · WHIP {whip}" if str(whip).strip() not in {"", "-"} else ""
            return f'{team}: {context.get("선발명", "-")} · {source_label}{era_text}{whip_text}'

        matchup = f'{row["기준팀"]} vs {row["상대팀"]}'
        cards[key] = {
            "game_id": game_status.get("game_id", key),
            "home_team": home_team,
            "away_team": away_team,
            "경기": matchup,
            "예측 구단": row["예측 구단"],
            "추천": f'{row["예측 구단"]} {tier["우세"]}',
            "예측승률": f"{confidence:.1%}",
            "신뢰도": tier["신뢰도"],
            "핵심 근거": row.get("예측 근거", ""),
            "투수 신호": f'{pick_context.get("투수 표시", "예상 선발: 추정 불가")} · 불펜 피로 {pick_context.get("불펜 피로", "-")}',
            "선발 매치업": f"선발 매치업: {starter_line(away_team)} / {starter_line(home_team)}",
            "선발 상태": status_label,
            "라인업 신호": f'{pick_lineup.get("status_label", "라인업 정보 미확인")} · 선발 WAR 합 {pick_lineup.get("lineup_war", "-")} · {pick_lineup.get("lineup_preview", "-")}',
            "판단": tier["판단"],
            "confidence_value": confidence,
        }
    return sorted(cards.values(), key=lambda row: row["confidence_value"], reverse=True)


def prediction_change_summary(current_probability: float, previous_probability, current_team: str, previous_team) -> str:
    if previous_probability is None or pd.isna(previous_probability):
        return "이전 예측 없음"
    if previous_team and current_team != previous_team:
        return f"예측 구단 변경: {previous_team} → {current_team}"
    delta = (current_probability - float(previous_probability)) * 100
    if abs(delta) < 1.0:
        return "직전 예측 대비 변화 거의 없음"
    if delta >= 3.0:
        return f"직전 예측 대비 우세 강화 +{delta:.1f}%p"
    if delta <= -3.0:
        return f"직전 예측 대비 우세 약화 {delta:.1f}%p"
    sign = "+" if delta > 0 else ""
    return f"직전 예측 대비 {sign}{delta:.1f}%p"


def append_pregame_prediction_history(prediction_cards: list[dict], status_payload: dict, lineup_context: dict, reference_datetime: datetime, update_stage: str):
    history_path = RESULTS_DIR / "pregame_prediction_history.csv"
    existing = pd.DataFrame()
    if history_path.exists():
        existing = pd.read_csv(history_path)

    previous_lookup = {}
    if not existing.empty and {"reference_date", "game_id", "run_time"}.issubset(existing.columns):
        same_day = existing[existing["reference_date"].astype(str) == str(status_payload.get("reference_date", ""))].copy()
        if not same_day.empty:
            same_day["run_time_sort"] = pd.to_datetime(same_day["run_time"], errors="coerce")
            for game_id, group in same_day.sort_values("run_time_sort").groupby("game_id"):
                previous_lookup[game_id] = group.iloc[-1].to_dict()

    rows = []
    for card in prediction_cards:
        game_id = card.get("game_id", "")
        previous = previous_lookup.get(game_id, {})
        lineup_sources = []
        for team in [card.get("away_team"), card.get("home_team")]:
            if team:
                lineup_sources.append(lineup_context.get(team, {}).get("lineup_source", "unknown"))
        lineup_status = "confirmed" if lineup_sources and set(lineup_sources) == {"confirmed"} else ("recent" if "recent" in lineup_sources else "unknown")
        current_probability = float(card.get("confidence_value", 0.0))
        previous_probability = previous.get("win_probability") if previous else None
        previous_team = previous.get("predicted_team") if previous else None
        change_summary = prediction_change_summary(current_probability, previous_probability, card.get("예측 구단", ""), previous_team)
        card["예측 변화"] = change_summary
        rows.append(
            {
                "run_time": reference_datetime.strftime("%Y-%m-%d %H:%M:%S"),
                "reference_date": status_payload.get("reference_date", ""),
                "update_stage": update_stage,
                "game_id": game_id,
                "away_team": card.get("away_team", ""),
                "home_team": card.get("home_team", ""),
                "predicted_team": card.get("예측 구단", ""),
                "win_probability": round(current_probability, 4),
                "starter_status": next((game.get("game_starter_status") for game in status_payload.get("games", []) if game.get("game_id") == game_id), ""),
                "lineup_status": lineup_status,
                "previous_predicted_team": previous_team or "",
                "previous_win_probability": "" if previous_probability is None or pd.isna(previous_probability) else round(float(previous_probability), 4),
                "change_summary": change_summary,
            }
        )

    if rows:
        history_path.parent.mkdir(parents=True, exist_ok=True)
        current = pd.DataFrame(rows)
        combined = pd.concat([existing, current], ignore_index=True) if not existing.empty else current
        combined = combined.drop_duplicates(subset=["run_time", "game_id"], keep="last")
        combined.to_csv(history_path, index=False, encoding="utf-8-sig")
    return prediction_cards


def today_summary(prediction_cards: list[dict]):
    if not prediction_cards:
        return {
            "headline": "오늘 예정 경기가 없거나 예측 가능한 경기 정보가 없습니다.",
            "possible_games": 0,
            "close_games": 0,
            "top_pick": "-",
        }
    top = prediction_cards[0]
    possible_games = sum(1 for row in prediction_cards if row["판단"] == "예측 가능")
    close_games = sum(1 for row in prediction_cards if row["판단"] in {"참고", "참고만"})
    strong_games = sum(1 for row in prediction_cards if row["판단"] == "과신 주의")
    if strong_games:
        headline = f'{top["추천"]}가 가장 높은 예측이지만, 60% 이상 구간은 과신 경향이 있어 참고 지표로 봐야 합니다.'
    elif possible_games:
        headline = f'오늘은 강한 정배보다 약우세 경기 중심입니다. 가장 높은 예측은 {top["추천"]}({top["예측승률"]})입니다.'
    else:
        headline = f'오늘은 대부분 박빙입니다. 가장 높은 예측도 {top["추천"]}({top["예측승률"]}) 수준입니다.'
    return {
        "headline": headline,
        "possible_games": possible_games,
        "close_games": close_games,
        "top_pick": f'{top["추천"]} · {top["예측승률"]}',
    }


def evaluate_model(training_games: pd.DataFrame, current_games: pd.DataFrame, cutoff: date, prediction_date: date):
    training_games = training_games.copy()
    training_games["date"] = pd.to_datetime(training_games["date"])
    completed = training_games[
        (training_games["status"] == "Final")
        & (
            (training_games["date"].dt.year < cutoff.year)
            | (training_games["date"].dt.date <= cutoff)
        )
    ].copy()
    model_input = DATA_DIR / "model_training_games.csv"
    completed.to_csv(model_input, index=False, encoding="utf-8-sig")
    features = build_features(model_input)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    features.to_csv(RESULTS_DIR / "features.csv", index=False, encoding="utf-8-sig")
    export_game_level_dataset(features, RESULTS_DIR / "game_level_features.csv")

    if len(features) < 20:
        return {"available": False, "reason": "학습 가능한 완료 경기가 부족합니다.", "training_cutoff": cutoff.isoformat()}

    x, y = prepare_matrix(features)
    split_index = max(int(len(x) * 0.8), 1)
    split_index = min(split_index, len(x) - 1)
    y_train, y_test = y[:split_index], y[split_index:]
    train_years = pd.to_datetime(features.iloc[:split_index]["date"]).dt.year
    max_train_year = int(train_years.max())
    recency_weight = (0.85 ** (max_train_year - train_years)).clip(lower=0.35).to_numpy(dtype=float)

    candidate_columns = {
        "기본 흐름 모델": [
            col for col in x.columns
            if col not in {"team_elo_pre", "opponent_elo_pre", "elo_diff", "games_last_7_days", "back_to_back"}
        ],
        "전력/일정 피로도 포함 모델": list(x.columns),
        "핵심 수치 모델": [
            col for col in [
                "is_home",
                "rest_days",
                "recent_5_win_rate",
                "recent_10_win_rate",
                "avg_run_diff_last_5",
                "avg_run_diff_last_10",
                "season_win_rate_prior",
                "opponent_recent_5_win_rate",
                "opponent_recent_10_win_rate",
                "opponent_avg_run_diff_last_5",
                "opponent_avg_run_diff_last_10",
                "season_win_rate_gap",
                "recent_5_win_rate_gap",
                "recent_10_win_rate_gap",
                "season_avg_run_diff_gap",
                "recent_run_diff_10_gap",
                "venue_win_rate_gap",
                "head_to_head_win_rate_gap",
                "elo_diff",
                "games_last_7_days",
                "back_to_back",
            ]
            if col in x.columns
        ],
    }
    best = None
    candidate_results = []
    for name, columns in candidate_columns.items():
        x_train, x_test = x.iloc[:split_index][columns], x.iloc[split_index:][columns]
        train_scaled, test_scaled, mean, std = standardize_train_test(x_train, x_test)
        weights, bias = train_logistic_regression(train_scaled.to_numpy(), y_train, lr=0.05, epochs=3500)
        raw_probability = sigmoid(test_scaled.to_numpy() @ weights + bias)
        probability = normalize_game_probabilities(features.iloc[split_index:], raw_probability)
        pred = (probability >= 0.5).astype(int)
        accuracy = round(float((pred == y_test).mean()), 3)
        score = probability_scores(y_test, probability)
        result = {
            "name": name,
            "columns": columns,
            "accuracy": accuracy,
            "score": score,
            "probability": probability,
            "pred": pred,
            "mean": mean,
            "std": std,
            "weights": weights,
            "bias": bias,
            "model_type": "from_scratch_logistic_regression",
            "prediction_unit": "team",
        }
        candidate_results.append({"모델": name, "검증 정확도": accuracy, "피처 수": len(columns), **score})
        best = pick_better_model(best, result)

    try:
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    except ImportError:
        sklearn_candidates = []
    else:
        sklearn_candidates = [
            (
                "RandomForest 비선형 모델",
                RandomForestClassifier(
                    n_estimators=500,
                    max_depth=7,
                    min_samples_leaf=8,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
                None,
            ),
            (
                "RandomForest 시간가중 모델",
                RandomForestClassifier(
                    n_estimators=500,
                    max_depth=7,
                    min_samples_leaf=8,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
                recency_weight,
            ),
            (
                "GradientBoosting 비선형 모델",
                HistGradientBoostingClassifier(
                    max_iter=220,
                    learning_rate=0.04,
                    max_leaf_nodes=15,
                    l2_regularization=0.08,
                    random_state=42,
                ),
                None,
            ),
            (
                "GradientBoosting 시간가중 모델",
                HistGradientBoostingClassifier(
                    max_iter=220,
                    learning_rate=0.04,
                    max_leaf_nodes=15,
                    l2_regularization=0.08,
                    random_state=42,
                ),
                recency_weight,
            ),
            (
                "GradientBoosting 확률보정(sigmoid)",
                CalibratedClassifierCV(
                    estimator=HistGradientBoostingClassifier(
                        max_iter=220,
                        learning_rate=0.04,
                        max_leaf_nodes=15,
                        l2_regularization=0.08,
                        random_state=42,
                    ),
                    method="sigmoid",
                    cv=3,
                ),
                None,
            ),
            (
                "GradientBoosting 확률보정(isotonic)",
                CalibratedClassifierCV(
                    estimator=HistGradientBoostingClassifier(
                        max_iter=220,
                        learning_rate=0.04,
                        max_leaf_nodes=15,
                        l2_regularization=0.08,
                        random_state=42,
                    ),
                    method="isotonic",
                    cv=3,
                ),
                None,
            ),
        ]

    for name, model, sample_weight in sklearn_candidates:
        columns = list(x.columns)
        x_train, x_test = x.iloc[:split_index][columns], x.iloc[split_index:][columns]
        train_scaled, test_scaled, mean, std = standardize_train_test(x_train, x_test)
        fit_kwargs = {"sample_weight": sample_weight} if sample_weight is not None else {}
        model.fit(train_scaled, y_train, **fit_kwargs)
        raw_probability = model.predict_proba(test_scaled)[:, 1]
        probability = normalize_game_probabilities(features.iloc[split_index:], raw_probability)
        pred = (probability >= 0.5).astype(int)
        accuracy = round(float((pred == y_test).mean()), 3)
        score = probability_scores(y_test, probability)
        result = {
            "name": name,
            "columns": columns,
            "accuracy": accuracy,
            "score": score,
            "probability": probability,
            "pred": pred,
            "mean": mean,
            "std": std,
            "model": model,
            "model_type": model.__class__.__name__,
            "prediction_unit": "team",
        }
        candidate_results.append({"모델": name, "검증 정확도": accuracy, "피처 수": len(columns), **score})
        best = pick_better_model(best, result)

    game_frame = build_game_level_frame(features).dropna(subset=["target_home_win"]).copy()
    if sklearn_candidates and len(game_frame) >= 20:
        gx, gy = prepare_game_level_matrix(game_frame)
        game_split = max(int(len(gx) * 0.8), 1)
        game_split = min(game_split, len(gx) - 1)
        gy_train, gy_test = gy[:game_split], gy[game_split:]
        game_years = pd.to_datetime(game_frame.iloc[:game_split]["date"]).dt.year
        max_game_year = int(game_years.max())
        game_recency_weight = (0.85 ** (max_game_year - game_years)).clip(lower=0.35).to_numpy(dtype=float)
        game_candidates = [
            (
                "경기 단위 RandomForest 모델",
                RandomForestClassifier(
                    n_estimators=500,
                    max_depth=7,
                    min_samples_leaf=8,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
                None,
            ),
            (
                "경기 단위 RandomForest 시간가중 모델",
                RandomForestClassifier(
                    n_estimators=500,
                    max_depth=7,
                    min_samples_leaf=8,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
                game_recency_weight,
            ),
            (
                "경기 단위 GradientBoosting 모델",
                HistGradientBoostingClassifier(
                    max_iter=220,
                    learning_rate=0.04,
                    max_leaf_nodes=15,
                    l2_regularization=0.08,
                    random_state=42,
                ),
                None,
            ),
            (
                "경기 단위 GradientBoosting 확률보정(sigmoid)",
                CalibratedClassifierCV(
                    estimator=HistGradientBoostingClassifier(
                        max_iter=220,
                        learning_rate=0.04,
                        max_leaf_nodes=15,
                        l2_regularization=0.08,
                        random_state=42,
                    ),
                    method="sigmoid",
                    cv=3,
                ),
                None,
            ),
        ]
        for name, model, sample_weight in game_candidates:
            columns = list(gx.columns)
            gx_train, gx_test = gx.iloc[:game_split][columns], gx.iloc[game_split:][columns]
            train_scaled, test_scaled, mean, std = standardize_train_test(gx_train, gx_test)
            fit_kwargs = {"sample_weight": sample_weight} if sample_weight is not None else {}
            model.fit(train_scaled, gy_train, **fit_kwargs)
            probability = model.predict_proba(test_scaled)[:, 1]
            pred = (probability >= 0.5).astype(int)
            accuracy = round(float((pred == gy_test).mean()), 3)
            score = probability_scores(gy_test, probability)
            result = {
                "name": name,
                "columns": columns,
                "accuracy": accuracy,
                "score": score,
                "probability": probability,
                "pred": pred,
                "mean": mean,
                "std": std,
                "model": model,
                "model_type": model.__class__.__name__,
                "prediction_unit": "game",
                "test_frame": game_frame.iloc[game_split:].copy(),
                "y_test": gy_test,
            }
            candidate_results.append({"모델": name, "검증 정확도": accuracy, "피처 수": len(columns), **score})
            best = pick_better_model(best, result)

    columns = best["columns"]
    probability = best["probability"]
    pred = best["pred"]
    accuracy = best["accuracy"]
    mean = best["mean"]
    std = best["std"]
    weights = best.get("weights")
    bias = best.get("bias")
    prediction_unit = best.get("prediction_unit", "team")
    y_eval = best.get("y_test", y_test)

    if prediction_unit == "game":
        recent = best["test_frame"].copy()
        recent["경기일"] = pd.to_datetime(recent["date"]).dt.strftime("%Y-%m-%d")
        recent["기준팀"] = recent["home_team"]
        recent["상대팀"] = recent["away_team"]
        recent["예측승률"] = [f"{max(p, 1 - p):.1%}" for p in probability]
        recent["예측 구단"] = np.where(probability >= 0.5, recent["home_team"], recent["away_team"])
        recent["예측"] = np.where(probability >= 0.5, "승리 예측", "패배 예측")
        recent["실제 승리 구단"] = np.where(y_eval == 1, recent["home_team"], recent["away_team"])
        recent["예측 근거"] = recent.apply(lambda row: game_prediction_reason(row, row["예측 구단"]), axis=1)
        train_rows = int(len(game_frame) - len(recent))
        test_rows = int(len(recent))
    else:
        recent = features.iloc[split_index:].copy()
        recent["경기일"] = pd.to_datetime(recent["date"]).dt.strftime("%Y-%m-%d")
        recent["기준팀"] = recent["team"]
        recent["상대팀"] = recent["opponent"]
        recent["예측승률"] = [f"{p:.1%}" for p in probability]
        recent["예측"] = np.where(pred == 1, "승리 예측", "패배 예측")
        recent["예측 구단"] = np.where(pred == 1, recent["team"], recent["opponent"])
        recent["실제 승리 구단"] = np.where(y_test == 1, recent["team"], recent["opponent"])
        recent["예측 근거"] = recent.apply(lambda row: prediction_reason(row, row["예측 구단"]), axis=1)
        train_rows = int(len(x_train))
        test_rows = int(len(x_test))

    prediction_input = DATA_DIR / "prediction_games.csv"
    current_games.to_csv(prediction_input, index=False, encoding="utf-8-sig")
    prediction_features = build_features(prediction_input, include_unlabeled=True)
    prediction_features["date_obj"] = pd.to_datetime(prediction_features["date"]).dt.date
    today_features = prediction_features[
        (prediction_features["date_obj"] == prediction_date)
        & (prediction_features["target_win"].isna())
    ].copy()
    today_predictions = []
    if not today_features.empty and prediction_unit == "game":
        game_prediction_frame = build_game_level_frame(prediction_features)
        game_prediction_frame["date_obj"] = pd.to_datetime(game_prediction_frame["date"]).dt.date
        today_games = game_prediction_frame[
            (game_prediction_frame["date_obj"] == prediction_date)
            & (game_prediction_frame["target_home_win"].isna())
        ].copy()
        if not today_games.empty:
            prediction_scaled = align_game_level_matrix(today_games.drop(columns=["date_obj"]), columns, mean, std)
            game_probability = best["model"].predict_proba(prediction_scaled)[:, 1]
            rows = []
            for (_, row), home_prob in zip(today_games.iterrows(), game_probability):
                home_pick = home_prob >= 0.5
                predicted_team = row["home_team"] if home_pick else row["away_team"]
                reason = game_prediction_reason(row, predicted_team)
                rows.append(
                    {
                        "경기일": row["date"],
                        "기준팀": row["home_team"],
                        "상대팀": row["away_team"],
                        "예측 구단": predicted_team,
                        "예측승률": f"{home_prob:.1%}",
                        "예측": "승리 예측" if home_pick else "패배 예측",
                        "예측 근거": reason,
                    }
                )
                rows.append(
                    {
                        "경기일": row["date"],
                        "기준팀": row["away_team"],
                        "상대팀": row["home_team"],
                        "예측 구단": predicted_team,
                        "예측승률": f"{1 - home_prob:.1%}",
                        "예측": "승리 예측" if not home_pick else "패배 예측",
                        "예측 근거": reason,
                    }
                )
            today_predictions = rows
    elif not today_features.empty:
        prediction_scaled = align_prediction_matrix(today_features.drop(columns=["date_obj"]), columns, mean, std)
        if best["model_type"] == "from_scratch_logistic_regression":
            raw_today_probability = sigmoid(prediction_scaled.to_numpy() @ weights + bias)
        else:
            raw_today_probability = best["model"].predict_proba(prediction_scaled)[:, 1]
        today_probability = normalize_game_probabilities(today_features, raw_today_probability)
        today_features["경기일"] = pd.to_datetime(today_features["date"]).dt.strftime("%Y-%m-%d")
        today_features["기준팀"] = today_features["team"]
        today_features["상대팀"] = today_features["opponent"]
        today_features["예측승률"] = [f"{p:.1%}" for p in today_probability]
        today_features["예측"] = np.where(today_probability >= 0.5, "승리 예측", "패배 예측")
        today_features["예측 구단"] = np.where(today_probability >= 0.5, today_features["team"], today_features["opponent"])
        today_features["예측 근거"] = today_features.apply(lambda row: prediction_reason(row, row["예측 구단"]), axis=1)
        today_predictions = today_features[["경기일", "기준팀", "상대팀", "예측 구단", "예측승률", "예측", "예측 근거"]].to_dict(orient="records")

    payload = {
        "available": True,
        "training_cutoff": cutoff.isoformat(),
        "training_start_year": int(completed["date"].dt.year.min()),
        "training_end_year": int(completed["date"].dt.year.max()),
        "train_rows": train_rows,
        "test_rows": test_rows,
        "accuracy": accuracy,
        "selected_model": best["name"],
        "candidate_results": candidate_results,
        "confidence_metrics": confidence_metrics(y_eval, probability),
        "calibration_table": calibration_table(y_eval, probability),
        "recent_backtest": recent[["경기일", "기준팀", "상대팀", "예측 구단", "예측승률", "예측", "실제 승리 구단", "예측 근거"]].tail(12).to_dict(orient="records"),
        "today_predictions": today_predictions,
        "source_note": "현재 주 경기는 적중/오답 집계에 포함하지 않습니다.",
        "feature_columns": columns,
        "model_type": best["model_type"],
        "prediction_unit": prediction_unit,
    }
    if best["model_type"] == "from_scratch_logistic_regression":
        payload["bias"] = round(float(bias), 6)
        payload["coefficients"] = {
            name: round(float(value), 6)
            for name, value in sorted(zip(columns, weights), key=lambda x: abs(x[1]), reverse=True)
        }
    elif hasattr(best.get("model"), "feature_importances_"):
        importances = best["model"].feature_importances_
        payload["feature_importance"] = {
            name: round(float(value), 6)
            for name, value in sorted(zip(columns, importances), key=lambda x: x[1], reverse=True)
        }
    (RESULTS_DIR / "win_predictor_model.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def table_html(rows, columns, limit=None):
    if isinstance(rows, pd.DataFrame):
        data = rows[columns].head(limit).to_dict(orient="records")
    else:
        data = rows[:limit] if limit else rows
    header = "".join(f"<th>{escape(col)}</th>" for col in columns)
    body = []
    for row in data:
        body.append("<tr>" + "".join(f"<td>{escape(str(row.get(col, '')))}</td>" for col in columns) + "</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def pct(value):
    return f"{value:.3f}"


def record_summary(frame: pd.DataFrame):
    wins = int((frame["result"] == "Win").sum())
    losses = int((frame["result"] == "Loss").sum())
    draws = int((frame["result"] == "Draw").sum())
    games = wins + losses + draws
    win_rate = wins / max(wins + losses, 1)
    return games, wins, losses, draws, win_rate


def team_context_metrics(hitters: pd.DataFrame):
    rows = []
    for team, frame in hitters.groupby("팀"):
        at_bats = frame["타수"].apply(to_float).sum()
        hits = frame["안타"].apply(to_float).sum()
        walks = frame["볼넷"].apply(to_float).sum()
        strikeouts = frame["삼진"].apply(to_float).sum()
        rows.append(
            {
                "팀": team,
                "팀타율": hits / max(at_bats, 1),
                "출루율": frame["출루율"].apply(to_float).replace(0, np.nan).mean(),
                "OPS": frame["OPS"].apply(to_float).replace(0, np.nan).mean(),
                "희생번트": int(frame["희생번트"].apply(to_float).sum()),
                "희생플라이": int(frame["희생플라이"].apply(to_float).sum()),
                "볼넷삼진비": walks / max(strikeouts, 1),
                "득점권타율": frame["득점권타율"].apply(to_float).replace(0, np.nan).mean(),
            }
        )
    metrics = pd.DataFrame(rows)
    for column in ["팀타율", "출루율", "OPS", "희생번트", "희생플라이", "볼넷삼진비", "득점권타율"]:
        metrics[f"{column}순위"] = metrics[column].rank(ascending=False, method="min").astype("Int64")
    return metrics


def team_pitching_metrics(pitchers: pd.DataFrame):
    rows = []
    for team, frame in pitchers.groupby("팀"):
        innings = frame["이닝"].apply(parse_innings).sum()
        earned_runs = frame["자책"].apply(to_float).sum()
        hits_allowed = frame["피안타"].apply(to_float).sum()
        walks = frame["볼넷"].apply(to_float).sum()
        rows.append(
            {
                "팀": team,
                "산출ERA": earned_runs * 9 / max(innings, 1),
                "산출WHIP": (hits_allowed + walks) / max(innings, 1),
            }
        )
    metrics = pd.DataFrame(rows)
    for column in ["산출ERA", "산출WHIP"]:
        metrics[f"{column}순위"] = metrics[column].rank(ascending=True, method="min").astype("Int64")
    return metrics


def rank_score(rank, total=10):
    if pd.isna(rank):
        return 50
    return int(round(100 - (int(rank) - 1) * (90 / max(total - 1, 1))))


def rank_text(rank):
    return "-" if pd.isna(rank) else f"{int(rank)}위"


def is_top_three(rank):
    return not pd.isna(rank) and int(rank) <= 3


def is_bottom_three(rank):
    return not pd.isna(rank) and int(rank) >= 8


def topic_particle(text: str):
    if not text:
        return "는"
    last = text[-1]
    code = ord(last) - 0xAC00
    if 0 <= code <= 11171 and code % 28 != 0:
        return "은"
    return "는"


def hitter_role(row: pd.Series):
    ops = to_float(row.get("OPS", 0))
    obp = to_float(row.get("출루율", 0))
    slg = to_float(row.get("장타율", 0))
    risp = to_float(row.get("득점권타율", 0))
    bb = to_float(row.get("볼넷", 0))
    strikeouts = to_float(row.get("삼진", 0))
    bb_k = bb / max(strikeouts, 1)
    if ops >= 0.850 and obp >= 0.380:
        return "출루와 장타를 동시에 만드는 중심 생산원"
    if risp >= 0.330:
        return "득점권 상황에서 점수 전환에 기여"
    if bb_k >= 0.75:
        return "볼넷 대비 삼진 관리가 좋은 타석 안정형"
    if slg >= 0.450:
        return "장타로 득점 기대값을 높이는 유형"
    return "팀 내 공격 생산을 보조하는 자원"


def pitcher_role(row: pd.Series):
    innings = parse_innings(row.get("이닝", 0))
    whip = to_float(row.get("WHIP", 0))
    saves_holds = to_float(row.get("세이브", 0)) + to_float(row.get("홀드", 0))
    if innings >= 35:
        return "긴 이닝을 소화해 불펜 부담을 줄이는 선발 자원"
    if saves_holds >= 5:
        return "승부처 등판 비중이 높은 핵심 불펜"
    if whip and whip <= 1.25:
        return "주자 허용을 억제하는 안정형 투수"
    return "마운드 운영 폭을 넓히는 투수 자원"


def key_player_summaries(team_hitters: pd.DataFrame, team_pitchers: pd.DataFrame, team_ops: float):
    hitters_frame = team_hitters.copy()
    pitchers_frame = team_pitchers.copy()
    for column in ["OPS", "출루율", "장타율", "득점권타율", "타석", "볼넷", "삼진"]:
        hitters_frame[f"{column}_num"] = hitters_frame[column].apply(to_float) if column in hitters_frame else 0
    hitters_eligible = hitters_frame[hitters_frame["타석_num"] >= 30].copy()
    if hitters_eligible.empty:
        hitters_eligible = hitters_frame.copy()
    hitters_eligible["impact_score"] = (
        hitters_eligible["OPS_num"] * 55
        + hitters_eligible["출루율_num"] * 25
        + hitters_eligible["장타율_num"] * 20
        + hitters_eligible["득점권타율_num"] * 10
        + hitters_eligible["타석_num"].clip(upper=150) / 150 * 10
    )
    hitters_top = hitters_eligible.sort_values("impact_score", ascending=False).head(3).copy()
    hitter_rows = []
    for _, row in hitters_top.iterrows():
        hitter_rows.append(
            {
                "선수": row.get("선수", "-"),
                "핵심 지표": f"OPS {to_float(row.get('OPS', 0)):.3f} · 출루율 {to_float(row.get('출루율', 0)):.3f} · 장타율 {to_float(row.get('장타율', 0)):.3f}",
                "역할 해석": hitter_role(row),
            }
        )

    for column in ["ERA", "WHIP", "이닝", "탈삼진", "볼넷", "세이브", "홀드"]:
        if column == "이닝":
            pitchers_frame[f"{column}_num"] = pitchers_frame[column].apply(parse_innings) if column in pitchers_frame else 0
        else:
            pitchers_frame[f"{column}_num"] = pitchers_frame[column].apply(to_float) if column in pitchers_frame else 0
    pitchers_eligible = pitchers_frame[pitchers_frame["이닝_num"] >= 5].copy()
    if pitchers_eligible.empty:
        pitchers_eligible = pitchers_frame.copy()
    kbb = pitchers_eligible["탈삼진_num"] / pitchers_eligible["볼넷_num"].replace(0, 1)
    pitchers_eligible["impact_score"] = (
        (6 - pitchers_eligible["ERA_num"]).clip(lower=0) / 6 * 35
        + (2 - pitchers_eligible["WHIP_num"]).clip(lower=0) / 2 * 25
        + pitchers_eligible["이닝_num"].clip(upper=50) / 50 * 25
        + kbb.clip(upper=4) / 4 * 10
        + (pitchers_eligible["세이브_num"] + pitchers_eligible["홀드_num"]).clip(upper=15) / 15 * 5
    )
    pitchers_top = pitchers_eligible.sort_values("impact_score", ascending=False).head(3).copy()
    pitcher_rows = []
    for _, row in pitchers_top.iterrows():
        pitcher_rows.append(
            {
                "선수": row.get("선수", "-"),
                "핵심 지표": f"ERA {to_float(row.get('ERA', 0)):.2f} · WHIP {to_float(row.get('WHIP', 0)):.2f} · 이닝 {parse_innings(row.get('이닝', 0)):.1f}",
                "역할 해석": pitcher_role(row),
            }
        )

    top3_ops = hitters_top["OPS_num"].mean() if not hitters_top.empty else 0
    ops_gap = top3_ops - team_ops
    total_innings = pitchers_frame["이닝_num"].sum()
    top3_innings_share = pitchers_frame.sort_values("이닝_num", ascending=False).head(3)["이닝_num"].sum() / max(total_innings, 1)
    if ops_gap >= 0.200:
        lineup_dependence = f"타선 의존도 높음: 핵심 타자 3명의 평균 OPS가 팀 평균보다 {ops_gap:.3f} 높아 중심타선 이탈 시 공격 하락 위험이 큽니다."
    elif ops_gap >= 0.100:
        lineup_dependence = f"타선 의존도 보통: 핵심 타자 3명의 평균 OPS가 팀 평균보다 {ops_gap:.3f} 높아 상위 생산원의 비중이 있습니다."
    else:
        lineup_dependence = "타선 의존도 낮음: 핵심 타자와 팀 평균 OPS 차이가 크지 않아 생산이 비교적 분산돼 있습니다."
    if top3_innings_share >= 0.45:
        pitching_dependence = f"마운드 의존도 높음: 이닝 상위 3명이 전체 이닝의 {top3_innings_share:.1%}를 맡아 선발/핵심 투수 의존도가 큽니다."
    elif top3_innings_share >= 0.35:
        pitching_dependence = f"마운드 의존도 보통: 이닝 상위 3명 비중이 {top3_innings_share:.1%}로 주요 투수 비중을 관리할 필요가 있습니다."
    else:
        pitching_dependence = f"마운드 의존도 낮음: 이닝 상위 3명 비중이 {top3_innings_share:.1%}로 투수 운영이 비교적 분산돼 있습니다."
    dependence_rows = [
        {"구분": "타선 의존도", "해석": lineup_dependence},
        {"구분": "마운드 의존도", "해석": pitching_dependence},
    ]
    return hitter_rows, pitcher_rows, dependence_rows


def build_team_analysis_pages(standings, vs_table, games, hitters, pitchers, rosters, generated_at: date):
    generated_pages = {}
    for team in standings["팀"]:
        generated_pages[team] = build_team_analysis_page(standings, vs_table, games, hitters, pitchers, rosters, team, generated_at)
    return generated_pages


def build_team_analysis_page(standings, vs_table, games, hitters, pitchers, rosters, team: str, generated_at: date):
    team_standing = standings[standings["팀"] == team].iloc[0]
    completed = games[(games["team"] == team) & (games["status"] == "Final")].copy()
    completed["date"] = pd.to_datetime(completed["date"])
    completed["득실차"] = completed["score_team"] - completed["score_opp"]
    games_count, wins, losses, draws, win_rate = record_summary(completed)
    runs_for = int(completed["score_team"].sum())
    runs_against = int(completed["score_opp"].sum())
    run_diff = runs_for - runs_against
    pythag = runs_for**2 / max(runs_for**2 + runs_against**2, 1)

    home_away = completed.groupby("home_away").agg(
        경기=("game_id", "count"),
        승=("result", lambda s: int((s == "Win").sum())),
        패=("result", lambda s: int((s == "Loss").sum())),
        평균득점=("score_team", "mean"),
        평균실점=("score_opp", "mean"),
        평균득실차=("득실차", "mean"),
    ).reset_index()
    home_away["구분"] = home_away["home_away"].map({"H": "홈", "A": "원정"})
    home_away["승률"] = home_away["승"] / (home_away["승"] + home_away["패"]).replace(0, pd.NA)
    home_away = home_away[["구분", "경기", "승", "패", "승률", "평균득점", "평균실점", "평균득실차"]].round(3)

    opponent = completed.groupby("opponent").agg(
        경기=("game_id", "count"),
        승=("result", lambda s: int((s == "Win").sum())),
        패=("result", lambda s: int((s == "Loss").sum())),
        평균득점=("score_team", "mean"),
        평균실점=("score_opp", "mean"),
        평균득실차=("득실차", "mean"),
    ).reset_index().rename(columns={"opponent": "상대"})
    opponent["승률"] = opponent["승"] / (opponent["승"] + opponent["패"]).replace(0, pd.NA)
    opponent = opponent.sort_values(["승률", "평균득실차"], ascending=False).round(3)

    monthly = completed.assign(월=completed["date"].dt.strftime("%Y-%m")).groupby("월").agg(
        경기=("game_id", "count"),
        승=("result", lambda s: int((s == "Win").sum())),
        패=("result", lambda s: int((s == "Loss").sum())),
        평균득점=("score_team", "mean"),
        평균실점=("score_opp", "mean"),
        평균득실차=("득실차", "mean"),
    ).reset_index()
    monthly["승률"] = monthly["승"] / (monthly["승"] + monthly["패"]).replace(0, pd.NA)
    monthly = monthly.round(3)

    recent = completed.sort_values("date", ascending=False).head(12).copy()
    recent["경기일"] = recent["date"].dt.strftime("%Y-%m-%d")
    recent["구분"] = recent["home_away"].map({"H": "홈", "A": "원정"})
    recent["결과"] = recent["result"].map({"Win": "승", "Loss": "패", "Draw": "무"})
    recent["스코어"] = recent["score_team"].astype("Int64").astype(str) + " - " + recent["score_opp"].astype("Int64").astype(str)

    team_hitters = hitters[hitters["팀"] == team].copy().head(12)
    team_pitchers = pitchers[pitchers["팀"] == team].copy().head(12)
    roster = rosters[rosters["팀"] == team].iloc[0].to_dict() if not rosters[rosters["팀"] == team].empty else {}
    top_hitter = team_hitters.iloc[0]["선수"] if not team_hitters.empty else "-"
    top_pitcher = team_pitchers.iloc[0]["선수"] if not team_pitchers.empty else "-"
    team_hitter_pool = hitters[hitters["팀"] == team].copy()
    team_pitcher_pool = pitchers[pitchers["팀"] == team].copy()
    team_avg = to_float(team_hitter_pool["안타"].apply(to_float).sum() / max(team_hitter_pool["타수"].apply(to_float).sum(), 1), 0)
    team_obp = to_float(team_hitter_pool["출루율"].apply(to_float).mean(), 0)
    team_ops = to_float(team_hitter_pool["OPS"].apply(to_float).mean(), 0)
    team_era = to_float(team_pitcher_pool["자책"].apply(to_float).sum() * 9 / max(team_pitcher_pool["이닝"].apply(parse_innings).sum(), 1), 0)
    context_metrics = team_context_metrics(hitters)
    context = context_metrics[context_metrics["팀"] == team].iloc[0].to_dict() if not context_metrics.empty else {}
    pitcher_metrics = team_pitching_metrics(pitchers)
    pitcher_context = pitcher_metrics[pitcher_metrics["팀"] == team].iloc[0].to_dict() if not pitcher_metrics.empty else {}
    sac_bunts = int(context.get("희생번트", 0))
    sac_flies = int(context.get("희생플라이", 0))
    bb_k = to_float(context.get("볼넷삼진비", 0))
    risp_avg = to_float(context.get("득점권타율", 0))
    ops_rank = context.get("OPS순위", pd.NA)
    bunt_rank = context.get("희생번트순위", pd.NA)
    sac_fly_rank = context.get("희생플라이순위", pd.NA)
    risp_rank = context.get("득점권타율순위", pd.NA)
    bb_k_rank = context.get("볼넷삼진비순위", pd.NA)
    era_rank = pitcher_context.get("산출ERA순위", pd.NA)
    whip_rank = pitcher_context.get("산출WHIP순위", pd.NA)
    operation_score = int(round((rank_score(bunt_rank) + rank_score(sac_fly_rank)) / 2))
    chance_score = rank_score(risp_rank)
    plate_score = rank_score(bb_k_rank)
    pitching_score = int(round((rank_score(era_rank) + rank_score(whip_rank)) / 2))
    pythag_gap = pythag - win_rate
    tags = []
    if int(team_standing["순위"]) <= 3:
        tags.append("상위권")
    if pythag_gap >= 0.03:
        tags.append("반등 여지")
    elif pythag_gap <= -0.03:
        tags.append("하락 위험")
    if is_top_three(ops_rank):
        tags.append("공격 생산형")
    if operation_score >= 75:
        tags.append("작전형")
    if chance_score >= 75:
        tags.append("찬스 강점")
    if plate_score >= 75:
        tags.append("타석 안정형")
    if pitching_score >= 75:
        tags.append("마운드 안정형")
    if run_diff <= -20:
        tags.append("득실 열세")
    if is_bottom_three(bb_k_rank):
        tags.append("타석 불안정")
    if is_bottom_three(era_rank) or is_bottom_three(whip_rank):
        tags.append("마운드 불안")
    if is_bottom_three(ops_rank) or is_bottom_three(context.get("팀타율순위", pd.NA)):
        tags.append("공격 침체")
    if not tags:
        tags.append("균형 점검형")
    strengths = []
    risks = []
    if run_diff >= 20:
        strengths.append(f"득실차가 {run_diff:+d}로 경기 내용이 확실히 승률을 뒷받침합니다.")
    elif run_diff > 0:
        strengths.append(f"득실차가 {run_diff:+d}으로 경기 내용도 소폭 우위입니다.")
    elif run_diff == 0:
        risks.append("득실차가 0으로 경기 내용은 거의 균형입니다.")
    elif run_diff < 0:
        risks.append(f"득실차 {run_diff:+d}라 현재 승률 유지에는 실점 억제가 필요합니다.")
    if operation_score >= 70:
        strengths.append(f"희생번트 {rank_text(bunt_rank)}, 희생플라이 {rank_text(sac_fly_rank)}로 작전/진루 생산이 강점입니다.")
    elif operation_score <= 40:
        risks.append(f"희생번트·희생플라이 지표가 하위권이라 작전 수행 결과가 뚜렷하지 않습니다.")
    if chance_score >= 70:
        strengths.append(f"득점권타율 {rank_text(risp_rank)}로 찬스 연결력이 좋습니다.")
    elif chance_score <= 40:
        risks.append(f"득점권타율 {rank_text(risp_rank)}라 찬스 대비 득점 전환을 점검해야 합니다.")
    if plate_score >= 70:
        strengths.append(f"BB/K {rank_text(bb_k_rank)}로 타석 운영 안정성이 좋습니다.")
    elif plate_score <= 40:
        risks.append(f"BB/K {rank_text(bb_k_rank)}라 볼넷 대비 삼진 관리가 필요합니다.")
    if pitching_score >= 70:
        strengths.append(f"산출 ERA {rank_text(era_rank)}, WHIP {rank_text(whip_rank)}로 마운드 지표가 안정적입니다.")
    elif pitching_score <= 40:
        risks.append(f"산출 ERA {rank_text(era_rank)}, WHIP {rank_text(whip_rank)}라 리드 유지 리스크가 있습니다.")
    if pythag_gap >= 0.03:
        strengths.append(f"실제 승률 {pct(win_rate)}보다 피타고리안 기대 승률 {pct(pythag)}이 높아 경기 내용상 반등 여지가 있습니다.")
    elif pythag_gap <= -0.03:
        risks.append(f"실제 승률 {pct(win_rate)}이 피타고리안 기대 승률 {pct(pythag)}보다 높아 득실 기반으로는 하락 위험이 있습니다.")
    if not strengths:
        strengths.append("리그 중간권 지표가 많아 특정 강점보다 균형 유지가 핵심입니다.")
    if not risks:
        risks.append("현재 공개 지표 기준 뚜렷한 하위권 리스크는 제한적입니다.")
    strengths = strengths[:3]
    risks = risks[:3]
    if int(team_standing["순위"]) <= 3 and chance_score >= 70 and plate_score >= 70 and pitching_score >= 70:
        conclusion = "찬스 수행, 타석 안정성, 마운드 안정성이 모두 상위권인 균형형 선두권 팀입니다."
    elif int(team_standing["순위"]) <= 3 and plate_score >= 70 and pitching_score >= 70:
        conclusion = "타석 운영과 마운드 안정성이 함께 받쳐주는 균형형 상위권 팀입니다."
    elif operation_score >= 70 and plate_score <= 50:
        conclusion = "작전 수행 결과는 좋지만 타석 안정성은 낮아 빅이닝 생산에는 보완이 필요합니다."
    elif chance_score >= 70 and run_diff < 0:
        conclusion = "찬스 수행력은 좋지만 득실차가 낮아 실점 억제와 기본 출루 생산을 함께 점검해야 합니다."
    elif pitching_score >= 70 and rank_score(ops_rank) <= 50:
        conclusion = "마운드 지표는 안정적이지만 공격 생산력이 제한돼 접전 의존도가 커질 수 있습니다."
    elif rank_score(ops_rank) >= 70 and pitching_score <= 50:
        conclusion = "공격 생산력은 강하지만 마운드 안정성이 낮아 리드 유지가 핵심 변수입니다."
    elif pythag_gap >= 0.03:
        conclusion = "실제 승률보다 경기 내용이 좋아 향후 반등 여지를 볼 수 있습니다."
    elif pythag_gap <= -0.03:
        conclusion = "현재 승률은 좋지만 득실 기반 기대 승률은 낮아 하락 위험을 함께 봐야 합니다."
    else:
        conclusion = "공격, 작전, 마운드 지표가 큰 한쪽 쏠림 없이 균형 점검이 필요한 팀입니다."
    summary_sentence = (
        f"{team}{topic_particle(team)} {rank_text(team_standing['순위'])}, 승률 {pct(win_rate)}의 팀입니다. "
        f"핵심 태그는 {', '.join(tags[:4])}이며, 작전 수행 지수 {operation_score}/100, "
        f"찬스 수행 지수 {chance_score}/100, 타석 안정성 {plate_score}/100으로 요약됩니다."
    )
    context_rows = [
        {"지표": "작전 수행 지수", "값": f"{operation_score}/100", "해석": f"희생번트 {sac_bunts}개({rank_text(bunt_rank)})와 희생플라이 {sac_flies}개({rank_text(sac_fly_rank)})를 함께 본 proxy입니다."},
        {"지표": "찬스 수행 지수", "값": f"{chance_score}/100", "해석": f"득점권타율 {risp_avg:.3f}, 리그 {rank_text(risp_rank)}입니다."},
        {"지표": "타석 안정성", "값": f"{plate_score}/100", "해석": f"BB/K {bb_k:.2f}, 리그 {rank_text(bb_k_rank)}입니다."},
        {"지표": "마운드 안정성", "값": f"{pitching_score}/100", "해석": f"산출 ERA {team_era:.2f}({rank_text(era_rank)}), 산출 WHIP {to_float(pitcher_context.get('산출WHIP', 0)):.2f}({rank_text(whip_rank)})입니다."},
    ]
    key_hitters, key_pitchers, dependence_rows = key_player_summaries(team_hitter_pool, team_pitcher_pool, team_ops)

    insight_rows = [
        {"인사이트": "시즌 위치", "내용": f"{team_standing['순위']}위, {wins}승 {losses}패 {draws}무, 승률 {pct(win_rate)}입니다."},
        {"인사이트": "득실 균형", "내용": f"득점 {runs_for}, 실점 {runs_against}, 득실차 {run_diff:+d}. 피타고리안 기대 승률은 {pct(pythag)}입니다."},
        {"인사이트": "팀 타격", "내용": f"등록 타자 기준 팀 타율 {team_avg:.3f}, 평균 출루율 {team_obp:.3f}, 평균 OPS {team_ops:.3f}입니다."},
        {"인사이트": "작전/상황 수행", "내용": f"희생번트 {sac_bunts}개, 희생플라이 {sac_flies}개, 득점권타율 {risp_avg:.3f}, BB/K {bb_k:.2f}입니다."},
        {"인사이트": "투수 운영", "내용": f"등록 투수 기준 산출 ERA {team_era:.2f}. 선발/불펜 후보군은 등록 투수 명단과 최근 이닝을 함께 봅니다."},
        {"인사이트": "벤치 구성", "내용": f"감독 {roster.get('감독', '-')}. 코치진은 {roster.get('코치', '-') or '-'}입니다."},
        {"인사이트": "선수 기여", "내용": f"타자 기록 상위는 {top_hitter}, 투수 기록 상위는 {top_pitcher}가 현재 테이블 최상단입니다."},
    ]

    html = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(team)} 2026 팀 분석</title>
  <style>
    body {{ margin:0; font-family:Arial,sans-serif; color:#1b1f24; background:#f4f6f8; }}
    header {{ background:#172033; color:white; padding:30px 32px; }}
    main {{ padding:24px 32px 48px; max-width:1440px; margin:0 auto; }}
    a {{ color:#1d4ed8; font-weight:700; text-decoration:none; }}
    h1,h2,h3 {{ margin:0 0 14px; }}
    .section {{ margin-top:22px; background:white; border:1px solid #dde3ea; border-radius:8px; padding:18px; }}
    .grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; }}
    .metric {{ border:1px solid #e1e7ef; border-radius:8px; padding:14px; background:#fbfcfe; }}
    .metric strong {{ display:block; font-size:24px; margin-top:6px; }}
    .tables {{ display:grid; grid-template-columns:1fr 1fr; gap:18px; }}
    table {{ width:100%; border-collapse:collapse; font-size:14px; }}
    th,td {{ padding:9px 10px; border-bottom:1px solid #e5e9f0; text-align:right; white-space:nowrap; }}
    th:first-child,td:first-child,td:nth-child(2) {{ text-align:left; }}
    th {{ background:#f0f3f7; }}
    .wrap-table td {{ white-space:normal; line-height:1.55; vertical-align:top; }}
    .wrap-table td:nth-child(2) {{ min-width:260px; }}
    .subsection {{ margin-top:18px; }}
    .lead {{ font-size:18px; line-height:1.6; margin:0 0 16px; }}
    .conclusion {{ margin:0 0 10px; font-size:20px; font-weight:700; line-height:1.5; }}
    .tags {{ display:flex; gap:8px; flex-wrap:wrap; margin:12px 0 18px; }}
    .tags span {{ border:1px solid #c8d2df; border-radius:999px; padding:6px 10px; background:#fff; font-weight:700; font-size:13px; }}
    .insight-list {{ margin:0; padding-left:20px; line-height:1.7; }}
    .note {{ color:#637083; font-size:13px; }}
    @media (max-width:960px) {{ .grid,.tables {{ grid-template-columns:1fr; }} main {{ padding:16px; }} }}
  </style>
</head>
<body>
<header>
  <h1>{escape(team)} 2026 팀 분석</h1>
  <div>생성일 {generated_at.isoformat()} · <a href="latest.html">KBO 리그 대시보드로 돌아가기</a></div>
</header>
<main>
  <section class="section">
    <h2>팀 분석 요약</h2>
    <p class="conclusion">{escape(conclusion)}</p>
    <p class="lead">{escape(summary_sentence)}</p>
    <div class="tags">{"".join(f"<span>{escape(tag)}</span>" for tag in tags[:4])}</div>
    <div class="grid">
      <div class="metric">순위<strong>{team_standing["순위"]}위</strong></div>
      <div class="metric">전적<strong>{wins}승 {losses}패 {draws}무</strong></div>
      <div class="metric">득실차<strong>{run_diff:+d}</strong></div>
      <div class="metric">팀 타율<strong>{team_avg:.3f}</strong></div>
    </div>
    {table_html(insight_rows, ["인사이트", "내용"])}
  </section>
  <section class="section">
    <h2>강점과 주의점</h2>
    <div class="tables">
      <div><h3>강점</h3><ol class="insight-list">{"".join(f"<li>{escape(item)}</li>" for item in strengths)}</ol></div>
      <div><h3>주의</h3><ol class="insight-list">{"".join(f"<li>{escape(item)}</li>" for item in risks)}</ol></div>
    </div>
  </section>
  <section class="section">
    <h2>핵심 선수 영향도</h2>
    <div class="tables">
      <div class="wrap-table"><h3>타선 핵심 TOP 3</h3>{table_html(key_hitters, ["선수", "핵심 지표", "역할 해석"])}</div>
      <div class="wrap-table"><h3>마운드 핵심 TOP 3</h3>{table_html(key_pitchers, ["선수", "핵심 지표", "역할 해석"])}</div>
    </div>
    <div class="subsection wrap-table">
      <h3>전력 의존도</h3>
      {table_html(dependence_rows, ["구분", "해석"])}
    </div>
    <p class="note">선수 영향도는 개별 선수명을 모델 피처로 직접 쓰지 않고, OPS·출루율·장타율·득점권타율·이닝·ERA·WHIP·K/BB·세이브/홀드 같은 공식 기록을 팀 내 역할 판단용으로 압축한 값입니다.</p>
  </section>
  <section class="section">
    <h2>감독·코치·등록 선수 구성</h2>
    <div class="grid">
      <div class="metric">감독<strong>{escape(str(roster.get("감독", "-")))}</strong></div>
      <div class="metric">투수 등록<strong>{len([p for p in str(roster.get("투수", "")).split(",") if p.strip()])}명</strong></div>
      <div class="metric">야수 등록<strong>{len([p for p in (str(roster.get("포수", "")) + "," + str(roster.get("내야수", "")) + "," + str(roster.get("외야수", ""))).split(",") if p.strip()])}명</strong></div>
      <div class="metric">산출 ERA<strong>{team_era:.2f}</strong></div>
    </div>
    <div class="wrap-table">{table_html([{"구분": "코치", "명단": roster.get("코치", "-")}, {"구분": "투수", "명단": roster.get("투수", "-")}, {"구분": "포수", "명단": roster.get("포수", "-")}, {"구분": "내야수", "명단": roster.get("내야수", "-")}, {"구분": "외야수", "명단": roster.get("외야수", "-")}], ["구분", "명단"])}</div>
    <p class="note">감독·코치·등록 선수 구성은 KBO 공식 전체 등록 현황 기준입니다.</p>
  </section>
  <section class="section">
    <h2>작전·상황 수행 지표</h2>
    {table_html(context_rows, ["지표", "값", "해석"])}
    <p class="note">작전 지시 수와 사인 성공률은 KBO 공식 공개 데이터에 없어 직접 확인할 수 없습니다. 대신 희생번트, 희생플라이, 득점권타율, BB/K를 작전·상황 수행 proxy로 사용합니다.</p>
    <p class="note">지수는 리그 내 순위를 100점으로 환산한 값이며 100점에 가까울수록 리그 상위권입니다. 작전 수행 지수는 희생번트와 희생플라이, 찬스 수행 지수는 득점권타율, 타석 안정성은 BB/K, 마운드 안정성은 ERA와 WHIP 순위를 기반으로 산출합니다.</p>
    <p class="note">태그 기준: OPS 또는 팀 타율 하위 3팀은 공격 침체, BB/K 하위 3팀은 타석 불안정, 득점권타율 상위 3팀은 찬스 강점, ERA 또는 WHIP 하위 3팀은 마운드 불안으로 표시합니다.</p>
  </section>
  <section class="section">
    <h2>경기력 분해</h2>
    <div class="tables">
      <div><h3>홈/원정</h3>{table_html(home_away, ["구분", "경기", "승", "패", "승률", "평균득점", "평균실점", "평균득실차"])}</div>
      <div><h3>월별 흐름</h3>{table_html(monthly, ["월", "경기", "승", "패", "승률", "평균득점", "평균실점", "평균득실차"])}</div>
    </div>
  </section>
  <section class="section">
    <h2>상대별 매치업</h2>
    {table_html(opponent, ["상대", "경기", "승", "패", "승률", "평균득점", "평균실점", "평균득실차"])}
  </section>
  <section class="section">
    <h2>최근 경기</h2>
    {table_html(recent, ["경기일", "opponent", "구분", "결과", "스코어", "득실차"])}
  </section>
  <section class="section">
    <h2>선수 지표</h2>
    <div class="tables">
      <div><h3>타자</h3>{table_html(team_hitters, ["선수", "경기", "타석", "타수", "안타", "홈런", "볼넷", "삼진", "타율", "출루율", "장타율", "OPS"])}</div>
      <div><h3>투수</h3>{table_html(team_pitchers, ["선수", "경기", "승", "패", "세이브", "홀드", "이닝", "자책", "탈삼진", "볼넷", "ERA", "WHIP"])}</div>
    </div>
  </section>
</main>
</body>
</html>"""
    slug = TEAM_PAGE_SLUGS.get(team, team)
    (DASHBOARD_DIR / f"{slug}.html").write_text(html, encoding="utf-8")
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    (PUBLIC_DIR / f"{slug}.html").write_text(html, encoding="utf-8")
    return f"{slug}.html"


def build_dashboard(standings, vs_table, games, hitters, pitchers, model_payload, generated_at: date, team_pages: dict[str, str] | None = None, reference_datetime: datetime | None = None, update_stage: str = "morning"):
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    team_data = {}
    team_pages = team_pages or {}
    standings_display = standings.rename(columns={"방문": "원정"})
    completed = games[games["status"] == "Final"].copy()
    completed["date"] = pd.to_datetime(completed["date"])
    lineup_context = build_lineup_context(games, generated_at)
    team_game_min = int(standings["경기"].min())
    team_game_max = int(standings["경기"].max())
    team_game_avg = round(float(standings["경기"].mean()), 1)
    league_leader = standings.iloc[0]
    today_predictions_by_team = {}
    for row in model_payload.get("today_predictions", []):
        today_predictions_by_team.setdefault(row["기준팀"], []).append(row)
    for team in standings["팀"]:
        team_games = completed[completed["team"] == team].sort_values("date", ascending=False).head(10).copy()
        team_games["경기일"] = team_games["date"].dt.strftime("%Y-%m-%d")
        team_games["상대"] = team_games["opponent"]
        team_games["구분"] = team_games["home_away"].map({"H": "홈", "A": "원정"})
        team_games["결과"] = team_games["result"].map({"Win": "승", "Loss": "패", "Draw": "무"})
        team_games["스코어"] = team_games["score_team"].astype("Int64").astype(str) + " - " + team_games["score_opp"].astype("Int64").astype(str)
        team_data[team] = {
            "standings": standings[standings["팀"] == team].to_dict(orient="records")[0],
            "vs": vs_table[vs_table["팀"] == team][["상대", "전적"]].to_dict(orient="records"),
            "recent": team_games[["경기일", "상대", "구분", "결과", "스코어"]].to_dict(orient="records"),
            "hitters": hitters[hitters["팀"] == team].head(15).to_dict(orient="records"),
            "pitchers": pitchers[pitchers["팀"] == team].head(15).to_dict(orient="records"),
            "today_predictions": today_predictions_by_team.get(team, []),
            "lineup": {
                "상태": lineup_context.get(team, {}).get("status_label", "라인업 정보 미확인"),
                "선발 WAR 합": lineup_context.get(team, {}).get("lineup_war", "-"),
                "라인업 요약": lineup_context.get(team, {}).get("lineup_preview", "-"),
                "players": lineup_context.get(team, {}).get("lineup", []),
            },
            "analysis_url": team_pages.get(team, f"{TEAM_PAGE_SLUGS.get(team, team)}.html"),
        }

    payload = json.dumps(team_data, ensure_ascii=False)
    model_rows = model_payload.get("recent_backtest", []) if model_payload.get("available") else []
    confidence_rows = model_payload.get("confidence_metrics", []) if model_payload.get("available") else []
    calibration_rows = model_payload.get("calibration_table", []) if model_payload.get("available") else []
    candidate_rows = model_payload.get("candidate_results", []) if model_payload.get("available") else []
    feature_rows = [
        {"피처": feature, "중요도": importance}
        for feature, importance in list((model_payload.get("feature_importance") or {}).items())[:10]
    ] if model_payload.get("available") else []
    pitching_context = build_pitching_context(games, pitchers, generated_at, reference_datetime, update_stage)
    export_pitching_context(pitching_context, DATA_DIR / "pitching_context.csv", generated_at)
    snapshot_status = append_pitching_daily_snapshot(games, pitching_context, DATA_DIR / "pitching_daily_snapshot.csv", RESULTS_DIR, generated_at, reference_datetime or datetime.now())
    update_pitching_snapshot_diagnostics(RESULTS_DIR, snapshot_status)
    snapshot_quality_path = RESULTS_DIR / "pitching_snapshot_quality_status.json"
    snapshot_quality = json.loads(snapshot_quality_path.read_text(encoding="utf-8")) if snapshot_quality_path.exists() else {}
    snapshot_time = reference_datetime or datetime.now()
    export_lineup_context(lineup_context, DATA_DIR / "lineup_context.csv", generated_at)
    append_lineup_daily_snapshot(
        lineup_context,
        DATA_DIR / "lineup_daily_snapshot.csv",
        generated_at,
        snapshot_time,
    )
    lineup_confirmed_count = sum(1 for values in lineup_context.values() if values.get("lineup_source") == "confirmed")
    lineup_recent_count = sum(1 for values in lineup_context.values() if values.get("lineup_source") == "recent")
    status_payload = build_pregame_update_status(games, pitching_context, generated_at, snapshot_time, update_stage)
    status_summary = status_payload["starter_status_summary"]
    lineup_unknown_count = max(status_payload["teams_checked"] - lineup_confirmed_count - lineup_recent_count, 0)
    status_payload["lineup_status_summary"] = {
        "confirmed": lineup_confirmed_count,
        "recent": lineup_recent_count,
        "unknown": lineup_unknown_count,
    }
    if update_stage == "morning":
        prediction_stage = "morning_estimated"
        update_stage_label = "오전 예측 · 추정 선발/최근 라인업"
    elif lineup_confirmed_count == status_payload["teams_checked"] and lineup_confirmed_count:
        prediction_stage = "pregame_lineup_confirmed"
        update_stage_label = "경기 전 재산출 · 확정 선발/금일 라인업"
    elif status_summary.get("confirmed", 0) == status_payload["teams_checked"]:
        prediction_stage = "pregame_starters_confirmed"
        update_stage_label = "경기 전 재산출 · 확정 선발/라인업 대기"
    else:
        prediction_stage = "pregame_partial"
        update_stage_label = "경기 전 재산출 · 일부 정보 확인"
    status_payload["prediction_stage"] = prediction_stage
    status_payload["probability_policy_note"] = (
        "승률은 승인된 운영 모델로 다시 산출합니다. 확정 라인업은 시점 스냅샷으로 수집하며, "
        "검증된 라인업 피처 artifact가 배포되기 전에는 승률 입력으로 사용하지 않습니다."
    )
    export_pregame_update_status(status_payload)
    changes = status_payload.get("changes", [])
    change_text = (
        f'최근 변경: {changes[0]["game"]} {changes[0]["field"]} {changes[0]["before"]} → {changes[0]["after"]}'
        if changes
        else "최근 선발 상태 변경 없음"
    )
    prediction_cards = build_prediction_cards(model_payload.get("today_predictions", []), pitching_context, status_payload, lineup_context)
    prediction_cards = append_pregame_prediction_history(prediction_cards, status_payload, lineup_context, reference_datetime or datetime.now(), update_stage)
    summary = today_summary(prediction_cards)
    confidence_threshold = float((model_payload.get("confidence_thresholds") or {}).get("top_20_percent_confidence", 0.58))
    recommendation_enabled = bool((model_payload.get("confidence_thresholds") or {}).get("recommendation_enabled", False))

    def prediction_tone(row):
        decision = str(row.get("판단", ""))
        confidence = float(row.get("confidence_value", 0))
        if recommendation_enabled and confidence >= confidence_threshold:
            return "tone-good"
        if "과신" in decision or "위험" in decision:
            return "tone-risk"
        return "tone-watch"

    def trust_level(row):
        confidence = float(row.get("confidence_value", 0))
        if recommendation_enabled and confidence >= confidence_threshold:
            return "높음"
        if confidence >= 0.56:
            return "보통"
        return "낮음"

    def recommendation_label(row):
        decision = str(row.get("판단", ""))
        starter_status = str(row.get("선발 상태", ""))
        if "미확인" in starter_status:
            return "정보 부족"
        if "과신" in decision or "위험" in decision:
            return "위험"
        if recommendation_enabled and float(row.get("confidence_value", 0)) >= confidence_threshold:
            return "추천"
        return "관망"

    def model_summary(row):
        team = row.get("예측 구단", "-")
        trust = trust_level(row)
        recommendation = recommendation_label(row)
        if recommendation == "추천":
            return f"예측 우세: {team} · 승률 우위가 있고 신뢰도는 {trust}입니다."
        if recommendation == "관망":
            return f"예측 우세: {team} · 승률 우위는 있으나 신뢰도는 {trust}이라 관망이 적절합니다."
        if recommendation == "정보 부족":
            return f"예측 우세: {team} · 선발 정보 확인 전까지 보수적으로 해석해야 합니다."
        return f"예측 우세: {team} · 확률이 높더라도 표본과 변동성을 함께 봐야 합니다."

    high_confidence_games = sum(1 for row in prediction_cards if recommendation_label(row) == "추천")
    watch_games = sum(1 for row in prediction_cards if recommendation_label(row) == "관망")
    average_confidence = (
        f'{sum(float(row.get("confidence_value", 0)) for row in prediction_cards) / len(prediction_cards):.1%}'
        if prediction_cards
        else "-"
    )
    snapshot_days = snapshot_quality.get("accumulated_snapshot_days", 0)
    snapshot_required_days = snapshot_quality.get("minimum_required_days", 30)
    snapshot_quality_label = snapshot_quality.get("quality_status", "-")
    snapshot_gate_label = "사용 가능" if snapshot_quality.get("safe_for_future_feature_use") else "차단됨"
    snapshot_gate_reason = snapshot_quality.get("reason_if_not_safe", "")
    featured_card = prediction_cards[0] if prediction_cards else {}
    featured_matchup = featured_card.get("경기", "-")
    featured_teams = [team.strip() for team in featured_matchup.split(" vs ", 1)]
    featured_left = featured_teams[0] if featured_teams else "-"
    featured_right = featured_teams[1] if len(featured_teams) > 1 else "-"
    featured_html = (
        f"""
        <div class="featured-match {prediction_tone(featured_card)}">
          <div class="featured-copy">
            <span class="featured-label">오늘의 핵심 예측</span>
            <div class="featured-teams">
              <span>{escape(featured_left)}</span>
              <em>vs</em>
              <span>{escape(featured_right)}</span>
            </div>
            <p class="featured-reason">오늘 경기 중 모델 신뢰도가 가장 높은 경기입니다.</p>
            <p>{escape(featured_card.get("핵심 근거", "오늘 표시할 핵심 예측이 없습니다."))}</p>
          </div>
          <div class="featured-result">
            <span class="badge-decision">{escape(recommendation_label(featured_card))}</span>
            <strong>{escape(featured_card.get("추천", "-"))}</strong>
            <div class="featured-prob">{escape(featured_card.get("예측승률", "-"))}</div>
            <p class="featured-trust">신뢰도 {escape(trust_level(featured_card))}</p>
          </div>
        </div>
        """
        if featured_card
        else '<div class="featured-match"><p class="note">오늘 표시할 핵심 예측이 없습니다.</p></div>'
    )

    prediction_cards_html = "".join(
        f"""
        <article class="prediction-card {prediction_tone(row)}">
          <div class="card-topline">
            <span class="game-chip">KBO · 경기 전</span>
            <span class="badge-decision">{escape(recommendation_label(row))}</span>
          </div>
          <div class="team-row">
            <span>{escape(row["경기"].split(" vs ", 1)[0])}</span>
            <em>vs</em>
            <span>{escape(row["경기"].split(" vs ", 1)[1] if " vs " in row["경기"] else row["경기"])}</span>
          </div>
          <div class="prediction-core">
            <span class="small-label">예상 우세</span>
            <strong>{escape(row["추천"])}</strong>
            <div class="win-probability">{escape(row["예측승률"])}</div>
          </div>
          <div class="confidence-block">
            <div class="confidence-label"><span>신뢰도 {escape(trust_level(row))}</span><span>{escape(row["예측승률"])}</span></div>
            <div class="confidence-track"><span style="width:{float(row.get("confidence_value", 0)) * 100:.0f}%"></span></div>
          </div>
          <div class="badges"><span class="badge-trust">승패 추천</span><span>핸디캡 관망</span><span>오버/언더 관망</span></div>
          <div class="judgement-box">
            <span class="small-label">모델 판단 요약</span>
            <p>{escape(model_summary(row))}</p>
          </div>
          <p class="reason-text">{escape(row["핵심 근거"])}</p>
          <div class="signal-list">
            <p><strong>판단 상태</strong> · {escape(row["판단"])} / 표시 등급 {escape(trust_level(row))}</p>
            <p>{escape(row["선발 상태"])}</p>
            <p>{escape(row.get("예측 변화", "이전 예측 없음"))}</p>
            <p>{escape(row["선발 매치업"])}</p>
            <p>{escape(row["투수 신호"])}</p>
            <p>{escape(row["라인업 신호"])}</p>
          </div>
          <p class="data-standard">기준: 공식 경기 데이터 및 모델 산출값 · 예측 결과는 참고용이며 실제 경기 결과와 다를 수 있습니다.</p>
        </article>
        """
        for row in prediction_cards
    ) or '<p class="note">오늘 표시할 예측 카드가 없습니다.</p>'
    team_buttons = "".join(
        f'<button type="button" class="team-button" data-team="{escape(team)}">{escape(team)}</button>'
        for team in standings["팀"]
    )
    run_expected_runs_pipeline(RUN_MODEL_INPUT, RUN_MODEL_RESULTS, 0.8, generated_at.isoformat(), RUN_MODEL_SCHEDULE_INPUT)
    run_model_html = render_prediction_board_embedded(RUN_MODEL_RESULTS)
    html = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>KBO 경기 예측 대시보드</title>
  <style>
    :root {{
      --bg: #F8FAFC;
      --bg-soft: #EEF4FF;
      --card: rgba(255, 255, 255, 0.92);
      --text: #0F172A;
      --muted: #64748B;
      --line: #E2E8F0;
      --blue: #1D4ED8;
      --navy: #0B1220;
      --blue-bg: #EFF6FF;
      --green: #059669;
      --green-bg: #ECFDF5;
      --orange: #D97706;
      --orange-bg: #FFF7ED;
      --red: #DC2626;
      --red-bg: #FEF2F2;
      --shadow: 0 24px 70px rgba(15, 23, 42, 0.08);
      --shadow-soft: 0 14px 35px rgba(15, 23, 42, 0.06);
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: Pretendard, Inter, "Apple SD Gothic Neo", "Noto Sans KR", "Malgun Gothic", sans-serif; color: var(--text); background: radial-gradient(circle at 18% 0%, rgba(219, 234, 254, 0.9), transparent 34%), linear-gradient(135deg, var(--bg) 0%, var(--bg-soft) 100%); line-height: 1.55; }}
    header {{ max-width: 1320px; margin: 0 auto; padding: 34px 28px 14px; color: var(--text); }}
    main {{ padding: 0 28px 60px; max-width: 1320px; margin: 0 auto; }}
    h1, h2, h3 {{ margin: 0 0 14px; letter-spacing: -0.01em; }}
    h1 {{ font-size: 36px; line-height: 1.16; font-weight:800; letter-spacing:-0.03em; }}
    h2 {{ font-size: 24px; }}
    h3 {{ font-size: 17px; }}
    .topbar {{ display:flex; justify-content:space-between; align-items:flex-start; gap:24px; }}
    .brand-kicker {{ display:inline-flex; align-items:center; gap:10px; margin-bottom:12px; color:var(--blue); font-weight:900; font-size:13px; letter-spacing:0.04em; }}
    .brand-mark {{ display:inline-grid; place-items:center; width:34px; height:34px; border-radius:12px; color:white; background:linear-gradient(135deg, var(--blue), #3B82F6); box-shadow:0 12px 26px rgba(29, 78, 216, 0.24); font-size:12px; letter-spacing:-0.01em; }}
    .meta-panel {{ display:flex; gap:10px; flex-wrap:wrap; justify-content:flex-end; min-width:360px; }}
    .meta-pill {{ border:1px solid rgba(226, 232, 240, 0.9); background:rgba(255,255,255,0.78); color:var(--muted); border-radius:999px; padding:8px 12px; font-size:13px; font-weight:800; box-shadow:var(--shadow-soft); }}
    .meta {{ color: var(--muted); margin-top: 8px; font-size: 15px; max-width: 760px; }}
    .section {{ margin-top: 22px; background: var(--card); border: 1px solid rgba(226, 232, 240, 0.72); border-radius: 26px; padding: 24px; box-shadow: var(--shadow); backdrop-filter: blur(10px); }}
    .hero-section {{ padding: 28px; overflow:hidden; position:relative; }}
    .hero-section::after {{ content:""; position:absolute; width:260px; height:260px; right:-90px; top:-120px; border-radius:50%; background:rgba(29, 78, 216, 0.08); pointer-events:none; }}
    .section-title {{ display: flex; align-items: baseline; justify-content: space-between; gap: 12px; margin-bottom: 14px; }}
    .eyebrow {{ color: var(--blue); font-size: 12px; font-weight: 900; letter-spacing: 0.06em; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; }}
    .metric {{ position:relative; border: 1px solid rgba(226, 232, 240, 0.9); border-radius: 18px; padding: 18px; background: rgba(248, 250, 252, 0.82); color: var(--muted); box-shadow: 0 10px 25px rgba(15, 23, 42, 0.035); }}
    .metric::before {{ content:""; display:block; width:8px; height:8px; border-radius:999px; background:var(--blue); margin-bottom:10px; opacity:.75; }}
    .metric strong {{ display: block; font-size: 24px; margin-top: 6px; color: var(--text); line-height: 1.22; font-weight:800; letter-spacing:-0.02em; }}
    .hero-metrics {{ margin-top:20px; }}
    .hero-metrics .metric {{ background:rgba(255,255,255,0.8); }}
    .hero-metrics .metric strong {{ font-size:28px; }}
    table {{ width: 100%; border-collapse: separate; border-spacing: 0; font-size: 14px; border: 1px solid rgba(226, 232, 240, 0.95); border-radius: 16px; overflow: hidden; background:white; }}
    th, td {{ padding: 14px 16px; border-bottom: 1px solid var(--line); text-align: right; white-space: nowrap; }}
    th:first-child, td:first-child, td:nth-child(2) {{ text-align: left; }}
    th {{ background: #F8FAFC; color: #334155; font-weight: 900; position:sticky; top:0; z-index:1; }}
    tbody tr:hover {{ background: #F8FAFC; }}
    tbody tr:last-child td {{ border-bottom: 0; }}
    select {{ padding: 10px 12px; border: 1px solid #D1D5DB; border-radius: 10px; font-size: 15px; background: white; }}
    .team-picker {{ display: grid; grid-template-columns: 220px 1fr; gap: 16px; align-items: start; margin-bottom: 18px; }}
    .team-buttons {{ display: grid; grid-template-columns: repeat(10, minmax(0, 1fr)); gap: 8px; }}
    .team-button {{ border: 1px solid var(--line); background: #fff; border-radius: 10px; padding: 10px 6px; cursor: pointer; font-weight: 800; color: var(--text); }}
    .team-button.active {{ background: var(--blue-bg); border-color: #BFDBFE; color: var(--blue); }}
    .subsection {{ margin-top: 18px; }}
    .action-link {{ display:inline-block; margin-top:10px; padding:10px 14px; border-radius:10px; background:var(--blue); color:white; font-weight:800; text-decoration:none; }}
    .tables {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }}
    .wide-table {{ overflow-x: auto; }}
    .note {{ color: var(--muted); font-size: 13px; margin-top: 10px; }}
    .insight-lead {{ font-size: 20px; line-height: 1.65; margin: 0 0 16px; max-width:850px; }}
    .featured-match {{ display:grid; grid-template-columns:minmax(0,1fr) 310px; gap:24px; align-items:center; margin:22px 0 8px; padding:24px; border:1px solid rgba(191, 219, 254, 0.7); border-radius:24px; background:linear-gradient(135deg, rgba(239,246,255,.95), rgba(255,255,255,.92)); box-shadow:0 22px 60px rgba(29,78,216,.10); }}
    .featured-label {{ display:inline-flex; width:max-content; margin-bottom:12px; border-radius:999px; padding:7px 10px; background:var(--blue-bg); color:var(--blue); font-weight:900; font-size:12px; }}
    .featured-teams {{ display:flex; align-items:center; gap:14px; flex-wrap:wrap; font-size:30px; font-weight:850; letter-spacing:-0.03em; }}
    .featured-teams em, .team-row em {{ color:var(--muted); font-style:normal; font-size:13px; font-weight:900; text-transform:uppercase; }}
    .featured-copy p {{ margin:12px 0 0; color:#475569; font-size:15px; max-width:760px; }}
    .featured-copy .featured-reason {{ color:var(--blue); font-weight:800; }}
    .featured-result {{ justify-self:end; min-width:260px; border-radius:22px; padding:20px; background:rgba(255,255,255,.86); border:1px solid rgba(226,232,240,.88); box-shadow:var(--shadow-soft); }}
    .featured-result strong {{ display:block; margin-top:14px; color:var(--navy); font-size:24px; line-height:1.25; }}
    .featured-prob {{ margin-top:12px; color:var(--blue); font-size:48px; line-height:1; font-weight:900; letter-spacing:-0.04em; }}
    .featured-trust {{ margin:10px 0 0; color:var(--muted); font-weight:800; }}
    .prediction-cards {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:18px; margin-top:22px; }}
    .prediction-card {{ border:1px solid rgba(226, 232, 240, 0.72); border-radius:22px; padding:20px; background:rgba(255,255,255,0.9); box-shadow: 0 18px 42px rgba(15, 23, 42, 0.07); transition:transform .18s ease, box-shadow .18s ease; }}
    .prediction-card:hover {{ transform:translateY(-3px); box-shadow:0 26px 58px rgba(15,23,42,.11); }}
    .prediction-card.tone-good {{ box-shadow: 0 18px 44px rgba(5, 150, 105, 0.10); }}
    .prediction-card.tone-watch {{ box-shadow: 0 18px 44px rgba(217, 119, 6, 0.10); }}
    .prediction-card.tone-risk {{ box-shadow: 0 18px 44px rgba(220, 38, 38, 0.10); }}
    .card-topline {{ display:flex; justify-content:space-between; align-items:center; gap:12px; margin-bottom:14px; }}
    .game-chip {{ color:var(--muted); background:#F8FAFC; border:1px solid var(--line); border-radius:999px; padding:6px 9px; font-size:12px; font-weight:900; }}
    .team-row {{ display:flex; align-items:center; justify-content:space-between; gap:10px; margin-bottom:16px; font-size:20px; font-weight:850; color:var(--navy); letter-spacing:-0.02em; }}
    .prediction-core {{ padding:18px; border-radius:18px; background:linear-gradient(135deg, rgba(239,246,255,0.95), rgba(255,255,255,0.95)); border:1px solid rgba(191, 219, 254, 0.7); }}
    .prediction-core strong {{ display:block; font-size:24px; line-height:1.28; color:var(--navy); margin-top:4px; }}
    .win-probability {{ margin-top:10px; font-size:40px; line-height:1; font-weight:850; color:var(--blue); letter-spacing:-0.04em; }}
    .small-label {{ display:block; color:var(--muted); font-size:12px; font-weight:900; letter-spacing:0.04em; }}
    .confidence-block {{ margin:16px 0 10px; }}
    .confidence-label {{ display:flex; justify-content:space-between; color:var(--muted); font-size:12px; font-weight:900; margin-bottom:8px; }}
    .confidence-track {{ height:6px; border-radius:999px; background:#E2E8F0; overflow:hidden; }}
    .confidence-track span {{ display:block; height:100%; border-radius:999px; background:linear-gradient(90deg, var(--blue), #60A5FA); }}
    .starter-status {{ color:#374151; font-size:13px; font-weight:700; margin:8px 0 0; }}
    .reason-text {{ color:var(--text); font-size:14px; font-weight:800; margin:14px 0 10px; }}
    .judgement-box {{ margin-top:14px; padding:14px; border-radius:16px; background:rgba(248,250,252,.86); border:1px solid rgba(226,232,240,.9); }}
    .judgement-box p {{ margin:6px 0 0; color:var(--text); font-size:14px; font-weight:700; line-height:1.55; }}
    .signal-list {{ border-top:1px solid var(--line); margin-top:14px; padding-top:12px; }}
    .signal-list p {{ color:#475569; font-size:13px; margin:7px 0 0; line-height:1.5; }}
    .data-standard {{ margin:14px 0 0; padding-top:12px; border-top:1px dashed rgba(148,163,184,.45); color:var(--muted); font-size:12px; line-height:1.5; }}
    .badges {{ display:flex; gap:7px; flex-wrap:wrap; margin:12px 0; }}
    .badges span, .badge-decision {{ border-radius:999px; padding:6px 10px; font-size:12px; font-weight:850; border:1px solid transparent; white-space:nowrap; }}
    .badge-trust {{ background:var(--blue-bg); color:var(--blue); border-color:#BFDBFE !important; }}
    .tone-good .badge-decision {{ background:var(--green-bg); color:var(--green); border-color:#BBF7D0 !important; }}
    .tone-watch .badge-decision {{ background:var(--orange-bg); color:#B45309; border-color:#FDE68A !important; }}
    .tone-risk .badge-decision {{ background:var(--red-bg); color:var(--red); border-color:#FECACA !important; }}
    .dashboard-tabs {{ display:flex; gap:10px; margin:0 0 22px; padding:8px; width:max-content; max-width:100%; border:1px solid rgba(226,232,240,.85); border-radius:18px; background:rgba(255,255,255,.76); box-shadow:var(--shadow-soft); }}
    .tab-button {{ border:0; border-radius:13px; padding:11px 16px; background:transparent; color:var(--muted); font-size:14px; font-weight:900; cursor:pointer; }}
    .tab-button.active {{ background:var(--blue); color:white; box-shadow:0 12px 26px rgba(29,78,216,.20); }}
    .tab-panel {{ display:none; }}
    .tab-panel.active {{ display:block; }}
    .run-model-panel {{ display:grid; gap:22px; }}
    .run-model-hero, .run-model-section {{ background:var(--card); border:1px solid rgba(226,232,240,.72); border-radius:26px; padding:24px; box-shadow:var(--shadow); backdrop-filter:blur(10px); }}
    .run-model-hero p {{ max-width:920px; color:#475569; margin:10px 0 0; }}
    .run-model-source {{ font-size:12px; color:var(--muted) !important; word-break:break-all; }}
    .run-model-grid, .run-model-panel .mini-grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:14px; }}
    .run-model-panel .metric {{ min-height:96px; }}
    .run-model-panel .metric p {{ margin:0 0 8px; color:var(--muted); font-size:13px; font-weight:800; }}
    .run-model-panel .metric span {{ display:block; margin-top:8px; color:var(--muted); font-size:12px; }}
    .run-model-tablewrap {{ overflow-x:auto; margin-top:12px; }}
    .run-model-split {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-top:16px; }}
    .run-model-panel .diagnostic-card {{ border:1px solid rgba(226,232,240,.9); border-radius:18px; padding:18px; background:rgba(248,250,252,.82); }}
    .run-model-panel .diagnostic-card p {{ color:var(--muted); line-height:1.6; }}
    .run-model-panel ul.features {{ columns:2; margin:8px 0 0; padding-left:20px; color:#334155; }}
    .run-model-panel .empty {{ color:var(--muted); }}
    .match-card-grid {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:16px; margin:14px 0 20px; }}
    .match-card {{ border:1px solid rgba(226,232,240,.85); border-radius:20px; padding:18px; background:rgba(255,255,255,.9); box-shadow:var(--shadow-soft); }}
    .match-card-top {{ display:flex; justify-content:space-between; align-items:center; gap:10px; color:var(--muted); font-size:12px; font-weight:850; }}
    .match-card h3 {{ margin-top:14px; font-size:19px; color:var(--navy); }}
    .score-line {{ margin:12px 0 16px; color:var(--blue); font-size:28px; line-height:1.2; font-weight:900; letter-spacing:-0.03em; }}
    .match-meta {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:8px; }}
    .match-meta div {{ border:1px solid var(--line); border-radius:14px; padding:10px; background:#F8FAFC; }}
    .match-meta span {{ display:block; color:var(--muted); font-size:11px; font-weight:850; }}
    .match-meta strong {{ display:block; margin-top:4px; font-size:14px; }}
    .match-pills {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:14px; }}
    .match-pills span, .match-badge {{ border-radius:999px; padding:6px 10px; font-size:12px; font-weight:850; border:1px solid var(--line); background:#F8FAFC; color:#334155; }}
    .match-badge.pick {{ background:var(--green-bg); color:var(--green); border-color:#BBF7D0; }}
    .match-badge.watch {{ background:var(--orange-bg); color:#B45309; border-color:#FDE68A; }}
    .match-badge.risk {{ background:var(--red-bg); color:var(--red); border-color:#FECACA; }}
    .run-model-diagnostics {{ background:var(--card); border:1px solid rgba(226,232,240,.72); border-radius:22px; padding:18px; box-shadow:var(--shadow-soft); }}
    details {{ margin-top:16px; }}
    summary {{ cursor:pointer; font-weight:800; color:var(--blue); }}
    @media (max-width: 960px) {{ .grid, .tables, .team-picker, .team-buttons, .prediction-cards, .featured-match, .run-model-grid, .run-model-panel .mini-grid, .run-model-split, .match-card-grid {{ grid-template-columns: 1fr; }} main {{ padding: 0 16px 40px; }} header {{ padding:24px 16px 8px; }} .topbar {{ flex-direction:column; }} .meta-panel {{ justify-content:flex-start; min-width:0; }} .featured-result {{ justify-self:stretch; }} .dashboard-tabs {{ width:100%; }} .tab-button {{ flex:1; }} }}
  </style>
</head>
<body>
<header>
  <div class="topbar">
    <div>
      <div class="brand-kicker"><span class="brand-mark">KBO</span><span>KBO PREDICTION LAB</span></div>
      <h1>KBO 경기 예측 대시보드</h1>
      <div class="meta">데이터 기반 KBO 경기 예측 리포트 · 오늘 경기의 우세 팀, 신뢰도, 선발/라인업 상태를 한 화면에서 확인합니다.</div>
    </div>
    <div class="meta-panel">
      <span class="meta-pill">기준일 {generated_at.isoformat()}</span>
      <span class="meta-pill">{escape(update_stage_label)}</span>
      <span class="meta-pill">오늘 경기 {len(prediction_cards)}경기</span>
      <span class="meta-pill">예측 학습 {escape(model_payload.get("prediction_training_cutoff", model_payload.get("training_cutoff", "")))}</span>
    </div>
  </div>
</header>
<main>
  <nav class="dashboard-tabs" aria-label="KBO dashboard tabs">
    <button type="button" class="tab-button active" data-tab="gamePrediction">경기 예측</button>
    <button type="button" class="tab-button" data-tab="runPrediction">득점 기반 승부 예측</button>
  </nav>
  <div id="gamePrediction" class="tab-panel active">
  <section class="section hero-section">
    <div class="eyebrow">TODAY · 오늘의 판단</div>
    <h2>오늘의 KBO 예측 요약</h2>
    <p class="insight-lead">{escape(summary["headline"])}</p>
    {featured_html}
    <div class="grid hero-metrics">
      <div class="metric">TOP PICK<strong>{escape(str(summary["top_pick"]))}</strong><span class="note">오늘 가장 강한 예측</span></div>
      <div class="metric">HIGH CONFIDENCE<strong>{high_confidence_games}</strong><span class="note">고신뢰 구간 경기 수</span></div>
      <div class="metric">WATCH LIST<strong>{watch_games}</strong><span class="note">관망 권장 경기 수</span></div>
      <div class="metric">AVG CONFIDENCE<strong>{average_confidence}</strong><span class="note">전체 평균 신뢰도</span></div>
    </div>
    <div class="subsection">
      <h3>업데이트 상태</h3>
      <div class="grid">
        <div class="metric">마지막 갱신<strong>{escape(status_payload["run_time"])}</strong></div>
        <div class="metric">업데이트 단계<strong>{escape(update_stage_label)}</strong></div>
        <div class="metric">선발 상태<strong>확정 {status_summary["confirmed"]} · 추정 {status_summary["estimated"]} · 미확인 {status_summary["unknown"]}</strong></div>
        <div class="metric">라인업 상태<strong>금일 {lineup_confirmed_count} · 최근 {lineup_recent_count} · 미확인 {lineup_unknown_count}</strong></div>
      </div>
      <p class="note">{escape(change_text)}</p>
    </div>
    <div class="prediction-cards">{prediction_cards_html}</div>
  </section>

  <section class="section">
    <div class="section-title">
      <div>
        <div class="eyebrow">STEP 1 · 리그 전체 상황</div>
        <h2>KBO 리그 전체 순위와 시즌 흐름</h2>
      </div>
    </div>
    <div class="grid">
      <div class="metric">1위<strong>{escape(str(league_leader["팀"]))}</strong></div>
      <div class="metric">팀별 경기 수 범위<strong>{team_game_min}~{team_game_max}</strong></div>
      <div class="metric">팀 평균 경기 수<strong>{team_game_avg}</strong></div>
      <div class="metric">업데이트일<strong>{generated_at.isoformat()}</strong></div>
    </div>
    <div class="wide-table">
      {table_html(standings_display, ["순위", "팀", "경기", "승", "패", "무", "승률", "게임차", "홈", "원정"])}
    </div>
  </section>

  <section class="section">
    <div class="section-title">
      <div>
        <div class="eyebrow">STEP 2 · 원하는 구단 선택</div>
        <h2 id="teamTitle">구단 상세 분석</h2>
      </div>
    </div>
    <div class="team-picker">
      <div class="note">구단 버튼을 선택하면 오늘 경기 예측, 최근 경기, 상대 전적, 선수 기록이 해당 구단 기준으로 바뀝니다.</div>
      <div class="team-buttons">{team_buttons}</div>
    </div>
    <div class="grid" id="teamMetrics"></div>
    <a class="action-link" id="teamAnalysisLink" href="kt.html">팀 분석 보기</a>
    <div class="subsection">
      <h3>오늘 경기 승패 예측</h3>
      <div id="todayPrediction"></div>
      <p class="note">예측은 경기 전 사용할 수 있는 팀 흐름, 상대 흐름, 홈/원정, 휴식일 기반 확률입니다.</p>
    </div>
    <div class="subsection">
      <h3>오늘 라인업</h3>
      <div id="lineupSummary"></div>
      <div id="lineupTable"></div>
      <p class="note">라인업은 KBO GameCenter 라인업 분석 기준입니다. KBO 응답이 금일 라인업을 확정하지 않은 경우 최근 라인업 기준으로 표시합니다.</p>
    </div>
    <div class="tables">
      <div><h3>최근 10경기</h3><div id="recentGames"></div></div>
      <div><h3>상대 전적</h3><div id="vsTable"></div></div>
    </div>
    <div class="tables">
      <div><h3>타자 주요 지표</h3><div id="hitterTable"></div></div>
      <div><h3>투수 주요 지표</h3><div id="pitcherTable"></div></div>
    </div>
  </section>

  <section class="section">
    <div class="eyebrow">DETAIL · 상세 데이터</div>
    <h2>상세 예측 데이터</h2>
    <div class="grid">
      <div class="metric">학습 시즌<strong>{model_payload.get("training_start_year", "-")}~{model_payload.get("training_end_year", "-")}</strong></div>
      <div class="metric">학습 행<strong>{model_payload.get("train_rows", "-")}</strong></div>
      <div class="metric">검증 행<strong>{model_payload.get("test_rows", "-")}</strong></div>
      <div class="metric">검증 정확도<strong>{model_payload.get("accuracy", "-")}</strong></div>
    </div>
    <div class="grid">
      <div class="metric">검증 cutoff<strong>{model_payload.get("validation_cutoff", model_payload.get("training_cutoff", "-"))}</strong></div>
      <div class="metric">예측 학습 cutoff<strong>{model_payload.get("prediction_training_cutoff", "-")}</strong></div>
      <div class="metric">최신 반영 경기일<strong>{model_payload.get("latest_completed_game_date_used", "-")}</strong></div>
      <div class="metric">이번 주 반영<strong>{model_payload.get("current_week_games_included_for_prediction", "-")}</strong></div>
    </div>
    <div class="grid">
      <div class="metric">투수 스냅샷 누적<strong>{snapshot_days}/{snapshot_required_days}일</strong></div>
      <div class="metric">스냅샷 품질<strong>{escape(str(snapshot_quality_label))}</strong></div>
      <div class="metric">모델 사용<strong>{escape(snapshot_gate_label)}</strong><span class="note">{escape(str(snapshot_gate_reason))}</span></div>
    </div>
    <p class="note">모델 상태: 전체 적중률 {model_payload.get("accuracy", "-")}, 55% 이상 예측 경기 적중률 {confidence_rows[1]["적중률"] if len(confidence_rows) > 1 else "-"}입니다. 현재 선택 모델은 단순 정확도 최고 모델이 아니라, Brier Score와 Log Loss를 함께 고려해 확률 품질이 상대적으로 안정적인 모델을 선택합니다. 60% 이상 구간은 평균 예측승률과 실제 승률이 비슷하더라도 표본 수가 작아 강한 정배보다는 참고 신호로 해석합니다.</p>
    <details>
      <summary>모델 검증 상세 보기</summary>
      <div class="subsection">
        <h3>확신 구간별 검증</h3>
        {table_html(confidence_rows, ["구간", "경기 수", "적중률"]) if confidence_rows else "<p>확신 구간 결과를 생성할 수 없습니다.</p>"}
      </div>
      <div class="subsection">
        <h3>확률 보정 검증</h3>
        {table_html(calibration_rows, ["예측승률 구간", "경기 수", "평균 예측승률", "실제 승률"]) if calibration_rows else "<p>확률 보정 결과를 생성할 수 없습니다.</p>"}
      </div>
      <div class="subsection">
        <h3>모델 후보 비교</h3>
        {table_html(candidate_rows, ["모델", "검증 정확도", "Brier Score", "Log Loss", "피처 수"]) if candidate_rows else "<p>모델 후보 결과를 생성할 수 없습니다.</p>"}
      </div>
      <div class="subsection">
        <h3>모델 중요 피처 TOP 10</h3>
        {table_html(feature_rows, ["피처", "중요도"]) if feature_rows else "<p>선택 모델에서 중요 피처를 추출할 수 없습니다.</p>"}
      </div>
      <div class="subsection">
        <h3>최근 검증 경기</h3>
        {table_html(model_rows, ["경기일", "경기", "예측 구단", "예측승률", "실제 승리 구단", "결과", "예측 근거"], limit=12) if model_rows else "<p>모델 결과를 생성할 수 없습니다.</p>"}
      </div>
    </details>
    <p class="note">예측 모델은 매일 오전 갱신 기준 완료 경기만 학습/검증에 사용합니다. 확신 구간으로 갈수록 전체 경기보다 높은 적중률을 보였지만, 58% 이상·60% 이상 구간은 표본 수가 줄어들기 때문에 장기 검증이 더 필요합니다. 불펜 피로와 휴식일은 경기 단위 모델 피처로 반영했고, 선발투수는 경기 전 업데이트에서 GameCenter 확정 선발을 확인합니다. 라인업은 대시보드 판단 정보로 표시하지만, 과거 시점별 라인업 스냅샷이 쌓이기 전까지 최종 승패 모델 피처에는 직접 반영하지 않습니다.</p>
  </section>
  </div>
  <div id="runPrediction" class="tab-panel">
    {run_model_html}
  </div>
</main>
<script>
const TEAM_DATA = {payload};
document.querySelectorAll('.tab-button').forEach(button => button.addEventListener('click', () => {{
  document.querySelectorAll('.tab-button').forEach(item => item.classList.toggle('active', item === button));
  document.querySelectorAll('.tab-panel').forEach(panel => panel.classList.toggle('active', panel.id === button.dataset.tab));
}}));
function renderTable(rows, cols) {{
  if (!rows || rows.length === 0) return '<p class="note">표시할 데이터가 없습니다.</p>';
  return '<table><thead><tr>' + cols.map(c => `<th>${{c}}</th>`).join('') + '</tr></thead><tbody>' +
    rows.map(r => '<tr>' + cols.map(c => `<td>${{r[c] ?? ''}}</td>`).join('') + '</tr>').join('') +
    '</tbody></table>';
}}
function renderTeam(team) {{
  const data = TEAM_DATA[team];
  const s = data.standings;
  document.getElementById('teamTitle').textContent = `${{team}} 구단 상세 분석`;
  document.querySelectorAll('.team-button').forEach(btn => btn.classList.toggle('active', btn.dataset.team === team));
  document.getElementById('teamMetrics').innerHTML = [
    ['순위', s['순위']], ['시즌 전적', `${{s['승']}}승 ${{s['패']}}패 ${{s['무']}}무`],
    ['승률', s['승률']], ['최근10경기', s['최근10경기']]
  ].map(([k,v]) => `<div class="metric">${{k}}<strong>${{v}}</strong></div>`).join('');
  const analysisLink = document.getElementById('teamAnalysisLink');
  analysisLink.href = data.analysis_url || 'kt.html';
  analysisLink.textContent = `${{team}} 팀 분석 보기`;
  document.getElementById('todayPrediction').innerHTML = renderTable(data.today_predictions, ['경기일','기준팀','상대팀','예측 구단','예측승률','예측','예측 근거']);
  const lineup = data.lineup || {{}};
  document.getElementById('lineupSummary').innerHTML = '<div class="grid">' + [
    ['상태', lineup['상태'] || '라인업 정보 미확인'],
    ['선발 WAR 합', lineup['선발 WAR 합'] ?? '-'],
    ['상위 타순', lineup['라인업 요약'] || '-'],
    ['선수 수', (lineup.players || []).length]
  ].map(([k,v]) => `<div class="metric">${{k}}<strong>${{v}}</strong></div>`).join('') + '</div>';
  document.getElementById('lineupTable').innerHTML = renderTable(lineup.players, ['타순','포지션','선수','WAR']);
  document.getElementById('recentGames').innerHTML = renderTable(data.recent, ['경기일','상대','구분','결과','스코어']);
  document.getElementById('vsTable').innerHTML = renderTable(data.vs, ['상대','전적']);
  document.getElementById('hitterTable').innerHTML = renderTable(data.hitters, ['선수','경기','타석','타수','안타','홈런','볼넷','삼진','타율','출루율','장타율','OPS']);
  document.getElementById('pitcherTable').innerHTML = renderTable(data.pitchers, ['선수','경기','승','패','세이브','홀드','이닝','자책','탈삼진','볼넷','ERA','WHIP']);
}}
document.querySelectorAll('.team-button').forEach(btn => btn.addEventListener('click', () => renderTeam(btn.dataset.team)));
renderTeam(document.querySelector('.team-button').dataset.team);
</script>
</body>
</html>"""
    (DASHBOARD_DIR / "latest.html").write_text(html, encoding="utf-8")
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    (PUBLIC_DIR / "index.html").write_text(html, encoding="utf-8")
    (PUBLIC_DIR / "latest.html").write_text(html, encoding="utf-8")
    (DASHBOARD_DIR / "latest_summary.md").write_text(
        "\n".join(
            [
                "# KBO 리그 분석 대시보드",
                f"- 생성일: {generated_at.isoformat()}",
                f"- KBO 공식 순위 팀 수: {len(standings)}",
                f"- 2026 공식 일정 팀별 행 수: {len(games)}",
                f"- 모델 학습 기준일: {model_payload.get('training_cutoff', '')}",
                f"- 모델 검증 정확도: {model_payload.get('accuracy', '-')}",
            ]
        ),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description="Build KBO dashboard from official KBO records.")
    parser.add_argument("--reference-date", default=date.today().isoformat())
    parser.add_argument("--reference-datetime", default="")
    parser.add_argument("--update-stage", choices=["morning", "pregame"], default="morning")
    parser.add_argument("--training-start-year", type=int, default=2021)
    args = parser.parse_args()
    reference_datetime = datetime.strptime(args.reference_datetime, "%Y-%m-%d %H:%M") if args.reference_datetime else datetime.now()
    ref_date = datetime.strptime(args.reference_date, "%Y-%m-%d").date()
    standings, vs_table = fetch_team_standings()
    games = fetch_schedule(ref_date.year, ref_date.month)
    training_games = fetch_training_schedule(args.training_start_year, ref_date)
    hitters, pitchers = fetch_player_stats()
    rosters = fetch_registered_rosters()
    export_sources(standings, vs_table, games, hitters, pitchers, rosters)
    load_official_tables_to_db(standings, vs_table, games, hitters, pitchers, rosters)
    model_payload = run_model_evaluation(training_games, games, previous_sunday(ref_date), ref_date, DATA_DIR, RESULTS_DIR)
    team_pages = build_team_analysis_pages(standings, vs_table, games, hitters, pitchers, rosters, ref_date)
    build_dashboard(standings, vs_table, games, hitters, pitchers, model_payload, ref_date, team_pages, reference_datetime, args.update_stage)
    print(
        f"[Success] official KBO dashboard generated: teams={len(standings)}, "
        f"current_game_rows={len(games)}, training_game_rows={len(training_games)}"
    )


if __name__ == "__main__":
    main()
