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
    return pd.DataFrame(rows)


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
            merged = basic.merge(advanced[["선수", "팀", "볼넷", "삼진", "장타율", "출루율", "OPS"]], on=["선수", "팀"], how="left")
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

        if starter is not None:
            starter_text = f"{starter['선수']} · ERA {starter['ERA']} · WHIP {starter['WHIP']}"
            starter_name = starter["선수"]
            starter_era = starter["ERA"]
            starter_whip = starter["WHIP"]
        else:
            starter_text = "추정 불가"
            starter_name = "-"
            starter_era = "-"
            starter_whip = "-"
        source, confirmed_at, quality = starter_source_for_game(games, team, prediction_date, reference_datetime, update_stage)
        source_label = "확정 선발" if source == "confirmed" else "예상 선발"
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


def build_prediction_cards(today_predictions: list[dict], pitching_context: dict | None = None):
    pitching_context = pitching_context or {}
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
        matchup = f'{row["기준팀"]} vs {row["상대팀"]}'
        cards[key] = {
            "경기": matchup,
            "추천": f'{row["예측 구단"]} {tier["우세"]}',
            "예측승률": f"{confidence:.1%}",
            "신뢰도": tier["신뢰도"],
            "핵심 근거": row.get("예측 근거", ""),
            "투수 신호": f'{pick_context.get("투수 표시", "예상 선발: 추정 불가")} · 불펜 피로 {pick_context.get("불펜 피로", "-")}',
            "판단": tier["판단"],
            "confidence_value": confidence,
        }
    return sorted(cards.values(), key=lambda row: row["confidence_value"], reverse=True)


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

    insight_rows = [
        {"인사이트": "시즌 위치", "내용": f"{team_standing['순위']}위, {wins}승 {losses}패 {draws}무, 승률 {pct(win_rate)}입니다."},
        {"인사이트": "득실 균형", "내용": f"득점 {runs_for}, 실점 {runs_against}, 득실차 {run_diff:+d}. 피타고리안 기대 승률은 {pct(pythag)}입니다."},
        {"인사이트": "팀 타격", "내용": f"등록 타자 기준 팀 타율 {team_avg:.3f}, 평균 출루율 {team_obp:.3f}, 평균 OPS {team_ops:.3f}입니다."},
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
    <div class="grid">
      <div class="metric">순위<strong>{team_standing["순위"]}위</strong></div>
      <div class="metric">전적<strong>{wins}승 {losses}패 {draws}무</strong></div>
      <div class="metric">득실차<strong>{run_diff:+d}</strong></div>
      <div class="metric">팀 타율<strong>{team_avg:.3f}</strong></div>
    </div>
    {table_html(insight_rows, ["인사이트", "내용"])}
  </section>
  <section class="section">
    <h2>감독·코치·등록 선수 구성</h2>
    <div class="grid">
      <div class="metric">감독<strong>{escape(str(roster.get("감독", "-")))}</strong></div>
      <div class="metric">투수 등록<strong>{len([p for p in str(roster.get("투수", "")).split(",") if p.strip()])}명</strong></div>
      <div class="metric">야수 등록<strong>{len([p for p in (str(roster.get("포수", "")) + "," + str(roster.get("내야수", "")) + "," + str(roster.get("외야수", ""))).split(",") if p.strip()])}명</strong></div>
      <div class="metric">산출 ERA<strong>{team_era:.2f}</strong></div>
    </div>
    {table_html([{"구분": "코치", "명단": roster.get("코치", "-")}, {"구분": "투수", "명단": roster.get("투수", "-")}, {"구분": "포수", "명단": roster.get("포수", "-")}, {"구분": "내야수", "명단": roster.get("내야수", "-")}, {"구분": "외야수", "명단": roster.get("외야수", "-")}], ["구분", "명단"])}
    <p class="note">감독·코치·등록 선수 구성은 KBO 공식 전체 등록 현황 기준입니다.</p>
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
            "analysis_url": team_pages.get(team, f"{TEAM_PAGE_SLUGS.get(team, team)}.html"),
        }

    payload = json.dumps(team_data, ensure_ascii=False)
    model_rows = model_payload.get("recent_backtest", []) if model_payload.get("available") else []
    confidence_rows = model_payload.get("confidence_metrics", []) if model_payload.get("available") else []
    calibration_rows = model_payload.get("calibration_table", []) if model_payload.get("available") else []
    candidate_rows = model_payload.get("candidate_results", []) if model_payload.get("available") else []
    pitching_context = build_pitching_context(games, pitchers, generated_at, reference_datetime, update_stage)
    export_pitching_context(pitching_context, DATA_DIR / "pitching_context.csv", generated_at)
    prediction_cards = build_prediction_cards(model_payload.get("today_predictions", []), pitching_context)
    summary = today_summary(prediction_cards)
    prediction_cards_html = "".join(
        f"""
        <article class="prediction-card">
          <div class="matchup">{escape(row["경기"])}</div>
          <h3>{escape(row["추천"])} <span>{escape(row["예측승률"])}</span></h3>
          <div class="badges"><span>신뢰도 {escape(row["신뢰도"])}</span><span>{escape(row["판단"])}</span></div>
          <p>{escape(row["핵심 근거"])}</p>
          <p class="pitching-signal">{escape(row["투수 신호"])}</p>
        </article>
        """
        for row in prediction_cards
    ) or '<p class="note">오늘 표시할 예측 카드가 없습니다.</p>'
    team_buttons = "".join(
        f'<button type="button" class="team-button" data-team="{escape(team)}">{escape(team)}</button>'
        for team in standings["팀"]
    )
    html = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>KBO 리그 분석 대시보드</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; color: #1b1f24; background: #f4f6f8; }}
    header {{ background: #172033; color: white; padding: 30px 32px; }}
    main {{ padding: 24px 32px 48px; max-width: 1440px; margin: 0 auto; }}
    h1, h2, h3 {{ margin: 0 0 14px; }}
    .meta {{ color: #d6e0ef; margin-top: 8px; }}
    .section {{ margin-top: 22px; background: white; border: 1px solid #dde3ea; border-radius: 8px; padding: 18px; }}
    .section-title {{ display: flex; align-items: baseline; justify-content: space-between; gap: 12px; margin-bottom: 14px; }}
    .eyebrow {{ color: #637083; font-size: 13px; font-weight: 700; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }}
    .metric {{ border: 1px solid #e1e7ef; border-radius: 8px; padding: 14px; background: #fbfcfe; }}
    .metric strong {{ display: block; font-size: 24px; margin-top: 6px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ padding: 9px 10px; border-bottom: 1px solid #e5e9f0; text-align: right; white-space: nowrap; }}
    th:first-child, td:first-child, td:nth-child(2) {{ text-align: left; }}
    th {{ background: #f0f3f7; font-weight: 700; }}
    select {{ padding: 9px 12px; border: 1px solid #bcc7d4; border-radius: 6px; font-size: 15px; }}
    .team-picker {{ display: grid; grid-template-columns: 220px 1fr; gap: 16px; align-items: start; margin-bottom: 18px; }}
    .team-buttons {{ display: grid; grid-template-columns: repeat(10, minmax(0, 1fr)); gap: 8px; }}
    .team-button {{ border: 1px solid #c8d2df; background: #fff; border-radius: 6px; padding: 9px 6px; cursor: pointer; font-weight: 700; }}
    .team-button.active {{ background: #172033; border-color: #172033; color: white; }}
    .subsection {{ margin-top: 18px; }}
    .action-link {{ display:inline-block; margin-top:10px; padding:9px 12px; border-radius:6px; background:#172033; color:white; font-weight:700; text-decoration:none; }}
    .tables {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }}
    .wide-table {{ overflow-x: auto; }}
    .note {{ color: #637083; font-size: 13px; margin-top: 10px; }}
    .insight-lead {{ font-size: 18px; line-height: 1.55; margin: 0 0 16px; }}
    .prediction-cards {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; }}
    .prediction-card {{ border:1px solid #d9e1ec; border-radius:8px; padding:16px; background:#fbfcfe; }}
    .prediction-card .matchup {{ color:#637083; font-size:13px; font-weight:700; margin-bottom:8px; }}
    .prediction-card h3 {{ display:flex; justify-content:space-between; gap:10px; font-size:20px; }}
    .prediction-card h3 span {{ color:#1d4ed8; white-space:nowrap; }}
    .pitching-signal {{ color:#374151; font-size:13px; margin-top:8px; }}
    .badges {{ display:flex; gap:6px; flex-wrap:wrap; margin:10px 0; }}
    .badges span {{ border:1px solid #c8d2df; border-radius:999px; padding:4px 8px; font-size:12px; font-weight:700; background:white; }}
    details {{ margin-top:16px; }}
    summary {{ cursor:pointer; font-weight:700; }}
    @media (max-width: 960px) {{ .grid, .tables, .team-picker, .team-buttons {{ grid-template-columns: 1fr; }} main {{ padding: 16px; }} }}
    @media (max-width: 960px) {{ .prediction-cards {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
<header>
  <h1>KBO 리그 분석 대시보드</h1>
  <div class="meta">KBO 공식 기록 기준 · 생성일 {generated_at.isoformat()} · 모델 학습 기준일 {escape(model_payload.get("training_cutoff", ""))}</div>
</header>
<main>
  <section class="section">
    <div class="eyebrow">TODAY · 오늘의 판단</div>
    <h2>오늘의 KBO 예측 요약</h2>
    <p class="insight-lead">{escape(summary["headline"])}</p>
    <div class="grid">
      <div class="metric">가장 높은 예측<strong>{escape(str(summary["top_pick"]))}</strong></div>
      <div class="metric">예측 가능 경기<strong>{escape(str(summary["possible_games"]))}</strong></div>
      <div class="metric">박빙/참고 경기<strong>{escape(str(summary["close_games"]))}</strong></div>
      <div class="metric">모델 한계<strong>선발·불펜 추정, 라인업 미반영</strong></div>
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
    <div class="eyebrow">ETC · 모델링 성능 참고</div>
    <h2>경기 승패 예측 모델 검증</h2>
    <div class="grid">
      <div class="metric">학습 시즌<strong>{model_payload.get("training_start_year", "-")}~{model_payload.get("training_end_year", "-")}</strong></div>
      <div class="metric">학습 행<strong>{model_payload.get("train_rows", "-")}</strong></div>
      <div class="metric">검증 행<strong>{model_payload.get("test_rows", "-")}</strong></div>
      <div class="metric">검증 정확도<strong>{model_payload.get("accuracy", "-")}</strong></div>
    </div>
    <p class="note">모델 상태: 전체 적중률 {model_payload.get("accuracy", "-")}, 55% 이상 예측 경기 적중률 {confidence_rows[1]["적중률"] if len(confidence_rows) > 1 else "-"}입니다. 60% 이상 구간은 과신 가능성이 있어 강한 정배가 아니라 참고 신호로 해석합니다.</p>
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
        <h3>최근 검증 경기</h3>
        {table_html(model_rows, ["경기일", "기준팀", "상대팀", "예측 구단", "예측승률", "예측", "실제 승리 구단", "예측 근거"], limit=12) if model_rows else "<p>모델 결과를 생성할 수 없습니다.</p>"}
      </div>
    </details>
    <p class="note">예측 모델은 매일 오전 갱신 기준 완료 경기만 학습/검증에 사용합니다. 55% 이상 구간은 전체보다 높은 적중률을 보였지만, 58% 이상·60% 이상 구간은 아직 안정적인 개선이 확인되지 않았습니다. 불펜 피로와 휴식일은 경기 단위 모델 피처로 반영했고, 선발투수는 경기 전 업데이트에서 GameCenter 확정 선발을 확인합니다. 미확인 시 누적 기록과 로테이션 순서 기반 예상 선발로 표시합니다. 확정 라인업과 엔트리 변동은 아직 직접 반영하지 않습니다.</p>
  </section>
</main>
<script>
const TEAM_DATA = {payload};
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
  analysisLink.textContent = `${team} 팀 분석 보기`;
  document.getElementById('todayPrediction').innerHTML = renderTable(data.today_predictions, ['경기일','기준팀','상대팀','예측 구단','예측승률','예측','예측 근거']);
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
