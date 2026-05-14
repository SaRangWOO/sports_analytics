from __future__ import annotations

import argparse
import json
import os
import re
from datetime import date, datetime, timedelta
from html import escape, unescape
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from modeling.feature_engineering import build_features
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


def fetch_schedule(year: int, through_month: int):
    rows = []
    for month in range(3, through_month + 1):
        rows.extend(fetch_schedule_month(year, month))
    return pd.DataFrame(rows)


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


def export_sources(standings, vs_table, games, hitters, pitchers):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    standings.to_csv(DATA_DIR / "team_standings.csv", index=False, encoding="utf-8-sig")
    vs_table.to_csv(DATA_DIR / "team_vs_team.csv", index=False, encoding="utf-8-sig")
    games.to_csv(DATA_DIR / "game_results.csv", index=False, encoding="utf-8-sig")
    hitters.to_csv(DATA_DIR / "hitter_stats.csv", index=False, encoding="utf-8-sig")
    pitchers.to_csv(DATA_DIR / "pitcher_stats.csv", index=False, encoding="utf-8-sig")


def load_official_tables_to_db(standings, vs_table, games, hitters, pitchers):
    from sqlalchemy import create_engine

    engine = create_engine(DB_URL)
    tables = {
        "game_results": games,
        "official_team_standings": standings,
        "official_team_vs_team": vs_table,
        "official_hitter_stats": hitters,
        "official_pitcher_stats": pitchers,
    }
    with engine.begin() as connection:
        for table_name, dataframe in tables.items():
            dataframe.to_sql(table_name, connection, if_exists="replace", index=False)


def evaluate_model(games: pd.DataFrame, cutoff: date):
    completed = games[(games["status"] == "Final") & (pd.to_datetime(games["date"]).dt.date <= cutoff)].copy()
    model_input = DATA_DIR / "model_training_games.csv"
    completed.to_csv(model_input, index=False, encoding="utf-8-sig")
    features = build_features(model_input)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    features.to_csv(RESULTS_DIR / "features.csv", index=False, encoding="utf-8-sig")

    if len(features) < 20:
        return {"available": False, "reason": "학습 가능한 완료 경기가 부족합니다.", "training_cutoff": cutoff.isoformat()}

    x, y = prepare_matrix(features)
    split_index = max(int(len(x) * 0.8), 1)
    split_index = min(split_index, len(x) - 1)
    x_train, x_test = x.iloc[:split_index], x.iloc[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    train_scaled, test_scaled, mean, std = standardize_train_test(x_train, x_test)
    weights, bias = train_logistic_regression(train_scaled.to_numpy(), y_train, lr=0.05, epochs=3500)
    probability = sigmoid(test_scaled.to_numpy() @ weights + bias)
    pred = (probability >= 0.5).astype(int)
    accuracy = round(float((pred == y_test).mean()), 3)
    recent = features.iloc[split_index:].copy()
    recent["예측승률"] = [f"{p:.1%}" for p in probability]
    recent["예측"] = np.where(pred == 1, "승리 예측", "패배 예측")
    recent["실제"] = np.where(y_test == 1, "승", "패")
    recent["date"] = pd.to_datetime(recent["date"]).dt.strftime("%Y-%m-%d")
    payload = {
        "available": True,
        "training_cutoff": cutoff.isoformat(),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "accuracy": accuracy,
        "recent_backtest": recent[["date", "team", "opponent", "예측승률", "예측", "실제"]].tail(12).to_dict(orient="records"),
        "source_note": "현재 주 경기는 적중/오답 집계에 포함하지 않습니다.",
        "feature_columns": list(x.columns),
        "bias": round(float(bias), 6),
        "coefficients": {name: round(float(value), 6) for name, value in sorted(zip(x.columns, weights), key=lambda x: abs(x[1]), reverse=True)},
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


def build_dashboard(standings, vs_table, games, hitters, pitchers, model_payload, generated_at: date):
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    team_data = {}
    completed = games[games["status"] == "Final"].copy()
    completed["date"] = pd.to_datetime(completed["date"])
    league_completed_games = completed["game_id"].str.rsplit("_", n=1).str[0].nunique()
    league_games = standings["경기"].sum() // 2
    league_leader = standings.iloc[0]
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
        }

    payload = json.dumps(team_data, ensure_ascii=False)
    model_rows = model_payload.get("recent_backtest", []) if model_payload.get("available") else []
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
    .tables {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }}
    .wide-table {{ overflow-x: auto; }}
    .note {{ color: #637083; font-size: 13px; margin-top: 10px; }}
    @media (max-width: 960px) {{ .grid, .tables, .team-picker, .team-buttons {{ grid-template-columns: 1fr; }} main {{ padding: 16px; }} }}
  </style>
</head>
<body>
<header>
  <h1>KBO 리그 분석 대시보드</h1>
  <div class="meta">KBO 공식 기록 기준 · 생성일 {generated_at.isoformat()} · 모델 학습 기준일 {escape(model_payload.get("training_cutoff", ""))}</div>
</header>
<main>
  <section class="section">
    <div class="section-title">
      <div>
        <div class="eyebrow">STEP 1 · 리그 전체 상황</div>
        <h2>KBO 리그 전체 순위와 시즌 흐름</h2>
      </div>
    </div>
    <div class="grid">
      <div class="metric">1위<strong>{escape(str(league_leader["팀"]))}</strong></div>
      <div class="metric">리그 완료 경기<strong>{league_completed_games}</strong></div>
      <div class="metric">순위표 기준 경기<strong>{league_games}</strong></div>
      <div class="metric">생성일<strong>{generated_at.isoformat()}</strong></div>
    </div>
    <div class="wide-table">
      {table_html(standings, ["순위", "팀", "경기", "승", "패", "무", "승률", "게임차", "최근10경기", "연속", "홈", "방문"])}
    </div>
  </section>

  <section class="section">
    <div class="section-title">
      <div>
        <div class="eyebrow">STEP 2 · 원하는 구단 선택</div>
        <h2 id="teamTitle">구단 상세 분석</h2>
      </div>
      <select id="teamSelect" aria-label="구단 선택">{"".join(f'<option value="{escape(team)}">{escape(team)}</option>' for team in standings["팀"])}</select>
    </div>
    <div class="team-picker">
      <div class="note">버튼이나 선택 상자에서 구단을 바꾸면 아래 최근 경기, 상대 전적, 선수 기록이 해당 구단 기준으로 바뀝니다.</div>
      <div class="team-buttons">{team_buttons}</div>
    </div>
    <div class="grid" id="teamMetrics"></div>
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
    <div class="eyebrow">STEP 3 · 모델링</div>
    <h2>경기 승패 예측 모델</h2>
    <div class="grid">
      <div class="metric">학습 행<strong>{model_payload.get("train_rows", "-")}</strong></div>
      <div class="metric">검증 행<strong>{model_payload.get("test_rows", "-")}</strong></div>
      <div class="metric">검증 정확도<strong>{model_payload.get("accuracy", "-")}</strong></div>
      <div class="metric">학습 기준<strong>{escape(model_payload.get("training_cutoff", ""))}</strong></div>
    </div>
    {table_html(model_rows, ["date", "team", "opponent", "예측승률", "예측", "실제"], limit=12) if model_rows else "<p>모델 결과를 생성할 수 없습니다.</p>"}
    <p class="note">예측 모델은 월요일 갱신 기준 지난주까지의 완료 경기만 학습/검증에 사용합니다. 현재 주 경기 결과는 적중률 계산에 섞지 않습니다.</p>
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
  document.getElementById('teamSelect').value = team;
  document.getElementById('teamMetrics').innerHTML = [
    ['순위', s['순위']], ['시즌 전적', `${{s['승']}}승 ${{s['패']}}패 ${{s['무']}}무`],
    ['승률', s['승률']], ['최근10경기', s['최근10경기']]
  ].map(([k,v]) => `<div class="metric">${{k}}<strong>${{v}}</strong></div>`).join('');
  document.getElementById('recentGames').innerHTML = renderTable(data.recent, ['경기일','상대','구분','결과','스코어']);
  document.getElementById('vsTable').innerHTML = renderTable(data.vs, ['상대','전적']);
  document.getElementById('hitterTable').innerHTML = renderTable(data.hitters, ['선수','경기','타석','타수','안타','홈런','볼넷','삼진','타율','출루율','장타율','OPS']);
  document.getElementById('pitcherTable').innerHTML = renderTable(data.pitchers, ['선수','경기','승','패','세이브','홀드','이닝','자책','탈삼진','볼넷','ERA','WHIP']);
}}
document.getElementById('teamSelect').addEventListener('change', e => renderTeam(e.target.value));
document.querySelectorAll('.team-button').forEach(btn => btn.addEventListener('click', () => renderTeam(btn.dataset.team)));
renderTeam(document.getElementById('teamSelect').value);
</script>
</body>
</html>"""
    (DASHBOARD_DIR / "latest.html").write_text(html, encoding="utf-8")
    (DASHBOARD_DIR / "latest_summary.md").write_text(
        "\n".join(
            [
                "# KBO 리그 분석 대시보드",
                f"- 생성일: {generated_at.isoformat()}",
                f"- KBO 공식 순위 팀 수: {len(standings)}",
                f"- 공식 일정 팀별 행 수: {len(games)}",
                f"- 모델 학습 기준일: {model_payload.get('training_cutoff', '')}",
                f"- 모델 검증 정확도: {model_payload.get('accuracy', '-')}",
            ]
        ),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description="Build KBO dashboard from official KBO records.")
    parser.add_argument("--reference-date", default=date.today().isoformat())
    args = parser.parse_args()
    ref_date = datetime.strptime(args.reference_date, "%Y-%m-%d").date()
    standings, vs_table = fetch_team_standings()
    games = fetch_schedule(ref_date.year, ref_date.month)
    hitters, pitchers = fetch_player_stats()
    export_sources(standings, vs_table, games, hitters, pitchers)
    load_official_tables_to_db(standings, vs_table, games, hitters, pitchers)
    model_payload = evaluate_model(games, previous_sunday(ref_date))
    build_dashboard(standings, vs_table, games, hitters, pitchers, model_payload, ref_date)
    print(f"[Success] official KBO dashboard generated: teams={len(standings)}, game_rows={len(games)}")


if __name__ == "__main__":
    main()
