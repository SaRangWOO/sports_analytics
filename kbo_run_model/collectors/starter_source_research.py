from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests


SOURCE_URLS = {
    "kbo": "https://www.koreabaseball.com/Schedule/ScoreBoard.aspx",
    "naver": "https://m.sports.naver.com/kbaseball/schedule/index",
    "daum": "https://sports.daum.net/schedule/KBO",
    "statiz": "https://www.statiz.co.kr/prediction/",
    "manual": "kbo_run_model/data/starter_pitchers.csv",
}
RESEARCH_COLUMNS = [
    "source_name",
    "source_url",
    "date_checked",
    "access_ok",
    "games_found",
    "home_team_found",
    "away_team_found",
    "home_starter_found",
    "away_starter_found",
    "pitcher_id_available",
    "game_id_match_possible",
    "requires_javascript",
    "robots_or_terms_risk",
    "rate_limit_risk",
    "parsing_stability",
    "implementation_difficulty",
    "viability_status",
    "blocker",
    "recommendation_rank",
]


def _get(url: str) -> tuple[bool, str, str]:
    try:
        response = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        return response.status_code == 200, response.text, response.url
    except requests.RequestException as exc:
        return False, "", f"{type(exc).__name__}: {exc}"


def _robots_risk(url: str) -> str:
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return "none"
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    ok, text, _ = _get(robots_url)
    if not ok:
        return "unknown"
    lowered = text.lower()
    if "disallow: /" in lowered:
        return "high"
    if "crawl-delay" in lowered or "disallow" in lowered:
        return "medium"
    return "low"


def _schedule_games(schedule: pd.DataFrame, target_date: date) -> pd.DataFrame:
    frame = schedule.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    frame = frame[frame["date"].eq(target_date)].copy()
    frame["game_key"] = frame["game_id"].astype(str).str.rsplit("_", n=1).str[0]
    rows = []
    for game_key, group in frame.groupby("game_key", sort=True):
        home = group[group["home_away"].eq("H")]
        away = group[group["home_away"].eq("A")]
        if home.empty or away.empty:
            continue
        rows.append({"game_id": game_key, "home_team": home.iloc[0]["team"], "away_team": away.iloc[0]["team"]})
    return pd.DataFrame(rows)


def _base_result(source: str, target_date: date) -> dict:
    return {
        "source_name": source,
        "source_url": SOURCE_URLS[source],
        "date_checked": target_date.isoformat(),
        "access_ok": False,
        "games_found": 0,
        "home_team_found": False,
        "away_team_found": False,
        "home_starter_found": False,
        "away_starter_found": False,
        "pitcher_id_available": False,
        "game_id_match_possible": False,
        "requires_javascript": False,
        "robots_or_terms_risk": "unknown",
        "rate_limit_risk": "unknown",
        "parsing_stability": "unknown",
        "implementation_difficulty": "unknown",
        "viability_status": "not_viable",
        "blocker": "",
        "recommendation_rank": 99,
    }


def _research_kbo(target_date: date, schedule_games: pd.DataFrame) -> dict:
    result = _base_result("kbo", target_date)
    ok, text, final_url = _get(SOURCE_URLS["kbo"])
    lowered = text.lower()
    result.update(
        {
            "source_url": final_url,
            "access_ok": ok,
            "games_found": int(len(schedule_games)) if ok else 0,
            "home_team_found": ok,
            "away_team_found": ok,
            "requires_javascript": False,
            "robots_or_terms_risk": _robots_risk(SOURCE_URLS["kbo"]),
            "rate_limit_risk": "low",
            "parsing_stability": "medium",
            "implementation_difficulty": "medium",
        }
    )
    if ok and not any(term in lowered for term in ["선발투수", "선발 투수", "starting pitcher", "probable pitcher"]):
        result["blocker"] = "공식 일정/스코어보드 HTML에서 선발투수 필드 확인 불가"
    return result


def _research_naver(target_date: date, schedule_games: pd.DataFrame) -> dict:
    result = _base_result("naver", target_date)
    ok, text, final_url = _get(SOURCE_URLS["naver"])
    lowered = text.lower()
    result.update(
        {
            "source_url": final_url,
            "access_ok": ok,
            "games_found": int(len(schedule_games)) if ok and target_date.isoformat() in lowered else 0,
            "requires_javascript": True,
            "robots_or_terms_risk": _robots_risk(SOURCE_URLS["naver"]),
            "rate_limit_risk": "medium",
            "parsing_stability": "low",
            "implementation_difficulty": "high",
            "blocker": "초기 HTML에서 2026-06-02 경기와 선발투수명 확인 불가, JavaScript/API 의존 가능성",
        }
    )
    return result


def _research_daum(target_date: date, schedule_games: pd.DataFrame) -> dict:
    result = _base_result("daum", target_date)
    ok, text, final_url = _get(SOURCE_URLS["daum"])
    lowered = text.lower()
    result.update(
        {
            "source_url": final_url,
            "access_ok": ok,
            "games_found": int(len(schedule_games)) if ok and target_date.isoformat() in lowered else 0,
            "requires_javascript": True,
            "robots_or_terms_risk": _robots_risk(SOURCE_URLS["daum"]),
            "rate_limit_risk": "medium",
            "parsing_stability": "low",
            "implementation_difficulty": "high",
            "viability_status": "partially_viable" if ok else "not_viable",
            "blocker": "뉴스 기사에서는 일부 선발명을 확인할 수 있으나 일정 페이지 초기 HTML에서 경기별 구조화 데이터 확인 불가",
            "recommendation_rank": 3 if ok else 99,
        }
    )
    return result


def _parse_statiz_prediction_page(s_no: int, target_date: date) -> dict | None:
    url = f"https://www.statiz.co.kr/prediction/?s_no={s_no}"
    ok, text, _ = _get(url)
    if not ok:
        return None
    game_date = re.search(r"gameDate2\s*=\s*'([^']+)'", text)
    if not game_date or game_date.group(1) != target_date.isoformat():
        return None
    names = re.findall(r'<div class="name">\s*([^<]+?)\s*</div>', text)
    team_block = re.search(r'<div class="team_name">(.*?)</div>\s*<div class="predict_text"', text, flags=re.S)
    teams = []
    if team_block:
        teams = [value.strip() for value in re.sub(r"<[^>]+>", " ", team_block.group(1)).split() if value.strip()]
    return {
        "s_no": s_no,
        "url": url,
        "starter_names": names[:2],
        "teams": teams,
        "pitcher_ids": re.findall(r"p_no=(\d+)", text)[:2],
    }


def _research_statiz(target_date: date, schedule_games: pd.DataFrame) -> dict:
    result = _base_result("statiz", target_date)
    ok, text, final_url = _get(SOURCE_URLS["statiz"])
    pages = []
    for s_no in range(20260250, 20260320):
        page = _parse_statiz_prediction_page(s_no, target_date)
        if page:
            pages.append(page)
    matchups = {
        (str(row.away_team), str(row.home_team))
        for row in schedule_games.itertuples(index=False)
    }
    matched_pages = [
        page
        for page in pages
        if len(page["teams"]) >= 2 and (str(page["teams"][0]), str(page["teams"][1])) in matchups
    ]
    pages = matched_pages
    games_found = len(pages)
    has_starters = bool(pages) and all(len(page["starter_names"]) >= 2 for page in pages)
    has_ids = bool(pages) and all(len(page["pitcher_ids"]) >= 2 for page in pages)
    result.update(
        {
            "source_url": final_url,
            "access_ok": ok,
            "games_found": games_found,
            "home_team_found": games_found >= min(len(schedule_games), 1),
            "away_team_found": games_found >= min(len(schedule_games), 1),
            "home_starter_found": has_starters,
            "away_starter_found": has_starters,
            "pitcher_id_available": has_ids,
            "game_id_match_possible": games_found == len(schedule_games) and len(schedule_games) > 0,
            "requires_javascript": False,
            "robots_or_terms_risk": _robots_risk(SOURCE_URLS["statiz"]),
            "rate_limit_risk": "medium",
            "parsing_stability": "medium",
            "implementation_difficulty": "medium",
            "viability_status": "viable" if games_found == len(schedule_games) and has_starters else "partially_viable",
            "blocker": "" if games_found == len(schedule_games) and has_starters else "날짜별 경기 전체 s_no 탐색 안정성 추가 검증 필요",
            "recommendation_rank": 1 if games_found == len(schedule_games) and has_starters else 2,
        }
    )
    if pages:
        result["source_url"] = " | ".join(page["url"] for page in pages[:5])
    return result


def _research_manual(target_date: date, schedule_games: pd.DataFrame) -> dict:
    result = _base_result("manual", target_date)
    result.update(
        {
            "access_ok": True,
            "games_found": int(len(schedule_games)),
            "home_team_found": True,
            "away_team_found": True,
            "home_starter_found": False,
            "away_starter_found": False,
            "pitcher_id_available": False,
            "game_id_match_possible": True,
            "requires_javascript": False,
            "robots_or_terms_risk": "none",
            "rate_limit_risk": "none",
            "parsing_stability": "high",
            "implementation_difficulty": "low",
            "viability_status": "manual_only",
            "blocker": "자동 수집은 아니며 사람이 선발투수명을 입력해야 함",
            "recommendation_rank": 2,
        }
    )
    return result


def research_sources(schedule: pd.DataFrame, target_date: date, source: str = "all") -> pd.DataFrame:
    schedule_games = _schedule_games(schedule, target_date)
    source_map = {
        "kbo": _research_kbo,
        "naver": _research_naver,
        "daum": _research_daum,
        "statiz": _research_statiz,
        "manual": _research_manual,
    }
    selected = list(source_map) if source == "all" else [source]
    rows = [source_map[name](target_date, schedule_games) for name in selected]
    return pd.DataFrame(rows)[RESEARCH_COLUMNS]


def summarize_source_research(results: pd.DataFrame) -> dict:
    viable = results[results["viability_status"].eq("viable")]
    recommended = results.sort_values(["recommendation_rank", "source_name"]).iloc[0] if not results.empty else {}
    blocker = "" if not viable.empty else "자동 수집 가능한 선발투수 원천 확정 필요"
    return {
        "starter_pitcher_source_research_completed": True,
        "starter_pitcher_sources_checked": int(len(results)),
        "viable_starter_pitcher_sources": int(len(viable)),
        "recommended_starter_pitcher_source": str(recommended.get("source_name", "")),
        "recommended_starter_pitcher_source_rank": int(recommended.get("recommendation_rank", 99)),
        "starter_pitcher_source_blocker": blocker,
        "next_recommended_starter_collection_step": "Statiz 승부예측 페이지의 날짜별 s_no 탐색과 팀명 매칭을 collector로 분리 검증합니다." if not results.empty else "선발투수 원천 조사를 먼저 실행합니다.",
    }


def write_source_research_outputs(schedule: pd.DataFrame, target_date: date, output_dir: Path, source: str = "all") -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    results = research_sources(schedule, target_date, source)
    path = output_dir / "starter_pitcher_source_research.csv"
    if source == "all" or not path.exists():
        output = results
    else:
        previous = pd.read_csv(path)
        output = pd.concat([previous[~previous["source_name"].isin(results["source_name"])], results], ignore_index=True)
    output = output.sort_values(["recommendation_rank", "source_name"]).reset_index(drop=True)
    output.to_csv(path, index=False, encoding="utf-8-sig")
    return summarize_source_research(output)


def load_source_research_summary(path: Path) -> dict:
    if not path.exists():
        return {
            "starter_pitcher_source_research_completed": False,
            "starter_pitcher_sources_checked": 0,
            "viable_starter_pitcher_sources": 0,
            "recommended_starter_pitcher_source": "",
            "recommended_starter_pitcher_source_rank": 99,
            "starter_pitcher_source_blocker": "선발투수 원천 조사 미실행",
            "next_recommended_starter_collection_step": "python scripts/research_starter_sources.py --date 2026-06-02를 실행합니다.",
        }
    return summarize_source_research(pd.read_csv(path))
