from __future__ import annotations

from pathlib import Path

import pandas as pd


SEARCH_TERMS = [
    "pitcher",
    "starter",
    "starting_pitcher",
    "player",
    "player_id",
    "era",
    "whip",
    "innings_pitched",
    "earned_runs",
    "strikeouts",
    "walks",
    "pitches",
]

V2_COLUMNS = [
    "season",
    "date",
    "game_id",
    "home_team",
    "away_team",
    "home_starter_name",
    "away_starter_name",
    "home_starter_id",
    "away_starter_id",
    "pitcher_id",
    "pitcher_name",
    "team",
    "opponent",
    "is_starter",
    "innings_pitched",
    "earned_runs",
    "hits_allowed",
    "walks",
    "strikeouts",
    "home_runs_allowed",
    "pitches",
]

COLUMN_ALIASES = {
    "player_id": "pitcher_id",
    "player_name": "pitcher_name",
    "strikeouts_pitched": "strikeouts",
    "walks_allowed": "walks",
    "이닝": "innings_pitched",
    "자책": "earned_runs",
    "피안타": "hits_allowed",
    "볼넷": "walks",
    "탈삼진": "strikeouts",
    "피홈런": "home_runs_allowed",
    "선수": "pitcher_name",
    "팀": "team",
    "ERA": "era",
    "WHIP": "whip",
}

REQUIRED_FOR_TRAINING = {
    "starter_mapping": ["date", "game_id", "home_team", "away_team", "home_starter_id", "away_starter_id"],
    "pitcher_logs": [
        "date",
        "game_id",
        "pitcher_id",
        "team",
        "opponent",
        "is_starter",
        "innings_pitched",
        "earned_runs",
        "hits_allowed",
        "walks",
        "strikeouts",
        "home_runs_allowed",
        "pitches",
    ],
}


def _normalized_columns(columns: list[str]) -> list[str]:
    return [COLUMN_ALIASES.get(column, column) for column in columns]


def _row_count(path: Path) -> int:
    return max(sum(1 for _ in path.open("r", encoding="utf-8-sig")) - 1, 0)


def _inspect_csv(path: Path, repo_dir: Path) -> dict:
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    normalized = _normalized_columns(columns)
    text = " ".join([path.name, *columns]).lower()
    matched_terms = [term for term in SEARCH_TERMS if term.lower() in text]
    mapped_columns = [column for column in V2_COLUMNS if column in normalized]
    starter_missing = [column for column in REQUIRED_FOR_TRAINING["starter_mapping"] if column not in normalized]
    log_missing = [column for column in REQUIRED_FOR_TRAINING["pitcher_logs"] if column not in normalized]
    sample_text = " ".join(pd.read_csv(path, dtype=str).head(5).fillna("").to_numpy().ravel()).upper()
    sample_only = "SAMPLE_" in sample_text or "MOCK_" in sample_text
    usable_as_starter_mapping = not starter_missing and _row_count(path) > 0 and not sample_only
    usable_as_pitcher_logs = not log_missing and _row_count(path) > 0 and not sample_only

    return {
        "file": str(path.relative_to(repo_dir)).replace("\\", "/"),
        "columns": columns,
        "normalized_columns": normalized,
        "matched_terms": matched_terms,
        "mapped_v2_columns": mapped_columns,
        "missing_starter_mapping_columns": starter_missing,
        "missing_pitcher_log_columns": log_missing,
        "row_count": _row_count(path),
        "sample_only": sample_only,
        "usable_as_starter_mapping": usable_as_starter_mapping,
        "usable_as_pitcher_logs": usable_as_pitcher_logs,
        "ready_to_use": bool(usable_as_starter_mapping or usable_as_pitcher_logs),
    }


def search_internal_pitcher_data(repo_dir: Path) -> dict:
    csv_files = sorted(repo_dir.glob("**/*.csv"))
    inspected = []
    for path in csv_files:
        relative = str(path.relative_to(repo_dir)).replace("\\", "/")
        if relative.startswith("kbo_run_model/results/"):
            continue
        result = _inspect_csv(path, repo_dir)
        if result["matched_terms"]:
            inspected.append(result)

    starter_ready = any(row["usable_as_starter_mapping"] for row in inspected)
    logs_ready = any(row["usable_as_pitcher_logs"] for row in inspected)
    internal_found = any(not row["file"].startswith("kbo_run_model/data/") for row in inspected)
    blocker = ""
    if not internal_found:
        blocker = "실제 데이터 없음"
    elif not starter_ready and not logs_ready:
        blocker = "경기별 선발투수 매핑과 필수 투수 로그 컬럼 부족"
    elif not starter_ready:
        blocker = "경기별 선발투수 매핑 컬럼 부족"
    elif not logs_ready:
        blocker = "필수 투수 등판 로그 컬럼 부족"

    return {
        "v2_data_search_completed": True,
        "internal_pitcher_data_found": bool(internal_found),
        "v2_ready_to_train": bool(starter_ready and logs_ready),
        "v2_blocker": blocker,
        "files": inspected,
    }
