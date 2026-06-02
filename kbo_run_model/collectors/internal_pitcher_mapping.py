from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError


SEARCH_TERMS = [
    "pitcher",
    "starter",
    "starting",
    "player",
    "batter",
    "lineup",
    "pitching",
    "innings",
    "era",
    "whip",
    "pitch_count",
    "confirmed_starters",
]
STARTER_TARGET_COLUMNS = [
    "season",
    "date",
    "game_id",
    "home_team",
    "away_team",
    "home_starter_name",
    "away_starter_name",
    "home_starter_id",
    "away_starter_id",
]
PITCHER_LOG_TARGET_COLUMNS = [
    "season",
    "date",
    "game_id",
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


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def _is_candidate(path: Path) -> bool:
    text = path.as_posix().lower()
    return path.suffix.lower() == ".csv" and any(term in text for term in SEARCH_TERMS)


def _columns_text(columns: list[str]) -> str:
    return ", ".join(columns)


def _find_column(columns: list[str], names: list[str]) -> str:
    lower_map = {column.lower(): column for column in columns}
    for name in names:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return ""


def _candidate_type(columns: list[str], path: Path) -> str:
    text = " ".join([path.name, *columns]).lower()
    if "confirmed_starters" in text or "starter_name" in text:
        return "starter_mapping"
    if "innings_pitched" in text or "earned_runs" in text or "pitches" in text:
        return "pitcher_game_log"
    if "pitcher" in text or "pitching" in text:
        return "pitcher_summary"
    if "player" in text or "batter" in text or "lineup" in text:
        return "player_context"
    return "unknown"


def scan_internal_pitcher_files(repo_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(repo_dir.rglob("*.csv")):
        relative_parts = path.relative_to(repo_dir).parts
        if "__pycache__" in relative_parts or relative_parts[:2] == ("kbo_run_model", "results"):
            continue
        if not _is_candidate(path):
            continue
        df = _read_csv(path)
        columns = df.columns.tolist()
        pitcher_name_col = _find_column(columns, ["pitcher_name", "starter_name", "player_name", "선수"])
        pitcher_id_col = _find_column(columns, ["pitcher_id", "player_id", "kbo_player_id"])
        candidate_type = _candidate_type(columns, path)
        has_game_id = "game_id" in columns
        has_date = "date" in columns
        has_team = "team" in columns or "팀" in columns
        has_innings = "innings_pitched" in columns or "이닝" in columns
        has_is_starter = "is_starter" in columns or "starter" in columns or "starter_name" in columns
        if candidate_type == "starter_mapping" and {"date", "team", "starter_name"}.issubset(columns):
            potential = "partial_starter_mapping"
        elif candidate_type == "pitcher_game_log" and {"game_id", "date", "player_id", "player_name", "team", "opponent", "innings_pitched"}.issubset(columns):
            potential = "partial_pitcher_log"
        else:
            potential = "not_enough_columns"
        rows.append(
            {
                "path": str(path.relative_to(repo_dir)).replace("\\", "/"),
                "rows": int(len(df)),
                "columns": _columns_text(columns),
                "has_game_id": has_game_id,
                "has_date": has_date,
                "has_team": has_team,
                "has_player_id": "player_id" in columns,
                "has_pitcher_id": "pitcher_id" in columns,
                "has_pitcher_name": bool(pitcher_name_col),
                "has_innings_pitched": has_innings,
                "has_is_starter": has_is_starter,
                "candidate_type": candidate_type,
                "mapping_potential": potential,
            }
        )
    return pd.DataFrame(rows)


def _starter_mapping_report(repo_dir: Path) -> dict:
    source = repo_dir / "kbo_analytics" / "data" / "manual" / "confirmed_starters.csv"
    df = _read_csv(source)
    columns = df.columns.tolist()
    missing = [column for column in ["date", "team", "starter_name"] if column not in columns]
    has_rows = len(df) > 0
    conversion_possible = False
    blocker = ""
    if missing:
        blocker = "필수 컬럼 누락: " + ", ".join(missing)
    elif not has_rows:
        blocker = "confirmed_starters.csv에 실제 row 없음"
    elif "game_id" not in columns:
        blocker = "game_id와 홈/원정 경기 묶음 컬럼 없음"
    elif not any(column in columns for column in ["starter_id", "pitcher_id", "player_id"]):
        blocker = "선발투수 ID 없음"
    return {
        "source_file": str(source.relative_to(repo_dir)).replace("\\", "/"),
        "target_schema": "starter_pitchers.csv",
        "required_columns_found": ", ".join([column for column in ["date", "team", "starter_name"] if column in columns]),
        "required_columns_missing": ", ".join(missing),
        "game_id_match_possible": "game_id" in columns,
        "date_team_match_possible": {"date", "team", "starter_name"}.issubset(columns),
        "conversion_possible": conversion_possible,
        "blocker": blocker,
    }


def _pitcher_log_report(repo_dir: Path) -> dict:
    source = repo_dir / "kbo_analytics" / "data" / "weekly" / "player_game_stats.csv"
    df = _read_csv(source)
    columns = df.columns.tolist()
    required = ["game_id", "date", "player_id", "player_name", "team", "opponent", "innings_pitched", "earned_runs", "hits_allowed", "walks_allowed", "strikeouts_pitched", "pitches"]
    missing = [column for column in required if column not in columns]
    blockers = []
    if missing:
        blockers.append("필수 컬럼 누락: " + ", ".join(missing))
    if "home_runs_allowed" not in columns:
        blockers.append("home_runs_allowed 컬럼 없음")
    if "is_starter" not in columns:
        blockers.append("명시적 is_starter 컬럼 없음")
    pitcher_rows = df[pd.to_numeric(df.get("innings_pitched", pd.Series(dtype=float)), errors="coerce").fillna(0).gt(0)] if "innings_pitched" in columns else pd.DataFrame()
    if pitcher_rows.empty:
        blockers.append("경기별 투수 등판 row 없음")
    conversion_possible = not blockers
    return {
        "source_file": str(source.relative_to(repo_dir)).replace("\\", "/"),
        "target_schema": "pitcher_game_logs.csv",
        "required_columns_found": ", ".join([column for column in required if column in columns]),
        "required_columns_missing": ", ".join(missing),
        "game_id_match_possible": "game_id" in columns,
        "date_team_match_possible": {"date", "team", "opponent"}.issubset(columns),
        "conversion_possible": conversion_possible,
        "blocker": "; ".join(blockers),
    }


def build_internal_pitcher_mapping_report(repo_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    inventory = scan_internal_pitcher_files(repo_dir)
    mapping = pd.DataFrame([_starter_mapping_report(repo_dir), _pitcher_log_report(repo_dir)])
    best_starter = mapping[mapping["target_schema"].eq("starter_pitchers.csv")].iloc[0].to_dict()
    best_log = mapping[mapping["target_schema"].eq("pitcher_game_logs.csv")].iloc[0].to_dict()
    starter_possible = bool(best_starter["conversion_possible"])
    log_possible = bool(best_log["conversion_possible"])
    blocker = "; ".join([value for value in [best_starter["blocker"], best_log["blocker"]] if value])
    summary = {
        "internal_pitcher_mapping_completed": True,
        "internal_pitcher_candidate_files": int(len(inventory)),
        "best_starter_source_file": best_starter["source_file"],
        "best_pitcher_log_source_file": best_log["source_file"],
        "starter_conversion_possible": starter_possible,
        "pitcher_log_conversion_possible": log_possible,
        "internal_pitcher_conversion_applied": False,
        "pitcher_data_ready_to_train_after_mapping": starter_possible and log_possible,
        "internal_pitcher_mapping_blocker": blocker,
        "next_recommended_pitcher_data_step": "home_runs_allowed와 명시적 is_starter가 포함된 경기별 투수 로그 및 pitcher_id 기반 선발 매핑을 확보합니다.",
    }
    return inventory, mapping, summary


def _conversion_check(target_file: str, attempted: bool, applied: bool, output_rows: int, validation_passed: bool, blocker: str) -> dict:
    return {
        "target_file": target_file,
        "conversion_attempted": attempted,
        "conversion_applied": applied,
        "output_rows": output_rows,
        "validation_passed": validation_passed,
        "blocker": blocker,
    }


def run_internal_pitcher_conversion(repo_dir: Path, apply: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    inventory, mapping, summary = build_internal_pitcher_mapping_report(repo_dir)
    checks = []
    for row in mapping.to_dict(orient="records"):
        checks.append(_conversion_check(row["target_schema"], apply, False, 0, bool(row["conversion_possible"]), row["blocker"]))
    summary["internal_pitcher_conversion_applied"] = False
    if apply and summary["starter_conversion_possible"] and summary["pitcher_log_conversion_possible"]:
        summary["internal_pitcher_conversion_applied"] = False
        summary["internal_pitcher_mapping_blocker"] = "변환 로직은 검증 가능 상태가 된 뒤 활성화해야 합니다."
    return inventory, mapping, pd.DataFrame(checks), summary


def write_internal_pitcher_mapping_outputs(repo_dir: Path, output_dir: Path, apply: bool = False) -> dict:
    inventory, mapping, checks, summary = run_internal_pitcher_conversion(repo_dir, apply)
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(output_dir / "internal_pitcher_data_inventory.csv", index=False, encoding="utf-8-sig")
    mapping.to_csv(output_dir / "internal_pitcher_mapping_report.csv", index=False, encoding="utf-8-sig")
    checks.to_csv(output_dir / "internal_pitcher_conversion_check.csv", index=False, encoding="utf-8-sig")
    return summary
