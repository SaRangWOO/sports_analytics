from __future__ import annotations

import html
import json
import os
import shutil
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import requests


KBO_BASE = "https://www.koreabaseball.com"
BOX_SCORE_URL = f"{KBO_BASE}/ws/Schedule.asmx/GetBoxScoreScroll"
GAME_LIST_URL = f"{KBO_BASE}/ws/Main.asmx/GetKboGameList"
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": f"{KBO_BASE}/Schedule/GameCenter/Main.aspx",
    "X-Requested-With": "XMLHttpRequest",
}
LOG_COLUMNS = [
    "game_date",
    "game_id",
    "team",
    "opponent",
    "home_away",
    "pitcher_index",
    "pitcher_id",
    "pitcher_name",
    "is_starter",
    "entry",
    "decision",
    "innings_outs",
    "batters_faced",
    "pitch_count",
    "at_bats",
    "hits_allowed",
    "home_runs_allowed",
    "walks_hbp",
    "strikeouts",
    "runs_allowed",
    "earned_runs",
    "game_era",
    "collected_at",
    "data_source",
]
LOG_KEY = ["game_id", "team", "pitcher_index"]


class PitcherGameLogError(RuntimeError):
    pass


def _text(value) -> str:
    return html.unescape(str(value or "")).replace("\xa0", " ").strip()


def _number(value, integer: bool = True):
    text = _text(value).replace(",", "")
    if not text or text == "-":
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return int(number) if integer else number


def innings_to_outs(value) -> int:
    text = _text(value)
    if not text or text == "-":
        return 0
    if " " in text:
        whole, fraction = text.split(None, 1)
        numerator, denominator = fraction.split("/", 1)
        return int(whole) * 3 + round(int(numerator) * 3 / int(denominator))
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        return round(int(numerator) * 3 / int(denominator))
    if "." in text:
        whole, partial = text.split(".", 1)
        if partial in {"0", "1", "2"}:
            return int(whole) * 3 + int(partial)
    return round(float(text) * 3)


def _table_rows(raw_table) -> list[list[str]]:
    if isinstance(raw_table, str):
        raw_table = json.loads(raw_table)
    if not isinstance(raw_table, dict):
        return []
    rows = []
    for item in raw_table.get("rows", []):
        cells = [_text(cell.get("Text")) for cell in item.get("row", [])]
        if cells:
            rows.append(cells)
    return rows


def fetch_game_list(game_date: date, session: requests.Session | None = None) -> list[dict]:
    client = session or requests.Session()
    response = client.post(
        GAME_LIST_URL,
        data={"leId": 1, "srId": 0, "date": game_date.strftime("%Y%m%d")},
        headers=HEADERS,
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    return payload.get("game", [])


def fetch_box_score(game_id: str, session: requests.Session | None = None) -> dict:
    client = session or requests.Session()
    response = client.post(
        BOX_SCORE_URL,
        data={"leId": 1, "srId": 0, "seasonId": int(game_id[:4]), "gameId": game_id},
        headers=HEADERS,
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if str(payload.get("code")) != "100":
        raise PitcherGameLogError(f"완료 경기 박스스코어가 없습니다: {game_id}")
    return payload


def parse_pitcher_box_score(game_date: date, game_meta: dict, payload: dict, collected_at: datetime) -> pd.DataFrame:
    game_id = str(game_meta.get("G_ID", "")).strip()
    if not game_id:
        raise PitcherGameLogError("공식 game_id가 없습니다.")
    sides = [
        {
            "team": _text(game_meta.get("AWAY_NM")),
            "opponent": _text(game_meta.get("HOME_NM")),
            "home_away": "A",
            "starter_id": _number(game_meta.get("T_PIT_P_ID")),
        },
        {
            "team": _text(game_meta.get("HOME_NM")),
            "opponent": _text(game_meta.get("AWAY_NM")),
            "home_away": "H",
            "starter_id": _number(game_meta.get("B_PIT_P_ID")),
        },
    ]
    groups = payload.get("arrPitcher", [])
    if len(groups) != 2:
        raise PitcherGameLogError(f"투수 박스스코어 팀 수가 올바르지 않습니다: {game_id}")

    rows = []
    for side, group in zip(sides, groups):
        if not side["team"] or not side["opponent"]:
            raise PitcherGameLogError(f"팀 매핑 정보가 없습니다: {game_id}")
        for pitcher_index, cells in enumerate(_table_rows(group.get("table")), start=1):
            if len(cells) < 17 or cells[0].upper() == "TOTAL":
                continue
            is_starter = cells[1] == "선발" or pitcher_index == 1
            rows.append(
                {
                    "game_date": game_date.isoformat(),
                    "game_id": game_id,
                    "team": side["team"],
                    "opponent": side["opponent"],
                    "home_away": side["home_away"],
                    "pitcher_index": pitcher_index,
                    "pitcher_id": side["starter_id"] if is_starter else None,
                    "pitcher_name": cells[0],
                    "is_starter": is_starter,
                    "entry": cells[1],
                    "decision": "" if cells[2] in {"&nbsp;", ""} else cells[2],
                    "innings_outs": innings_to_outs(cells[6]),
                    "batters_faced": _number(cells[7]),
                    "pitch_count": _number(cells[8]),
                    "at_bats": _number(cells[9]),
                    "hits_allowed": _number(cells[10]),
                    "home_runs_allowed": _number(cells[11]),
                    "walks_hbp": _number(cells[12]),
                    "strikeouts": _number(cells[13]),
                    "runs_allowed": _number(cells[14]),
                    "earned_runs": _number(cells[15]),
                    "game_era": _number(cells[16], integer=False),
                    "collected_at": collected_at.isoformat(timespec="seconds"),
                    "data_source": "KBO GetBoxScoreScroll",
                }
            )
    frame = pd.DataFrame(rows, columns=LOG_COLUMNS)
    if frame.empty or frame.groupby(["game_id", "team"])["is_starter"].sum().ne(1).any():
        raise PitcherGameLogError(f"팀별 선발투수 한 명을 확인하지 못했습니다: {game_id}")
    return frame


def merge_pitcher_logs(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    frames = [frame for frame in [existing, incoming] if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame(columns=LOG_COLUMNS)
    merged = pd.concat(frames, ignore_index=True).reindex(columns=LOG_COLUMNS)
    merged = merged.drop_duplicates(LOG_KEY, keep="last")
    merged = merged.sort_values(["game_date", "game_id", "team", "pitcher_index"]).reset_index(drop=True)
    if merged.duplicated(LOG_KEY).any():
        raise PitcherGameLogError("투수 경기 로그 업무 키가 중복됩니다.")
    return merged


def save_pitcher_logs(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    backup = output_path.with_suffix(output_path.suffix + ".bak")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    reloaded = pd.read_csv(temporary)
    if len(reloaded) != len(frame) or reloaded.duplicated(LOG_KEY).any():
        temporary.unlink(missing_ok=True)
        raise PitcherGameLogError("임시 투수 로그 검증에 실패했습니다.")
    if output_path.exists():
        shutil.copy2(output_path, backup)
    os.replace(temporary, output_path)


def collect_pitcher_game_logs(
    games: pd.DataFrame,
    output_path: Path,
    start_date: date,
    end_date: date,
    session: requests.Session | None = None,
) -> tuple[pd.DataFrame, dict]:
    frame = games.copy()
    frame["game_date"] = pd.to_datetime(frame["date"]).dt.date
    frame["actual_game_id"] = frame["game_id"].astype(str).str.rsplit("_", n=1).str[0]
    targets = frame[
        frame["status"].eq("Final")
        & frame["game_date"].between(start_date, end_date)
    ][["game_date", "actual_game_id"]].drop_duplicates().sort_values(["game_date", "actual_game_id"])
    existing = pd.read_csv(output_path) if output_path.exists() else pd.DataFrame(columns=LOG_COLUMNS)
    existing_ids = set(existing["game_id"].astype(str)) if not existing.empty else set()
    pending = targets[~targets["actual_game_id"].isin(existing_ids)]
    client = session or requests.Session()
    collected_at = datetime.now()
    collected = []
    failures = []
    for game_date, date_targets in pending.groupby("game_date", sort=True):
        try:
            metadata = {str(row.get("G_ID")): row for row in fetch_game_list(game_date, client)}
        except (requests.RequestException, ValueError) as exc:
            failures.extend({"game_id": game_id, "error": type(exc).__name__} for game_id in date_targets["actual_game_id"])
            continue
        for game_id in date_targets["actual_game_id"]:
            game_meta = metadata.get(str(game_id))
            if not game_meta:
                failures.append({"game_id": str(game_id), "error": "official_game_mapping_missing"})
                continue
            try:
                payload = fetch_box_score(str(game_id), client)
                collected.append(parse_pitcher_box_score(game_date, game_meta, payload, collected_at))
            except (requests.RequestException, ValueError, KeyError, PitcherGameLogError) as exc:
                failures.append({"game_id": str(game_id), "error": str(exc)[:160]})
    incoming = pd.concat(collected, ignore_index=True) if collected else pd.DataFrame(columns=LOG_COLUMNS)
    merged = merge_pitcher_logs(existing, incoming)
    if len(merged) < len(existing):
        raise PitcherGameLogError("기존 투수 경기 로그 행이 감소했습니다.")
    if not incoming.empty:
        save_pitcher_logs(merged, output_path)
    status = {
        "generated_at": collected_at.isoformat(timespec="seconds"),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "target_games": int(len(targets)),
        "existing_games": int(len(existing_ids)),
        "new_games": int(incoming["game_id"].nunique()) if not incoming.empty else 0,
        "total_games": int(merged["game_id"].nunique()) if not merged.empty else 0,
        "total_rows": int(len(merged)),
        "failed_games": failures,
        "status": "pass" if not failures else "warning",
        "output_file": str(output_path),
    }
    return merged, status
