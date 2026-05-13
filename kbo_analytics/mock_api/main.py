from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
from typing import Optional, List
from pydantic import BaseModel
import zlib

app = FastAPI(title="Mock KBO API", version="1.0.0")

DATA_FILE = "data.csv"
ROSTER_FILE = "player_roster_mapping.csv"
POSITIONS = ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH"]
KT_PLAYERS = [
    ("KT01", "KT Batter 1"), ("KT02", "KT Batter 2"), ("KT03", "KT Batter 3"),
    ("KT04", "KT Batter 4"), ("KT05", "KT Batter 5"), ("KT06", "KT Batter 6"),
    ("KT07", "KT Batter 7"), ("KT08", "KT Batter 8"), ("KT09", "KT Batter 9"),
    ("KT10", "KT Starter"), ("KT11", "KT Reliever 1"), ("KT12", "KT Reliever 2"),
]

# 데이터 로딩
try:
    df = pd.read_csv(DATA_FILE)
    print(f"✅ Data Loaded: {len(df)} games")
except Exception as e:
    print(f"❌ Failed to load data: {e}")
    df = pd.DataFrame()

try:
    roster_df = pd.read_csv(ROSTER_FILE)
    roster_df = roster_df.sort_values(["team", "slot"])
    print(f"✅ Roster Loaded: {len(roster_df)} players")
except Exception as e:
    print(f"❌ Failed to load roster mapping: {e}")
    roster_df = pd.DataFrame()

# 모델 정의
class Game(BaseModel):
    game_id: str
    date: str
    team: str
    opponent: str
    home_away: str
    status: str
    result: Optional[str] = None
    score_team: Optional[float] = None
    score_opp: Optional[float] = None
    note: Optional[str] = None

class PlayerGameStat(BaseModel):
    game_id: str
    date: str
    player_id: str
    player_name: str
    team: str
    opponent: str
    home_away: str
    position: str
    batting_order: Optional[int] = None
    plate_appearances: int
    at_bats: int
    hits: int
    doubles: int
    triples: int
    home_runs: int
    rbi: int
    runs: int
    walks: int
    strikeouts: int
    stolen_bases: int
    innings_pitched: float
    pitches: int
    earned_runs: int
    strikeouts_pitched: int
    walks_allowed: int
    hits_allowed: int

def _filter_games(
    source_df: pd.DataFrame,
    date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    team: Optional[str] = None,
) -> pd.DataFrame:
    filtered_df = source_df.copy()

    if date:
        filtered_df = filtered_df[filtered_df["date"] == date]
    if start_date:
        filtered_df = filtered_df[filtered_df["date"] >= start_date]
    if end_date:
        filtered_df = filtered_df[filtered_df["date"] <= end_date]
    if team:
        filtered_df = filtered_df[filtered_df["team"] == team]

    return filtered_df

def _rng_for(*parts: str) -> np.random.Generator:
    seed = zlib.crc32("|".join(parts).encode("utf-8"))
    return np.random.default_rng(seed)

def _opponent_players(opponent: str) -> list[tuple[str, str]]:
    players = _team_players(opponent)
    if players:
        return players
    prefix = opponent[:3].upper()
    return [(f"{prefix}{i:02d}", f"{opponent} Player {i}") for i in range(1, 13)]

def _team_players(team: str) -> list[tuple[str, str]]:
    if roster_df.empty:
        return KT_PLAYERS if team == "KT" else []

    team_rows = roster_df[roster_df["team"] == team].sort_values("slot")
    if len(team_rows) < 12:
        return KT_PLAYERS if team == "KT" else []

    return list(zip(team_rows["player_id"], team_rows["player_name"]))

def _make_batter_row(game: dict, player_id: str, player_name: str, idx: int, team: str, opponent: str, home_away: str):
    rng = _rng_for(game["game_id"], player_id, "bat")
    plate_appearances = int(rng.integers(3, 6))
    walks = int(rng.binomial(plate_appearances, 0.08))
    at_bats = max(plate_appearances - walks, 0)
    hits = int(rng.binomial(at_bats, 0.27)) if at_bats else 0
    doubles = int(rng.binomial(hits, 0.18)) if hits else 0
    triples = int(rng.binomial(max(hits - doubles, 0), 0.03)) if hits else 0
    home_runs = int(rng.binomial(max(hits - doubles - triples, 0), 0.08)) if hits else 0

    return {
        "game_id": game["game_id"],
        "date": game["date"],
        "player_id": player_id,
        "player_name": player_name,
        "team": team,
        "opponent": opponent,
        "home_away": home_away,
        "position": POSITIONS[idx - 1],
        "batting_order": idx,
        "plate_appearances": plate_appearances,
        "at_bats": at_bats,
        "hits": hits,
        "doubles": doubles,
        "triples": triples,
        "home_runs": home_runs,
        "rbi": int(rng.integers(0, 4)),
        "runs": int(rng.integers(0, 3)),
        "walks": walks,
        "strikeouts": int(rng.binomial(plate_appearances, 0.18)),
        "stolen_bases": int(rng.binomial(1, 0.06)),
        "innings_pitched": 0.0,
        "pitches": 0,
        "earned_runs": 0,
        "strikeouts_pitched": 0,
        "walks_allowed": 0,
        "hits_allowed": 0,
    }

def _make_pitcher_row(game: dict, player_id: str, player_name: str, idx: int, team: str, opponent: str, home_away: str):
    rng = _rng_for(game["game_id"], player_id, "pit")
    innings = [5.0, 2.0, 2.0][idx]

    return {
        "game_id": game["game_id"],
        "date": game["date"],
        "player_id": player_id,
        "player_name": player_name,
        "team": team,
        "opponent": opponent,
        "home_away": home_away,
        "position": "P",
        "batting_order": None,
        "plate_appearances": 0,
        "at_bats": 0,
        "hits": 0,
        "doubles": 0,
        "triples": 0,
        "home_runs": 0,
        "rbi": 0,
        "runs": 0,
        "walks": 0,
        "strikeouts": 0,
        "stolen_bases": 0,
        "innings_pitched": innings,
        "pitches": int(rng.integers(22, 91)),
        "earned_runs": int(rng.integers(0, 5)),
        "strikeouts_pitched": int(rng.integers(0, 8)),
        "walks_allowed": int(rng.integers(0, 4)),
        "hits_allowed": int(rng.integers(0, 8)),
    }

def _generate_player_stats(games_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for game in games_df.to_dict(orient="records"):
        if game.get("status") != "Final":
            continue

        teams = [
            (game["team"], game["opponent"], game["home_away"], _team_players(game["team"])),
            (game["opponent"], game["team"], "A" if game["home_away"] == "H" else "H", _opponent_players(game["opponent"])),
        ]

        for team, opponent, home_away, players in teams:
            for idx, (player_id, player_name) in enumerate(players[:9], start=1):
                rows.append(_make_batter_row(game, player_id, player_name, idx, team, opponent, home_away))
            for idx, (player_id, player_name) in enumerate(players[9:12]):
                rows.append(_make_pitcher_row(game, player_id, player_name, idx, team, opponent, home_away))

    return pd.DataFrame(rows)

@app.get("/")
def health_check():
    return {"status": "ok", "games_loaded": len(df)}

@app.get("/games", response_model=List[Game])
def get_games(
    date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    team: Optional[str] = None,
):
    filtered_df = _filter_games(df, date=date, start_date=start_date, end_date=end_date, team=team)
    # NaN -> None 변환
    filtered_df = filtered_df.replace({np.nan: None})
    
    result = filtered_df.to_dict(orient="records")
    return result

@app.get("/player-stats", response_model=List[PlayerGameStat])
def get_player_stats(
    date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    team: Optional[str] = None,
    game_id: Optional[str] = None,
):
    filtered_games = _filter_games(df, date=date, start_date=start_date, end_date=end_date)
    if game_id:
        filtered_games = filtered_games[filtered_games["game_id"] == game_id]

    player_df = _generate_player_stats(filtered_games)
    if team and not player_df.empty:
        player_df = player_df[player_df["team"] == team]

    player_df = player_df.replace({np.nan: None})
    return player_df.to_dict(orient="records")

@app.get("/games/{game_id}", response_model=Game)
def get_game_detail(game_id: str):
    row = df[df['game_id'] == game_id]
    if row.empty:
        # [수정된 부분] 여기가 끊겨 있었습니다. 따옴표와 괄호를 닫았습니다.
        raise HTTPException(status_code=404, detail="Game not found")
    
    # 마지막 줄: 상세 정보 반환
    row_dict = row.iloc[0].replace({np.nan: None}).to_dict()
    return row_dict
