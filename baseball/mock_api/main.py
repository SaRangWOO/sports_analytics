from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
from typing import Optional, List
from pydantic import BaseModel

app = FastAPI(title="Mock KBO API", version="1.0.0")

DATA_FILE = "data.csv"

# 데이터 로딩
try:
    df = pd.read_csv(DATA_FILE)
    print(f"✅ Data Loaded: {len(df)} games")
except Exception as e:
    print(f"❌ Failed to load data: {e}")
    df = pd.DataFrame()

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

@app.get("/")
def health_check():
    return {"status": "ok", "games_loaded": len(df)}

@app.get("/games", response_model=List[Game])
def get_games(date: Optional[str] = None, team: Optional[str] = None):
    filtered_df = df.copy()

    if date:
        filtered_df = filtered_df[filtered_df['date'] == date]
    if team:
        filtered_df = filtered_df[filtered_df['team'] == team]
    
    # NaN -> None 변환
    filtered_df = filtered_df.replace({np.nan: None})
    
    result = filtered_df.to_dict(orient="records")
    return result

@app.get("/games/{game_id}", response_model=Game)
def get_game_detail(game_id: str):
    row = df[df['game_id'] == game_id]
    if row.empty:
        # [수정된 부분] 여기가 끊겨 있었습니다. 따옴표와 괄호를 닫았습니다.
        raise HTTPException(status_code=404, detail="Game not found")
    
    # 마지막 줄: 상세 정보 반환
    row_dict = row.iloc[0].replace({np.nan: None}).to_dict()
    return row_dict