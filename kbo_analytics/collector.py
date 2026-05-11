import requests
import pandas as pd
from datetime import date, datetime, timedelta
import argparse
import os

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
DB_URL = os.getenv("DB_URL", "postgresql://tera:tera@localhost:5432/baseball")

def get_previous_week_window(reference_date: date | None = None):
    reference_date = reference_date or date.today()
    this_monday = reference_date - timedelta(days=reference_date.weekday())
    last_monday = this_monday - timedelta(days=7)
    last_sunday = this_monday - timedelta(days=1)
    return last_monday.isoformat(), last_sunday.isoformat()

def fetch_endpoint(endpoint: str, start_date: str, end_date: str):
    url = f"{API_BASE_URL}/{endpoint}"
    params = {"start_date": start_date, "end_date": end_date}
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()

def replace_date_range(engine, table_name: str, df: pd.DataFrame, start_date: str, end_date: str):
    from sqlalchemy import inspect
    from sqlalchemy import text

    if df.empty:
        print(f"[Skip] {table_name}: 적재할 데이터가 없습니다.")
        return

    with engine.begin() as connection:
        if inspect(connection).has_table(table_name):
            connection.execute(
                text(f"DELETE FROM {table_name} WHERE date BETWEEN :start_date AND :end_date"),
                {"start_date": start_date, "end_date": end_date},
            )
        df.to_sql(name=table_name, con=connection, if_exists="append", index=False)

def run_pipeline(start_date: str | None = None, end_date: str | None = None):
    if not start_date or not end_date:
        start_date, end_date = get_previous_week_window()

    print(f"[Start] 지난주 경기 데이터 수집 파이프라인 시작: {start_date} ~ {end_date}")

    try:
        games = fetch_endpoint("games", start_date, end_date)
        player_stats = fetch_endpoint("player-stats", start_date, end_date)
        print(f"[Extract] 경기 {len(games)}건, 선수 경기 기록 {len(player_stats)}건을 가져왔습니다.")
    except Exception as e:
        print(f"[Extract Error] 데이터 수집 실패: {e}")
        return
    
    try:
        games_df = pd.DataFrame(games)
        player_stats_df = pd.DataFrame(player_stats)
        print(f"[Transform] 경기 {len(games_df)}행, 선수 기록 {len(player_stats_df)}행 변환 완료.")
    except Exception as e:
        print(f"[Transform Error] 데이터 변환 실패: {e}")
        return
    
    try:
        from sqlalchemy import create_engine

        engine = create_engine(DB_URL)
        replace_date_range(engine, "game_results", games_df, start_date, end_date)
        replace_date_range(engine, "player_game_stats", player_stats_df, start_date, end_date)
        print("[Load] 데이터베이스 누적 적재 완료.")

    except Exception as e:
        print(f"[Load Error] 데이터 저장 실패: {e}")
        return
    
    print("[Success] 데이터 수집 파이프라인 완료.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect last week's KBO mock game and player data.")
    parser.add_argument("--start-date", help="YYYY-MM-DD. 생략하면 지난주 월요일")
    parser.add_argument("--end-date", help="YYYY-MM-DD. 생략하면 지난주 일요일")
    parser.add_argument("--reference-date", help="YYYY-MM-DD. 지난주 계산 기준일")
    args = parser.parse_args()

    if args.reference_date and not (args.start_date and args.end_date):
        reference_date = datetime.strptime(args.reference_date, "%Y-%m-%d").date()
        args.start_date, args.end_date = get_previous_week_window(reference_date)

    run_pipeline(args.start_date, args.end_date)
