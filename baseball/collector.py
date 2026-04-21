import requests
import pandas as pd
from sqlalchemy import create_engine

API_URL = "http://localhost:8000/games"
DB_URL = "postgresql://tera:123123@localhost:5432/baseball"

def run_pipeline():
    print("[Start] 데이터 수집 파이프라인 시작...")

    try:
        response = requests.get(API_URL)
        response.raise_for_status()
        data = response.json()
        print(f"✅ [Extract] API에서 {len(data)}개의 데이터를 가져왔습니다.")
    except Exception as e:
        print(f"❌ [Extract Error] 데이터 수집 실패: {e}")
        return
    
    try:
        df = pd.DataFrame(data)
        print(f"✅ [Transform] 데이터프레임으로 변환 완료. 행 수: {len(df)}")
    except Exception as e:
        print(f"❌ [Transform Error] 데이터 변환 실패: {e}")
        return
    
    try:
        engine = create_engine(DB_URL)
        df.to_sql(name='game_results', con=engine, if_exists='replace', index=False)

        print(f"✅ [Load] 데이터베이스에 {len(df)}개의 데이터를 저장했습니다.")

    except Exception as e:
        print(f"❌ [Load Error] 데이터 저장 실패: {e}")
        return
    
    print("[Success] 데이터 수집 파이프라인 완료.")

if __name__ == "__main__":
    run_pipeline()
