from airflow import DAG
from airflow.operators.python import PythonOperator
import pendulum
from datetime import datetime, timedelta
import requests
import pandas as pd
from sqlalchemy import create_engine

# ==========================================
# 1. 설정 (Settings)
# ==========================================
# [중요] Airflow 컨테이너 안에서 Host(내 컴퓨터)로 나가기 위한 주소: 172.17.0.1
API_URL = "http://172.17.0.1:8000/games"
DB_URL = "postgresql://tera:tera@172.17.0.1:5432/baseball"

# 시간대 설정 (한국 시간)
local_tz = pendulum.timezone("Asia/Seoul")

# ==========================================
# 2. ETL 함수 정의 (우리가 짠 로직)
# ==========================================
def run_etl_process(**context):
    print("🚀 [Start] ETL Process Started by Airflow")
    
    # (1) Extract: API 호출
    try:
        response = requests.get(API_URL)
        response.raise_for_status()
        data = response.json()
        print(f"✅ [Extract] API에서 {len(data)}개의 데이터를 가져왔습니다.")
    except Exception as e:
        print(f"❌ [Extract Error] API 접속 실패: {e}")
        raise e 

    # (2) Transform: 데이터프레임 변환
    try:
        df = pd.DataFrame(data)
        print(f"✅ [Transform] 데이터 변환 완료 (Rows: {len(df)})")
    except Exception as e:
        print(f"❌ [Transform Error] 데이터 변환 실패: {e}")
        raise e

    # (3) Load: DB 적재
    try:
        engine = create_engine(DB_URL)
        # 테스트니까 기존 거 지우고 덮어쓰기 (replace)
        df.to_sql(name='game_results', con=engine, if_exists='replace', index=False)
        print("✅ [Load] DB 적재 완료! (Table: game_results)")
    except Exception as e:
        print(f"❌ [Load Error] DB 저장 실패: {e}")
        raise e

# ==========================================
# 3. DAG 정의 (작업 명세서)
# ==========================================
default_args = {
    'owner': 'tera',
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

with DAG(
    dag_id='baseball_pipeline_v1',      # 웹 화면에 뜰 이름
    default_args=default_args,
    description='KBO Baseball Data ETL Pipeline',
    schedule_interval='0 9 * * *',      # 매일 아침 9시 실행
    start_date=datetime(2026, 3, 20, tzinfo=local_tz), # 시작일
    catchup=False,                      # 과거 데이터 한꺼번에 돌리기 방지
    tags=['kbo', 'etl'],
) as dag:

    # 파이썬 함수를 실행하는 오퍼레이터
    etl_task = PythonOperator(
        task_id='extract_transform_load_task',
        python_callable=run_etl_process
    )
