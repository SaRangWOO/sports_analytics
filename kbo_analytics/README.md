# KBO Analytics

목적은 두 가지입니다.

1. KBO 팀 성과 대시보드 생성
2. 경기 승패 예측 모델 생성

## 운영 구조

```text
External game/player data API
        |
        v
collector.py
        |
        v
PostgreSQL
  - game_results
  - player_game_stats
        |
        v
weekly_update.py
        |
        +-- dashboard/latest.html
        +-- dashboard/latest_summary.md
        +-- modeling/results/win_predictor_model.json
        +-- modeling/results/features.csv
```

## 핵심 파일

```text
kbo_analytics/
├── collector.py                 # 지난주 경기/선수 기록 수집 후 PostgreSQL 적재
├── weekly_update.py             # DB 기반 대시보드와 승패 예측 결과 생성
├── docker-compose.yaml          # API, PostgreSQL, Metabase 실행
├── mock_api/                    # 실제 데이터 API 연결 전까지 쓰는 임시 데이터 API
├── data/weekly/                 # DB 내용을 CSV로 내보낸 결과
├── dashboard/                   # 확인용 HTML/Markdown 대시보드
├── modeling/                    # 승패 예측 모델
├── pg_data/                     # PostgreSQL 실제 데이터 디렉터리
└── sql/analysis_queries.sql     # Metabase/SQL 분석 쿼리
```

## 실행

```bash
cd /home/tera/1.project/1.sports_analytics/kbo_analytics
docker compose up -d
API_BASE_URL=http://localhost:8000 DB_URL=postgresql://user:password@localhost:5432/baseball .venv/bin/python collector.py
DB_URL=postgresql://user:password@localhost:5432/baseball .venv/bin/python weekly_update.py
```

`collector.py`는 실행일 기준 지난주 월요일부터 일요일까지의 데이터를 가져옵니다. 특정 기간을 다시 적재하려면 아래처럼 실행합니다.

```bash
API_BASE_URL=http://localhost:8000 DB_URL=postgresql://user:password@localhost:5432/baseball .venv/bin/python collector.py --start-date 2026-03-30 --end-date 2026-04-05
DB_URL=postgresql://user:password@localhost:5432/baseball .venv/bin/python weekly_update.py --start-date 2026-03-30 --end-date 2026-04-05
```

## 결과 확인

- HTML 대시보드: `dashboard/latest.html`
- 요약 리포트: `dashboard/latest_summary.md`
- 예측 모델 결과: `modeling/results/win_predictor_model.json`
- Metabase: PostgreSQL의 `game_results`, `player_game_stats` 테이블 연결

## 현재 확인된 한계

현재 서버에는 실제 KBO 공식/외부 API 연동 코드가 아니라 `mock_api/` 기반 임시 API가 있습니다. 실제 데이터 소스가 정해지면 `collector.py`의 API 호출 대상만 교체하고, 이후 DB 테이블과 대시보드/모델 구조는 그대로 유지합니다.
