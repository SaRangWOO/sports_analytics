# KBO Analytics

목적은 두 가지입니다.

1. KBO 팀 성과 대시보드 생성
2. 경기 승패 예측 모델 생성

## 운영 구조

```text
Mock KBO game/player API
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
        +-- modeling/results/model_history.json
```

## 핵심 파일

```text
kbo_analytics/
├── collector.py                 # 시즌 누적 또는 지난주 경기/선수 기록 수집 후 PostgreSQL 적재
├── weekly_update.py             # DB 기반 대시보드와 승패 예측 결과 생성
├── docker-compose.yaml          # API, PostgreSQL, Metabase 실행
├── mock_api/                    # 실제 데이터 API 연결 전까지 쓰는 임시 데이터 API
│   └── player_roster_mapping.csv # KBO 기록실 기준 선수명 매핑
├── scripts/
│   ├── build_player_roster_mapping.py # KBO 기록실에서 선수명 매핑 생성
│   └── weekly_kbo_update.sh     # 매주 월요일 증분 적재/대시보드/모델/GitHub push
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

처음 운영을 시작하거나 데이터베이스를 다시 만들었을 때는 시즌 시작일부터 오늘까지 누적 적재합니다.

```bash
API_BASE_URL=http://localhost:8000 DB_URL=postgresql://user:password@localhost:5432/baseball .venv/bin/python collector.py --season-to-date
DB_URL=postgresql://user:password@localhost:5432/baseball .venv/bin/python weekly_update.py
```

이후 `collector.py`는 실행일 기준 지난주 월요일부터 일요일까지의 데이터를 가져옵니다. 매주 월요일 cron은 이 방식으로 지난주 경기와 선수 기록만 교체 적재하고, 대시보드와 모델 결과를 다시 생성합니다. 특정 기간을 다시 적재하려면 아래처럼 실행합니다.

```bash
API_BASE_URL=http://localhost:8000 DB_URL=postgresql://user:password@localhost:5432/baseball .venv/bin/python collector.py --start-date 2026-03-30 --end-date 2026-04-05
DB_URL=postgresql://user:password@localhost:5432/baseball .venv/bin/python weekly_update.py --start-date 2026-03-30 --end-date 2026-04-05
```

서버 IP, 계정, 비밀번호 같은 운영 접속 정보는 README와 Git에 기록하지 않습니다. 실행 환경에서만 환경변수로 지정합니다.

## 결과 확인

- HTML 대시보드: `dashboard/latest.html`
- 요약 리포트: `dashboard/latest_summary.md`
- 예측 모델 결과: `modeling/results/win_predictor_model.json`
- 예측 모델 성능 이력: `modeling/results/model_history.json`
- Metabase: PostgreSQL의 `game_results`, `player_game_stats` 테이블 연결

서버에서는 HTML 대시보드가 dashboard 컨테이너를 통해 제공됩니다. 운영 접속 정보는 Git에 기록하지 않고 서버의 실행 환경과 `.env`에서만 관리합니다.

## 대시보드 구성

- 경기 흐름: 최근 경기 결과, 스코어, 득실차
- 팀 성과: 주간 전적, 승률, 평균 득점/실점, 상대팀별 성적, 홈/원정 성적, 월별 흐름
- 타자 지표: 타율, 출루율, 장타율 proxy, OPS proxy
- 투수 지표: ERA, WHIP, K/9, 투구수, 탈삼진, 볼넷, 피안타
- 승패 예측: 후보 모델별 정확도, 선택 모델 주요 변수, 최근 경기별 예측 확률

타자/투수 표의 선수명은 `scripts/build_player_roster_mapping.py`가 KBO 기록실의 팀별 타자/투수 기록 페이지에서 가져온 `mock_api/player_roster_mapping.csv`를 기준으로 표시합니다. mock 경기 기록은 실제 경기 상세 box score가 아니므로, 선수별 수치는 분석 파이프라인 검증용 합성 기록입니다.

## 승패 예측 모델

`weekly_update.py`는 실행할 때마다 여러 로지스틱 회귀 후보를 비교합니다. 전체 변수 모델, 공격/실점 흐름 중심 모델, 구장/일정 포함 모델을 검증 구간에서 비교하고 가장 좋은 정확도와 F1을 보인 모델을 선택합니다.

선택된 모델과 성능은 `win_predictor_model.json`에 저장되고, 최근 실행 이력은 `model_history.json`에 누적됩니다. 매주 월요일 자동 실행 스크립트가 새 데이터 적재 후 대시보드와 모델 결과를 다시 만들고 GitHub에 push합니다.

## 현재 확인된 한계

현재 서버에는 실제 KBO 공식/외부 API 연동 코드가 아니라 `mock_api/` 기반 임시 API가 있습니다. 실제 데이터 소스가 정해지면 `collector.py`의 API 호출 대상만 교체하고, 이후 DB 테이블과 대시보드/모델 구조는 그대로 유지합니다.
