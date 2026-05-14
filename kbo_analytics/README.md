# KBO Analytics

목적은 두 가지입니다.

1. KBO 팀 성과 대시보드 생성
2. 경기 승패 예측 모델 생성

## 운영 구조

```text
KBO official record/schedule pages
        |
        v
official_kbo_dashboard.py
        |
        +-- PostgreSQL
        |     +-- game_results
        |     +-- official_team_standings
        |     +-- official_team_vs_team
        |     +-- official_hitter_stats
        |     +-- official_pitcher_stats
        +-- dashboard/latest.html
        +-- docs/index.html
        +-- dashboard/latest_summary.md
        +-- data/official/*.csv
        +-- modeling/results/win_predictor_model.json
        +-- modeling/results/features.csv
        +-- modeling/results/model_history.json
```

## 핵심 파일

```text
kbo_analytics/
├── collector.py                 # 시즌 누적 또는 지난주 경기/선수 기록 수집 후 PostgreSQL 적재
├── official_kbo_dashboard.py    # KBO 공식 기록 기반 대시보드와 승패 예측 결과 생성
├── weekly_update.py             # 기존 DB 기반 mock 분석 스크립트
├── docker-compose.yaml          # API, PostgreSQL, Metabase 실행
├── mock_api/                    # 실제 데이터 API 연결 전까지 쓰는 임시 데이터 API
│   └── player_roster_mapping.csv # KBO 기록실 기준 선수명 매핑
├── scripts/
│   ├── build_player_roster_mapping.py # KBO 기록실에서 선수명 매핑 생성
│   └── weekly_kbo_update.sh     # 매주 월요일 증분 적재/대시보드/모델/GitHub push
├── data/official/               # KBO 공식 순위/일정/선수 기록 스냅샷
├── data/weekly/                 # 기존 mock DB 내용을 CSV로 내보낸 결과
├── dashboard/                   # 확인용 HTML/Markdown 대시보드
├── modeling/                    # 승패 예측 모델
├── pg_data/                     # PostgreSQL 실제 데이터 디렉터리
└── sql/analysis_queries.sql     # Metabase/SQL 분석 쿼리
```

## 실행

```bash
cd /home/tera/1.project/1.sports_analytics/kbo_analytics
docker compose up -d
.venv/bin/python official_kbo_dashboard.py
```

현재 운영 대시보드는 KBO 공식 팀 순위, 일정/결과, 선수 기록 페이지를 직접 조회해 만듭니다. `official_kbo_dashboard.py`는 공식 데이터를 CSV로 저장하고 PostgreSQL에도 다시 적재합니다. 기존 `mock_api`는 파이프라인 테스트용으로만 남겨두며, 리그 현황 대시보드의 기준 데이터로 사용하지 않습니다.

기존 mock 파이프라인을 다시 검증해야 할 때만 시즌 시작일부터 오늘까지 누적 적재합니다.

```bash
API_BASE_URL=http://localhost:8000 DB_URL=postgresql://user:password@localhost:5432/baseball .venv/bin/python collector.py --season-to-date
DB_URL=postgresql://user:password@localhost:5432/baseball .venv/bin/python weekly_update.py
```

매주 월요일 cron은 `official_kbo_dashboard.py`를 실행해 공식 기록 스냅샷, 대시보드, 모델 결과를 다시 생성하고 GitHub에 push합니다. 승패 예측 모델은 실행일 기준 지난주 일요일까지의 완료 경기만 학습/검증에 사용합니다. 특정 mock 기간을 다시 적재하려면 아래처럼 실행합니다.

```bash
API_BASE_URL=http://localhost:8000 DB_URL=postgresql://user:password@localhost:5432/baseball .venv/bin/python collector.py --start-date 2026-03-30 --end-date 2026-04-05
DB_URL=postgresql://user:password@localhost:5432/baseball .venv/bin/python weekly_update.py --start-date 2026-03-30 --end-date 2026-04-05
```

서버 IP, 계정, 비밀번호 같은 운영 접속 정보는 README와 Git에 기록하지 않습니다. 실행 환경에서만 환경변수로 지정합니다.

## 결과 확인

- HTML 대시보드: `dashboard/latest.html`
- GitHub Pages 배포용 정적 대시보드: `docs/index.html`
- 요약 리포트: `dashboard/latest_summary.md`
- 공식 데이터 스냅샷: `data/official/`
- PostgreSQL 공식 테이블: `game_results`, `official_team_standings`, `official_team_vs_team`, `official_hitter_stats`, `official_pitcher_stats`
- 예측 모델 결과: `modeling/results/win_predictor_model.json`
- 예측 모델 성능 이력: `modeling/results/model_history.json`
- Metabase: PostgreSQL의 `game_results`, `player_game_stats` 테이블 연결

서버에서는 HTML 대시보드가 dashboard 컨테이너를 통해 제공됩니다. 운영 접속 정보는 Git에 기록하지 않고 서버의 실행 환경과 `.env`에서만 관리합니다.

## 대시보드 구성

- 첫 화면: KBO 리그 전체 순위
- 구단 선택: 선택 구단 순위, 오늘 경기 승패 예측, 최근 10경기, 상대 전적
- 타자 지표: KBO 공식 타자 기록의 경기, 타석, 타수, 안타, 홈런, 볼넷, 삼진, 타율, 출루율, 장타율, OPS
- 투수 지표: KBO 공식 투수 기록의 경기, 승, 패, 세이브, 홀드, 이닝, 자책, 탈삼진, 볼넷, ERA, WHIP
- 승패 예측: 지난주 일요일까지의 완료 경기만 사용한 검증 결과

## 승패 예측 모델

`official_kbo_dashboard.py`는 KBO 공식 일정/결과에서 2021년 이후 각 경기를 양 팀 관점의 학습 행으로 변환합니다. 최근 5경기 승률, 최근 5경기 평균 득점/실점/득실차, 시즌 누적 승률, 상대팀 최근 흐름, 홈/원정, 상대팀을 사용해 로지스틱 회귀 모델을 학습합니다.

모델 학습 기준일은 실행일 기준 지난주 일요일입니다. 예를 들어 목요일에 수동 실행해도 현재 주 화/수 경기 결과를 적중률 계산에 섞지 않습니다.

## 외부 접근

`docs/index.html`은 GitHub Pages 배포용 정적 파일입니다. `main` 브랜치에 push되면 `.github/workflows/deploy-dashboard.yml`이 `docs/`를 GitHub Pages로 배포합니다. 서버가 열려 있지 않아도 GitHub Pages 주소로 같은 대시보드를 볼 수 있습니다.

## 현재 확인된 한계

KBO 공식 페이지 구조가 바뀌면 `official_kbo_dashboard.py`의 HTML/웹서비스 파싱 로직을 조정해야 합니다.
