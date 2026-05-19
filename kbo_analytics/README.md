# KBO Analytics

KBO 공식 기록을 수집해 구단별 분석 리포트와 경기 승패 예측 대시보드를 생성하는 프로젝트입니다.

이 프로젝트는 단순히 순위표를 보여주기보다, 공식 기록을 바탕으로 아래 질문에 답하는 것을 목표로 합니다.

- 이 팀은 공격형 팀인가, 마운드 중심 팀인가?
- 득실차 기준으로 반등 여지나 하락 위험이 있는가?
- 작전 수행이나 찬스 상황에서 강점이 있는가?
- 어떤 선수에게 공격/마운드 전력이 집중되어 있는가?
- 오늘 경기에서 어느 팀이 약우세인지, 그 예측을 얼마나 믿을 수 있는가?

## 1. 현재 구현된 내용

### 1.1 KBO 공식 데이터 수집

`official_kbo_dashboard.py`가 KBO 공식 페이지를 조회해 데이터를 수집합니다.

수집 대상:

- 팀 순위
- 팀 간 상대전적
- 경기 일정과 경기 결과
- 타자 기록
- 투수 기록
- 감독/코치/등록 선수 명단

수집된 데이터는 `data/official/` 아래 CSV 파일로 저장됩니다. PostgreSQL 접속이 가능하면 DB에도 적재합니다. DB 적재가 실패해도 CSV 저장과 HTML 대시보드 생성은 계속 진행됩니다.

주요 CSV:

| 파일 | 내용 |
| --- | --- |
| `data/official/team_standings.csv` | KBO 팀 순위 |
| `data/official/team_vs_team.csv` | 팀 간 상대전적 |
| `data/official/game_results.csv` | 경기 일정/결과 |
| `data/official/hitter_stats.csv` | 타자 기록 |
| `data/official/pitcher_stats.csv` | 투수 기록 |
| `data/official/registered_rosters.csv` | 감독/코치/등록 선수 명단 |
| `data/official/pitching_context.csv` | 추정/확정 선발과 불펜 피로 proxy |
| `data/official/lineup_context.csv` | KBO GameCenter 라인업 분석 기준 타순/포지션/선수/WAR |

### 1.2 리그 전체 대시보드

`dashboard/latest.html`에는 KBO 리그 전체 상황과 오늘 경기 예측 요약이 들어갑니다.

주요 내용:

- 오늘의 KBO 예측 요약
- 가장 높은 예측 승률
- 예측 가능 경기 수
- 박빙/참고 경기 수
- KBO 리그 전체 순위
- 구단 선택형 상세 분석
- 모델 검증 결과

### 1.3 구단별 분석 리포트

각 구단은 별도 HTML 파일로 생성됩니다.

예:

- `dashboard/kt.html`
- `dashboard/lg.html`
- `dashboard/samsung.html`
- `dashboard/kiwoom.html`

구단별 페이지는 아래 흐름으로 구성됩니다.

```text
한 줄 결론
→ 팀 스타일 태그
→ 강점/주의점
→ 핵심 선수 영향도
→ 작전/상황 수행 지수
→ 감독/코치/등록 선수 구성
→ 홈/원정, 월별 흐름, 상대별 매치업
→ 선수 상세 기록표
```

#### 한 줄 결론

팀의 현재 상태를 짧은 문장으로 요약합니다.

예:

```text
KT는 찬스 수행, 타석 안정성, 마운드 안정성이 모두 상위권인 균형형 선두권 팀입니다.
```

#### 팀 스타일 태그

공식 기록의 리그 내 순위를 기준으로 태그를 생성합니다.

예:

- `상위권`
- `공격 생산형`
- `찬스 강점`
- `타석 안정형`
- `마운드 안정형`
- `마운드 불안`
- `득실 열세`
- `공격 침체`
- `반등 여지`
- `하락 위험`

태그 기준 예시:

| 태그 | 기준 |
| --- | --- |
| `공격 침체` | OPS 또는 팀 타율 하위 3팀 |
| `타석 불안정` | BB/K 하위 3팀 |
| `찬스 강점` | 득점권타율 상위권 |
| `마운드 불안` | ERA 또는 WHIP 하위 3팀 |
| `반등 여지` | 피타고리안 기대 승률이 실제 승률보다 0.030 이상 높음 |
| `하락 위험` | 실제 승률이 피타고리안 기대 승률보다 0.030 이상 높음 |

### 1.4 작전/상황 수행 지표

KBO 공식 공개 데이터에는 `작전 지시 수`, `사인 성공률`, `실제 작전 성공률`이 없습니다.

따라서 없는 데이터를 임의로 만들지 않고, 공식 기록으로 확인 가능한 proxy를 사용합니다.

| 지표 | 해석 |
| --- | --- |
| 희생번트 | 작전 개입 빈도 proxy |
| 희생플라이 | 진루/득점 상황 수행 proxy |
| 득점권타율 | 찬스 수행 proxy |
| BB/K | 타석 운영 안정성 proxy |
| ERA / WHIP | 마운드 안정성 proxy |

각 지표는 리그 내 순위를 100점 지수로 변환합니다. 100점에 가까울수록 리그 상위권입니다.

예:

```text
작전 수행 지수: 95/100
찬스 수행 지수: 50/100
타석 안정성: 40/100
마운드 안정성: 60/100
```

### 1.5 핵심 선수 영향도

개별 선수 페이지를 만들기 전에, 구단 페이지 안에서 팀 전력에 영향을 주는 핵심 선수를 요약합니다.

#### 타선 핵심 TOP 3

타자 기록을 바탕으로 팀 내 공격 영향도가 큰 선수를 뽑습니다.

반영 지표:

- OPS
- 출루율
- 장타율
- 득점권타율
- 타석 수

예:

```text
안현민: OPS 1.161 · 출루율 0.507 · 장타율 0.654
허경민: OPS 1.055 · 출루율 0.455 · 장타율 0.600
유준규: OPS 0.991 · 출루율 0.455 · 장타율 0.536
```

#### 마운드 핵심 TOP 3

투수 기록을 바탕으로 팀 내 마운드 영향도가 큰 선수를 뽑습니다.

반영 지표:

- ERA
- WHIP
- 이닝
- K/BB
- 세이브
- 홀드

역할 해석 예:

- 긴 이닝을 소화해 불펜 부담을 줄이는 선발 자원
- 승부처 등판 비중이 높은 핵심 불펜
- 주자 허용을 억제하는 안정형 투수

#### 전력 의존도

좋은 선수를 나열하는 데서 끝내지 않고, 팀이 특정 선수에게 얼마나 의존하는지도 봅니다.

예:

```text
타선 의존도 높음:
핵심 타자 3명의 평균 OPS가 팀 평균보다 0.491 높아 중심타선 이탈 시 공격 하락 위험이 큽니다.

마운드 의존도 보통:
이닝 상위 3명 비중이 36.7%로 주요 투수 비중을 관리할 필요가 있습니다.
```

#### 선수 영향도 피처 export

구단 리포트에서 계산한 선수 영향도는 경기 단위 모델 피처로도 연결할 수 있도록 별도 파일로 저장합니다.

| 파일 | 내용 |
| --- | --- |
| `modeling/results/player_team_context.csv` | 팀별 타선 핵심 OPS, 타자 의존도, 마운드 핵심 점수, 투수 의존도 |
| `modeling/results/game_level_player_features.csv` | 경기별 홈/원정 선수 영향도 차이값 |

현재 이 피처는 `참고 실험`으로만 사용합니다. 이유는 현재 `hitter_stats.csv`, `pitcher_stats.csv`가 최신 공식 기록 스냅샷이기 때문입니다. 과거 경기 학습에 최신 선수 기록을 그대로 넣으면 미래 정보를 과거 예측에 사용한 것처럼 보일 수 있습니다. 그래서 과거 시점별 선수 기록 스냅샷이 쌓이기 전까지는 최종 선택 모델에는 자동 반영하지 않고, 후보 비교와 진단용 export로만 남깁니다.

### 1.6 경기 전 라인업 표시

경기 전 업데이트에서는 KBO GameCenter의 라인업 분석 데이터를 조회합니다.

대시보드에는 다음 정보가 표시됩니다.

- 라인업 상태: `금일 라인업 기준` 또는 `최근 라인업 기준`
- 타순
- 포지션
- 선수명
- WAR
- 선발 라인업 WAR 합

KBO 응답의 `LINEUP_CK`가 `true`이면 금일 라인업으로 표시하고, `false`이면 KBO가 제공하는 최근 라인업 기준으로 표시합니다. 현재 라인업은 대시보드 판단 정보로 사용하지만, 과거 시점별 라인업 스냅샷이 쌓이기 전까지 최종 승패 모델 피처에는 직접 반영하지 않습니다.

저장 파일:

| 파일 | 내용 |
| --- | --- |
| `data/official/lineup_context.csv` | 경기일, 경기 ID, 팀, 홈/원정, 라인업 기준, 타순, 포지션, 선수, WAR |

### 1.7 득점 예측용 데이터셋

승패 모델은 `이긴다/진다`를 바로 예측합니다. 하지만 야구 분석에서는 예상 득점과 예상 득실차가 함께 있어야 예측 이유를 더 잘 설명할 수 있습니다.

그래서 먼저 득점 예측 모델을 만들기 위한 경기 단위 데이터셋을 생성합니다.

| 파일 | 내용 |
| --- | --- |
| `modeling/results/run_expectancy_features.csv` | 홈/원정 실제 득점, 득실차, 총득점, 경기 전 득점/실점 흐름 피처 |

한 행은 한 경기입니다.

주요 목표값:

- `home_score`: 홈팀 실제 득점
- `away_score`: 원정팀 실제 득점
- `run_diff`: 홈팀 득점 - 원정팀 득점
- `total_runs`: 양 팀 총득점

주요 피처:

- 최근 5경기 평균 득점/실점
- 시즌 평균 득점/실점
- 최근 득실차
- 시즌 득실차
- 최근 10경기 승률
- 휴식일
- 홈/원정 승률 차이
- Elo 차이

이 파일은 아직 최종 승패 모델에 직접 들어가지 않습니다. 다음 단계에서 `home_expected_runs`, `away_expected_runs`, `expected_run_diff`를 계산하는 baseline/ML 득점 모델의 입력으로 사용할 예정입니다.

## 2. 승패 예측 모델

### 2.1 목적

완료된 경기 결과를 학습해 오늘 경기의 승리 가능성을 예측합니다.

야구는 단일 경기 변동성이 크기 때문에, 전체 정확도 하나만 보지 않습니다. 모델은 아래 관점으로 평가합니다.

- 전체 경기 적중률
- 55% 이상 확신 경기 적중률
- 58% 이상 확신 경기 적중률
- 60% 이상 구간의 과신 여부
- Brier Score
- Log Loss
- Calibration Table

즉, 목표는 `모든 경기를 억지로 맞히는 모델`이 아니라 `예측 가능한 경기와 참고만 할 경기를 구분하는 모델`입니다.

### 2.2 주요 피처

현재 모델은 경기 결과에서 아래 정보를 숫자로 변환해 사용합니다.

팀 흐름:

- 최근 5경기 승률
- 최근 10경기 승률
- 최근 5경기 평균 득점
- 최근 5경기 평균 실점
- 최근 득실차
- 시즌 누적 승률
- 시즌 평균 득점/실점

상대 비교:

- 상대 최근 흐름
- 시즌 승률 차이
- 최근 5경기 승률 차이
- 최근 10경기 승률 차이
- 시즌 득실차 차이
- 상대전적 차이

경기 환경:

- 홈/원정
- 휴식일
- 최근 7일 경기 수
- 백투백 여부
- 홈/원정 승률 차이

전력/컨텍스트:

- Elo 점수
- Elo 차이
- 불펜 피로 proxy
- 추정/확정 선발 컨텍스트 일부
- 선수 영향도 피처 참고 실험

선수 영향도 피처는 다음처럼 홈팀과 원정팀의 차이값으로 변환합니다.

- `top3_hitter_ops_avg_gap`
- `top3_hitter_impact_score_gap`
- `hitter_dependency_gap`
- `pitcher_core_score_gap`
- `pitcher_dependency_gap`

모델에는 선수 이름 자체를 넣지 않습니다. 대신 `홈팀 핵심 타자 OPS 평균 - 원정팀 핵심 타자 OPS 평균`처럼 경기 단위 전력 차이로 압축합니다.

### 2.3 모델 후보

`modeling/model_training.py`에서 여러 후보 모델을 학습하고 비교합니다.

현재 비교하는 후보 예:

- 기본 흐름 모델
- 전력/일정 피로도 포함 모델
- 핵심 수치 모델
- RandomForest
- RandomForest 시간가중 모델
- GradientBoosting
- GradientBoosting 시간가중 모델
- GradientBoosting 확률보정 모델
- 경기 단위 RandomForest
- 경기 단위 GradientBoosting

모델 선택은 정확도만 보지 않고, 확률 품질도 함께 봅니다.

확률 품질 지표:

- Brier Score
- Log Loss
- Calibration Table

### 2.4 팀 기준 모델과 경기 기준 모델

현재 프로젝트는 두 가지 예측 단위를 모두 실험합니다.

팀 기준 모델:

```text
한 경기 = 두 행
LG 기준 행
삼성 기준 행
```

경기 기준 모델:

```text
한 경기 = 한 행
home_team
away_team
target_home_win
```

경기 기준 모델은 실제 예측 구조에 더 자연스럽지만, 현재 성능 비교를 위해 두 방식을 모두 후보로 둡니다.

## 3. 프로젝트 작동 순서

초보자 관점에서는 `official_kbo_dashboard.py`를 실행하면 전체 작업이 순서대로 진행된다고 보면 됩니다.

```text
1. KBO 공식 페이지에서 데이터 수집
2. 수집 데이터를 data/official/*.csv로 저장
3. 가능하면 PostgreSQL 테이블로 적재
4. 팀 리포트용 지표 계산
5. 작전/상황 수행 지수 계산
6. 핵심 선수 영향도 계산
7. 모델 학습용 피처 생성
8. 후보 모델 학습과 검증
9. 오늘 경기 예측 생성
10. dashboard/*.html 생성
11. docs/*.html 정적 배포 파일 생성
```

## 4. 실행 방법

### 4.1 프로젝트 폴더 이동

```bash
cd /home/tera/1.project/1.sports_analytics/kbo_analytics
```

### 4.2 Docker 서비스 실행

```bash
docker compose up -d
```

주요 서비스:

- PostgreSQL
- dashboard 정적 파일 서버
- 기존 mock API 컨테이너

현재 공식 KBO 대시보드의 기준 데이터는 KBO 공식 페이지에서 직접 가져옵니다. `mock_api`는 과거 파이프라인 검증용 성격이 강합니다.

### 4.3 대시보드 생성

```bash
.venv/bin/python official_kbo_dashboard.py --training-start-year 2016
```

특정 날짜를 기준으로 실행하려면:

```bash
.venv/bin/python official_kbo_dashboard.py \
  --reference-date 2026-05-19 \
  --training-start-year 2016
```

경기 전 업데이트 모드:

```bash
.venv/bin/python official_kbo_dashboard.py \
  --reference-date 2026-05-19 \
  --update-stage pregame \
  --training-start-year 2016
```

옵션 설명:

| 옵션 | 의미 |
| --- | --- |
| `--reference-date` | 분석 기준일 |
| `--reference-datetime` | 분석 기준 시각 |
| `--update-stage` | `morning` 또는 `pregame` |
| `--training-start-year` | 모델 학습 시작 연도 |

### 4.4 문법 확인

코드를 수정한 뒤에는 먼저 문법 오류를 확인합니다.

```bash
.venv/bin/python -m py_compile official_kbo_dashboard.py
```

모델링 파일까지 함께 확인:

```bash
.venv/bin/python -m py_compile \
  official_kbo_dashboard.py \
  modeling/game_level_features.py \
  modeling/model_evaluation.py \
  modeling/model_training.py
```

### 4.5 HTML 확인

dashboard 컨테이너가 떠 있으면 아래 주소에서 확인할 수 있습니다.

```text
http://127.0.0.1:8501/latest.html
```

직접 임시 서버를 띄울 수도 있습니다.

```bash
python -m http.server 8501 -d dashboard
```

## 5. 자동 운영

운영 서버에서는 cron으로 자동 갱신합니다. 아래 스케줄은 저장소 코드만 보고 추측한 내용이 아니라, 운영 서버의 `crontab`과 실행 로그를 기준으로 확인한 운영 정보입니다.

현재 의도한 운영 방식:

```text
매일 08:00
  - 공식 데이터 수집
  - 모델 학습/검증
  - 대시보드 생성
  - GitHub push

매일 11:00~18:30, 30분 간격
  - 경기 전 pregame 업데이트
  - 확정 선발 확인 시 반영
  - 대시보드 재생성
  - 변경 시 GitHub push
```

현재 서버 crontab:

```cron
0 8 * * * /usr/bin/flock -n /tmp/kbo_daily_update.lock /home/tera/1.project/1.sports_analytics/kbo_analytics/scripts/daily_kbo_update.sh
0,30 11-18 * * * /usr/bin/flock -n /tmp/kbo_pregame_update.lock /home/tera/1.project/1.sports_analytics/kbo_analytics/scripts/pregame_kbo_update.sh
```

운영 의도는 다음과 같습니다.

| 시간 | 역할 |
| --- | --- |
| 매일 08:00 | 하루 기준 정식 데이터 수집, 모델 학습/검증, 대시보드 생성 |
| 매일 11:00~18:30 | 30분 간격 경기 전 업데이트 |
| 경기 시작 약 2시간 전 | 확정 선발 반영 가능 구간 |
| 변경 발생 시 | 대시보드 재생성 후 GitHub push |

현재 구조는 운영상 충분히 동작하지만, 향후에는 `11:00~18:30` 전체 반복을 더 세분화할 수 있습니다.

```text
08:00  정식 학습/검증
11:00  1차 예측
16:00~18:30  확정 선발 집중 업데이트
경기 종료 후 결과 반영은 다음날 08:00
```

다음 개선점은 pregame 업데이트에서 선발 정보가 `estimated`에서 `confirmed`로 바뀌었는지 로그와 대시보드에 더 명확히 표시하는 것입니다.

실행 스크립트:

| 파일 | 역할 |
| --- | --- |
| `scripts/daily_kbo_update.sh` | 매일 오전 전체 갱신 |
| `scripts/pregame_kbo_update.sh` | 경기 전 반복 갱신 |

각 스크립트는 `.env`가 있으면 환경변수를 로드합니다. 서버 IP, 계정, 비밀번호 같은 운영 접속 정보는 README와 Git에 기록하지 않습니다.

## 6. 주요 결과 파일

| 파일 | 설명 |
| --- | --- |
| `dashboard/latest.html` | 최신 리그 대시보드 |
| `dashboard/kt.html` | KT 구단 리포트 |
| `dashboard/lg.html` | LG 구단 리포트 |
| `dashboard/*.html` | 구단별 리포트 |
| `dashboard/latest_summary.md` | 최신 대시보드 요약 |
| `../docs/index.html` | GitHub 정적 배포용 메인 |
| `../docs/latest.html` | GitHub 정적 배포용 최신 페이지 |
| `data/official/*.csv` | 공식 데이터 스냅샷 |
| `modeling/results/features.csv` | 팀 기준 모델 피처 |
| `modeling/results/game_level_features.csv` | 경기 기준 모델 피처 |
| `modeling/results/win_predictor_model.json` | 모델 검증/예측 결과 |
| `modeling/results/model_history.json` | 모델 성능 이력 |

## 7. 폴더 구조

```text
sports_analytics/
├── README.md
├── docs/
│   ├── index.html
│   ├── latest.html
│   ├── kt.html
│   └── ...
└── kbo_analytics/
    ├── official_kbo_dashboard.py
    ├── collector.py
    ├── weekly_update.py
    ├── docker-compose.yaml
    ├── requirements.txt
    ├── scripts/
    │   ├── daily_kbo_update.sh
    │   ├── pregame_kbo_update.sh
    │   └── weekly_kbo_update.sh
    ├── data/
    │   └── official/
    ├── dashboard/
    │   ├── latest.html
    │   ├── kt.html
    │   └── ...
    ├── modeling/
    │   ├── feature_engineering.py
    │   ├── game_level_features.py
    │   ├── model_evaluation.py
    │   ├── model_training.py
    │   ├── train_win_predictor.py
    │   └── results/
    └── sql/
```

## 8. 주요 코드 역할

### `official_kbo_dashboard.py`

전체 실행의 중심 파일입니다.

주요 역할:

- KBO 공식 데이터 수집
- CSV 저장
- PostgreSQL 적재 시도
- 팀 리포트 생성
- 작전/상황 수행 지수 계산
- 핵심 선수 영향도 계산
- 모델 학습 호출
- HTML 대시보드 생성
- `docs/` 정적 배포 파일 생성

초보자 관점에서는 이 파일을 실행하면 전체 대시보드가 만들어진다고 보면 됩니다.

### `modeling/feature_engineering.py`

경기 결과를 모델이 학습할 수 있는 숫자 피처로 바꿉니다.

예:

- 최근 5경기 승률
- 최근 10경기 승률
- 시즌 승률
- 최근 득실차
- 홈/원정 여부
- 상대전적
- Elo 점수
- 휴식일

### `modeling/game_level_features.py`

한 경기를 한 행으로 보는 경기 단위 데이터를 만듭니다.

예:

```text
KT vs 삼성
home_team = 삼성
away_team = KT
target_home_win = 삼성 승리 여부
```

### `modeling/model_evaluation.py`

모델 성능을 평가합니다.

평가 지표:

- Accuracy
- Brier Score
- Log Loss
- 확신 구간별 적중률
- Calibration Table

### `modeling/model_training.py`

여러 후보 모델을 학습하고 비교합니다. 모델 학습/검증 로직은 대시보드 코드에서 분리되어 이 파일에서 관리됩니다.

## 9. 데이터 해석 시 주의점

### 9.1 없는 데이터는 만들지 않는다

KBO 공식 공개 데이터에 없는 항목은 임의로 만들지 않습니다.

예:

- 작전 지시 수
- 사인 성공률
- 실제 작전 성공률

대신 공식 기록으로 확인 가능한 proxy를 사용합니다.

### 9.2 proxy는 실제 지표가 아니라 대체 지표다

희생번트는 `작전 지시 수`가 아닙니다. 하지만 공식 기록에서 확인 가능한 작전 관련 결과이므로, 작전 개입 빈도를 간접적으로 보는 proxy로 사용합니다.

### 9.3 전체 정확도보다 선별 정확도가 중요하다

야구는 단일 경기 변동성이 큽니다. 모든 경기를 강제로 예측하기보다, 모델이 비교적 확신하는 경기만 따로 보는 것이 중요합니다.

예:

```text
전체 경기 정확도: 참고 지표
55% 이상 확신 경기: 실전 추천 지표
58% 이상 확신 경기: 고신뢰 추천 후보
60% 이상 경기: 과신 여부 점검
```

## 10. 확률적/분석적 인사이트 확장 방향

현재 구단 리포트는 선수 영향도까지 보여줍니다. 다음 단계는 이 정보를 경기 예측 모델의 입력값으로 연결하는 것입니다.

### 10.1 선수 영향도를 승률 변화 요인으로 연결

현재:

```text
안현민: OPS 1.161 · 출루율 0.507 · 장타율 0.654
```

다음 단계:

```text
KT는 핵심 타자 3명의 평균 OPS가 팀 평균보다 높아 중심타선 의존도가 큽니다.
이 선수들이 라인업에서 빠질 경우 공격 생산력이 크게 떨어질 수 있습니다.
```

### 10.2 선수명을 직접 모델 피처로 쓰지 않는다

나쁜 예:

```text
안현민_OPS
허경민_OPS
오스틴_OPS
```

좋은 예:

```text
home_top3_hitter_ops_avg
away_top3_hitter_ops_avg
top3_hitter_ops_gap
```

모델은 `누가 누구냐`보다 `홈팀이 원정팀보다 어느 부분에서 얼마나 우위인가`를 학습하는 편이 안정적입니다.

### 10.3 다음에 추가할 모델 피처

타선:

- `lineup_ops_gap`
- `top3_hitter_ops_gap`
- `lineup_obp_gap`
- `lineup_slg_gap`
- `runners_in_scoring_position_avg_gap`
- `bb_k_gap`

선발투수:

- `starter_era_gap`
- `starter_whip_gap`
- `starter_recent3_era_gap`
- `starter_recent3_ip_avg_gap`
- `starter_rest_days_gap`
- `starter_score_gap`
- `starter_source_confirmed`

불펜:

- `core_bullpen_ip_last_3d_gap`
- `closer_used_yesterday_gap`
- `setup_used_yesterday_gap`
- `top3_reliever_back_to_back_gap`
- `bullpen_fatigue_gap`

전력 의존도:

- `hitter_dependency_gap`
- `pitcher_dependency_gap`
- `top3_hitter_ops_share`
- `top3_pitcher_innings_share`

## 11. 배포 확인

GitHub에 push된 정적 HTML은 RawGitHack 형태로 확인할 수 있습니다.

```text
https://raw.githack.com/SaRangWOO/sports_analytics/main/docs/
```

특정 커밋 기준:

```text
https://raw.githack.com/SaRangWOO/sports_analytics/<commit-hash>/docs/
```

## 12. 프로젝트 요약

이 프로젝트는 아래 흐름을 자동화합니다.

```text
공식 기록 수집
→ 팀 스타일 분석
→ 작전/상황 수행 지수화
→ 핵심 선수 영향도 계산
→ 경기 승패 예측 모델 검증
→ HTML 대시보드 생성
→ 정적 페이지 배포
```

현재는 구단 분석 리포트와 기본 승패 예측 모델이 구현되어 있습니다. 다음 단계는 선수 영향도를 경기 예측 모델의 피처로 연결하고, 어떤 선수 기반 피처가 실제로 정확도와 확률 품질을 개선하는지 ablation test로 검증하는 것입니다.
