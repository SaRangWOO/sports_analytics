# KBO Analytics

KBO 공식 기록을 수집하고 승패 확률, 예상 득점, 선발·라인업 상태를 하나의 HTML 대시보드로 생성하는 운영 파이프라인입니다.

[대시보드](https://raw.githack.com/SaRangWOO/sports_analytics/main/docs/latest.html) · [루트 소개](../README.md) · [아키텍처](../docs/ARCHITECTURE.md) · [모델 검증 원칙](../docs/MODEL_GOVERNANCE.md)

## 제공 기능

- KBO 일정, 결과, 순위, 상대전적, 타자·투수 기록 수집
- 취소·우천·폭염 등 공식 경기 상태 반영
- GameCenter 기반 확정 선발과 라인업 수집
- 팀 흐름, Elo, 득실차, 일정·피로 피처 생성
- 승패 모델과 독립 득점 모델 비교
- 경기 전 예측 변화 이력 저장
- 모델 아티팩트 검증·승격·롤백
- predict-only 경기 전 갱신
- pipeline health와 recommendation outcome 감사

## 데이터 흐름

```text
KBO 공식 원천
→ official_kbo_dashboard.py
→ data/official/*.csv
→ modeling/feature_engineering.py
→ modeling/model_training.py
→ modeling/results/*.csv|json
→ run_model/run_prediction_model.py
→ dashboard/latest.html
→ ../docs/latest.html
```

## 주요 데이터

| 파일 | 역할 |
| --- | --- |
| `data/official/game_results.csv` | 일정·완료 결과·취소 상태 |
| `data/official/model_training_games.csv` | 승패·득점 모델 학습 입력 |
| `data/official/prediction_games.csv` | 기준일 예정 경기 |
| `data/official/team_standings.csv` | 팀 순위 |
| `data/official/hitter_stats.csv` | 현재 타자 기록 |
| `data/official/pitcher_stats.csv` | 현재 투수 기록 |
| `data/official/pitching_context.csv` | 현재 선발·불펜 컨텍스트 |
| `data/official/lineup_context.csv` | 현재·최근 라인업 |
| `data/official/pitching_daily_snapshot.csv` | 예측 시점별 투수 정보 |
| `data/official/pitching_snapshot_schedule.csv` | 공식 경기 ID·시작 시각 |

현재 선수 통계 스냅샷은 오늘 경기 설명에 사용할 수 있지만, 과거 경기 전체의 학습 피처로 직접 결합하지 않습니다. 과거 시점에는 알 수 없었던 누적 기록이 들어갈 수 있기 때문입니다.

## 모델 구조

### 운영 승패 모델

현재 선택 모델은 `RandomForest 보수 시간가중 모델`입니다. 결과는 `modeling/results/win_predictor_model.json`에 기록됩니다.

주요 피처 그룹:

- Elo와 시즌 전력
- 최근 5·10경기 승률과 득실차
- 홈·원정 성과
- 휴식일과 경기 밀도
- 연승·연패와 득점 변동성
- 상대전적과 월별 흐름

모든 rolling 피처는 현재 경기 이전 데이터만 사용합니다.

### 독립 득점 모델

`run_model/`은 승패 모델 예측값을 사용하지 않고 팀별 득점을 별도로 추정합니다.

출력:

- `home_expected_runs`
- `away_expected_runs`
- `expected_run_diff`
- `home_win_probability`

오늘 예정 경기 결과는 `today_expected_runs_predictions.csv`, validation 결과는 `expected_runs_predictions.csv`로 분리합니다.

### 투수 스냅샷 challenger

투수 스냅샷 저장 조건:

```text
snapshot_time < scheduled_start_datetime
```

공식 일정에 매핑되지 않거나 경기 시작 이후 생성된 행은 canonical 파일에 저장하지 않습니다. 후보 모델은 별도 rolling validation을 수행하며 production gate를 통과하기 전까지 운영 모델에 연결하지 않습니다.

## 주요 결과 파일

| 파일 | 내용 |
| --- | --- |
| `modeling/results/win_predictor_model.json` | 선택 모델, 검증 지표, 오늘 예측 |
| `modeling/results/production_model_gate_audit.json` | 운영 승격 판정 |
| `modeling/results/model_insight_summary.json` | 피처·세그먼트·데이터 공백 요약 |
| `modeling/results/daily_pipeline_health_status.json` | 일일 파이프라인 상태 |
| `modeling/results/pitching_snapshot_candidate_gate_audit.json` | 투수 challenger 판정 |
| `run_model/results/expected_runs_model.json` | 득점 모델 성능 |
| `run_model/results/today_expected_runs_predictions.csv` | 기준일 예정 경기 예상 득점 |

## 실행

### 환경 구성

```bash
cd sports_analytics/kbo_analytics
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
docker compose up -d
```

### 전체 갱신

```bash
.venv/bin/python official_kbo_dashboard.py \
  --reference-date 2026-08-25 \
  --training-start-year 2016 \
  --update-stage morning
```

### 승인 artifact 기반 갱신

```bash
.venv/bin/python scripts/predict_only_dashboard_hybrid.py \
  --reference-date 2026-08-25 \
  --update-stage pregame
```

### 작업 CLI

```bash
.venv/bin/python scripts/kbo_tasks.py --help
.venv/bin/python scripts/kbo_tasks.py smoke
.venv/bin/python scripts/kbo_tasks.py full --reference-date 2026-08-25
.venv/bin/python scripts/kbo_tasks.py predict-only --reference-date 2026-08-25
```

### HTML 확인

```bash
python3 -m http.server 8501 -d dashboard
```

`http://127.0.0.1:8501/latest.html`에서 확인합니다.

## 테스트

```bash
.venv/bin/python -m unittest \
  modeling.test_model_artifacts \
  modeling.test_predict_only \
  modeling.test_pitching_snapshot_storage \
  modeling.test_pitching_snapshot_candidate_validation \
  modeling.test_hybrid_probability_policy -v
```

## 폴더 구조

```text
kbo_analytics/
├── official_kbo_dashboard.py
├── config/
├── data/
│   ├── official/
│   └── manual/
├── modeling/
│   ├── artifacts/
│   ├── results/
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── model_artifacts.py
│   ├── prediction_runtime.py
│   └── predict_only.py
├── run_model/
├── scripts/
├── dashboard/
├── reports/
└── sql/
```

## 운영 원칙

1. 완료 경기만 학습에 사용합니다.
2. 미래 결과가 섞일 수 있는 최신 선수 기록은 과거 학습에 일괄 결합하지 않습니다.
3. 모델 확률과 대시보드 추천 문구를 분리합니다.
4. Accuracy만으로 모델을 승격하지 않습니다.
5. 검증 실패 시 기존 production artifact와 CSV를 유지합니다.
6. PostgreSQL 장애는 경고로 기록하되 CSV·JSON·HTML 생성은 계속합니다.

## 현재 한계

- 장기간의 확정 라인업 스냅샷이 부족합니다.
- 실제 불펜 투구 수와 핵심 불펜 연투 피처가 필요합니다.
- 투수 challenger는 최소 표본과 bootstrap 안정성 gate를 통과하지 못했습니다.
- 모델 정확도는 강한 상업 예측 성능으로 해석할 수준이 아닙니다.

다음 우선순위는 선발 최근 3경기·휴식일·불펜 실제 투구 수를 as-of 기준으로 축적하고, walk-forward 검증으로 재평가하는 것입니다.
