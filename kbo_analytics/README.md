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
- 동일 승인 artifact 기반 predict-only 오전·경기 전 갱신
- pipeline health와 recommendation outcome 감사

## 데이터 흐름

```text
운영: KBO 공식 원천 → data/official → 승인 artifact 추론 → 독립 득점 모델 → HTML
연구: 완료 경기 → 피처 → 후보 선택 → 미사용 최종 검증 → 승격 심사
```

오전과 경기 전 작업은 `scripts/predict_only_dashboard_hybrid.py`를 사용합니다. 데이터 수집과 승인 모델 재학습은 별개이며, 매일 새 모델을 자동 선택하지 않습니다. 세 갱신 shell 진입점은 공통 잠금을 사용하고 승패·득점 결과를 함께 동기화합니다.

## 주요 데이터

| 파일 | 역할 |
| --- | --- |
| `data/official/game_results.csv` | 일정·완료 결과·취소 상태 |
| `data/official/model_training_games.csv` | 승패·득점 모델 학습 입력 |
| `data/official/prediction_games.csv` | 추론 피처용 일정·이력 입력; 출력 단계에서 기준일 예정 경기만 선택 |
| `data/official/team_standings.csv` | 팀 순위 |
| `data/official/hitter_stats.csv` | 현재 타자 기록 |
| `data/official/pitcher_stats.csv` | 현재 투수 기록 |
| `data/official/pitching_context.csv` | 현재 선발·불펜 컨텍스트 |
| `data/official/lineup_context.csv` | 현재·최근 라인업 |
| `data/official/pitching_daily_snapshot.csv` | 예측 시점별 투수 정보 |
| `data/official/pitching_snapshot_schedule.csv` | 공식 경기 ID·시작 시각 |
| `data/official/pitcher_game_logs.csv` | KBO 박스스코어 기반 투수별 이닝·투구 수·실점 로그 |

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

rolling 피처는 현재 경기 이전 데이터만 사용하는 것이 원칙입니다. 과거 기본 피처의 전기간 평균 결측 대체는 추가 감사 대상이며, 아직 전체 피처의 누수 감사를 완료했다는 의미는 아닙니다.

평가 지표는 `model_id`·`metrics_model_id`가 같은 승인 아티팩트에서만 가져옵니다. 구형 아티팩트의 검증 기간이 없으면 미기록 상태로 남기고, 30일이 지난 아티팩트는 경고합니다. 노후화 경고만으로 모델을 교체하지 않습니다.

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

완료 경기 로그의 이전 날짜 등판으로 선발 최근 3경기 기록과 불펜 최근 1·3일 사용량을 만듭니다. WHIP 필드는 BB/HBP 합계 기반 proxy이고 휴식일은 직전 선발 등판과의 날짜 차이입니다. 결측값은 실측 0과 구분합니다. 실제 등판 선발과 경기 전 확정 선발은 같다고 가정하지 않습니다.

후보 선택 기간과 최종 검증 기간은 분리하고 bootstrap은 경기일 단위로 수행합니다. [승격 검토 기록](../docs/reviews/PITCHER_PROMOTION_20261013.md)은 실제 승인 아티팩트의 동일 경기 재현과 `snapshot_time <= prediction_reference_datetime < scheduled_start_datetime` 검사를 함께 기록합니다.

PostgreSQL은 CSV 원천을 대체하지 않고 조회·감사용 feature store로 사용합니다. `pitcher_game_logs`, `pregame_pitching_snapshots`, `pregame_lineup_snapshots` 테이블은 `sql/002_feature_store.sql`로 관리하며 DB 장애는 모델·HTML 생성을 막지 않습니다.

## 주요 결과 파일

| 파일 | 내용 |
| --- | --- |
| `modeling/results/win_predictor_model.json` | 선택 모델, 검증 지표, 오늘 예측 |
| `modeling/results/production_model_gate_audit.json` | 운영 승격 판정 |
| `modeling/results/model_insight_summary.json` | 피처·세그먼트·데이터 공백 요약 |
| `modeling/results/daily_pipeline_health_status.json` | 일일 파이프라인 상태 |
| `modeling/results/pitching_snapshot_candidate_gate_audit.json` | 투수 challenger 판정 |
| `modeling/results/pitcher_workload_candidate_gate_audit.json` | 경기별 투수 사용량 challenger 판정 |
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

### 연구용 전체 평가

격리된 작업 경로에서만 실행합니다. 운영 갱신 명령이 아니며 자동 승격하지 않습니다.

```bash
.venv/bin/python official_kbo_dashboard.py \
  --reference-date YYYY-MM-DD \
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
- 실제 투구 수는 수집 중이며 선발 정체성의 as-of 검증과 운영 피처 계약이 추가로 필요합니다.
- 재학습 기준 모델 대비 개선을 실제 배포 모델 대비 개선으로 간주하지 않습니다.
- 득점 모델의 OOF 보정은 `--temporal-oof --output-dir <격리된 경로>`로만 실험합니다.
- 모델 정확도는 강한 상업 예측 성능으로 해석할 수준이 아닙니다.

다음 우선순위는 선발 최근 3경기·휴식일·불펜 실제 투구 수를 as-of 기준으로 축적하고, walk-forward 검증으로 재평가하는 것입니다.
