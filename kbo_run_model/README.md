# KBO Run Model Research

KBO 경기의 홈·원정 예상 득점과 예상 득실차를 먼저 계산한 뒤 승패 확률로 변환하는 연구용 프로토타입입니다.

현재 운영 대시보드에서 사용하는 독립 득점 모델은 [`../kbo_analytics/run_model`](../kbo_analytics/run_model)에 통합되어 있습니다. 이 디렉터리는 초기 피처·데이터 소스·오차 분석 실험을 재현하기 위해 보존합니다.

## 연구 질문

- 시즌 평균과 최근 득점 흐름으로 팀 득점을 얼마나 설명할 수 있는가?
- 구장 효과와 팀 bias가 점수 MAE를 줄이는가?
- 예상 득실차를 승률로 변환했을 때 직접 승패 분류와 어떤 차이가 있는가?
- 선발·불펜 데이터가 없을 때 모델의 오차 유형은 무엇인가?

## 출력

| 파일 | 내용 |
| --- | --- |
| `results/match_predictions.csv` | 경기별 예상 득점과 승률 |
| `results/model_scores.csv` | 회귀 모델 후보 비교 |
| `results/season_metrics.csv` | 시즌별 성능 |
| `results/team_error_metrics.csv` | 팀별 오차 |
| `results/total_runs_error_metrics.csv` | 총득점 구간별 오차 |
| `results/report.html` | 연구 결과 대시보드 |

## 모델 흐름

```text
완료 경기
→ 경기당 팀 2행 변환
→ shift(1) rolling 피처
→ 득점 회귀 후보 비교
→ 홈·원정 예상 득점 결합
→ expected_run_diff
→ 승률 변환
```

후보 모델:

- Ridge
- PoissonRegressor
- TweedieRegressor
- RandomForestRegressor
- HistGradientBoostingRegressor

## 실행

```bash
cd sports_analytics/kbo_run_model
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python run_pipeline.py --target-date 2026-08-25
```

## 데이터 누수 방지

- 현재 경기 결과를 rolling 피처에 포함하지 않습니다.
- 같은 날짜의 경기 결과가 다른 경기 피처에 먼저 반영되지 않도록 날짜 단위로 갱신합니다.
- 시즌과 팀을 분리해 expanding/rolling 통계를 계산합니다.
- 투수 데이터는 이름이 아니라 안정적인 선수 ID로 연결하는 것을 원칙으로 합니다.

## 한계

- 초기 프로토타입의 핸디캡과 오버/언더는 실제 시장 라인 기반 운영 기능이 아닙니다.
- 과거 시점별 선발·라인업·불펜 기록이 충분하지 않습니다.
- 고득점 경기를 과소예측하고 저득점 경기를 과대예측하는 평균 회귀 성향이 있습니다.
- 운영 모델 승격 판단은 이 디렉터리가 아니라 `kbo_analytics/modeling/results/production_model_gate_audit.json`을 기준으로 합니다.
