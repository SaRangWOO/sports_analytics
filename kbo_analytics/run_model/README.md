# Independent KBO Run Model

이 폴더는 기존 KBO 대시보드와 기존 승패 예측 모델에서 완전히 분리된 득점 기반 예측 모델입니다.

## 목적

기존 모델처럼 승패를 바로 분류하지 않고, 먼저 팀별 예상 득점을 예측합니다.

흐름은 다음과 같습니다.

```text
완료 경기 원천 데이터
→ 팀별 득점/실점 흐름 피처 생성
→ 팀별 예상 득점 회귀 모델 학습
→ 홈/원정 예상 득점 차이 계산
→ 예상 득실차를 홈팀 승률로 변환
```

## 분리 원칙

- 기존 `official_kbo_dashboard.py`는 수정하지 않습니다.
- 기존 `modeling/model_training.py`와 기존 승패 모델은 사용하지 않습니다.
- 기존 `modeling/results/` 산출물은 읽거나 쓰지 않습니다.
- 완료 경기 원천 데이터인 `data/official/model_training_games.csv`만 읽기 전용으로 사용합니다.
- 새 코드와 결과물은 모두 `kbo_analytics/run_model/` 아래에만 저장합니다.

## 생성 파일

```text
run_model/
├── run_prediction_model.py
├── results/
│   ├── run_model_features.csv
│   ├── expected_runs_predictions.csv
│   └── expected_runs_model.json
├── dashboard.html
└── README.md
```

## 실행 방법

프로젝트 서버에서:

```bash
cd /home/tera/1.project/1.sports_analytics/kbo_analytics
.venv/bin/python run_model/run_prediction_model.py
```

모델 결과를 HTML 대시보드로 확인하려면:

```bash
.venv/bin/python run_model/run_model_dashboard.py
```

옵션을 직접 지정할 수도 있습니다.

```bash
.venv/bin/python run_model/run_prediction_model.py \
  --input data/official/model_training_games.csv \
  --output-dir run_model/results \
  --train-ratio 0.8
```

## 결과 파일 설명

| 파일 | 설명 |
| --- | --- |
| `results/run_model_features.csv` | 팀 기준 2행 구조의 득점 예측 피처 |
| `results/expected_runs_predictions.csv` | 검증 구간 경기별 예상 득점, 예상 득실차, 홈팀 승률 |
| `results/expected_runs_model.json` | 후보 모델 성능, 선택 모델, 사용 피처, 실행 메타데이터 |
| `results/run_model_error_analysis.csv` | 선택 모델의 경기별 득점 오차, 오차 구간, 오차 태그 |
| `results/error_tag_summary.csv` | 오차 태그별 경기 수, 평균 오차, 방향 적중률 요약 |
| `results/run_model_feature_importance.csv` | 선택 모델의 permutation importance |
| `dashboard.html` | 모델링 전용 HTML 대시보드 |

## 모델 후보

팀별 `target_runs`를 예측하는 회귀 모델 후보는 다음과 같습니다.

- Poisson Regression
- Ridge Regression
- Tweedie Regression
- RandomForest Regressor
- HistGradientBoosting Regressor

각 모델은 먼저 팀별 예상 득점을 예측합니다. 이후 홈팀 예상 득점과 원정팀 예상 득점의 차이인 `expected_run_diff`를 만들고, 이 값을 Logistic Regression으로 홈팀 승률에 변환합니다.

## 평가 지표

득점 예측 성능:

- MAE
- RMSE

득점 기반 승률 변환 성능:

- Accuracy
- Brier Score
- Log Loss
- 예상 득실차 방향 적중률

## 오차 분석

선택 모델에 대해서는 별도 오차 분석 파일을 생성합니다.

`run_model_error_analysis.csv`에는 다음과 같은 태그가 들어갑니다.

| 태그 | 의미 |
| --- | --- |
| `LOW_SCORING_MISS` | 실제 저득점 경기에서 승패 예측이 빗나간 경우 |
| `HIGH_SCORING_MISS` | 실제 고득점 경기에서 승패 예측이 빗나간 경우 |
| `RUN_DIFF_DIRECTION_MISS` | 예상 득실차 방향과 실제 득실차 방향이 다른 경우 |
| `BLOWOUT_UNDERPREDICTED` | 실제 큰 점수 차 경기를 박빙으로 예측한 경우 |
| `CLOSE_GAME_NOISE` | 실제 접전 경기에서 승패 예측이 빗나간 경우 |

`run_model_feature_importance.csv`는 선택 모델의 피처를 하나씩 섞어 보며 MAE가 얼마나 나빠지는지 계산합니다.
이 값은 모델이 어떤 입력 신호에 민감한지 확인하기 위한 진단용입니다.

## 해석 방법

`expected_runs_predictions.csv`의 핵심 컬럼은 다음과 같습니다.

| 컬럼 | 의미 |
| --- | --- |
| `home_expected_runs` | 홈팀 예상 득점 |
| `away_expected_runs` | 원정팀 예상 득점 |
| `expected_run_diff` | 홈팀 예상 득점 - 원정팀 예상 득점 |
| `home_win_probability` | 예상 득실차 기반 홈팀 승률 |
| `predicted_winner` | 예상 승리팀 |
| `actual_winner` | 실제 승리팀 |
| `prediction_result` | 적중 여부 |

이 모델의 목적은 기존 승패 모델을 대체하는 것이 아니라, “왜 이 팀이 우세한가”를 예상 득점과 예상 득실차로 설명하는 별도 분석 축을 만드는 것입니다.
