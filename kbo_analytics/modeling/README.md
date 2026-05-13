# KBO Win Prediction Model

PostgreSQL에서 내보낸 `data/weekly/game_results.csv`를 사용해 경기 전 정보 기반 승패 예측 모델을 학습합니다.

## Features

- 홈/원정 여부
- 상대팀
- 월
- 시리즈 내 경기 번호
- 휴식일
- 최근 5경기 승률
- 최근 5경기 평균 득점
- 최근 5경기 평균 실점
- 최근 5경기 평균 득실차

## Run

```bash
cd kbo_analytics/modeling
python train_win_predictor.py --input ../data/weekly/game_results.csv
python evaluate_model.py
```

## Notes

- 모델은 추가 ML 패키지 없이 실행할 수 있도록 로지스틱 회귀를 직접 구현했습니다.
- `weekly_update.py`는 여러 후보 모델과 판정 기준을 비교한 뒤 검증 정확도와 F1이 가장 좋은 설정을 선택합니다.
- 선택 결과는 `modeling/results/win_predictor_model.json`에 저장되고, 실행 이력은 `modeling/results/model_history.json`에 누적됩니다.
- `Draw` 경기는 이진 분류 목표가 아니므로 학습 데이터에서 제외합니다.
- 실제 운영 흐름에서는 `weekly_update.py`가 DB에서 CSV를 갱신한 뒤 모델 결과를 다시 생성합니다.
