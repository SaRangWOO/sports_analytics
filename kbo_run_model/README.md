# KBO 승패·핸디캡·오버/언더 예측 대시보드

`kbo_run_model`은 KBO 경기 일정에 맞춰 경기별 예상 스코어, 승패 확률, 핸디캡 추천, 오버/언더 추천, 신뢰도를 산출하는 독립 예측 프로젝트입니다. 기존 `kbo_analytics` 대시보드와 기존 모델 파일은 수정하지 않습니다.

## 주요 출력

- 예상 홈 득점 / 예상 원정 득점
- 예상 득실차
- 홈 승률 / 원정 승률
- 승/패 추천
- 핸디캡 추천
- 오버/언더 추천
- 신뢰도

대시보드:

```text
results/report.html
```

경기별 예측 CSV:

```text
results/match_predictions.csv
```

## 실행

기본 실행은 오늘 날짜 경기 일정을 찾습니다. 오늘 경기가 없으면 다음 예정 경기, 이후 일정도 없으면 가장 최근 경기 날짜를 기준으로 리포트를 만듭니다.

```bash
cd "C:\Users\PCuser\Documents\New project\sports_analytics\kbo_run_model"
python run_pipeline.py
```

특정 날짜를 지정할 수 있습니다.

```bash
python run_pipeline.py --target-date 2026-05-27
```

## 결과 파일

- `results/match_predictions.csv`
- `results/report.html`
- `results/summary.json`
- `results/model_scores.csv`
- `results/season_metrics.csv`
- `results/team_bias_metrics.csv`
- `results/team_game_features.csv`

## 예측 방식

1. 완료 경기만 사용해 팀 단위 득점 예측 모델을 학습합니다.
2. 일정 CSV에서 예측 기준 날짜의 경기 목록을 읽습니다.
3. 각 팀의 현재 경기 이전 기록만 사용해 rolling 피처를 생성합니다.
4. 홈팀 예상 득점과 원정팀 예상 득점을 각각 예측합니다.
5. 예상 득실차를 승률로 변환합니다.
6. 승/패, 핸디캡, 오버/언더, 신뢰도를 계산합니다.

## 추천 기준

핸디캡 기준값:

- `1.5`
- `2.5`
- `3.5`

기본 화면은 `2.5` 기준을 표시합니다.

오버/언더 기준값:

- `7.5`
- `8.5`
- `9.5`
- `10.5`

기본 화면은 `8.5` 기준을 표시합니다.

신뢰도 기준:

- 예상 득실차 절대값 `0.0 ~ 0.5`: 낮음
- 예상 득실차 절대값 `0.5 ~ 1.0`: 보통
- 예상 득실차 절대값 `1.0 이상`: 높음

## 데이터 누수 방지

- rolling 피처는 현재 경기 이전 데이터만 사용합니다.
- 같은 날짜의 미래 경기 결과가 현재 경기 피처에 섞이지 않도록 날짜 단위로 이력을 갱신합니다.
- 시즌별, 팀별 그룹을 분리해 이전 시즌 경기 결과가 현재 시즌 첫 경기 rolling 계산에 직접 섞이지 않습니다.
- 선수/투수 피처 계산 시 `pitcher_name`이 아니라 `pitcher_id`를 기준으로 groupby, rolling, join을 수행합니다.

## 선수 데이터 확장 상태

현재 팀 기반 baseline은 동작합니다. 선발투수 모델은 실제 선발투수 매핑과 투수 등판 로그가 부족해 아직 학습하지 않습니다. 아래 sample CSV는 스키마 예시와 적재 테스트용 mock 데이터이며 실제 학습 데이터가 아닙니다.

- `data/starter_pitchers_sample.csv`
- `data/pitcher_game_logs_sample.csv`

실제 학습을 위해 필요한 선발 매핑 컬럼:

- `season`
- `date`
- `game_id`
- `home_team`
- `away_team`
- `home_starter_name`
- `away_starter_name`
- `home_starter_id`
- `away_starter_id`

실제 학습을 위해 필요한 투수 로그 컬럼:

- `season`
- `date`
- `game_id`
- `pitcher_id`
- `pitcher_name`
- `team`
- `opponent`
- `is_starter`
- `innings_pitched`
- `earned_runs`
- `hits_allowed`
- `walks`
- `strikeouts`
- `home_runs_allowed`
- `pitches`

## 이닝 처리

`innings_pitched`의 `5.1`, `5.2`는 십진수가 아니라 각각 5⅓, 5⅔ 이닝입니다. ERA, WHIP, 평균 이닝 계산은 아웃 카운트 기반으로 변환한 뒤 수행합니다.

구현 함수:

- `features/starter_features.py`의 `ip_to_outs()`
- `features/starter_features.py`의 `outs_to_decimal_ip()`
- `features/starter_features.py`의 `normalize_innings_pitched()`

## 불펜 확장 계획

`pitcher_game_logs_sample.csv`는 선발투수 전용이 아니라 통합 투수 로그 스키마입니다. 향후 `is_starter=False` 등판 기록을 사용해 불펜 피로도 피처를 계산할 수 있습니다.

예시 피처:

- `bullpen_ip_last_1d`
- `bullpen_ip_last_3d`
- `bullpen_er_last_3d`
- `bullpen_pitch_count_last_3d`
- `bullpen_pitchers_used_last_3d`
- `closer_used_yesterday`
- `setup_man_used_yesterday`
