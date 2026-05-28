# KBO 투수 데이터 수집/적재 파이프라인

## 목적

현재 KBO 예측 대시보드는 팀 단위 득점 흐름을 기준으로 예상 스코어, 승률, 핸디캡, 오버/언더를 계산한다. 진단 결과 총득점 예측 오차가 가장 큰 약점으로 확인되었고, 다음 개선에는 선발투수와 불펜 등판 로그가 필요하다.

선발투수와 불펜 데이터는 다음 정보를 모델에 제공한다.

- 당일 선발투수의 실점 억제력
- 선발투수 최근 컨디션과 휴식일
- 최근 불펜 소모량
- 오버/언더와 총득점 예측의 변동 요인

실제 데이터가 없으면 모델 학습을 진행하지 않는다. 빈 파일이나 sample/mock 데이터는 스키마 검증용으로만 사용한다.

## 입력 파일

실제 입력 파일은 sample 파일과 분리한다.

- `data/starter_pitchers.csv`
- `data/pitcher_game_logs.csv`

sample 파일은 구조 예시용이다.

- `data/starter_pitchers_sample.csv`
- `data/pitcher_game_logs_sample.csv`

## `starter_pitchers.csv` 스키마

필수 컬럼:

- `season`
- `date`
- `game_id`
- `home_team`
- `away_team`
- `home_starter_name`
- `away_starter_name`
- `home_starter_id`
- `away_starter_id`

`starter_name`은 표시용이며 계산과 매칭에는 `starter_id`를 사용한다.

## `pitcher_game_logs.csv` 스키마

필수 컬럼:

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

`is_starter` 값으로 선발투수와 불펜투수를 구분한다. 향후 불펜 피로도는 `is_starter=False`인 등판 기록에서 계산한다.

## `game_id` 매칭 원칙

- `starter_pitchers.csv`와 `pitcher_game_logs.csv`의 `game_id`는 일정/결과 데이터의 `game_id`와 같은 경기 단위를 가리켜야 한다.
- 팀 단위 데이터가 `_H`, `_A` 같은 suffix를 쓰는 경우에도 base game id가 같아야 한다.
- 검증기는 schedule의 경기 목록과 입력 CSV의 `game_id` 매칭률을 계산한다.

## `pitcher_id` 사용 원칙

- 동명이인을 방지하기 위해 rolling, groupby, join 계산은 `pitcher_name`이 아니라 `pitcher_id` 기준으로 수행한다.
- `pitcher_name`은 대시보드 표시와 수동 점검용으로만 사용한다.

## 이닝 변환 주의사항

야구 기록의 `innings_pitched`에서 `5.1`, `5.2`는 십진수가 아니다.

- `5.1` = 5와 1/3이닝
- `5.2` = 5와 2/3이닝

ERA, WHIP, 평균 이닝 계산은 아웃 카운트로 변환한 뒤 수행한다. 현재 변환 함수는 `features/starter_features.py`의 `ip_to_outs()`, `outs_to_decimal_ip()`, `normalize_innings_pitched()`를 사용한다.

## 실제 데이터 원천 후보

- 기존 저장소 내부 CSV: `kbo_analytics/data/official/pitcher_stats.csv`, `kbo_analytics/data/weekly/player_game_stats.csv`
- KBO 공식 기록 페이지: 팀별 투수 기록, 선수 ID 매핑
- KBO GameCenter 또는 경기별 박스스코어: 경기별 선발투수, 투수 등판 로그
- 수동 CSV 입력: 공식 파싱 실패 시 `starter_pitchers.csv`, `pitcher_game_logs.csv` 직접 적재

현재 저장소 내부 후보 중 일부는 투수 누적 기록 또는 mock/player 기록을 포함하지만, `starter_pitchers.csv`와 `pitcher_game_logs.csv`의 필수 스키마를 동시에 만족하는 실제 학습용 데이터는 아직 준비되지 않았다.

## 실행 흐름

1. 실제 선발투수 매핑을 `data/starter_pitchers.csv`에 적재한다.
2. 실제 투수 등판 로그를 `data/pitcher_game_logs.csv`에 적재한다.
3. `python run_pipeline.py`를 실행한다.
4. `results/summary.json`의 `pitcher_data_ready_to_train`을 확인한다.
5. `results/report.html` 하단의 “투수 데이터 준비 상태”를 확인한다.
6. 데이터가 학습 가능 상태일 때만 선발투수/불펜 피처 학습 실험을 진행한다.

## 데이터가 없을 때 원칙

- 모델 학습을 억지로 진행하지 않는다.
- sample/mock 데이터로 성능을 만들지 않는다.
- 기존 baseline 예측 결과를 바꾸지 않는다.
- 대시보드에는 “투수 데이터 미수집”과 blocker를 표시한다.
