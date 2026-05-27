# V2 선발투수 데이터 스키마

현재 V1 모델은 팀 단위 득점 예측 baseline입니다. V2는 선발투수 데이터를 추가해 득점 예측 성능 개선 여부를 검증하기 위한 단계입니다.

## 경기별 선발투수 매핑 파일

파일: `data/starter_pitchers_sample.csv`

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

`pitcher_name`은 표시용입니다. 동명이인 방지를 위해 계산, rolling, join 기준은 반드시 `pitcher_id`를 사용합니다.

## 통합 투수 등판 로그 파일

파일: `data/pitcher_game_logs_sample.csv`

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

이 파일은 V2 선발투수 전용 테이블이 아니라 통합 투수 로그 테이블입니다. `is_starter` 컬럼을 통해 선발투수와 불펜투수를 구분합니다.

## 이닝 처리 규칙

야구 기록의 `innings_pitched` 값에서 `5.1`, `5.2`는 십진수가 아닙니다.

- `5.1`은 5와 1/3이닝입니다.
- `5.2`는 5와 2/3이닝입니다.
- ERA, WHIP, 평균 이닝 계산은 아웃 카운트 기반으로 변환한 뒤 수행해야 합니다.

구현 함수:

- `ip_to_outs()`
- `outs_to_decimal_ip()`
- `normalize_innings_pitched()`

## V2 생성 예정 피처

- `starter_era`
- `starter_whip`
- `starter_recent_3g_era`
- `starter_rest_days`
- `starter_avg_ip`

모든 rolling 계산은 현재 경기 이전 등판 기록만 사용합니다. 정렬 기준은 `season`, `pitcher_id`, `date`, `game_id`입니다. 같은 날짜의 미래 경기 결과가 현재 경기 피처에 섞이지 않도록 날짜 단위로 이력을 갱신합니다.

## V3 불펜 확장

`pitcher_game_logs_sample.csv`는 V3 불펜 피로도 피처에도 재사용할 수 있어야 합니다. V3에서는 `is_starter=False`인 등판 기록을 사용해 팀 단위 불펜 상태를 계산합니다.

V3 예시 피처:

- `bullpen_ip_last_1d`
- `bullpen_ip_last_3d`
- `bullpen_er_last_3d`
- `bullpen_pitch_count_last_3d`
- `bullpen_pitchers_used_last_3d`
- `closer_used_yesterday`
- `setup_man_used_yesterday`
