# 내부 투수 데이터 매핑 분석

## 목적

외부 크롤러를 만들기 전에 저장소 내부 CSV만으로 `starter_pitchers.csv`와 `pitcher_game_logs.csv`를 구성할 수 있는지 확인합니다. 변환이 검증되지 않으면 실제 학습에는 사용하지 않습니다.

## 주요 후보 파일

| 파일 | 판단 |
| --- | --- |
| `kbo_analytics/data/manual/confirmed_starters.csv` | 선발투수 매핑 후보지만 현재 실제 row가 없고 `game_id`, 홈/원정 경기 묶음, pitcher_id가 없습니다. |
| `kbo_analytics/data/weekly/player_game_stats.csv` | 경기별 투수 등판 row가 있고 `player_id`, `player_name`, `innings_pitched`, `earned_runs`, `pitches`가 있습니다. 다만 `home_runs_allowed`와 명시적 `is_starter`가 없습니다. |
| `kbo_analytics/data/official/pitcher_stats.csv` | 투수 시즌/기간 누적 기록으로 보이며 경기별 `game_id`, `date`가 없습니다. 경기별 로그로 직접 사용할 수 없습니다. |
| `kbo_analytics/modeling/results/game_level_player_features.csv` | 이미 집계된 경기 단위 피처입니다. 선발투수 이름/ID나 경기별 등판 원천 로그가 아닙니다. |
| `kbo_analytics/data/official/pitching_context.csv` | 일부 날짜의 선발 추정 정보와 ERA/WHIP가 있지만 `game_id`, pitcher_id, 경기별 등판 로그가 없습니다. |

## 변환 가능 여부

### starter_pitchers.csv

필요한 핵심 필드:
- `game_id`
- `date`
- `home_team`
- `away_team`
- `home_starter_id`
- `away_starter_id`
- `home_starter_name`
- `away_starter_name`

현재 `confirmed_starters.csv`는 `date`, `team`, `starter_name` 구조만 있고 row가 없습니다. 홈/원정 선발을 한 경기 단위로 묶을 수 없고 pitcher_id도 없어 변환을 적용하지 않습니다.

### pitcher_game_logs.csv

필요한 핵심 필드:
- `game_id`
- `date`
- `team`
- `opponent`
- `pitcher_id`
- `pitcher_name`
- `is_starter`
- `innings_pitched`
- `earned_runs`
- `hits_allowed`
- `walks`
- `strikeouts`
- `home_runs_allowed`
- `pitches`

`player_game_stats.csv`는 경기별 투수 row를 포함하지만 `home_runs_allowed`와 명시적 `is_starter`가 없습니다. 선발 여부를 이닝 순서로 추정할 수는 있지만, 추정값을 실제 학습용 데이터로 저장하지 않습니다.

## game_id 매칭 문제

`player_game_stats.csv`는 `game_id`가 있으므로 투수 로그 후보로는 강점이 있습니다. 반면 `confirmed_starters.csv`와 `pitching_context.csv`는 `game_id`가 없어 `date + team` 수준의 느슨한 매칭만 가능합니다. 더블헤더나 같은 날짜 복수 경기에서는 이 방식이 불안정합니다.

## pitcher_id/name 원칙

동명이인 방지를 위해 계산 키는 `pitcher_name`이 아니라 `pitcher_id`여야 합니다. `player_game_stats.csv`는 `player_id`가 있어 투수 ID 후보로 사용할 수 있지만, 선발 매핑 쪽에는 ID가 없어 양쪽을 안정적으로 연결할 수 없습니다.

## 경기별 로그와 누적 기록 차이

`pitcher_stats.csv`는 ERA, WHIP, 이닝 등 누적 기록입니다. 경기별 `game_id`와 등판 단위 기록이 없으므로 rolling 피처 생성에 직접 사용할 수 없습니다. 누적 기록을 현재 경기 피처에 넣으면 데이터 시점이 불명확해 누수 위험이 큽니다.

## 실제 학습으로 넘어가기 위한 조건

1. 선발투수 매핑에 `game_id`, 홈/원정 팀, 홈/원정 starter_id가 있어야 합니다.
2. 투수 로그에 `pitcher_id`, `game_id`, `is_starter`, `innings_pitched`, `earned_runs`, `hits_allowed`, `walks`, `strikeouts`, `home_runs_allowed`, `pitches`가 있어야 합니다.
3. `innings_pitched`는 야구식 5.1/5.2 표기를 아웃 카운트 기준으로 변환할 수 있어야 합니다.
4. 현재 경기 기록이 현재 경기 피처에 들어가지 않도록 날짜와 game_id 기준 정렬 후 과거 경기만 rolling 계산해야 합니다.

현재 결론은 내부 데이터만으로 완전한 선발투수/투수 로그 스키마를 만들기에는 부족하다는 것입니다. 다음 단계는 `home_runs_allowed`, 명시적 `is_starter`, pitcher_id 기반 선발 매핑이 포함된 외부 수집 또는 보강 CSV 확보입니다.
