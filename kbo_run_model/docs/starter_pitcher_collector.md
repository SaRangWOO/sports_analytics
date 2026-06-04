# KBO 선발투수 수집기

## 목적

`starter_pitchers.csv`를 실제 경기별 선발투수 매핑 데이터로 채우기 위한 수집/검증 구조입니다. 이번 단계는 선발투수 매핑만 다루며, 투수 경기 로그, 불펜, 타자 라인업, 모델 학습 적용은 범위에서 제외합니다.

## 수집 원천

현재 점검 대상 원천은 KBO 공식 일정/스코어보드 페이지입니다.

- `https://www.koreabaseball.com/Schedule/ScoreBoard.aspx`
- `https://www.koreabaseball.com/Schedule/Schedule.aspx`
- `https://eng.koreabaseball.com/Schedule/DailySchedule.aspx`

페이지 접근은 가능하지만 현재 확인 가능한 HTML에서는 경기별 선발투수 필드가 직접 노출되지 않습니다. 따라서 수집기는 기존 파일을 변경하지 않고 blocker를 기록합니다.

## 출력 스키마

`kbo_run_model/data/starter_pitchers.csv`는 기존 로더 호환성을 위해 `season`을 유지하고, 아래 컬럼을 포함해야 합니다.

- `season`
- `game_id`
- `date`
- `home_team`
- `away_team`
- `home_starter_id`
- `away_starter_id`
- `home_starter_name`
- `away_starter_name`
- `source`
- `collected_at`

## game_id 매칭 원칙

수집된 선발투수 데이터는 `prediction_games.csv`에서 만든 경기 단위 `game_id`와 매칭되어야 합니다. 팀/상대/home_away만으로 매칭하는 방식은 더블헤더나 같은 날짜 복수 경기에서 불안정하므로 최종 학습 가능 판정에는 `game_id` 매칭률을 사용합니다.

## pitcher_id가 없을 때

선발투수 ID를 확보할 수 없으면 ID 컬럼은 비워둘 수 있습니다. 다만 이 경우 `pitcher_id_missing_count`가 증가하고 학습 가능 상태로 보지 않습니다. 동명이인 방지를 위해 모델 피처 계산은 이름이 아니라 pitcher_id 기준으로 진행해야 합니다.

## 사용법

```powershell
cd kbo_run_model
python scripts/collect_starter_pitchers.py --check-only
python scripts/collect_starter_pitchers.py --date 2026-06-02 --check-only
python scripts/collect_starter_pitchers.py --date 2026-06-02 --apply
python scripts/collect_starter_pitchers.py --start-date 2026-06-01 --end-date 2026-06-30 --apply
```

`--check-only`는 외부 원천 접근 가능 여부와 수집 가능성만 확인하고 data 파일을 변경하지 않습니다. `--apply`는 수집 결과가 검증을 통과할 때만 기존 파일을 백업한 뒤 교체합니다.

## 검증 실패 시 보호 로직

아래 조건을 만족하지 못하면 `starter_pitchers.csv`는 유지됩니다.

- 필수 컬럼 존재
- date 파싱 가능
- `game_id`가 `prediction_games.csv`와 매칭 가능
- home/away 팀이 일정과 일치
- 같은 `game_id` 중복 없음
- 홈/원정 선발 이름 존재
- 홈/원정 선발 ID 존재
- source 존재
- collected_at 존재

## 아직 모델 학습에 적용하지 않는 이유

현재 공식 일정/스코어보드 페이지에서 선발투수 필드를 확인하지 못했기 때문에 실제 row를 수집하지 못했습니다. 빈 데이터나 mock 이름으로 학습하면 성능 지표가 왜곡될 수 있으므로, 이번 단계에서는 수집 상태와 blocker만 대시보드와 summary에 남기고 사용자용 예측에는 반영하지 않습니다.
