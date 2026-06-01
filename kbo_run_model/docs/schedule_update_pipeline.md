# KBO 일정 갱신 파이프라인

## prediction_games.csv 역할

`prediction_games.csv`는 사용자용 예측 대시보드가 경기 대상 목록을 선택하는 일정 파일입니다. `run_pipeline.py`는 이 파일에서 KST 기준 오늘 경기, 가장 가까운 미래 경기, 사용자가 지정한 날짜 경기를 읽어 `match_predictions.csv`와 `report.html`을 생성합니다.

## 최신화가 필요한 이유

일정 파일이 최신 상태가 아니면 사용자용 예측표를 생성할 수 없습니다. 기본 실행은 과거 경기로 자동 fallback하지 않으므로, 오늘 이후 일정이 없으면 대시보드에는 일정 업데이트 필요 상태가 표시됩니다.

## 상태 점검

```powershell
cd kbo_run_model
python scripts/update_schedule.py --check-only
```

점검 결과는 `results/schedule_update_report.csv`에 저장됩니다.

주요 항목:
- `current_date_kst`
- `schedule_min_date`
- `schedule_max_date`
- `total_rows`
- `total_games`
- `future_games`
- `today_games`
- `schedule_is_stale`
- `stale_schedule_days`

## 갱신 실행

현재 저장소에는 공식 일정을 직접 가져오는 독립 수집기가 아직 없습니다. 실제 일정 CSV를 확보한 경우에만 아래처럼 source CSV를 지정해 갱신할 수 있습니다.

```powershell
cd kbo_run_model
python scripts/update_schedule.py --update --source-csv path\to\prediction_games_new.csv
```

`--source-csv`를 지정하지 않으면 기존 파일을 변경하지 않고 blocker를 기록합니다.

## 보호 로직

갱신은 다음 순서로만 진행됩니다.

1. 새 CSV 필수 컬럼 검증
2. 날짜 파싱 검증
3. 팀명과 상대팀 결측/공백 검증
4. `home_away` 값이 `A` 또는 `H`인지 검증
5. 같은 경기의 홈/원정 2행 pairing 검증
6. 동일 경기 중복 검증
7. 새 파일 최신 날짜가 기존보다 오래되지 않았는지 검증
8. 검증 통과 시 기존 파일 백업
9. 백업 후 새 파일로 교체

검증 실패 또는 수집 실패 시 기존 `prediction_games.csv`는 유지됩니다.

## stale schedule 처리 흐름

`run_pipeline.py`는 KST 오늘 날짜를 기준으로 일정을 선택합니다.

- 오늘 경기가 있으면 오늘 경기 표시
- 오늘 경기가 없고 미래 경기가 있으면 가장 가까운 미래 경기 표시
- 오늘 이후 일정이 없으면 사용자용 예측표 미생성
- `--target-date`를 명시하면 과거 날짜도 표시 가능
- `--allow-past-fallback`을 명시한 경우에만 최신 과거 경기 기준 리포트 허용

## 현재 한계

이 PR은 일정 상태 점검, 외부 CSV 적재, 검증, 백업 구조를 추가합니다. KBO 공식 사이트나 외부 API에서 실시간 일정을 직접 수집하는 크롤러는 아직 포함하지 않았습니다. 실제 수집 원천이 준비되기 전까지 `--update`는 source CSV 없이 기존 파일을 변경하지 않습니다.
