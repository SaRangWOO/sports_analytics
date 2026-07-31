# KBO 운영 자동화

이 문서는 KBO 수집, predict-only 예측, 대시보드 게시, 사후 평가와 challenger 검증을 운영하는 자동화 코드의 계약을 설명한다. 자동화 코드는 기존 모델 학습 로직을 변경하지 않으며 승인된 production artifact만 운영 예측에 사용한다.

## 안전 기본값

- `KBO_AUTO_PROMOTE_ENABLED=false`
- `KBO_AUTO_ROLLBACK_ENABLED=true`
- 실제 runtime 파일은 `/home/wsr/1.project/runtime/kbo_automation` 아래에 둔다.
- 설정 파일에는 토큰, 비밀번호, webhook URL을 넣지 않는다.
- candidate는 Gate와 서로 다른 기준일의 shadow 검증 3회를 통과해도 자동 승격 환경 변수가 명시적으로 활성화되지 않으면 승격되지 않는다.
- morning과 pregame은 전체 모델을 재학습하지 않는다.

## 실행 흐름

```text
morning-update
  공식 데이터 수집
  -> canonical 투수 snapshot 및 품질 검사
  -> production artifact validate
  -> 격리 predict-only
  -> 출력 검증
  -> 원자적 publish

automation-dispatch
  오늘 공식 일정
  -> 경기 시작 120~15분 전 경기 선택
  -> official gameId별 pregame-update
  -> 입력 checksum이 같은 성공 실행은 skip

postgame-update
  완료 경기만 연결
  -> Accuracy/Brier/Log Loss/calibration 리포트

challenger-evaluate
  snapshot/표본 Gate
  -> candidate build/validate
  -> shadow predict-only
  -> promotion Gate
```

`predict_only_dashboard.py`는 데이터, 결과, run_model 결과, dashboard와 public 경로를 인자로 받을 수 있다. 자동화는 이 계약을 사용해 shadow 디렉터리에서 먼저 생성하고 검증된 파일만 백업 후 `os.replace`로 게시한다.

## CLI

```bash
cd /home/wsr/1.project/1.sports_analytics/kbo_analytics

.venv/bin/python scripts/kbo_automation.py automation-status --json
.venv/bin/python scripts/kbo_automation.py automation-dispatch --dry-run --json
.venv/bin/python scripts/kbo_automation.py morning-update --dry-run --json
.venv/bin/python scripts/kbo_automation.py pregame-update \
  --game-id 20260731KTLG0_LG --source-checksum example --dry-run --json
.venv/bin/python scripts/kbo_automation.py postgame-update --dry-run --json
.venv/bin/python scripts/kbo_automation.py challenger-evaluate --dry-run --json
.venv/bin/python scripts/kbo_automation.py cleanup-runtime --dry-run --json
.venv/bin/python scripts/kbo_automation.py automation-smoke --dry-run --json
```

공통 기준 시각은 `--reference-date`와 `--reference-datetime`으로 고정할 수 있다. 상태 키는 task, reference date, official gameId, update stage, input checksum으로 구성된다.

## Runtime 구조

```text
/home/wsr/1.project/runtime/kbo_automation/
  state/
    runs/
    automation_status.json
  locks/
  logs/
  backups/
  shadow/
  quarantine/
  reports/
```

상태 JSON과 게시 파일은 임시 파일 작성, 검증, checksum, 백업, `os.replace`, 사후 검증 순서로 처리한다. 잠금은 `fcntl.flock`을 사용한다. 동일한 성공 상태 키는 재실행하지 않고, 오래 남은 running 상태는 stale 실패로 전환한다.

## Artifact Gate와 rollback

승격 가능 판정에는 snapshot 품질, leakage audit, 최소 표본, artifact checksum/schema/runtime 호환성, shadow 3회, fit 호출 0건, Brier와 Log Loss 개선, Accuracy 허용 기준이 필요하다. 자동 승격은 기본적으로 비활성화되어 있다.

승격 직후 smoke가 실패하면 auto rollback 설정에 따라 previous artifact를 복구하고 다음 정보를 runtime report에 남긴다.

- 실패 artifact
- 복구 artifact
- 실패 단계와 원인
- rollback 시작/완료 시각
- rollback 상태

## Scheduler 템플릿

`deploy/systemd/`에는 dispatcher, morning, postgame, challenger, runtime cleanup service/timer 템플릿이 있다. 이 저장소 작업은 템플릿만 제공하며 설치, enable, start를 수행하지 않는다.

현재 서버 crontab은 morning과 pregame 작업을 이미 실행한다. systemd 전환 전에 기존 cron을 중지하지 않으면 중복 실행 구성이 된다. `automation-status`의 scheduler 검사에서 cron과 systemd가 같은 역할을 동시에 보유하면 conflict로 기록한다.

권장 전환 순서:

1. 격리 설정으로 smoke와 shadow 검증
2. systemd unit 내용과 절대 경로 검토
3. 기존 cron과 신규 timer의 역할별 중복 확인
4. 유지보수 시간에 한 scheduler만 선택
5. timer 설치 후 `systemctl list-timers`와 runtime 상태 확인

## 보존 정책

- 일반 로그 30일
- 실패 로그 90일
- CSV 백업 최근 30개
- candidate artifact 최근 10개
- previous artifact 최근 3개
- production/current 삭제 금지
- quarantine 자동 삭제 금지

`cleanup-runtime --dry-run`으로 삭제 후보를 먼저 확인한다.

## 검증

```bash
python -m py_compile automation/*.py scripts/kbo_automation.py
python -m unittest \
  modeling.test_model_artifacts \
  modeling.test_predict_only \
  modeling.test_pitching_snapshot_storage \
  automation.test_automation -v
python scripts/kbo_automation.py --help
python scripts/kbo_automation.py automation-smoke --dry-run --json
```

실제 배포 전에는 운영 artifact 존재, KBO 외부 수집 응답, PostgreSQL 연결, publish 권한을 별도로 확인한다. PostgreSQL 적재 실패가 HTML/CSV/JSON 생성과 분리되는 현재 정책은 유지하지만 상태 경고로 기록해야 한다.
