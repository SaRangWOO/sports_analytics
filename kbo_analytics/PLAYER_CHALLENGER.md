# KBO 선수 능력치 Challenger

선수 challenger는 운영 승패 모델과 분리된 shadow 검증 경로다. 운영 확률, production artifact, 대시보드와 예약 작업을 변경하지 않는다.

## 시간 기준

- 선수 경기 기록은 경기 종료 후 데이터로 간주하고 `기록일 < 예측 기준일`인 행만 사용한다.
- 투수 스냅샷은 canonical 저장 계층에서 경기 시작 전 행만 허용하며, 예측 기준시각 이전 최신 행을 선택한다.
- 라인업은 예측 기준시각 이전 최신 행을 사용한다. 저장 데이터만으로 경기 시작시각을 검증할 수 없으면 leakage Gate를 차단한다.
- 최신 누적 타자·투수 CSV를 과거 경기에 일괄 결합하지 않는다.
- 선수명과 선수 ID는 매핑과 화면 설명에만 사용하며 모델 입력에는 넣지 않는다.

## 후보 피처

- baseline: 시즌 승률, 최근 5경기 승률, 최근 득실차, 휴식일 차이
- 선발: 스냅샷 ERA/WHIP, 정보 품질, 최근 3·5경기 ERA, K/BB, 휴식일
- 불펜: 1·3일 이닝과 투구 수, 연투 인원, 가용 투수 수, fatigue proxy
- 라인업: order-weighted WAR, OPS/OBP/SLG, 최근 7·14·30일 OPS, 상·중·하위 타순 강도

표본이 없거나 최근 기록이 없으면 중립 사전값과 availability 피처를 함께 사용한다. 실제 값이 없는 세부 지표는 만들지 않는다.

## 비교와 Gate

동일 완료 경기와 expanding time split으로 다음 세트를 비교한다.

1. `production_baseline_proxy`
2. `baseline_plus_pitching`
3. `baseline_plus_full_player`

Accuracy, Brier Score, Log Loss, calibration error, 55% 이상 적중률, 최근 30·60일 성능을 기록한다. 최소 150경기, coverage, leakage audit, 동일 경기 집합, 3회 shadow 검증과 확률 품질 개선을 모두 통과해야 후보 자격만 얻는다. 자동 승격은 항상 꺼져 있다.

현재 저장 데이터는 라인업 이력이 짧고 경기 시작시각 검증이 불가능해 Gate가 차단되는 것이 정상이다.

## CLI

```bash
python scripts/kbo_automation.py player-feature-build --dry-run --json
python scripts/kbo_automation.py player-feature-quality --dry-run --json
python scripts/kbo_automation.py player-challenger-evaluate --dry-run --json
python scripts/kbo_automation.py player-contribution-report --dry-run --json
```

비 dry-run 산출물은 기본적으로 `runtime/reports/player_challenger/`에 원자적으로 기록한다. Git 추적 결과 디렉터리에는 저장하지 않는다.

## 기여도 해석

기여도는 baseline에서 선발, 불펜, 라인업 피처 그룹을 순서대로 반영한 확률 차이다. 각 그룹 합은 최종 challenger 확률과 baseline 확률의 차이와 일치한다. 이는 모델 설명값이며 인과 효과가 아니다.
