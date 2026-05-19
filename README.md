# Sports Analytics Portfolio

KBO 리그 데이터를 수집하고, 구단 리포트와 경기 승패 예측 대시보드를 자동 생성하는 스포츠 데이터 분석 포트폴리오입니다.

현재 저장소는 `kbo_analytics` 프로젝트에 집중합니다. 단순 순위표가 아니라 공식 기록을 기반으로 팀 스타일, 작전/상황 수행 지수, 핵심 선수 영향도, 승패 예측 모델 검증 결과를 HTML 대시보드로 생성합니다.

## 주요 결과

- KBO 리그 전체 대시보드: `kbo_analytics/dashboard/latest.html`
- 구단별 분석 페이지: `kbo_analytics/dashboard/kt.html`, `lg.html`, `samsung.html` 등
- GitHub 정적 배포 파일: `docs/index.html`, `docs/latest.html`
- 모델 결과: `kbo_analytics/modeling/results/win_predictor_model.json`

## 핵심 기능

1. KBO 공식 기록 수집
   - 팀 순위
   - 경기 일정/결과
   - 상대전적
   - 타자/투수 기록
   - 감독/코치/등록 선수 명단

2. 구단별 리포트
   - 한 줄 결론
   - 팀 스타일 태그
   - 강점/주의점
   - 작전/상황 수행 지수
   - 핵심 선수 영향도
   - 타선/마운드 의존도

3. 승패 예측 모델
   - 완료 경기 기반 학습
   - 팀 기준 모델과 경기 기준 모델 후보 비교
   - 정확도, Brier Score, Log Loss, Calibration Table 검증
   - 오늘 경기 예측 카드 생성

4. 자동 운영
   - 매일 오전 데이터 수집/모델링/대시보드 갱신
   - 경기 전 pregame 업데이트
   - 변경 결과 GitHub push

## 실행

```bash
cd /home/tera/1.project/1.sports_analytics/kbo_analytics
docker compose up -d
.venv/bin/python official_kbo_dashboard.py --training-start-year 2016
```

HTML 확인:

```bash
python -m http.server 8501 -d dashboard
```

브라우저:

```text
http://127.0.0.1:8501/latest.html
```

서버 IP, 계정, 비밀번호 등 운영 접속 정보는 Git에 기록하지 않습니다. 실행 환경과 `.env`에서만 관리합니다.

## 자세한 문서

프로젝트 구조, 데이터 흐름, 지표 산식, 모델링 방식, 자동화 스크립트 설명은 아래 문서를 참고하세요.

[kbo_analytics/README.md](kbo_analytics/README.md)
