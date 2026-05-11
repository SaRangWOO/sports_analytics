# Sports Analytics Portfolio

이 저장소는 KBO 분석 포트폴리오에 집중합니다.

## 목표

1. KBO 팀 성과 대시보드 생성
2. 경기 승패 예측 모델 생성

## 구조

```text
sports_analytics/
└── kbo_analytics/
    ├── collector.py
    ├── weekly_update.py
    ├── docker-compose.yaml
    ├── mock_api/
    ├── data/weekly/
    ├── dashboard/
    ├── modeling/
    ├── pg_data/
    └── sql/
```

## 서버 실행

```bash
cd /home/tera/1.project/1.sports_analytics/kbo_analytics
docker compose up -d
API_BASE_URL=http://localhost:8000 DB_URL=postgresql://user:password@localhost:5432/baseball .venv/bin/python collector.py
DB_URL=postgresql://user:password@localhost:5432/baseball .venv/bin/python weekly_update.py
```

## 결과 확인

- 대시보드: `http://<server-ip>:8501/latest.html`
- Metabase: `http://<server-ip>:3000`
- 모델 결과: `kbo_analytics/modeling/results/win_predictor_model.json`

서버 IP, 계정, 비밀번호는 README와 Git에 기록하지 않고 실행 환경에서만 사용합니다.
