from __future__ import annotations

from pathlib import Path

import pandas as pd


POLICIES = {"production_only", "hybrid_50_50"}


def _percent(value: str) -> float:
    return float(str(value).strip().removesuffix("%")) / 100


def apply_probability_policy(
    payload: dict,
    expected_runs_path: Path,
    policy: str,
) -> dict:
    if policy not in POLICIES:
        raise ValueError(f"unsupported KBO probability policy: {policy}")
    payload = dict(payload)
    predictions = [dict(row) for row in payload.get("today_predictions", [])]
    if policy == "production_only" or not predictions:
        payload["operational_probability_policy"] = "production_only"
        payload["operational_probability_policy_note"] = (
            "기존 승패 모델 확률을 그대로 사용합니다."
        )
        return payload

    expected = pd.read_csv(expected_runs_path, encoding="utf-8-sig")
    required = {
        "date",
        "home_team",
        "away_team",
        "home_expected_runs",
        "away_expected_runs",
        "home_win_probability",
    }
    missing = required - set(expected.columns)
    if missing:
        raise ValueError(f"expected-runs predictions missing columns: {sorted(missing)}")
    expected["date"] = expected["date"].astype(str)
    index: dict[tuple[str, frozenset[str]], dict] = {}
    for row in expected.to_dict(orient="records"):
        key = (row["date"], frozenset([row["home_team"], row["away_team"]]))
        if key in index:
            raise ValueError(f"duplicate expected-runs matchup: {key}")
        index[key] = row

    updated = []
    matched_games: set[tuple[str, frozenset[str]]] = set()
    for row in predictions:
        game_date = str(row["경기일"])
        team = str(row["기준팀"])
        opponent = str(row["상대팀"])
        key = (game_date, frozenset([team, opponent]))
        run_row = index.get(key)
        if run_row is None:
            raise ValueError(f"expected-runs prediction not found: {game_date} {team} vs {opponent}")
        production_probability = _percent(row["예측승률"])
        home_probability = float(run_row["home_win_probability"])
        if team == run_row["home_team"]:
            run_probability = home_probability
            team_runs = float(run_row["home_expected_runs"])
            opponent_runs = float(run_row["away_expected_runs"])
        elif team == run_row["away_team"]:
            run_probability = 1 - home_probability
            team_runs = float(run_row["away_expected_runs"])
            opponent_runs = float(run_row["home_expected_runs"])
        else:
            raise ValueError(f"team mapping failed: {game_date} {team} vs {opponent}")
        hybrid_probability = (production_probability + run_probability) / 2
        predicted_team = team if hybrid_probability >= 0.5 else opponent
        existing_reason = str(row.get("예측 근거", "")).strip()
        policy_reason = (
            f"승패 모델 {production_probability:.1%} · 득점 모델 {run_probability:.1%} · "
            f"예상 스코어 {team_runs:.1f}-{opponent_runs:.1f}"
        )
        row.update(
            {
                "예측 구단": predicted_team,
                "예측승률": f"{hybrid_probability:.1%}",
                "예측": "승리 예측" if hybrid_probability >= 0.5 else "패배 예측",
                "예측 근거": f"{policy_reason}; {existing_reason}" if existing_reason else policy_reason,
                "기존모델승률": f"{production_probability:.1%}",
                "득점모델승률": f"{run_probability:.1%}",
                "예상득점": round(team_runs, 1),
                "상대예상득점": round(opponent_runs, 1),
                "예상스코어": f"{team_runs:.1f} - {opponent_runs:.1f}",
                "확률정책": "승패·득점 모델 50:50 결합",
            }
        )
        updated.append(row)
        matched_games.add(key)

    expected_games = len(predictions) // 2
    if len(matched_games) != expected_games:
        raise ValueError(
            f"hybrid matchup count mismatch: matched={len(matched_games)} expected={expected_games}"
        )
    payload["today_predictions"] = updated
    payload["operational_probability_policy"] = policy
    payload["operational_probability_policy_note"] = (
        "2026 시간순 516경기 검증에서 기존 운영형 대비 정확도 +0.58%p, "
        "최근 100경기 +1.0%p를 보인 승패·득점 모델 50:50 결합 정책입니다. "
        "bootstrap 신뢰구간은 0을 포함하므로 롤백 가능한 canary로 운영합니다."
    )
    payload["operational_probability_policy_rollback"] = (
        "KBO_WIN_PROBABILITY_POLICY=production_only"
    )
    return payload
