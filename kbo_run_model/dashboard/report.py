from __future__ import annotations

import html
from pathlib import Path

import pandas as pd


SCORE_COLUMNS = {
    "model": "모델",
    "run_mae": "득점 MAE",
    "run_rmse": "득점 RMSE",
    "home_win_accuracy": "홈팀 승패 정확도",
    "brier_score": "브라이어 점수",
}

BUCKET_COLUMNS = {
    "bucket": "예상 득실차 구간",
    "games": "경기 수",
    "accuracy": "적중률",
    "avg_brier_score": "평균 브라이어 점수",
}

SEASON_COLUMNS = {
    "season": "시즌",
    "games": "경기 수",
    "run_mae": "득점 MAE",
    "run_rmse": "득점 RMSE",
    "home_win_accuracy": "홈팀 승패 정확도",
    "brier_score": "브라이어 점수",
}

TEAM_COLUMNS = {
    "team": "팀",
    "games": "경기 수",
    "actual_avg_runs": "실제 평균 득점",
    "predicted_avg_runs": "예측 평균 득점",
    "mae": "MAE",
    "bias": "예측 편향",
}

PREDICTION_COLUMNS = {
    "date": "날짜",
    "game_key": "경기 ID",
    "home_team": "홈팀",
    "away_team": "원정팀",
    "expected_score": "예상 스코어",
    "home_expected_runs": "홈 예상 득점",
    "away_expected_runs": "원정 예상 득점",
    "expected_run_diff": "예상 득실차",
    "expected_total_runs": "예상 총득점",
    "home_win_probability": "홈팀 승률",
    "confidence": "신뢰도",
    "predicted_winner": "예상 승리팀",
    "actual_winner": "실제 승리팀",
    "home_actual_runs": "홈 실제 득점",
    "away_actual_runs": "원정 실제 득점",
}

SEARCH_COLUMNS = {
    "file": "파일",
    "row_count": "행 수",
    "mapped_v2_columns": "매핑 가능 컬럼",
    "missing_starter_mapping_columns": "부족한 선발 매핑 컬럼",
    "missing_pitcher_log_columns": "부족한 투수 로그 컬럼",
    "ready_to_use": "바로 사용 가능",
}

MATCH_COLUMNS = {
    "date": "경기일시",
    "league": "리그",
    "away_team": "원정팀",
    "home_team": "홈팀",
    "predicted_score": "예상 스코어",
    "predicted_winner": "예상 승리팀",
    "home_win_probability_display": "홈 승률",
    "away_win_probability_display": "원정 승률",
    "expected_run_diff": "예상 득실차",
    "moneyline_pick": "승/패 추천",
    "handicap_line": "핸디캡 기준",
    "handicap_pick": "핸디캡 추천",
    "total_expected_runs": "예상 총득점",
    "over_under_line": "오버/언더 기준",
    "over_under_pick": "오버/언더 추천",
    "confidence_level": "신뢰도",
}


def write_html_report(
    output_path: Path,
    summary: dict,
    candidate_scores: list[dict],
    bucket_scores: list[dict],
    abs_bucket_scores: list[dict],
    season_scores: list[dict],
    team_scores: list[dict],
    predictions: pd.DataFrame,
    match_predictions: pd.DataFrame,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_table = pd.DataFrame(candidate_scores).rename(columns=SCORE_COLUMNS).to_html(index=False, classes="table", border=0)
    bucket_table = pd.DataFrame(bucket_scores).rename(columns=BUCKET_COLUMNS).to_html(index=False, classes="table", border=0)
    abs_bucket_table = pd.DataFrame(abs_bucket_scores).rename(columns=BUCKET_COLUMNS).to_html(index=False, classes="table", border=0)
    season_table = pd.DataFrame(season_scores).rename(columns=SEASON_COLUMNS).to_html(index=False, classes="table", border=0)
    team_table = pd.DataFrame(team_scores).rename(columns=TEAM_COLUMNS).to_html(index=False, classes="table", border=0)
    prediction_table = predictions.head(40).rename(columns=PREDICTION_COLUMNS).to_html(index=False, classes="table", border=0)
    match_display = match_predictions.copy()
    if not match_display.empty:
        match_display["league"] = "KBO"
        match_display["home_win_probability_display"] = (match_display["home_win_probability"] * 100).round(1).astype(str) + "%"
        match_display["away_win_probability_display"] = (match_display["away_win_probability"] * 100).round(1).astype(str) + "%"
        match_display["expected_run_diff"] = match_display["expected_run_diff"].round(2)
        match_display["total_expected_runs"] = match_display["total_expected_runs"].round(2)
        match_display = match_display[
            [
                "date",
                "league",
                "away_team",
                "home_team",
                "predicted_score",
                "predicted_winner",
                "home_win_probability_display",
                "away_win_probability_display",
                "expected_run_diff",
                "moneyline_pick",
                "handicap_line",
                "handicap_pick",
                "total_expected_runs",
                "over_under_line",
                "over_under_pick",
                "confidence_level",
            ]
        ]
        match_table = match_display.rename(columns=MATCH_COLUMNS).to_html(index=False, classes="table", border=0)
        cards = []
        for row in match_predictions.itertuples(index=False):
            home_pct = f"{row.home_win_probability * 100:.1f}%"
            away_pct = f"{row.away_win_probability * 100:.1f}%"
            watch_badges = []
            if row.moneyline_pick == "관망":
                watch_badges.append("승패 관망")
            if row.handicap_pick == "관망":
                watch_badges.append("핸디캡 관망")
            if row.over_under_pick == "관망":
                watch_badges.append("오버/언더 관망")
            badges = "".join(f'<span class="badge watch">{html.escape(value)}</span>' for value in watch_badges)
            confidence_class = {"낮음": "low", "보통": "mid", "높음": "high"}.get(row.confidence_level, "low")
            cards.append(
                f"""
      <article class="game-card">
        <div class="game-head">
          <div>
            <div class="game-date">{html.escape(str(row.date))} · KBO</div>
            <h3>{html.escape(str(row.away_team))} <span>원정</span> vs {html.escape(str(row.home_team))} <span>홈</span></h3>
          </div>
          <span class="badge {confidence_class}">신뢰도 {html.escape(str(row.confidence_level))}</span>
        </div>
        <div class="score">{html.escape(str(row.predicted_score))}</div>
        <div class="pick-grid">
          <div><div class="label">예상 승리팀</div><div class="value">{html.escape(str(row.predicted_winner))}</div></div>
          <div><div class="label">승률</div><div class="value">홈 {home_pct} · 원정 {away_pct}</div></div>
          <div><div class="label">승/패 추천</div><div class="value">{html.escape(str(row.moneyline_pick))}</div></div>
          <div><div class="label">핸디캡 {row.handicap_line}</div><div class="value">{html.escape(str(row.handicap_pick))}</div></div>
          <div><div class="label">오버/언더 {row.over_under_line}</div><div class="value">{html.escape(str(row.over_under_pick))}</div></div>
          <div><div class="label">예상 총득점</div><div class="value">{row.total_expected_runs:.1f}점</div></div>
        </div>
        <div class="badges">{badges}</div>
      </article>
"""
            )
        match_cards = "\n".join(cards)
    else:
        match_table = '<p class="empty">해당 날짜에 예정된 KBO 경기가 없습니다.</p>'
        match_cards = match_table

    if match_predictions.empty:
        high_confidence = "없음"
        watch_games = "없음"
        high_total = "없음"
        close_games = "없음"
    else:
        sorted_confidence = match_predictions.assign(abs_diff=match_predictions["expected_run_diff"].abs()).sort_values("abs_diff", ascending=False)
        high_row = sorted_confidence.iloc[0]
        high_confidence = f"{high_row['away_team']} vs {high_row['home_team']} ({high_row['confidence_level']}, {high_row['predicted_winner']} 우세)"
        watch = match_predictions[
            match_predictions["moneyline_pick"].eq("관망")
            | match_predictions["handicap_pick"].eq("관망")
            | match_predictions["over_under_pick"].eq("관망")
        ]
        watch_games = ", ".join((watch["away_team"] + " vs " + watch["home_team"]).head(5).tolist()) or "없음"
        total_row = match_predictions.sort_values("total_expected_runs", ascending=False).iloc[0]
        high_total = f"{total_row['away_team']} vs {total_row['home_team']} ({total_row['total_expected_runs']:.1f}점)"
        close = match_predictions.assign(abs_diff=match_predictions["expected_run_diff"].abs()).sort_values("abs_diff").head(3)
        close_games = ", ".join((close["away_team"] + " vs " + close["home_team"]).tolist()) or "없음"
    search_rows = []
    for row in summary["internal_pitcher_data_search"]["files"]:
        search_rows.append(
            {
                "file": row["file"],
                "row_count": row["row_count"],
                "mapped_v2_columns": ", ".join(row["mapped_v2_columns"]),
                "missing_starter_mapping_columns": ", ".join(row["missing_starter_mapping_columns"]),
                "missing_pitcher_log_columns": ", ".join(row["missing_pitcher_log_columns"]),
                "ready_to_use": "예" if row["ready_to_use"] else "아니오",
            }
        )
    search_table = pd.DataFrame(search_rows).rename(columns=SEARCH_COLUMNS).to_html(index=False, classes="table", border=0)
    selected = html.escape(str(summary["selected_model"]["model"]))
    generated_at = html.escape(str(summary["generated_at"]))
    starter_schema = summary["starter_schema_inspection"]
    starter_status = html.escape(starter_schema["status"])
    available_columns = ", ".join(starter_schema["available_columns"])
    candidate_columns = ", ".join(starter_schema["candidate_columns"])
    game_columns = ", ".join(starter_schema["required_game_starter_columns"])
    pitcher_columns = ", ".join(starter_schema["required_pitcher_log_columns"])
    starter_features = ", ".join(starter_schema["future_starter_features"])
    starter_data_status = summary["starter_data_status"]
    over_predicted = ", ".join(f"{row['team']}({row['bias']})" for row in summary["team_bias_summary"]["over_predicted_teams"]) or "없음"
    under_predicted = ", ".join(f"{row['team']}({row['bias']})" for row in summary["team_bias_summary"]["under_predicted_teams"]) or "없음"
    target_context = summary["target_context"]
    schedule_check = summary["schedule_selection_check"]

    document = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>KBO 승패·핸디캡·오버/언더 예측 대시보드</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 28px; color: #202124; background: #f6f7f9; }}
    h1, h2 {{ margin: 0 0 14px; }}
    section {{ margin-top: 24px; background: #fff; border: 1px solid #dadce0; padding: 18px; }}
    .meta {{ color: #5f6368; margin-bottom: 16px; line-height: 1.6; }}
    .table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
    .table th, .table td {{ border-bottom: 1px solid #dadce0; padding: 8px; text-align: right; }}
    .table th:first-child, .table td:first-child {{ text-align: left; }}
    .note {{ line-height: 1.6; }}
    .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }}
    .summary-item {{ border: 1px solid #e5e7eb; padding: 12px; }}
    .label {{ color: #5f6368; font-size: 13px; margin-bottom: 6px; }}
    .value {{ font-size: 16px; font-weight: 700; }}
    .empty {{ color: #5f6368; }}
    .game-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 14px; }}
    .game-card {{ border: 1px solid #d8dde6; padding: 16px; background: #fff; }}
    .game-head {{ display: flex; justify-content: space-between; gap: 12px; align-items: flex-start; }}
    .game-date {{ color: #5f6368; font-size: 13px; margin-bottom: 4px; }}
    .game-card h3 {{ margin: 0; font-size: 18px; }}
    .game-card h3 span {{ color: #6b7280; font-size: 12px; font-weight: 400; }}
    .score {{ font-size: 26px; font-weight: 800; margin: 16px 0; }}
    .pick-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }}
    .badge {{ display: inline-block; padding: 4px 8px; border: 1px solid #cbd5e1; font-size: 12px; font-weight: 700; white-space: nowrap; }}
    .badge.low {{ background: #f8fafc; color: #475569; }}
    .badge.mid {{ background: #fff7ed; color: #9a3412; }}
    .badge.high {{ background: #ecfdf5; color: #047857; }}
    .badge.watch {{ margin-right: 6px; margin-top: 10px; background: #f1f5f9; color: #334155; }}
    .detail-table {{ margin-top: 18px; overflow-x: auto; }}
  </style>
</head>
<body>
  <h1>KBO 승패·핸디캡·오버/언더 예측 대시보드</h1>
    <div class="meta">
      예상 득점, 예상 득실차, 승률, 핸디캡, 오버/언더를 한눈에 확인하는 화면<br>
      생성 시각={generated_at} | 예측 기준 날짜={target_context["target_date"]} | 해당 날짜 경기 수={target_context["game_count"]} | 데이터 기준={html.escape(target_context["report_mode"])}
    </div>
  <section>
    <h2>일정 선택 상태</h2>
    <div class="summary-grid">
      <div class="summary-item"><div class="label">선택 모드</div><div class="value">{html.escape(target_context["report_mode"])}</div></div>
      <div class="summary-item"><div class="label">일정 원본 행 수</div><div class="value">{schedule_check["schedule_rows"]}</div></div>
      <div class="summary-item"><div class="label">예상 경기 수 / 생성 경기 수</div><div class="value">{schedule_check["expected_games"]} / {schedule_check["predicted_games"]}</div></div>
      <div class="summary-item"><div class="label">홈/원정 매칭</div><div class="value">{"정상" if schedule_check["home_away_pairing_ok"] else "확인 필요"}</div></div>
      <div class="summary-item"><div class="label">중복 경기</div><div class="value">{schedule_check["duplicate_games"]}건</div></div>
      <div class="summary-item"><div class="label">검증 상태</div><div class="value">{html.escape(schedule_check["status"])}</div></div>
    </div>
  </section>
  <section>
    <h2>KBO 경기 예측표</h2>
    <div class="game-grid">
      {match_cards}
    </div>
    <div class="detail-table">
      {match_table}
    </div>
  </section>
  <section>
    <h2>예측 결과 요약</h2>
    <div class="summary-grid">
      <div class="summary-item"><div class="label">가장 신뢰도 높은 경기</div><div class="value">{html.escape(high_confidence)}</div></div>
      <div class="summary-item"><div class="label">관망 경기 목록</div><div class="value">{html.escape(watch_games)}</div></div>
      <div class="summary-item"><div class="label">예상 총득점 높은 경기</div><div class="value">{html.escape(high_total)}</div></div>
      <div class="summary-item"><div class="label">예상 접전 경기</div><div class="value">{html.escape(close_games)}</div></div>
    </div>
  </section>
  <section>
    <h2>모델 검증 정보</h2>
    <p class="note">선택 모델 {selected} 기준 득점 MAE {summary["selected_model"]["run_mae"]}, 득점 RMSE {summary["selected_model"]["run_rmse"]}, 홈팀 승패 정확도 {summary["selected_model"]["home_win_accuracy"]}, 브라이어 점수 {summary["selected_model"]["brier_score"]}입니다.</p>
    <h3>모델별 성능 비교</h3>
    {candidate_table}
    <h3>시즌별 성능</h3>
    {season_table}
    <h3>팀별 예측 편향</h3>
    <p class="note">과대예측 팀: {html.escape(over_predicted)}<br>과소예측 팀: {html.escape(under_predicted)}</p>
    {team_table}
    <h3>예상 득실차 구간별 적중률</h3>
    {abs_bucket_table}
    <h3>기존 득실차 방향 구간별 적중률</h3>
    {bucket_table}
  </section>
  <section>
    <h2>데이터 상태</h2>
    <p class="note">
      선발투수 데이터 상태: {starter_status}<br>
      현재 CSV에서 확인된 컬럼: {html.escape(available_columns)}<br>
      검사한 후보 컬럼: {html.escape(candidate_columns)}<br>
      필요한 경기 단위 선발투수 컬럼: {html.escape(game_columns)}<br>
      필요한 투수 등판 기록 컬럼: {html.escape(pitcher_columns)}<br>
      향후 추가 가능한 선발투수 피처: {html.escape(starter_features)}<br>
      선발투수 피처 사용 여부: 사용 안 함
    </p>
    <p class="note">
      선수 데이터 확장 상태: 데이터 수집 스키마 준비, 모델 미학습<br>
      사유: 선발투수/투수 등판 기록 데이터 미수집<br>
      starter_pitchers row_count={starter_data_status["starter_pitchers"]["row_count"]}, pitcher_game_logs row_count={starter_data_status["pitcher_game_logs"]["row_count"]}
    </p>
    <p class="note">
      내부 투수 데이터 검색 완료: 예<br>
      내부 투수 관련 데이터 발견: {"예" if summary["internal_pitcher_data_found"] else "아니오"}<br>
      실제 학습 가능 여부: {"가능" if summary["v2_ready_to_train"] else "불가"}<br>
      차단 사유: {html.escape(summary["v2_blocker"])}<br>
      현재 sample CSV는 스키마와 적재 예시용 mock 데이터이며 실제 학습에는 사용하지 않습니다.
    </p>
    <h3>저장소 내부 투수 데이터 검색 결과</h3>
    {search_table}
    <h3>실제 데이터 필요 여부</h3>
    <p class="note">현재 내부 후보 파일만으로는 pitcher_id 기반 경기별 선발투수 매핑과 필수 투수 등판 로그가 완성되지 않아 외부 수집 또는 별도 적재가 필요합니다.</p>
    <h3>필요한 최소 컬럼</h3>
    <p class="note">
      선발 매핑: season, date, game_id, home_team, away_team, home_starter_name, away_starter_name, home_starter_id, away_starter_id<br>
      투수 로그: season, date, game_id, pitcher_id, pitcher_name, team, opponent, is_starter, innings_pitched, earned_runs, hits_allowed, walks, strikeouts, home_runs_allowed, pitches
    </p>
    <h3>불펜 확장 안내</h3>
    <p class="note">
      동일한 pitcher_game_logs 스키마를 사용해 불펜 피로도 계산이 가능합니다.<br>
      향후 is_starter=False인 등판 기록을 활용할 예정입니다.
    </p>
  </section>
  <section>
    <h2>기술 검증용 예측 결과</h2>
    {prediction_table}
    <p class="note">대시보드 경로: kbo_run_model/results/report.html</p>
  </section>
</body>
</html>
"""
    output_path.write_text(document, encoding="utf-8")
