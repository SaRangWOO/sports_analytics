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

ERROR_GAME_COLUMNS = {
    "date": "날짜",
    "game_key": "경기 ID",
    "away_team": "원정팀",
    "home_team": "홈팀",
    "away_actual_runs": "원정 실제 득점",
    "away_expected_runs": "원정 예측 득점",
    "away_run_error": "원정 오차",
    "home_actual_runs": "홈 실제 득점",
    "home_expected_runs": "홈 예측 득점",
    "home_run_error": "홈 오차",
    "run_mae": "득점 MAE",
    "actual_total_runs": "실제 총득점",
    "expected_total_runs": "예상 총득점",
    "total_run_error": "총득점 오차",
    "expected_run_diff": "예상 득실차",
    "actual_run_diff": "실제 득실차",
    "win_correct": "승패 적중",
}

WIN_BUCKET_COLUMNS = {
    "bucket": "예측 승률 구간",
    "games": "경기 수",
    "accuracy": "승패 적중률",
    "avg_expected_run_diff_abs": "평균 예상 득실차 절대값",
    "avg_run_mae": "평균 득점 MAE",
}

TOTAL_RUNS_COLUMNS = {
    "category": "구분",
    "games": "경기 수",
    "actual_total_runs_avg": "평균 실제 총득점",
    "expected_total_runs_avg": "평균 예상 총득점",
    "total_runs_mae": "총득점 MAE",
    "over_under_accuracy_8_5": "오버/언더 8.5 적중률",
}

HANDICAP_COLUMNS = {
    "handicap_pick": "핸디캡 추천",
    "games": "경기 수",
    "recommended_games": "추천 경기 수",
    "accuracy_2_5": "핸디캡 2.5 적중률",
}

TEAM_ERROR_COLUMNS = {
    "team": "팀",
    "games": "경기 수",
    "actual_avg_runs": "평균 실제 득점",
    "predicted_avg_runs": "평균 예측 득점",
    "run_mae": "득점 MAE",
    "run_bias": "득점 예측 편향",
    "opponent_actual_avg_runs": "상대 실제 평균 득점",
    "opponent_predicted_avg_runs": "상대 예측 평균 득점",
    "opponent_run_mae": "상대 득점 MAE",
    "opponent_run_bias": "상대 득점 예측 편향",
    "bias_direction": "경향",
}

BALLPARK_COLUMNS = {
    "ballpark": "구장",
    "games": "경기 수",
    "actual_total_runs_avg": "평균 실제 총득점",
    "expected_total_runs_avg": "평균 예상 총득점",
    "total_runs_mae": "총득점 MAE",
    "total_runs_bias": "총득점 예측 편향",
    "bias_direction": "경향",
}

MONTHLY_COLUMNS = {
    "month": "월/구간",
    "games": "경기 수",
    "run_mae": "득점 MAE",
    "total_runs_mae": "총득점 MAE",
    "win_accuracy": "승패 정확도",
}

IMPROVEMENT_COLUMNS = {
    "metric": "지표",
    "baseline": "Baseline",
    "improved": "Improved",
    "delta": "변화량",
    "direction": "판정",
}

PARK_FACTOR_COLUMNS = {
    "ballpark": "구장",
    "games": "경기 수",
    "avg_total_runs": "평균 총득점",
    "avg_home_runs": "평균 홈 득점",
    "run_factor": "득점 팩터",
    "sample_note": "표본 상태",
}

TEAM_BIAS_FEATURE_COLUMNS = {
    "team": "팀",
    "games": "경기 수",
    "avg_prediction_error": "평균 예측 오차",
    "avg_abs_prediction_error": "평균 절대 예측 오차",
    "avg_allowed_prediction_error": "평균 허용득점 예측 오차",
    "avg_abs_allowed_prediction_error": "평균 절대 허용득점 오차",
    "bias_direction": "편향 보정 후보",
}

IMPROVEMENT_MODEL_COLUMNS = {
    "model_version": "실험 구분",
    "model": "모델",
    "run_mae": "득점 MAE",
    "run_rmse": "득점 RMSE",
    "home_win_accuracy": "홈팀 승패 정확도",
    "brier_score": "브라이어 점수",
}

PITCHER_VALIDATION_COLUMNS = {
    "dataset": "데이터셋",
    "file_exists": "파일 존재",
    "schema_valid": "스키마 유효",
    "row_count": "행 수",
    "game_match_rate": "경기 매칭률",
    "duplicate_rows": "중복 행",
    "valid": "학습 가능",
    "message": "상태",
}

INTERNAL_PITCHER_INVENTORY_COLUMNS = {
    "path": "파일 경로",
    "rows": "행 수",
    "candidate_type": "후보 유형",
    "mapping_potential": "매핑 가능성",
    "has_game_id": "game_id",
    "has_date": "date",
    "has_team": "team",
    "has_player_id": "player_id",
    "has_pitcher_name": "투수명",
    "has_innings_pitched": "이닝",
    "has_is_starter": "선발 여부",
}

INTERNAL_PITCHER_MAPPING_COLUMNS = {
    "source_file": "원천 파일",
    "target_schema": "목표 스키마",
    "required_columns_found": "확인 컬럼",
    "required_columns_missing": "부족 컬럼",
    "game_id_match_possible": "game_id 매칭",
    "date_team_match_possible": "날짜/팀 매칭",
    "conversion_possible": "변환 가능",
    "blocker": "차단 사유",
}

INTERNAL_PITCHER_CONVERSION_COLUMNS = {
    "target_file": "대상 파일",
    "conversion_attempted": "변환 시도",
    "conversion_applied": "변환 적용",
    "output_rows": "출력 행 수",
    "validation_passed": "검증 통과",
    "blocker": "차단 사유",
}


def _format_generated_at(value: str) -> str:
    return value.replace("T", " ")[:16]


def _badge(text: str, css_class: str) -> str:
    return f'<span class="badge {css_class}">{html.escape(text)}</span>'


def _pick_badge(text: str) -> str:
    if text == "관망":
        return _badge(text, "watch")
    if text == "오버":
        return _badge(text, "over")
    if text == "언더":
        return _badge(text, "under")
    return _badge(text, "pick")


def _confidence_badge(text: str) -> str:
    return _badge(f"신뢰도 {text}", {"낮음": "low", "보통": "mid", "높음": "high"}.get(text, "low"))


def _score_for_card(predicted_score: str) -> str:
    return html.escape(predicted_score.replace(" - ", " : "))


def _win_probability_text(row) -> str:
    away_pct = f"{row.away_win_probability * 100:.1f}%"
    home_pct = f"{row.home_win_probability * 100:.1f}%"
    return f"{html.escape(str(row.away_team))} {away_pct} · {html.escape(str(row.home_team))} {home_pct}"


def _explanation(row) -> str:
    abs_diff = abs(float(row.expected_run_diff))
    total_gap = float(row.total_expected_runs) - float(row.over_under_line)
    if abs_diff < 0.5:
        return "예상 득실차가 작아 승패와 핸디캡은 관망이 적절합니다."
    if row.over_under_pick == "오버":
        return "예상 총득점이 기준점보다 높아 오버 쪽으로 기울어집니다."
    if row.over_under_pick == "언더":
        return "예상 총득점이 기준점보다 낮아 언더 쪽으로 기울어집니다."
    if abs(total_gap) < 0.4:
        return "예상 총득점이 기준점과 가까워 오버/언더는 관망이 적절합니다."
    return f"{html.escape(str(row.predicted_winner))} 쪽 우세가 예상되지만 추천 강도는 신뢰도와 함께 확인해야 합니다."


def _limited_match_list(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "없음"
    names = (frame["away_team"] + " vs " + frame["home_team"]).head(3).tolist()
    remaining = len(frame) - len(names)
    suffix = f" 외 {remaining}경기" if remaining > 0 else ""
    return ", ".join(names) + suffix


def _render_game_cards(match_predictions: pd.DataFrame) -> str:
    if match_predictions.empty:
        return '<p class="empty">예측 가능한 최신 일정이 없어 사용자용 경기 예측표를 생성하지 않았습니다.</p>'

    cards = []
    for row in match_predictions.itertuples(index=False):
        watch_badges = []
        if row.moneyline_pick == "관망":
            watch_badges.append("승패 관망")
        if row.handicap_pick == "관망":
            watch_badges.append("핸디캡 관망")
        if row.over_under_pick == "관망":
            watch_badges.append("오버/언더 관망")
        badges = "".join(_badge(value, "watch") for value in watch_badges)
        cards.append(
            f"""
      <article class="game-card">
        <div class="game-head">
          <div>
            <div class="game-date">{html.escape(str(row.date))} · KBO</div>
            <h3><span class="team-name">{html.escape(str(row.away_team))}</span> <span class="muted">원정</span> vs <span class="team-name">{html.escape(str(row.home_team))}</span> <span class="muted">홈</span></h3>
          </div>
          {_confidence_badge(str(row.confidence_level))}
        </div>
        <div class="score">{_score_for_card(str(row.predicted_score))}</div>
        <div class="pick-grid">
          <div><div class="label">승패</div><div class="value">{_pick_badge(str(row.moneyline_pick))}</div></div>
          <div><div class="label">승률</div><div class="value small">{_win_probability_text(row)}</div></div>
          <div><div class="label">핸디캡 {row.handicap_line}</div><div class="value">{_pick_badge(str(row.handicap_pick))}</div></div>
          <div><div class="label">오버/언더 {row.over_under_line}</div><div class="value">{_pick_badge(str(row.over_under_pick))}</div></div>
          <div><div class="label">예상 승리팀</div><div class="value">{html.escape(str(row.predicted_winner))}</div></div>
          <div><div class="label">예상 총득점</div><div class="value">{row.total_expected_runs:.1f}점</div></div>
        </div>
        <p class="interpretation">{html.escape(_explanation(row))}</p>
        <div class="badges">{badges}</div>
      </article>
"""
        )
    return "\n".join(cards)


def _render_compact_match_table(match_predictions: pd.DataFrame) -> str:
    if match_predictions.empty:
        return '<p class="empty">예측 가능한 최신 일정이 없어 사용자용 상세 예측표를 생성하지 않았습니다.</p>'
    compact = pd.DataFrame(
        {
            "경기일시": match_predictions["date"],
            "매치업": match_predictions["away_team"] + " vs " + match_predictions["home_team"],
            "예상 스코어": match_predictions["predicted_score"],
            "승패 추천": match_predictions["moneyline_pick"],
            "승률": [
                f"{row.away_team} {row.away_win_probability * 100:.1f}% / {row.home_team} {row.home_win_probability * 100:.1f}%"
                for row in match_predictions.itertuples(index=False)
            ],
            "핸디캡 추천": match_predictions["handicap_pick"],
            "오버/언더 추천": match_predictions["over_under_pick"],
            "신뢰도": match_predictions["confidence_level"],
        }
    )
    return compact.to_html(index=False, classes="table compact-table", border=0)


def _table(frame: pd.DataFrame, columns: dict[str, str], limit: int | None = None) -> str:
    data = frame.head(limit) if limit else frame
    return data.rename(columns=columns).to_html(index=False, classes="table", border=0)


def _schedule_notice(target_context: dict) -> str:
    current_date = html.escape(str(target_context.get("current_date_kst", "")))
    latest_date = html.escape(str(target_context.get("schedule_latest_date", "")))
    selected_date = html.escape(str(target_context.get("selected_target_date", "")))
    if target_context.get("schedule_is_stale") and not target_context.get("user_prediction_available"):
        return f"""
  <section class="notice warning">
    <h2>예측 가능한 최신 KBO 일정이 없습니다</h2>
    <p class="note">현재 일정 데이터의 최신 경기일은 {latest_date}입니다.<br>
    현재 기준일은 {current_date}입니다.<br>
    최신 KBO 일정 데이터가 업데이트되지 않아 사용자용 예측표를 생성하지 않았습니다.<br>
    prediction_games.csv를 최신 일정으로 갱신한 뒤 다시 실행하세요.</p>
  </section>
"""
    if target_context.get("report_mode") == "과거 경기 기준 리포트":
        return """
  <section class="notice warning">
    <h2>과거 경기 기준 리포트</h2>
    <p class="note">주의: 최신 예정 일정이 없어 과거 경기 기준으로 표시 중입니다.</p>
  </section>
"""
    if target_context.get("schedule_selection_reason") == "사용자가 지정한 과거 경기 기준":
        return f"""
  <section class="notice info">
    <h2>사용자가 지정한 과거 경기 기준</h2>
    <p class="note">지정한 예측 기준일 {selected_date}은 현재 기준일 {current_date}보다 과거입니다.</p>
  </section>
"""
    return ""


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
    error_analysis: dict,
    improvement_experiment: dict,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_table = pd.DataFrame(candidate_scores).rename(columns=SCORE_COLUMNS).to_html(index=False, classes="table", border=0)
    bucket_table = pd.DataFrame(bucket_scores).rename(columns=BUCKET_COLUMNS).to_html(index=False, classes="table", border=0)
    abs_bucket_table = pd.DataFrame(abs_bucket_scores).rename(columns=BUCKET_COLUMNS).to_html(index=False, classes="table", border=0)
    season_table = pd.DataFrame(season_scores).rename(columns=SEASON_COLUMNS).to_html(index=False, classes="table", border=0)
    team_table = pd.DataFrame(team_scores).rename(columns=TEAM_COLUMNS).to_html(index=False, classes="table", border=0)
    prediction_table = predictions.head(40).rename(columns=PREDICTION_COLUMNS).to_html(index=False, classes="table", border=0)
    error_games = error_analysis["game_errors"]
    top_error_table = _table(error_analysis["top_errors"][list(ERROR_GAME_COLUMNS)], ERROR_GAME_COLUMNS)
    win_bucket_table = _table(error_analysis["win_probability_buckets"], WIN_BUCKET_COLUMNS)
    total_runs_table = _table(error_analysis["total_runs"], TOTAL_RUNS_COLUMNS)
    handicap_table = _table(error_analysis["handicap"], HANDICAP_COLUMNS)
    team_error_table = _table(error_analysis["team"], TEAM_ERROR_COLUMNS)
    ballpark_error_table = _table(error_analysis["ballpark"], BALLPARK_COLUMNS)
    monthly_error_table = _table(error_analysis["monthly"], MONTHLY_COLUMNS)
    improvement_table = _table(improvement_experiment["comparison"], IMPROVEMENT_COLUMNS)
    park_factor_table = _table(improvement_experiment["park_metrics"], PARK_FACTOR_COLUMNS)
    team_bias_feature_table = _table(improvement_experiment["bias_metrics"], TEAM_BIAS_FEATURE_COLUMNS)
    improvement_model_table = _table(improvement_experiment["model_scores"], IMPROVEMENT_MODEL_COLUMNS)
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
        match_table = match_display.rename(columns=MATCH_COLUMNS).to_html(index=False, classes="table full-table", border=0)
        compact_match_table = _render_compact_match_table(match_predictions)
        match_cards = _render_game_cards(match_predictions)
    else:
        match_cards = _render_game_cards(match_predictions)
        compact_match_table = _render_compact_match_table(match_predictions)
        match_table = '<p class="empty">표시할 사용자용 예측 상세 수치가 없습니다.</p>'

    if match_predictions.empty:
        high_confidence = "신뢰도 높은 경기 없음"
        watch_games = "현재 기준으로는 강한 추천 경기 없음"
        high_total = "없음"
        close_games = "없음"
    else:
        sorted_confidence = match_predictions.assign(abs_diff=match_predictions["expected_run_diff"].abs()).sort_values("abs_diff", ascending=False)
        high_candidates = sorted_confidence[sorted_confidence["confidence_level"].ne("낮음")]
        if high_candidates.empty:
            high_confidence = "신뢰도 높은 경기 없음"
        else:
            high_row = high_candidates.iloc[0]
            high_confidence = f"{high_row['away_team']} vs {high_row['home_team']} ({high_row['confidence_level']}, {high_row['predicted_winner']} 우세)"
        watch = match_predictions[
            match_predictions["moneyline_pick"].eq("관망")
            | match_predictions["handicap_pick"].eq("관망")
            | match_predictions["over_under_pick"].eq("관망")
        ]
        all_watch = len(watch) == len(match_predictions)
        watch_games = "현재 기준으로는 강한 추천 경기 없음" if all_watch else _limited_match_list(watch)
        total_row = match_predictions.sort_values("total_expected_runs", ascending=False).iloc[0]
        high_total = f"{total_row['away_team']} vs {total_row['home_team']} ({total_row['total_expected_runs']:.1f}점)"
        close = match_predictions.assign(abs_diff=match_predictions["expected_run_diff"].abs()).sort_values("abs_diff").head(3)
        close_games = ", ".join(
            f"{row.away_team} vs {row.home_team} (득실차 {abs(row.expected_run_diff):.2f})"
            for row in close.itertuples(index=False)
        ) or "없음"
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
    generated_at = html.escape(_format_generated_at(str(summary["generated_at"])))
    starter_schema = summary["starter_schema_inspection"]
    starter_status = html.escape(starter_schema["status"])
    available_columns = ", ".join(starter_schema["available_columns"])
    candidate_columns = ", ".join(starter_schema["candidate_columns"])
    game_columns = ", ".join(starter_schema["required_game_starter_columns"])
    pitcher_columns = ", ".join(starter_schema["required_pitcher_log_columns"])
    starter_features = ", ".join(starter_schema["future_starter_features"])
    starter_data_status = summary["starter_data_status"]
    pitcher_validation = summary["pitcher_data_validation"]
    pitcher_validation_rows = []
    for dataset, row in pitcher_validation.items():
        pitcher_validation_rows.append(
            {
                "dataset": dataset,
                "file_exists": "예" if row["file_exists"] else "아니오",
                "schema_valid": "예" if row["schema_valid"] else "아니오",
                "row_count": row["row_count"],
                "game_match_rate": row["game_match_rate"],
                "duplicate_rows": row["duplicate_rows"],
                "valid": "예" if row["valid"] else "아니오",
                "message": row["message"],
            }
        )
    pitcher_validation_table = _table(pd.DataFrame(pitcher_validation_rows), PITCHER_VALIDATION_COLUMNS)
    results_dir = output_path.parent
    internal_inventory = pd.read_csv(results_dir / "internal_pitcher_data_inventory.csv")
    internal_mapping = pd.read_csv(results_dir / "internal_pitcher_mapping_report.csv")
    internal_conversion = pd.read_csv(results_dir / "internal_pitcher_conversion_check.csv")
    internal_inventory_table = _table(internal_inventory[list(INTERNAL_PITCHER_INVENTORY_COLUMNS)], INTERNAL_PITCHER_INVENTORY_COLUMNS, limit=20)
    internal_mapping_table = _table(internal_mapping, INTERNAL_PITCHER_MAPPING_COLUMNS)
    internal_conversion_table = _table(internal_conversion, INTERNAL_PITCHER_CONVERSION_COLUMNS)
    pitcher_ready = "학습 가능" if summary["pitcher_data_ready_to_train"] else "학습 불가"
    pitcher_collection_status = "투수 데이터 준비 완료" if summary["pitcher_data_ready_to_train"] else "투수 데이터 미수집"
    mapping_train_ready = "학습 가능" if summary["pitcher_data_ready_to_train_after_mapping"] else "학습 불가"
    mapping_applied = "적용" if summary["internal_pitcher_conversion_applied"] else "미적용"
    starter_collection_ready = "학습 가능" if summary["starter_pitcher_data_ready_to_train"] else "학습 불가"
    starter_source_available = "예" if summary["starter_pitcher_source_available"] else "아니오"
    over_predicted = ", ".join(f"{row['team']}({row['bias']})" for row in summary["team_bias_summary"]["over_predicted_teams"]) or "없음"
    under_predicted = ", ".join(f"{row['team']}({row['bias']})" for row in summary["team_bias_summary"]["under_predicted_teams"]) or "없음"
    target_context = summary["target_context"]
    schedule_check = summary["schedule_selection_check"]
    schedule_update = summary.get("schedule_update_status", {})
    schedule_notice = _schedule_notice(target_context)
    target_date_display = html.escape(str(target_context["target_date"] or "없음"))
    schedule_update_needed = "필요" if schedule_update.get("schedule_update_needed") else "불필요"
    schedule_update_blocker = html.escape(str(schedule_update.get("schedule_update_blocker") or "없음"))
    win_accuracy = error_games["win_correct"].mean()
    close_games = error_games[error_games["expected_run_diff"].abs().lt(0.5)]
    strong_games = error_games[error_games["expected_run_diff"].abs().ge(1.0)]
    close_accuracy = close_games["win_correct"].mean() if not close_games.empty else 0
    strong_accuracy = strong_games["win_correct"].mean() if not strong_games.empty else 0
    weakness_lines = [
        f"가장 큰 오차 범주는 {summary['biggest_error_category']}입니다.",
        f"팀 단위 최약 예측 대상은 {summary['weakest_team_prediction'] or '확인 불가'}입니다.",
        f"총득점 MAE는 {summary['total_runs_mae']}이고 오버/언더 8.5 적중률은 {summary['over_under_accuracy_8_5']}입니다.",
        f"접전 예측 경기 승패 적중률은 {close_accuracy:.4f}, 강한 우세 예측 경기 적중률은 {strong_accuracy:.4f}입니다.",
    ]
    weakness_summary = "<br>".join(html.escape(line) for line in weakness_lines)
    improved_metrics = ", ".join(summary["improved_metrics"]) or "없음"
    worsened_metrics = ", ".join(summary["worsened_metrics"]) or "없음"
    final_applied = "적용" if summary["final_model_applied"] else "미적용"

    document = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>KBO 승부 예측 대시보드</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ font-family: Arial, sans-serif; margin: 28px; color: #202124; background: #f6f7f9; word-break: keep-all; }}
    h1, h2 {{ margin: 0 0 14px; }}
    section {{ margin-top: 24px; background: #fff; border: 1px solid #dadce0; padding: 18px; }}
    .subtitle {{ color: #5f6368; margin: 0 0 18px; line-height: 1.6; }}
    .table-wrap {{ overflow-x: auto; }}
    .table {{ border-collapse: collapse; width: 100%; min-width: 760px; font-size: 14px; }}
    .table th, .table td {{ border-bottom: 1px solid #dadce0; padding: 9px; text-align: right; white-space: nowrap; word-break: keep-all; }}
    .table th:first-child, .table td:first-child {{ text-align: left; }}
    .note {{ line-height: 1.6; }}
    .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }}
    .summary-item {{ border: 1px solid #e5e7eb; padding: 12px; }}
    .label {{ color: #5f6368; font-size: 13px; margin-bottom: 6px; }}
    .value {{ font-size: 16px; font-weight: 700; }}
    .value.small {{ font-size: 14px; line-height: 1.5; }}
    .empty {{ color: #5f6368; }}
    .game-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 14px; }}
    .game-card {{ border: 1px solid #d8dde6; padding: 16px; background: #fff; }}
    .game-head {{ display: flex; justify-content: space-between; gap: 12px; align-items: flex-start; }}
    .game-date {{ color: #5f6368; font-size: 13px; margin-bottom: 4px; }}
    .game-card h3 {{ margin: 0; font-size: 18px; white-space: nowrap; }}
    .team-name {{ white-space: nowrap; }}
    .muted {{ color: #6b7280; font-size: 12px; font-weight: 400; }}
    .score {{ font-size: 30px; font-weight: 800; margin: 16px 0; white-space: nowrap; }}
    .pick-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }}
    .badge {{ display: inline-block; padding: 4px 8px; border: 1px solid #cbd5e1; font-size: 12px; font-weight: 700; white-space: nowrap; }}
    .badge.pick {{ background: #eff6ff; color: #1d4ed8; border-color: #bfdbfe; }}
    .badge.over {{ background: #fef2f2; color: #b91c1c; border-color: #fecaca; }}
    .badge.under {{ background: #eef2ff; color: #3730a3; border-color: #c7d2fe; }}
    .badge.low {{ background: #f8fafc; color: #475569; }}
    .badge.mid {{ background: #fff7ed; color: #9a3412; }}
    .badge.high {{ background: #ecfdf5; color: #047857; }}
    .badge.watch {{ margin-right: 6px; margin-top: 10px; background: #f1f5f9; color: #334155; }}
    .interpretation {{ margin: 14px 0 0; color: #374151; line-height: 1.5; }}
    .detail-table {{ margin-top: 18px; overflow-x: auto; }}
    details {{ margin-top: 14px; }}
    summary {{ cursor: pointer; font-weight: 700; }}
    .notice.warning {{ border-color: #f59e0b; background: #fffbeb; }}
    .notice.info {{ border-color: #93c5fd; background: #eff6ff; }}
  </style>
</head>
<body>
  <h1>KBO 승부 예측 대시보드</h1>
  <p class="subtitle">KBO 경기 일정에 맞춰 예상 스코어, 승패 확률, 핸디캡, 오버/언더를 자동 계산합니다.</p>
  <section>
    <h2>기준 정보</h2>
    <div class="summary-grid">
      <div class="summary-item"><div class="label">생성 시각</div><div class="value">{generated_at}</div></div>
      <div class="summary-item"><div class="label">예측 기준일</div><div class="value">{target_date_display}</div></div>
      <div class="summary-item"><div class="label">경기 수</div><div class="value">{target_context["game_count"]}경기</div></div>
      <div class="summary-item"><div class="label">데이터 기준</div><div class="value">{html.escape(target_context["report_mode"])}</div></div>
    </div>
  </section>
  {schedule_notice}
  <section>
    <h2>일정 선택 상태</h2>
    <div class="summary-grid">
      <div class="summary-item"><div class="label">선택 모드</div><div class="value">{html.escape(target_context["report_mode"])}</div></div>
      <div class="summary-item"><div class="label">일정 원본 행 수</div><div class="value">{schedule_check["schedule_rows"]}</div></div>
      <div class="summary-item"><div class="label">예상 경기 수 / 생성 경기 수</div><div class="value">{schedule_check["expected_games"]} / {schedule_check["predicted_games"]}</div></div>
      <div class="summary-item"><div class="label">홈/원정 매칭</div><div class="value">{"정상" if schedule_check["home_away_pairing_ok"] else "확인 필요"}</div></div>
      <div class="summary-item"><div class="label">중복 경기</div><div class="value">{schedule_check["duplicate_games"]}건</div></div>
      <div class="summary-item"><div class="label">검증 상태</div><div class="value">{html.escape(schedule_check["status"])}</div></div>
      <div class="summary-item"><div class="label">일정 파일 최신 날짜</div><div class="value">{html.escape(str(schedule_update.get("schedule_max_date", "확인 불가") or "확인 불가"))}</div></div>
      <div class="summary-item"><div class="label">현재 KST 날짜</div><div class="value">{html.escape(str(schedule_update.get("current_date_kst", "확인 불가") or "확인 불가"))}</div></div>
      <div class="summary-item"><div class="label">오늘 경기 수</div><div class="value">{schedule_update.get("today_games", 0)}경기</div></div>
      <div class="summary-item"><div class="label">미래 경기 수</div><div class="value">{schedule_update.get("future_games", 0)}경기</div></div>
      <div class="summary-item"><div class="label">일정 stale 여부</div><div class="value">{"예" if schedule_update.get("schedule_is_stale") else "아니오"}</div></div>
      <div class="summary-item"><div class="label">stale 경과일</div><div class="value">{schedule_update.get("stale_schedule_days", 0)}일</div></div>
      <div class="summary-item"><div class="label">일정 갱신 필요</div><div class="value">{schedule_update_needed}</div></div>
      <div class="summary-item"><div class="label">일정 갱신 blocker</div><div class="value">{schedule_update_blocker}</div></div>
    </div>
  </section>
  <section>
    <h2>KBO 경기 예측표</h2>
    <div class="game-grid">
      {match_cards}
    </div>
    <div class="detail-table">
      <h3>상세 예측표</h3>
      <div class="table-wrap">{compact_match_table}</div>
      <details>
        <summary>전체 상세 수치 보기</summary>
        <div class="table-wrap">{match_table}</div>
      </details>
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
    <h2>성능 개선 진단</h2>
    <p class="note">
      {weakness_summary}<br>
      다음 개선 추천 순서: {html.escape(str(summary["recommended_next_improvement"]))}
    </p>
    <div class="summary-grid">
      <div class="summary-item"><div class="label">검증 경기 승패 정확도</div><div class="value">{win_accuracy:.4f}</div></div>
      <div class="summary-item"><div class="label">총득점 MAE</div><div class="value">{summary["total_runs_mae"]}</div></div>
      <div class="summary-item"><div class="label">오버/언더 8.5 적중률</div><div class="value">{summary["over_under_accuracy_8_5"]}</div></div>
      <div class="summary-item"><div class="label">핸디캡 2.5 적중률</div><div class="value">{summary["handicap_accuracy_2_5"]}</div></div>
    </div>
    <h3>오차가 큰 경기 TOP 10</h3>
    <div class="table-wrap">{top_error_table}</div>
    <h3>승률 구간별 적중률</h3>
    <div class="table-wrap">{win_bucket_table}</div>
    <h3>총득점 예측 오차</h3>
    <div class="table-wrap">{total_runs_table}</div>
    <h3>핸디캡/오버언더 적중률</h3>
    <div class="table-wrap">{handicap_table}</div>
    <h3>팀별 과대/과소 예측 경향</h3>
    <div class="table-wrap">{team_error_table}</div>
    <h3>구장별 오차</h3>
    <div class="table-wrap">{ballpark_error_table}</div>
    <h3>월별/시즌 구간별 오차</h3>
    <div class="table-wrap">{monthly_error_table}</div>
  </section>
  <section>
    <h2>성능 개선 실험 결과</h2>
    <p class="note">
      실험 내용: 구장 득점 팩터와 팀별 공격/실점 예측 편향 보정 피처를 추가해 baseline과 같은 검증 구간에서 비교했습니다.<br>
      최종 적용 여부: {final_applied}<br>
      적용 판단 사유: {html.escape(str(summary["final_model_reason"]))}<br>
      개선된 지표: {html.escape(improved_metrics)}<br>
      악화된 지표: {html.escape(worsened_metrics)}<br>
      다음 개선 추천: {html.escape(str(summary["next_recommended_improvement"]))}
    </p>
    <div class="summary-grid">
      <div class="summary-item"><div class="label">Baseline 총득점 MAE</div><div class="value">{summary["baseline_total_runs_mae"]}</div></div>
      <div class="summary-item"><div class="label">Improved 총득점 MAE</div><div class="value">{summary["improved_total_runs_mae"]}</div></div>
      <div class="summary-item"><div class="label">총득점 MAE 변화량</div><div class="value">{summary["total_runs_mae_delta"]}</div></div>
      <div class="summary-item"><div class="label">Brier 변화</div><div class="value">{summary["baseline_brier_score"]} → {summary["improved_brier_score"]}</div></div>
    </div>
    <h3>Baseline vs Improved 성능 비교</h3>
    <div class="table-wrap">{improvement_table}</div>
    <h3>개선 실험 모델별 성능</h3>
    <div class="table-wrap">{improvement_model_table}</div>
    <h3>구장 팩터 효과</h3>
    <div class="table-wrap">{park_factor_table}</div>
    <h3>팀 편향 보정 효과</h3>
    <div class="table-wrap">{team_bias_feature_table}</div>
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
    <h3>투수 데이터 준비 상태</h3>
    <p class="note">
      현재 상태: {html.escape(pitcher_collection_status)}<br>
      선발투수 매핑 데이터 존재 여부: {"예" if summary["starter_pitchers_file_exists"] else "아니오"}<br>
      투수 등판 로그 존재 여부: {"예" if summary["pitcher_game_logs_file_exists"] else "아니오"}<br>
      필수 컬럼 충족 여부: 선발 매핑 {"예" if summary["starter_pitchers_schema_valid"] else "아니오"}, 투수 로그 {"예" if summary["pitcher_game_logs_schema_valid"] else "아니오"}<br>
      경기 일정 매칭률: 선발 매핑 {summary["starter_schedule_match_rate"]}, 투수 로그 {summary["pitcher_logs_game_match_rate"]}<br>
      학습 가능/불가능: {html.escape(pitcher_ready)}<br>
      차단 사유: {html.escape(str(summary["pitcher_data_blocker"]))}<br>
      다음 필요한 작업: 실제 경기별 선발투수 매핑과 투수 등판 로그를 pitcher_id 기준으로 적재한 뒤 재검증합니다.
    </p>
    <div class="table-wrap">{pitcher_validation_table}</div>
    <h3>선발투수 수집 상태</h3>
    <p class="note">
      수집 원천 사용 가능 여부: {starter_source_available}<br>
      수집 row 수: {summary["starter_pitcher_rows_collected"]}<br>
      일정 매칭률: {summary["starter_pitcher_schedule_match_rate"]}<br>
      양쪽 선발 확인 경기 비율: {summary["starter_pitcher_full_match_rate"]}<br>
      ID 누락 수: {summary["starter_pitcher_id_missing_count"]}<br>
      학습 가능 여부: {starter_collection_ready}<br>
      blocker: {html.escape(str(summary["starter_pitcher_collection_blocker"]))}<br>
      다음 필요 작업: 선발투수 이름과 가능한 pitcher_id를 제공하는 공식 또는 신뢰 가능한 원천을 확보한 뒤 수집 검증을 재실행합니다.
    </p>
    <h3>내부 투수 데이터 매핑 분석</h3>
    <p class="note">
      발견된 투수/선수 관련 파일 수: {summary["internal_pitcher_candidate_files"]}<br>
      가장 유망한 선발투수 데이터 파일: {html.escape(str(summary["best_starter_source_file"]))}<br>
      가장 유망한 투수 로그 데이터 파일: {html.escape(str(summary["best_pitcher_log_source_file"]))}<br>
      starter_pitchers.csv 변환 가능 여부: {"예" if summary["starter_conversion_possible"] else "아니오"}<br>
      pitcher_game_logs.csv 변환 가능 여부: {"예" if summary["pitcher_log_conversion_possible"] else "아니오"}<br>
      변환 적용 여부: {mapping_applied}<br>
      학습 가능 여부: {mapping_train_ready}<br>
      차단 사유: {html.escape(str(summary["internal_pitcher_mapping_blocker"]))}<br>
      다음 권장 작업: {html.escape(str(summary["next_recommended_pitcher_data_step"]))}
    </p>
    <h4>내부 후보 파일</h4>
    <div class="table-wrap">{internal_inventory_table}</div>
    <h4>목표 스키마 매핑 판단</h4>
    <div class="table-wrap">{internal_mapping_table}</div>
    <h4>변환 적용 점검</h4>
    <div class="table-wrap">{internal_conversion_table}</div>
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
