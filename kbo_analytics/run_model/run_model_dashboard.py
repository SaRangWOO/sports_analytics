from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import pandas as pd


RUN_MODEL_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS = RUN_MODEL_DIR / "results"
DEFAULT_OUTPUT = RUN_MODEL_DIR / "dashboard.html"


def fmt_float(value, digits=3):
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def fmt_pct(value, digits=1):
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "-"


def table_html(rows: list[dict], columns: list[str], limit: int | None = None):
    rows = rows[:limit] if limit else rows
    if not rows:
        return '<p class="empty">표시할 데이터가 없습니다.</p>'
    header = "".join(f"<th>{html.escape(col)}</th>" for col in columns)
    body = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(str(row.get(col, '')))}</td>" for col in columns)
        body.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def metric_card(label: str, value: str, note: str = ""):
    note_html = f"<span>{html.escape(note)}</span>" if note else ""
    return f'<div class="metric"><p>{html.escape(label)}</p><strong>{html.escape(value)}</strong>{note_html}</div>'


def model_score_rows(payload: dict):
    by_model: dict[str, dict] = {}
    for row in payload.get("run_regression_scores", []):
        by_model.setdefault(row["model"], {})["모델"] = row["model"]
        by_model[row["model"]]["MAE"] = fmt_float(row.get("mae"), 4)
        by_model[row["model"]]["RMSE"] = fmt_float(row.get("rmse"), 4)
    for row in payload.get("win_conversion_scores", []):
        by_model.setdefault(row["model"], {})["모델"] = row["model"]
        by_model[row["model"]]["승패 정확도"] = fmt_pct(row.get("accuracy"))
        by_model[row["model"]]["Brier"] = fmt_float(row.get("brier_score"), 4)
        by_model[row["model"]]["LogLoss"] = fmt_float(row.get("log_loss"), 4)
        by_model[row["model"]]["득실차 방향"] = fmt_pct(row.get("run_diff_direction_accuracy"))
    return list(by_model.values())


def prediction_summary(predictions: pd.DataFrame):
    if predictions.empty:
        return {}
    df = predictions.copy()
    df["abs_run_diff"] = df["expected_run_diff"].abs()
    df["correct_flag"] = df["prediction_result"].eq("correct")
    return {
        "games": len(df),
        "avg_home_runs": df["home_expected_runs"].mean(),
        "avg_away_runs": df["away_expected_runs"].mean(),
        "avg_abs_run_diff": df["abs_run_diff"].mean(),
        "accuracy": df["correct_flag"].mean(),
        "big_edge_games": int((df["abs_run_diff"] >= 1.0).sum()),
        "big_edge_accuracy": df.loc[df["abs_run_diff"] >= 1.0, "correct_flag"].mean() if (df["abs_run_diff"] >= 1.0).any() else None,
    }


def confidence_rows(predictions: pd.DataFrame):
    if predictions.empty:
        return []
    df = predictions.copy()
    df["abs_run_diff"] = df["expected_run_diff"].abs()
    df["correct_flag"] = df["prediction_result"].eq("correct")
    bins = [
        ("0.0~0.5점", df["abs_run_diff"] < 0.5),
        ("0.5~1.0점", (df["abs_run_diff"] >= 0.5) & (df["abs_run_diff"] < 1.0)),
        ("1.0~1.5점", (df["abs_run_diff"] >= 1.0) & (df["abs_run_diff"] < 1.5)),
        ("1.5점 이상", df["abs_run_diff"] >= 1.5),
    ]
    rows = []
    for label, mask in bins:
        subset = df[mask]
        rows.append(
            {
                "예상 득실차 구간": label,
                "경기 수": len(subset),
                "적중률": fmt_pct(subset["correct_flag"].mean()) if len(subset) else "-",
                "평균 홈승률": fmt_pct(subset["home_win_probability"].mean()) if len(subset) else "-",
            }
        )
    return rows


def error_bucket_rows(payload: dict):
    rows = []
    for row in payload.get("error_analysis_summary", {}).get("score_bucket_error", []):
        rows.append(
            {
                "득점 구간": row.get("bucket", "-"),
                "경기 수": row.get("games", "-"),
                "홈 MAE": fmt_float(row.get("home_mae"), 3),
                "원정 MAE": fmt_float(row.get("away_mae"), 3),
                "합계 MAE": fmt_float(row.get("total_mae"), 3),
                "득실차 MAE": fmt_float(row.get("run_diff_mae"), 3),
                "승패 적중률": fmt_pct(row.get("accuracy")),
            }
        )
    return rows


def error_tag_rows(payload: dict):
    rows = []
    for row in payload.get("error_analysis_summary", {}).get("error_tag_counts", []):
        rows.append({"오차 태그": row.get("tag", "-"), "경기 수": row.get("games", "-")})
    return rows


def tag_summary_rows(payload: dict):
    rows = []
    for row in payload.get("error_tag_summary", []):
        rows.append(
            {
                "유형": row.get("tag", "-"),
                "경기 수": row.get("games", "-"),
                "평균 MAE": fmt_float(row.get("mean_mae"), 3),
                "RMSE": fmt_float(row.get("rmse"), 3),
                "평균 실제 득점": fmt_float(row.get("mean_actual_runs"), 2),
                "평균 예측 득점": fmt_float(row.get("mean_expected_runs"), 2),
                "평균 오차": fmt_float(row.get("mean_error"), 2),
                "득실 방향 적중률": fmt_pct(row.get("direction_accuracy")),
            }
        )
    return rows


def score_summary_card(title: str, summary: dict):
    if not summary:
        return '<p class="empty">요약 데이터가 없습니다.</p>'
    feature_means = summary.get("feature_means", {})
    feature_lines = "".join(
        f"<li>{html.escape(key)}: {fmt_float(value, 2)}</li>"
        for key, value in list(feature_means.items())[:6]
    )
    return f"""
      <div class="diagnostic-card">
        <h3>{html.escape(title)}</h3>
        <p>{html.escape(summary.get("interpretation", ""))}</p>
        <div class="mini-grid">
          {metric_card("경기 수", str(summary.get("games", "-")))}
          {metric_card("평균 실제 총득점", fmt_float(summary.get("avg_actual_total_runs"), 2))}
          {metric_card("평균 예측 총득점", fmt_float(summary.get("avg_expected_total_runs"), 2))}
          {metric_card("평균 총득점 오차", fmt_float(summary.get("avg_total_error"), 2))}
        </div>
        <ul class="features">{feature_lines}</ul>
      </div>
    """


def importance_rows(payload: dict, limit: int = 20):
    rows = []
    for row in payload.get("feature_importance_top20", [])[:limit]:
        rows.append(
            {
                "피처": row.get("feature", "-"),
                "중요도 평균": fmt_float(row.get("importance_mean"), 6),
                "표준편차": fmt_float(row.get("importance_std"), 6),
            }
        )
    return rows


def load_run_model_results(results_dir: Path):
    model_path = results_dir / "expected_runs_model.json"
    prediction_path = results_dir / "expected_runs_predictions.csv"
    if not model_path.exists() or not prediction_path.exists():
        raise FileNotFoundError("Run model results are missing. Run run_prediction_model.py first.")
    payload = json.loads(model_path.read_text(encoding="utf-8"))
    predictions = pd.read_csv(prediction_path)
    return payload, predictions, model_path, prediction_path


def prediction_board_rows(predictions: pd.DataFrame):
    if predictions.empty:
        return []
    latest_date = predictions["date"].max()
    df = predictions[predictions["date"].eq(latest_date)].copy()
    df["abs_run_diff"] = df["expected_run_diff"].abs()
    df = df.sort_values("abs_run_diff", ascending=False)
    rows = []
    for _, row in df.iterrows():
        home_prob = float(row["home_win_probability"])
        away_prob = 1 - home_prob
        home_pick = row["predicted_winner"] == row["home_team"]
        pick_prob = home_prob if home_pick else away_prob
        expected_total = float(row["home_expected_runs"]) + float(row["away_expected_runs"])
        run_diff = float(row["expected_run_diff"])
        if pick_prob >= 0.60:
            confidence = "높음"
            decision = "추천"
        elif pick_prob >= 0.55:
            confidence = "보통"
            decision = "관망"
        else:
            confidence = "낮음"
            decision = "정보 부족"

        if abs(run_diff) < 0.7:
            handicap_pick = "관망"
        elif abs(run_diff) >= 2.5:
            handicap_pick = f'{row["predicted_winner"]} -2.5'
        else:
            underdog = row["away_team"] if home_pick else row["home_team"]
            handicap_pick = f"{underdog} +2.5"

        if expected_total >= 8.8:
            total_pick = "오버"
        elif expected_total <= 8.2:
            total_pick = "언더"
        else:
            total_pick = "관망"

        rows.append(
            {
                "경기일": row["date"],
                "원정팀 vs 홈팀": f'{row["away_team"]} vs {row["home_team"]}',
                "예상 스코어": f'{row["away_team"]} {fmt_float(row["away_expected_runs"], 1)} - {row["home_team"]} {fmt_float(row["home_expected_runs"], 1)}',
                "승패 확률": f'{row["away_team"]} {fmt_pct(away_prob)} / {row["home_team"]} {fmt_pct(home_prob)}',
                "신뢰도": confidence,
                "판단": decision,
                "핸디캡": handicap_pick,
                "오버/언더": total_pick,
                "예측 구단": row["predicted_winner"],
                "예측 승률": fmt_pct(pick_prob),
                "예상 총득점": fmt_float(expected_total, 1),
            }
        )
    return rows


def board_status(payload: dict, predictions: pd.DataFrame, model_path: Path, prediction_path: Path):
    latest_date = predictions["date"].max() if not predictions.empty else "-"
    latest_rows = predictions[predictions["date"].eq(latest_date)] if latest_date != "-" else predictions
    duplicate_games = int(latest_rows["game_key"].duplicated().sum()) if "game_key" in latest_rows else 0
    return {
        "generated_at": payload.get("generated_at", "-"),
        "prediction_date": latest_date,
        "game_count": len(latest_rows),
        "total_rows": len(predictions),
        "pair_rows": len(latest_rows) * 2,
        "duplicate_games": duplicate_games,
        "home_away_match": "완료" if len(latest_rows) else "확인 필요",
        "validation_status": "완료 경기 검증 데이터",
        "data_source": f"{model_path.name}, {prediction_path.name}",
    }


def match_card_html(row: dict):
    tone = "pick" if row["판단"] == "추천" else "watch" if row["판단"] == "관망" else "risk"
    return f"""
      <article class="match-card {tone}">
        <div class="match-card-top">
          <span>{html.escape(row["경기일"])}</span>
          <span class="match-badge {tone}">{html.escape(row["판단"])}</span>
        </div>
        <h3>{html.escape(row["원정팀 vs 홈팀"])}</h3>
        <div class="score-line">{html.escape(row["예상 스코어"])}</div>
        <div class="match-meta">
          <div><span>예측 구단</span><strong>{html.escape(row["예측 구단"])}</strong></div>
          <div><span>예측 승률</span><strong>{html.escape(row["예측 승률"])}</strong></div>
          <div><span>신뢰도</span><strong>{html.escape(row["신뢰도"])}</strong></div>
        </div>
        <div class="match-pills">
          <span>핸디캡 {html.escape(row["핸디캡"])}</span>
          <span>O/U 8.5 {html.escape(row["오버/언더"])}</span>
        </div>
      </article>
    """


def render_model_diagnostics_embedded(results_dir: Path = DEFAULT_RESULTS):
    payload, predictions, model_path, prediction_path = load_run_model_results(results_dir)
    selected = payload.get("selected_model", {})
    summary = prediction_summary(predictions)
    cards = "".join(
        [
            metric_card("선택 모델", selected.get("model", "-"), "득점 MAE 기준 우선 선택"),
            metric_card("MAE", fmt_float(selected.get("mae"), 4), "득점 평균 절대 오차"),
            metric_card("RMSE", fmt_float(selected.get("rmse"), 4), "큰 오차 반영"),
            metric_card("승패 변환 정확도", fmt_pct(selected.get("accuracy")), "예상 득실차 기반"),
            metric_card("Brier Score", fmt_float(selected.get("brier_score"), 4), "확률 품질"),
            metric_card("검증 경기", str(payload.get("validation_games", "-")), f'cutoff {payload.get("training_cutoff", "-")}'),
        ]
    )
    summary_cards = "".join(
        [
            metric_card("평균 홈 예상 득점", fmt_float(summary.get("avg_home_runs"), 2)),
            metric_card("평균 원정 예상 득점", fmt_float(summary.get("avg_away_runs"), 2)),
            metric_card("평균 절대 예상 득실차", fmt_float(summary.get("avg_abs_run_diff"), 2)),
            metric_card("1점 이상 우세 경기", str(summary.get("big_edge_games", "-")), f'적중률 {fmt_pct(summary.get("big_edge_accuracy"))}'),
        ]
    )
    return f"""
      <div class="run-model-panel">
        <section class="run-model-hero">
          <div>
            <div class="eyebrow">INDEPENDENT RUN MODEL</div>
            <h2>득점 기반 승부 예측</h2>
            <p>독립 득점 기반 모델 결과입니다. 기존 경기 승패 모델과 결과를 섞지 않고, 팀별 예상 득점을 먼저 예측한 뒤 예상 득실차를 홈팀 승률로 변환합니다.</p>
            <p class="run-model-source">생성 시각: {html.escape(str(payload.get("generated_at", "-")))} · 결과 JSON: {html.escape(str(model_path))} · 예측 CSV: {html.escape(str(prediction_path))}</p>
          </div>
        </section>

        <section class="run-model-section">
          <div class="section-title">
            <div>
              <div class="eyebrow">SELECTED MODEL</div>
              <h2>선택 모델 요약</h2>
            </div>
          </div>
          <div class="run-model-grid">{cards}</div>
          <p class="note">이 모델은 기존 승패 모델을 대체하지 않는 별도 실험 모델입니다. 성능이 기존 모델보다 낮으면 낮은 그대로 표시합니다.</p>
        </section>

        <section class="run-model-section">
          <div class="eyebrow">MODEL CANDIDATES</div>
          <h2>후보 모델 비교</h2>
          <p class="note">선택 모델은 득점 예측 오차(MAE)를 우선 기준으로 고릅니다. 승패 정확도와 Brier Score는 예상 득점 차이를 승률로 변환했을 때의 참고 성능입니다.</p>
          <div class="run-model-tablewrap">{table_html(model_score_rows(payload), ["모델", "MAE", "RMSE", "승패 정확도", "Brier", "LogLoss", "득실차 방향"])}</div>
        </section>

        <section class="run-model-section">
          <div class="eyebrow">EXPECTED RUNS</div>
          <h2>예상 득점 예측 결과</h2>
          <div class="run-model-grid">{summary_cards}</div>
          <h3>예상 득실차 구간별 적중률</h3>
          <div class="run-model-tablewrap">{table_html(confidence_rows(predictions), ["예상 득실차 구간", "경기 수", "적중률", "평균 홈승률"])}</div>
          <h3>최근 검증 경기 예측</h3>
          <div class="run-model-tablewrap">{table_html(recent_prediction_rows(predictions), ["경기일", "경기", "원정 예상득점", "홈 예상득점", "홈 기준 득실차", "예측 승률", "예측", "실제", "결과"])}</div>
        </section>

        <section class="run-model-section">
          <div class="eyebrow">ERROR ANALYSIS</div>
          <h2>오차 태그 요약</h2>
          <p class="note">득점 기반 모델이 어떤 경기에서 크게 틀리는지 보기 위한 진단 리포트입니다. 예측값을 보정하지 않고 오차 유형만 분해합니다.</p>
          <div class="run-model-tablewrap">{table_html(tag_summary_rows(payload), ["유형", "경기 수", "평균 MAE", "RMSE", "평균 실제 득점", "평균 예측 득점", "평균 오차", "득실 방향 적중률"])}</div>
          <div class="run-model-split">
            {score_summary_card("고득점 경기 예측 한계", payload.get("high_score_error_summary", {}))}
            {score_summary_card("저득점 경기 예측 한계", payload.get("low_score_error_summary", {}))}
          </div>
          <p class="note">{html.escape(str(payload.get("run_model_next_step_note", "")))}</p>
        </section>

        <section class="run-model-section">
          <div class="eyebrow">FEATURE IMPORTANCE</div>
          <h2>피처 중요도 TOP 20</h2>
          <p class="note">Permutation importance를 사용해 MAE 기준 중요도를 계산했습니다. 값이 클수록 해당 피처를 섞었을 때 득점 예측 오차가 커졌다는 뜻입니다.</p>
          <div class="run-model-tablewrap">{table_html(importance_rows(payload, 20), ["피처", "중요도 평균", "표준편차"])}</div>
        </section>
      </div>
    """


def render_prediction_board_embedded(results_dir: Path = DEFAULT_RESULTS):
    payload, predictions, model_path, prediction_path = load_run_model_results(results_dir)
    selected = payload.get("selected_model", {})
    status = board_status(payload, predictions, model_path, prediction_path)
    rows = prediction_board_rows(predictions)
    cards = "".join(match_card_html(row) for row in rows)
    status_cards = "".join(
        [
            metric_card("생성 시간", str(status["generated_at"])),
            metric_card("예측 기준일", str(status["prediction_date"])),
            metric_card("경기 수", str(status["game_count"])),
            metric_card("데이터 기준", str(status["data_source"])),
        ]
    )
    schedule_cards = "".join(
        [
            metric_card("선택 모드", "최신 결과 기준", "expected_runs_predictions.csv 최신 경기일"),
            metric_card("일정 원본 행 수", str(status["total_rows"])),
            metric_card("예상 경기 수 / 쌍생성 경기 수", f'{status["game_count"]} / {status["pair_rows"]}'),
            metric_card("홈/원정 매칭", str(status["home_away_match"])),
            metric_card("중복 경기", str(status["duplicate_games"])),
            metric_card("검증 상태", str(status["validation_status"])),
        ]
    )
    diagnostics = render_model_diagnostics_embedded(results_dir)
    return f"""
      <div class="run-model-panel">
        <section class="run-model-hero match-board-hero">
          <div>
            <div class="eyebrow">KBO MATCH PREDICTION</div>
            <h2>KBO 승부 예측 대시보드</h2>
            <p>KBO 경기 일정에 맞춰 예상 스코어, 승패 확률, 핸디캡, 오버/언더를 자동 계산합니다.</p>
            <p class="run-model-source">독립 득점 기반 모델 결과 · 선택 모델 {html.escape(str(selected.get("model", "-")))} · 결과 JSON: {html.escape(str(model_path))} · 예측 CSV: {html.escape(str(prediction_path))}</p>
          </div>
        </section>

        <section class="run-model-section">
          <div class="eyebrow">BASE INFO</div>
          <h2>기준 정보</h2>
          <div class="run-model-grid">{status_cards}</div>
        </section>

        <section class="run-model-section">
          <div class="eyebrow">SCHEDULE STATUS</div>
          <h2>일정 선택 상태</h2>
          <div class="run-model-grid schedule-grid">{schedule_cards}</div>
          <p class="note">이 탭은 `expected_runs_predictions.csv`의 최신 경기일을 기준으로 표시합니다. 현재 결과 파일은 완료 경기 검증 구간을 포함하므로 실제 운영 예측과 구분해서 해석해야 합니다.</p>
        </section>

        <section class="run-model-section">
          <div class="eyebrow">MATCH BOARD</div>
          <h2>KBO 경기 예측표</h2>
          <p class="note">예상 스코어는 `away_expected_runs`, `home_expected_runs`를 사용합니다. 승패 확률은 `home_win_probability`로 홈/원정 확률을 계산합니다. 핸디캡은 예상 득실차 2.5, 오버/언더는 예상 총득점 8.5 기준의 표시용 판단입니다.</p>
          <div class="match-card-grid">{cards or '<p class="empty">표시할 경기 예측이 없습니다.</p>'}</div>
          <div class="run-model-tablewrap">{table_html(rows, ["경기일", "원정팀 vs 홈팀", "예상 스코어", "승패 확률", "신뢰도", "판단", "핸디캡", "오버/언더"])}</div>
        </section>

        <details class="run-model-diagnostics">
          <summary>모델 진단 보기</summary>
          {diagnostics}
        </details>
      </div>
    """


def render_embedded_dashboard(results_dir: Path = DEFAULT_RESULTS):
    return render_model_diagnostics_embedded(results_dir)


def recent_prediction_rows(predictions: pd.DataFrame):
    if predictions.empty:
        return []
    df = predictions.copy().tail(20)
    rows = []
    for _, row in df.iterrows():
        home_pick = row["predicted_winner"] == row["home_team"]
        predicted_win_probability = row["home_win_probability"] if home_pick else 1 - row["home_win_probability"]
        rows.append(
            {
                "경기일": row["date"],
                "경기": f'{row["away_team"]} @ {row["home_team"]}',
                "원정 예상득점": fmt_float(row["away_expected_runs"], 1),
                "홈 예상득점": fmt_float(row["home_expected_runs"], 1),
                "홈 기준 득실차": f'{row["expected_run_diff"]:+.2f}',
                "예측 승률": fmt_pct(predicted_win_probability),
                "예측": row["predicted_winner"],
                "실제": row["actual_winner"],
                "결과": "적중" if row["prediction_result"] == "correct" else "오답",
            }
        )
    return rows


def render_dashboard(results_dir: Path, output_path: Path):
    board_html = render_prediction_board_embedded(results_dir)
    document = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>KBO 승부 예측 대시보드</title>
  <style>
    :root {{
      --bg: #f6f7f9; --panel: #ffffff; --text: #17202a; --muted: #64748b;
      --line: #dbe3ea; --accent: #0f766e; --blue: #1d4ed8;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: Arial, 'Noto Sans KR', sans-serif; background: var(--bg); color: var(--text); line-height: 1.55; }}
    main {{ max-width: 1240px; margin: 0 auto; padding: 24px; }}
    h1, h2, h3 {{ margin: 0 0 10px; letter-spacing: 0; }}
    .eyebrow {{ color: var(--accent); font-size: 12px; font-weight: 700; text-transform: uppercase; }}
    .run-model-panel {{ display: grid; gap: 18px; }}
    .run-model-hero, .run-model-section {{ background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 22px; }}
    .run-model-hero {{ background: #10202f; color: #fff; }}
    .run-model-hero p {{ color: #cbd5e1; max-width: 980px; }}
    .run-model-source {{ font-size: 12px; word-break: break-all; }}
    .run-model-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; }}
    .mini-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; margin-top: 12px; }}
    .metric {{ border: 1px solid var(--line); border-radius: 8px; padding: 14px; background: #fbfcfd; min-height: 92px; }}
    .metric p {{ margin: 0 0 8px; color: var(--muted); font-size: 13px; }}
    .metric strong {{ display: block; font-size: 22px; }}
    .metric span {{ display: block; margin-top: 8px; color: var(--muted); font-size: 12px; }}
    .note {{ color: var(--muted); line-height: 1.6; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ padding: 9px 10px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }}
    th {{ color: #334155; background: #f1f5f9; }}
    .run-model-tablewrap {{ overflow-x: auto; margin-top: 12px; }}
    .match-card-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px; margin-bottom: 18px; }}
    .match-card {{ border: 1px solid var(--line); border-radius: 12px; padding: 16px; background: #fff; }}
    .match-card-top {{ display:flex; justify-content:space-between; gap: 10px; color: var(--muted); font-size: 12px; font-weight: 700; }}
    .match-badge {{ border-radius: 999px; padding: 4px 9px; background: #f1f5f9; color: #334155; }}
    .match-badge.pick {{ background: #ecfdf5; color: #047857; }}
    .match-badge.watch {{ background: #fffbeb; color: #b45309; }}
    .match-badge.risk {{ background: #fef2f2; color: #b91c1c; }}
    .score-line {{ font-size: 24px; font-weight: 800; margin: 12px 0; color: var(--blue); }}
    .match-meta {{ display:grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }}
    .match-meta span {{ display:block; color: var(--muted); font-size: 12px; }}
    .match-meta strong {{ display:block; margin-top:4px; }}
    .match-pills {{ display:flex; gap: 8px; flex-wrap:wrap; margin-top: 14px; }}
    .match-pills span {{ border:1px solid var(--line); border-radius:999px; padding:6px 9px; font-size:12px; color:#334155; background:#f8fafc; }}
    .run-model-split {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    .run-model-diagnostics {{ background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 18px; }}
    .diagnostic-card {{ border: 1px solid var(--line); border-radius: 8px; padding: 16px; background: #fbfcfd; margin-top: 12px; }}
    .diagnostic-card p {{ color: var(--muted); line-height: 1.6; }}
    ul.features {{ columns: 2; margin: 8px 0 0; padding-left: 20px; color: #334155; }}
    .empty {{ color: var(--muted); }}
    @media (max-width: 760px) {{ main {{ padding: 12px; }} .run-model-split {{ grid-template-columns: 1fr; }} ul.features {{ columns: 1; }} }}
  </style>
</head>
<body>
  <main>
    {board_html}
  </main>
</body>
</html>
"""
    output_path.write_text(document, encoding="utf-8")
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Render the independent KBO run model dashboard")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main():
    args = parse_args()
    output = render_dashboard(args.results_dir, args.output)
    print(f"Run model dashboard generated: {output}")


if __name__ == "__main__":
    main()
