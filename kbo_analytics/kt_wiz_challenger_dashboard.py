from __future__ import annotations

import argparse
import json
from datetime import date
from html import escape
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "modeling" / "results"
DASHBOARD_DIR = BASE_DIR / "dashboard"


def load_json(path: Path):
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path, limit: int | None = None):
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    return frame.head(limit) if limit else frame


def render_table(frame: pd.DataFrame, columns: list[str] | None = None, limit: int = 20):
    if frame.empty:
        return "<p class='muted'>데이터가 아직 생성되지 않았습니다.</p>"
    view = frame.copy()
    if columns:
        view = view[[column for column in columns if column in view.columns]]
    view = view.head(limit)
    header = "".join(f"<th>{escape(str(column))}</th>" for column in view.columns)
    rows = []
    for _, row in view.iterrows():
        rows.append("<tr>" + "".join(f"<td>{escape(str(value))}</td>" for value in row.tolist()) + "</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def metric_card(label: str, value):
    return f"<div class='metric'><span>{escape(label)}</span><strong>{escape(str(value))}</strong></div>"


def best_precision_row(precision: pd.DataFrame):
    if precision.empty:
        return {}
    frame = precision.copy()
    precision_column = "precision" if "precision" in frame.columns else "accuracy"
    games_column = "games" if "games" in frame.columns else "picked_games"
    frame = frame[frame[precision_column].notna()]
    if frame.empty:
        return {}
    frame = frame.sort_values([precision_column, games_column], ascending=False)
    return frame.iloc[0].to_dict()


def build_dashboard(reference_date: date):
    summary = load_json(RESULTS_DIR / "kt_wiz_performance_summary.json")
    dataset = load_json(RESULTS_DIR / "kt_wiz_dataset_summary.json")
    roadmap = load_json(RESULTS_DIR / "kt_wiz_85_percent_target_roadmap.json")
    experiments = load_csv(RESULTS_DIR / "kt_wiz_model_experiment_report.csv")
    strategies = load_csv(RESULTS_DIR / "kt_wiz_selective_pick_strategy_report.csv")
    comparison = load_csv(RESULTS_DIR / "kt_wiz_vs_production_comparison_report.csv")
    rolling = load_csv(RESULTS_DIR / "kt_wiz_rolling_backtest_report.csv")
    precision = load_csv(RESULTS_DIR / "kt_wiz_precision_target_report.csv")
    game_audit = load_csv(RESULTS_DIR / "kt_wiz_precision_target_game_audit.csv")
    feature_schema = load_csv(RESULTS_DIR / "kt_wiz_pregame_feature_schema.csv")
    availability_audit = load_csv(RESULTS_DIR / "kt_wiz_pregame_data_availability_audit.csv")

    best = summary.get("best_kt_challenger", {})
    policy = summary.get("recommended_kt_prediction_policy", "no_kt_specific_edge_found")
    target = summary.get("kt_85_percent_target_summary", {})
    best_precision = roadmap or best_precision_row(precision)
    current_best_segment = best_precision.get("current_best_display_segment", best_precision.get("display_segment_name", best_precision.get("segment_name", "not_available")))
    current_best_precision = best_precision.get("current_best_segment_precision", best_precision.get("precision", best_precision.get("accuracy", "not_available")))
    current_best_games = best_precision.get("current_best_segment_games", best_precision.get("games", "not_available"))
    current_gap = best_precision.get("current_gap_to_target", best_precision.get("gap_to_target", "not_available"))
    cards = [
        metric_card("KT 분석 경기", summary.get("total_games_analyzed")),
        metric_card("2026 KT 경기", summary.get("current_season_games_analyzed")),
        metric_card("추천 정책", policy),
        metric_card("실험 후보", best.get("model_name", "not_available")),
        metric_card("후보 정확도", best.get("accuracy", "not_available")),
        metric_card("85% 목표", "달성" if target.get("target_met") else "미달"),
        metric_card("최고 구간", current_best_segment),
        metric_card("최고 precision", current_best_precision),
        metric_card("대상 경기", current_best_games),
        metric_card("목표까지 차이", current_gap),
        metric_card("운영 반영", "미반영"),
    ]

    html = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>KT Wiz Focused Challenger Dashboard</title>
  <style>
    body {{ margin: 0; font-family: Arial, "Malgun Gothic", sans-serif; background: #f5f7fb; color: #18202f; }}
    header {{ background: #111827; color: white; padding: 28px 32px; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 24px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    h2 {{ margin-top: 32px; font-size: 20px; }}
    .label {{ display: inline-block; padding: 6px 10px; background: #f59e0b; color: #111827; font-weight: 700; border-radius: 4px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin: 20px 0; }}
    .metric {{ background: white; border: 1px solid #d8dee9; border-radius: 6px; padding: 14px; }}
    .metric span {{ display: block; color: #5b6472; font-size: 12px; margin-bottom: 8px; }}
    .metric strong {{ font-size: 18px; }}
    section {{ background: white; border: 1px solid #d8dee9; border-radius: 6px; padding: 18px; margin-bottom: 18px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #e5e7eb; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #f3f4f6; color: #374151; }}
    .muted {{ color: #667085; }}
    .note {{ background: #fff7ed; border: 1px solid #fed7aa; padding: 12px; border-radius: 6px; }}
    .warning {{ background: #fef2f2; border: 1px solid #fecaca; padding: 12px; border-radius: 6px; }}
  </style>
</head>
<body>
  <header>
    <h1>KT Wiz Focused Challenger Dashboard</h1>
    <div class="label">Experimental / Offline Monitoring Only</div>
    <p>운영 예측에는 아직 미반영. 전체 모델과 KT 특화 모델이 같은 방향일 때만 참고합니다.</p>
  </header>
  <main>
    <div class="grid">{''.join(cards)}</div>
    <section>
      <h2>KT Challenger Status</h2>
      <div class="note">KT challenger는 offline monitoring 전용입니다. production KBO-wide model을 대체하지 않으며, 운영 pick으로 사용하지 않습니다.</div>
      <div class="grid">
        {metric_card("Best challenger", best.get("model_name", "not_available"))}
        {metric_card("Production comparison", summary.get("kt_vs_production_summary", {}).get("accuracy_delta", "not_available"))}
        {metric_card("Policy", policy)}
        {metric_card("Report only", summary.get("report_only", True))}
      </div>
    </section>
    <section>
      <h2>KT Dataset Summary</h2>
      <p class="muted">기준일: {escape(reference_date.isoformat())}</p>
      <div class="grid">
        {metric_card("시즌", dataset.get("seasons_covered"))}
        {metric_card("KT 승률", dataset.get("kt_win_rate"))}
        {metric_card("홈 경기", dataset.get("home_games"))}
        {metric_card("원정 경기", dataset.get("away_games"))}
      </div>
    </section>
    <section>
      <h2>Best KT Challenger</h2>
      {render_table(pd.DataFrame([best]) if best else pd.DataFrame(), limit=1)}
    </section>
    <section>
      <h2>KT vs Production Comparison</h2>
      {render_table(comparison, ["comparison_scope", "games", "production_accuracy", "kt_challenger_accuracy", "accuracy_delta", "agreement_games", "agreement_accuracy", "interpretation"], 20)}
    </section>
    <section>
      <h2>Selective Pick Strategy</h2>
      {render_table(strategies.sort_values(["pick_accuracy", "coverage_rate"], ascending=False) if not strategies.empty else strategies, ["strategy_name", "picked_games", "coverage_rate", "pick_accuracy", "avg_probability", "current_season_pick_accuracy", "interpretation"], 20)}
    </section>
    <section>
      <h2>85% Precision Target</h2>
      <p class="muted">Target definition: selected-pick precision / selected-pick accuracy for selected KT recommendation segments only. It is not full-season accuracy, all-game accuracy, or generic classification precision.</p>
      <p class="muted">85% 목표 유지. 현재 미달성. 최고 구간은 {escape(str(current_best_segment))}, precision은 {escape(str(current_best_precision))}, 경기 수는 {escape(str(current_best_games))}, gap은 {escape(str(current_gap))}입니다.</p>
      <div class="warning">선발/불펜/라인업 정보의 누수 없는 축적이 필요합니다. 이 화면은 운영 반영이 아닌 KT 전용 offline monitoring입니다.</div>
      {render_table(precision.sort_values([("precision" if "precision" in precision.columns else "accuracy"), ("games" if "games" in precision.columns else "picked_games")], ascending=False) if not precision.empty else precision, ["evaluation_mode", "segment_name", "display_segment_name", "games", "correct", "precision", "target_precision", "gap_to_target", "meets_target", "sample_size_bucket", "statistically_actionable", "wilson_ci_low", "wilson_ci_high", "target_met_by_point_estimate", "target_met_by_lower_bound", "segment_search_count", "selected_after_segment_search", "multiple_testing_risk", "requires_forward_validation", "interpretation"], 20)}
    </section>
    <section>
      <h2>Why 85% Is Not Met</h2>
      <div class="warning">The current best segment is the result of historical segment search and must be forward-validated before operational use.</div>
      <p class="muted">현재 모델은 confirmed starter, bullpen usage, lineup, catcher status, key hitter availability 신호가 부족합니다. 모델 튜닝만으로 gap을 닫기는 어렵고, cutoff 이전에 확인된 pregame 데이터 축적이 필요합니다.</p>
    </section>
    <section>
      <h2>85% Precision Roadmap</h2>
      <div class="grid">
        {metric_card("목표 precision", roadmap.get("target_precision", 0.85))}
        {metric_card("현재 최고 구간", roadmap.get("current_best_display_segment", roadmap.get("current_best_segment", "not_available")))}
        {metric_card("현재 최고 precision", roadmap.get("current_best_segment_precision", current_best_precision))}
        {metric_card("현재 gap", roadmap.get("current_gap_to_target", current_gap))}
      </div>
      <p>{escape(str(roadmap.get("why_current_model_is_insufficient", "85% is not currently achieved.")))}</p>
      <p class="muted">다음 필요 정보: {escape(str(roadmap.get("required_new_information", [])))}</p>
      <p class="muted">표본 기준: {escape(str(roadmap.get("minimum_sample_size_policy", {})))}</p>
    </section>
    <section>
      <h2>Pregame Feature Store Plan</h2>
      <p class="muted">예측 cutoff 이전에 확인 가능한 정보만 후보 feature로 둡니다. 경기 후 기록과 당일 경기 결과는 평가 전용입니다.</p>
      {render_table(feature_schema, ["feature_name", "feature_group", "data_type", "allowed_timing", "leakage_risk", "current_implementation_status", "feature_available_at", "prediction_cutoff_at", "is_available_before_cutoff", "expected_impact", "notes"], 40)}
    </section>
    <section>
      <h2>Pregame Availability Audit</h2>
      {render_table(availability_audit, ["priority", "source_group", "feature_name", "required_for_85_target_experiment", "currently_available", "historical_backfill_possible", "live_collection_needed", "estimated_coverage", "blocking_issue", "interpretation"], 40)}
    </section>
    <section>
      <h2>Precision Target Game Audit</h2>
      {render_table(game_audit.sort_values(["included_in_best_segment", "game_date"], ascending=[False, False]) if not game_audit.empty else game_audit, ["game_date", "season", "home_team", "away_team", "evaluation_mode", "segment_name", "display_segment_name", "included_in_best_segment", "prediction_cutoff_type", "predicted_winner", "predicted_probability", "actual_winner", "correct", "production_predicted_winner", "production_predicted_probability", "kt_challenger_predicted_winner", "kt_challenger_probability", "model_agrees_with_production", "feature_set", "data_available_before_cutoff", "leakage_audit_passed", "interpretation"], 40)}
    </section>
    <section>
      <h2>Rolling Backtest</h2>
      {render_table(rolling.sort_values("prediction_date", ascending=False) if not rolling.empty else rolling, ["prediction_date", "opponent", "is_home", "train_games_before_date", "predicted_winner", "predicted_probability", "actual_winner", "correct", "production_predicted_winner", "model_agrees_with_production"], 30)}
    </section>
    <section>
      <h2>Limitations & Leakage Audit</h2>
      <div class="note">현재는 포트폴리오 검증용 실험 모델입니다. KT 특화 실험 후보: offline monitoring. 메인 예측과 운영 모델을 대체하지 않습니다.</div>
      <p>{escape(str(summary.get("leakage_audit_summary", {})))}</p>
      <p>{escape(str(summary.get("limitations", [])))}</p>
      <div class="warning">Final policy: offline monitoring only, no production replacement, no betting recommendation.</div>
    </section>
  </main>
</body>
</html>"""
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    output = DASHBOARD_DIR / "kt_wiz_challenger.html"
    output.write_text(html, encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description="Build the KT Wiz offline challenger dashboard.")
    parser.add_argument("--reference-date", default=date.today().isoformat())
    parser.add_argument("--training-start-year", default="2016")
    args = parser.parse_args()
    output = build_dashboard(date.fromisoformat(args.reference_date))
    print(f"[Success] KT Wiz challenger dashboard generated: {output}")


if __name__ == "__main__":
    main()
