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
    model_path = results_dir / "expected_runs_model.json"
    prediction_path = results_dir / "expected_runs_predictions.csv"
    if not model_path.exists() or not prediction_path.exists():
        raise FileNotFoundError("Run model results are missing. Run run_prediction_model.py first.")

    payload = json.loads(model_path.read_text(encoding="utf-8"))
    predictions = pd.read_csv(prediction_path)
    selected = payload.get("selected_model", {})
    summary = prediction_summary(predictions)
    score_rows = model_score_rows(payload)
    conf_rows = confidence_rows(predictions)
    recent_rows = recent_prediction_rows(predictions)
    features = payload.get("feature_columns", [])

    cards = "".join(
        [
            metric_card("선택 모델", selected.get("model", "-"), "득점 MAE 기준 우선 선택"),
            metric_card("득점 MAE", fmt_float(selected.get("mae"), 4), "낮을수록 좋음"),
            metric_card("득점 RMSE", fmt_float(selected.get("rmse"), 4), "큰 오차 반영"),
            metric_card("승패 정확도", fmt_pct(selected.get("accuracy")), "예상 득실차 기반"),
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
    feature_list = "".join(f"<li>{html.escape(col)}</li>" for col in features)
    selected_note = (
        "현재 모델은 승패를 직접 맞히는 모델이 아니라 득점을 먼저 예측하는 기준선 모델입니다. "
        "승패 정확도는 참고 지표이며, 모델링 관점에서는 MAE/RMSE와 예상 득실차 구간별 성능을 함께 봐야 합니다."
    )

    document = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>KBO 득점 기반 예측 모델</title>
  <style>
    :root {{
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #17202a;
      --muted: #64748b;
      --line: #dbe3ea;
      --accent: #0f766e;
      --accent2: #b45309;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: Arial, 'Noto Sans KR', sans-serif; background: var(--bg); color: var(--text); }}
    header {{ padding: 28px 32px 18px; background: #10202f; color: #fff; }}
    header p {{ color: #cbd5e1; max-width: 980px; line-height: 1.6; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 24px; }}
    section {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 20px; margin-bottom: 18px; }}
    h1, h2, h3 {{ margin: 0 0 10px; letter-spacing: 0; }}
    .eyebrow {{ color: var(--accent); font-size: 12px; font-weight: 700; text-transform: uppercase; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; }}
    .metric {{ border: 1px solid var(--line); border-radius: 8px; padding: 14px; background: #fbfcfd; min-height: 92px; }}
    .metric p {{ margin: 0 0 8px; color: var(--muted); font-size: 13px; }}
    .metric strong {{ display: block; font-size: 22px; }}
    .metric span {{ display: block; margin-top: 8px; color: var(--muted); font-size: 12px; }}
    .note {{ color: var(--muted); line-height: 1.6; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ padding: 9px 10px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }}
    th {{ color: #334155; background: #f1f5f9; }}
    .split {{ display: grid; grid-template-columns: minmax(0, 1.2fr) minmax(280px, 0.8fr); gap: 16px; }}
    ul.features {{ columns: 2; margin: 8px 0 0; padding-left: 20px; color: #334155; }}
    .empty {{ color: var(--muted); }}
    @media (max-width: 760px) {{ main {{ padding: 12px; }} header {{ padding: 22px 16px; }} .split {{ grid-template-columns: 1fr; }} ul.features {{ columns: 1; }} }}
  </style>
</head>
<body>
  <header>
    <div class="eyebrow">Independent Run Model</div>
    <h1>KBO 득점 기반 예측 모델</h1>
    <p>이 화면은 기존 대시보드와 기존 승패 모델에서 분리된 모델링 전용 대시보드입니다. 팀별 예상 득점을 먼저 예측하고, 예상 득실차를 홈팀 승률로 변환하는 별도 실험 결과만 보여줍니다.</p>
  </header>
  <main>
    <section>
      <div class="eyebrow">Selected Model</div>
      <h2>선택 모델 요약</h2>
      <div class="grid">{cards}</div>
      <p class="note">{html.escape(selected_note)}</p>
      <p class="note">생성 시각: {html.escape(str(payload.get("generated_at", "-")))} · 학습 행 {payload.get("train_rows", "-")} · 검증 행 {payload.get("validation_rows", "-")} · 원천: {html.escape(str(payload.get("input_file", "-")))}</p>
    </section>

    <section>
      <div class="eyebrow">Model Candidates</div>
      <h2>후보 모델 비교</h2>
      <p class="note">선택 모델은 득점 예측 오차(MAE)를 우선 기준으로 고릅니다. 승패 정확도와 Brier Score는 예상 득점 차이를 승률로 변환했을 때의 참고 성능입니다.</p>
      {table_html(score_rows, ["모델", "MAE", "RMSE", "승패 정확도", "Brier", "LogLoss", "득실차 방향"])}
    </section>

    <section>
      <div class="eyebrow">Expected Runs</div>
      <h2>예상 득점 기반 검증 요약</h2>
      <div class="grid">{summary_cards}</div>
      <h3>예상 득실차 구간별 적중률</h3>
      {table_html(conf_rows, ["예상 득실차 구간", "경기 수", "적중률", "평균 홈승률"])}
    </section>

    <section>
      <div class="eyebrow">Predictions</div>
      <h2>최근 검증 경기 예측</h2>
      <p class="note">득실차는 홈팀 기준입니다. `+`면 홈팀 예상 우세, `-`면 원정팀 예상 우세입니다. 예측 승률은 예측 구단 기준 확률입니다.</p>
      {table_html(recent_rows, ["경기일", "경기", "원정 예상득점", "홈 예상득점", "홈 기준 득실차", "예측 승률", "예측", "실제", "결과"])}
    </section>

    <section class="split">
      <div>
        <div class="eyebrow">Features</div>
        <h2>사용 피처</h2>
        <p class="note">모든 피처는 완료 경기 원천 데이터에서 현재 경기 이전 정보만 사용하도록 rolling/expanding 방식으로 생성합니다.</p>
        <ul class="features">{feature_list}</ul>
      </div>
      <div>
        <div class="eyebrow">Scope</div>
        <h2>분리 원칙</h2>
        <p class="note">기존 대시보드 코드와 기존 승패 모델 결과 JSON은 사용하지 않습니다. 이 대시보드는 `run_model/results` 산출물만 읽어 생성됩니다.</p>
      </div>
    </section>
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
