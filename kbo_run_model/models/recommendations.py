from __future__ import annotations


HANDICAP_LINES = [1.5, 2.5, 3.5]
DEFAULT_HANDICAP_LINE = 2.5
OVER_UNDER_LINES = [7.5, 8.5, 9.5, 10.5]
DEFAULT_OVER_UNDER_LINE = 8.5
WATCH_RUN_DIFF_THRESHOLD = 0.5
WATCH_TOTAL_THRESHOLD = 0.4


def confidence_level(expected_run_diff: float) -> str:
    margin = abs(expected_run_diff)
    if margin < 0.5:
        return "낮음"
    if margin < 1.0:
        return "보통"
    return "높음"


def moneyline_pick(home_team: str, away_team: str, home_win_probability: float) -> str:
    if abs(home_win_probability - 0.5) < 0.03:
        return "관망"
    return home_team if home_win_probability >= 0.5 else away_team


def handicap_pick(home_team: str, away_team: str, expected_run_diff: float, line: float = DEFAULT_HANDICAP_LINE) -> str:
    if abs(expected_run_diff) < WATCH_RUN_DIFF_THRESHOLD:
        return "관망"
    if expected_run_diff > line:
        return f"{home_team} -{line}"
    if expected_run_diff < -line:
        return f"{away_team} -{line}"
    return f"{away_team} +{line}" if expected_run_diff > 0 else f"{home_team} +{line}"


def over_under_pick(total_expected_runs: float, line: float = DEFAULT_OVER_UNDER_LINE) -> str:
    diff = total_expected_runs - line
    if abs(diff) < WATCH_TOTAL_THRESHOLD:
        return "관망"
    return "오버" if diff > 0 else "언더"
