#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import random
from datetime import datetime, timedelta
import pandas as pd

# ==========================================
# Defaults
# ==========================================
DEFAULT_START_DATE = "2026-03-23"
DEFAULT_TEAM = "KT"
DEFAULT_TOTAL_FINAL_GAMES = 144

DEFAULT_OPPONENTS = ["LG", "NC", "Doosan", "KIA", "Lotte", "Samsung", "Hanwha", "Kiwoom", "SSG"]

# 간단한 “브레이크” (올스타 느낌) - 필요 없으면 0으로 가능
DEFAULT_BREAK_START = "2026-07-13"
DEFAULT_BREAK_DAYS = 5
# ==========================================


def daterange_generator(start_date: datetime):
    """무한 날짜 generator (start_date부터 하루씩 증가)"""
    d = start_date
    while True:
        yield d
        d += timedelta(days=1)


def is_break_day(date_obj: datetime, break_start: datetime | None, break_days: int) -> bool:
    if break_start is None or break_days <= 0:
        return False
    return break_start <= date_obj < (break_start + timedelta(days=break_days))


def make_series_plan(opponents: list[str], games_per_opp: int, seed: int | None):
    """
    3연전(3-game blocks) 위주로 시리즈 블록을 만들고, 블록 순서를 섞는다.
    games_per_opp=16이면: 3+3+3+3+3+1 식으로 블록 생성.
    """
    rng = random.Random(seed)

    blocks = []
    for opp in opponents:
        remaining = games_per_opp
        while remaining > 0:
            block_size = 3 if remaining >= 3 else remaining
            blocks.append([opp] * block_size)
            remaining -= block_size

    rng.shuffle(blocks)

    # flatten
    plan = []
    for b in blocks:
        plan.extend(b)

    return plan  # 길이 = len(opponents) * games_per_opp (=144)


def make_home_away_plan(n: int, seed: int | None):
    """H/A 균등(가능한 범위)하게 섞어서 반환"""
    rng = random.Random(seed)
    half = n // 2
    plan = ["H"] * half + ["A"] * (n - half)
    rng.shuffle(plan)
    return plan


def generate_score(rng: random.Random, is_win: bool, allow_draw: bool, draw_prob: float):
    """
    점수 생성:
      - 승/패는 최소 1점 차이
      - 무승부는 allow_draw일 때 draw_prob로 발생
    """
    if allow_draw and rng.random() < draw_prob:
        s = rng.randint(1, 10)
        return "Draw", s, s

    if is_win:
        kt = rng.randint(2, 12)
        opp = rng.randint(0, kt - 1)
        return "Win", kt, opp
    else:
        opp = rng.randint(2, 12)
        kt = rng.randint(0, opp - 1)
        return "Loss", kt, opp


def main():
    parser = argparse.ArgumentParser(
        description="KT 2026 Virtual Season Mock Data Generator (pipeline-friendly)"
    )
    parser.add_argument("--team", default=DEFAULT_TEAM, help="My team name (default: KT)")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE, help="Season start date YYYY-MM-DD")
    parser.add_argument("--total-final", type=int, default=DEFAULT_TOTAL_FINAL_GAMES,
                        help="Number of FINAL games to generate (default: 144)")
    parser.add_argument("--winrate", type=float, default=0.58, help="Win probability for my team (default: 0.58)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")

    parser.add_argument("--include-canceled", action="store_true",
                        help="If set, include CANCELED rows in output (Final 144 still guaranteed)")
    parser.add_argument("--rain-prob", type=float, default=0.10,
                        help="Rain cancel probability during Jun-Aug (default: 0.10)")

    parser.add_argument("--draw-prob", type=float, default=0.01, help="Draw probability (default: 0.01)")
    parser.add_argument("--format", choices=["csv", "parquet"], default="csv", help="Output format")
    parser.add_argument("--output", default="kt_2026_virtual_season.csv", help="Output file path")

    parser.add_argument("--break-start", default=DEFAULT_BREAK_START,
                        help="Break start date YYYY-MM-DD (default: 2026-07-13). Use 'none' to disable.")
    parser.add_argument("--break-days", type=int, default=DEFAULT_BREAK_DAYS,
                        help="Number of break days (default: 5)")

    args = parser.parse_args()

    rng = random.Random(args.seed)

    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")

    break_start = None
    if args.break_start.lower() != "none":
        break_start = datetime.strptime(args.break_start, "%Y-%m-%d")

    opponents = DEFAULT_OPPONENTS[:]
    # 9팀 균등 분포: total_final=144이면 games_per_opp=16이 딱 떨어짐.
    if args.total_final % len(opponents) != 0:
        raise SystemExit(
            f"total_final({args.total_final}) must be divisible by number of opponents({len(opponents)}) "
            "to keep balanced distribution. (e.g., 144)"
        )

    games_per_opp = args.total_final // len(opponents)

    # 144경기 상대 분포 균등 + 3연전 블록 기반 “시리즈 계획”
    opp_plan = make_series_plan(opponents, games_per_opp, seed=args.seed)

    # 홈/원정 균등
    ha_plan = make_home_away_plan(args.total_final, seed=args.seed + 1)

    data_rows = []
    final_count = 0
    scheduled_index = 0

    date_iter = daterange_generator(start_date)

    series_id = 0
    last_opp = None
    last_ha = None

    print(f"⚾ Generating mock season for {args.team} ...")
    print(f"- Final games target: {args.total_final}")
    print(f"- Winrate: {args.winrate:.3f}, Seed: {args.seed}")
    print(f"- Include canceled rows: {args.include_canceled}")
    print(f"- Break: {('disabled' if break_start is None else break_start.strftime('%Y-%m-%d'))} for {args.break_days} days")

    while final_count < args.total_final:
        current_date = next(date_iter)

        # 월요일 휴식
        if current_date.weekday() == 0:
            continue

        # 브레이크 스킵
        if is_break_day(current_date, break_start, args.break_days):
            continue

        # 이번에 생성할 “예정 경기”의 상대/홈원정 (Final 144개 기준 플랜)
        opponent = opp_plan[final_count]  # final_count 기준으로 확정 분포 유지
        home_away = ha_plan[final_count]

        # 시리즈 id: 상대 또는 홈/원정이 바뀌면 새 시리즈로
        if (opponent != last_opp) or (home_away != last_ha):
            series_id += 1
            last_opp = opponent
            last_ha = home_away

        # 우천 취소 시뮬레이션 (6~8월에만)
        canceled = False
        if 6 <= current_date.month <= 8:
            if rng.random() < args.rain_prob:
                canceled = True

        game_id = f"{current_date.strftime('%Y%m%d')}_{args.team}_{opponent}_{series_id:03d}"

        if canceled:
            # 취소 row는 옵션일 때만 남김. Final 카운트는 증가하지 않음.
            if args.include_canceled:
                data_rows.append({
                    "game_id": game_id,
                    "series_id": series_id,
                    "date": current_date.strftime("%Y-%m-%d"),
                    "team": args.team,
                    "opponent": opponent,
                    "home_away": home_away,
                    "status": "Canceled",
                    "result": None,
                    "score_team": None,
                    "score_opp": None,
                    "note": "Rainout"
                })
            continue

        # Final 경기 생성 (144개 채울 때까지)
        is_win = (rng.random() < args.winrate)
        result, team_score, opp_score = generate_score(
            rng=rng,
            is_win=is_win,
            allow_draw=True,
            draw_prob=args.draw_prob
        )

        data_rows.append({
            "game_id": game_id,
            "series_id": series_id,
            "date": current_date.strftime("%Y-%m-%d"),
            "team": args.team,
            "opponent": opponent,
            "home_away": home_away,
            "status": "Final",
            "result": result,
            "score_team": int(team_score),
            "score_opp": int(opp_score),
            "note": None
        })

        final_count += 1
        scheduled_index += 1

    df = pd.DataFrame(data_rows)

    # 정렬: 날짜, status(Final 우선), game_id
    status_order = {"Final": 0, "Canceled": 1}
    df["_status_rank"] = df["status"].map(status_order).fillna(9)
    df = df.sort_values(["date", "_status_rank", "game_id"]).drop(columns=["_status_rank"]).reset_index(drop=True)

    # 통계 출력
    final_df = df[df["status"] == "Final"].copy()
    wins = (final_df["result"] == "Win").sum()
    losses = (final_df["result"] == "Loss").sum()
    draws = (final_df["result"] == "Draw").sum()
    final_total = len(final_df)

    print("=" * 60)
    print(f"✅ Done. Rows total: {len(df)} (Final: {final_total}, Canceled: {(df['status']=='Canceled').sum()})")
    if final_total > 0:
        print(f"📊 Final W-L-D = {wins}-{losses}-{draws} | Win% (W/Final) = {wins/final_total:.3f}")
    print("🔎 Head:")
    print(df.head(10).to_string(index=False))

    # 저장
    if args.format == "csv":
        out = args.output
        if not out.lower().endswith(".csv"):
            out += ".csv"
        df.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"📂 Saved CSV: {out}")
    else:
        out = args.output
        if not out.lower().endswith(".parquet"):
            out += ".parquet"
        try:
            df.to_parquet(out, index=False)
            print(f"📂 Saved Parquet: {out}")
        except Exception as e:
            raise SystemExit(
                f"Parquet save failed (need pyarrow or fastparquet). Error: {e}\n"
                "Tip: pip install pyarrow"
            )


if __name__ == "__main__":
    main()
