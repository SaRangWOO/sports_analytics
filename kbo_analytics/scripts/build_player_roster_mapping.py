from __future__ import annotations

import csv
import re
from html import unescape
from pathlib import Path

import requests


BASE_URL = "https://www.koreabaseball.com/Record/Player"
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": f"{BASE_URL}/HitterBasic/Basic1.aspx",
}
TEAM_CODES = {
    "KT": ("KT", "KT"),
    "Samsung": ("SS", "삼성"),
    "LG": ("LG", "LG"),
    "SSG": ("SK", "SSG"),
    "Doosan": ("OB", "두산"),
    "Hanwha": ("HH", "한화"),
    "KIA": ("HT", "KIA"),
    "NC": ("NC", "NC"),
    "Lotte": ("LT", "롯데"),
    "Kiwoom": ("WO", "키움"),
}
TEAM_ID_PREFIXES = {
    "KT": "KT",
    "Samsung": "SAM",
    "LG": "LG",
    "SSG": "SSG",
    "Doosan": "DOO",
    "Hanwha": "HAN",
    "KIA": "KIA",
    "NC": "NC",
    "Lotte": "LOT",
    "Kiwoom": "KIW",
}


def _hidden_fields(html: str):
    fields = {}
    for match in re.finditer(r'<input type="hidden" name="([^"]+)"[^>]*>', html):
        tag = match.group(0)
        value_match = re.search(r'value="([^"]*)"', tag)
        fields[match.group(1)] = unescape(value_match.group(1)) if value_match else ""
    return fields


def _fetch_team_page(session: requests.Session, path: str, team_code: str):
    url = f"{BASE_URL}/{path}"
    response = session.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()

    payload = _hidden_fields(response.text)
    payload.update(
        {
            "__EVENTTARGET": "ctl00$ctl00$ctl00$cphContents$cphContents$cphContents$ddlTeam$ddlTeam",
            "__EVENTARGUMENT": "",
            "ctl00$ctl00$ctl00$cphContents$cphContents$cphContents$ddlSeason$ddlSeason": "2026",
            "ctl00$ctl00$ctl00$cphContents$cphContents$cphContents$ddlSeries$ddlSeries": "0",
            "ctl00$ctl00$ctl00$cphContents$cphContents$cphContents$ddlTeam$ddlTeam": team_code,
            "ctl00$ctl00$ctl00$cphContents$cphContents$cphContents$ddlPos$ddlPos": "",
            "ctl00$ctl00$ctl00$cphContents$cphContents$cphContents$ddlSituation$ddlSituation": "",
            "ctl00$ctl00$ctl00$cphContents$cphContents$cphContents$ddlSituationDetail$ddlSituationDetail": "",
        }
    )
    response = session.post(url, headers=HEADERS, data=payload, timeout=30)
    response.raise_for_status()
    return response.text


def _parse_players(html: str):
    pattern = re.compile(
        r"<td>\d+</td>\s*"
        r'<td><a href="[^"]*playerId=(?P<kbo_player_id>\d+)[^"]*">(?P<name>[^<]+)</a></td>\s*'
        r"<td>(?P<team>[^<]+)</td>",
        re.S,
    )
    players = []
    seen = set()
    for match in pattern.finditer(html):
        name = unescape(match.group("name")).strip()
        if name in seen:
            continue
        seen.add(name)
        players.append(
            {
                "kbo_player_id": match.group("kbo_player_id"),
                "name": name,
                "kbo_team": unescape(match.group("team")).strip(),
            }
        )
    return players


def build_mapping():
    rows = []
    session = requests.Session()

    for team, (team_code, team_ko) in TEAM_CODES.items():
        prefix = TEAM_ID_PREFIXES[team]
        hitter_html = _fetch_team_page(session, "HitterBasic/Basic1.aspx", team_code)
        pitcher_html = _fetch_team_page(session, "PitcherBasic/Basic1.aspx", team_code)
        hitters = _parse_players(hitter_html)[:9]
        pitchers = _parse_players(pitcher_html)[:3]

        if len(hitters) < 9 or len(pitchers) < 3:
            raise RuntimeError(f"{team} roster is incomplete: hitters={len(hitters)}, pitchers={len(pitchers)}")

        for slot, player in enumerate(hitters, start=1):
            rows.append(
                {
                    "player_id": f"{prefix}{slot:02d}",
                    "team": team,
                    "team_ko": team_ko,
                    "slot": slot,
                    "role": "batter",
                    "player_name": player["name"],
                    "kbo_player_id": player["kbo_player_id"],
                    "source": "KBO Record HitterBasic Basic1",
                }
            )

        for offset, player in enumerate(pitchers, start=10):
            rows.append(
                {
                    "player_id": f"{prefix}{offset:02d}",
                    "team": team,
                    "team_ko": team_ko,
                    "slot": offset,
                    "role": "pitcher",
                    "player_name": player["name"],
                    "kbo_player_id": player["kbo_player_id"],
                    "source": "KBO Record PitcherBasic Basic1",
                }
            )

    return rows


def main():
    output_path = Path(__file__).resolve().parents[1] / "mock_api" / "player_roster_mapping.csv"
    rows = build_mapping()
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "player_id",
                "team",
                "team_ko",
                "slot",
                "role",
                "player_name",
                "kbo_player_id",
                "source",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"written {len(rows)} players to {output_path}")


if __name__ == "__main__":
    main()
