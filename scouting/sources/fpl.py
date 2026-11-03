"""Fantasy Premier League API adapter (unofficial, no API key).

`bootstrap-static/` returns every player in one call: season totals (minutes,
goals, assists, xG/xA, ICT components), price, and current availability status.
That makes it the practical source of per-player *profiles* for free.

Caveats worth knowing:
  * The API is undocumented and can change without notice. Parsing is defensive:
    missing fields become NaN instead of crashing.
  * It only exposes the current season; schedule the refresh job to build history.
  * Position is FPL's coarse 4-way split (wingers are 'MID', centre-forwards 'FWD').
  * Numeric fields such as xG and ICT arrive as strings.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..config import RAW_DIR
from ._http import JsonClient

log = logging.getLogger(__name__)

BASE_URL = "https://fantasy.premierleague.com/api"
ELEMENT_TYPE = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}  # anything else (e.g. managers) is dropped

# unified column -> FPL "element" field
STAT_MAP = {
    "minutes": "minutes", "starts": "starts", "goals": "goals_scored", "assists": "assists",
    "xg": "expected_goals", "xa": "expected_assists", "xgc": "expected_goals_conceded",
    "clean_sheets": "clean_sheets", "goals_conceded": "goals_conceded", "saves": "saves",
    "yellow_cards": "yellow_cards", "red_cards": "red_cards", "bonus": "bonus", "bps": "bps",
    "influence": "influence", "creativity": "creativity", "threat": "threat",
    "defensive_contribution": "defensive_contribution",
    "total_points": "total_points", "selected_by_percent": "selected_by_percent",
}


def parse_bootstrap(data: dict) -> pd.DataFrame:
    teams = {t["id"]: t for t in data.get("teams", [])}
    finished_gws = sum(1 for ev in data.get("events", []) if ev.get("finished"))
    rows = []
    for e in data.get("elements", []):
        position = ELEMENT_TYPE.get(e.get("element_type"))
        if position is None:
            continue
        team = teams.get(e.get("team"), {})
        cost = e.get("now_cost")
        row = {
            "player_id": f"fpl:{e.get('id')}",
            "name": f"{e.get('first_name', '')} {e.get('second_name', '')}".strip(),
            "web_name": e.get("web_name", ""),
            "club": team.get("name", ""),
            "club_short": team.get("short_name", ""),
            "position_group": position,
            "data_source": "fpl",
            "fpl_price_m": cost / 10 if cost is not None else np.nan,
            "status": e.get("status"),
            "news": e.get("news") or "",
            "birth_date": e.get("birth_date"),  # not always populated
            # Approximation of games played: finished gameweeks (ignores blank/double
            # gameweeks). Overwritten by real standings when football-data.org is used.
            "club_played": finished_gws,
        }
        for col, key in STAT_MAP.items():
            row[col] = e.get(key, np.nan)
        rows.append(row)

    df = pd.DataFrame(rows)
    for col in STAT_MAP:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    log.info("FPL: parsed %d players across %d finished gameweeks", len(df), finished_gws)
    return df


class FplClient:
    def __init__(self, client: JsonClient | None = None, max_age_hours: float = 1.0):
        self.http = client or JsonClient(
            BASE_URL,
            headers={"User-Agent": "pl-scouting-student-project/0.1 (educational)"},
            cache_dir=RAW_DIR / "fpl",
            max_age_hours=max_age_hours,
        )

    def players(self) -> pd.DataFrame:
        return parse_bootstrap(self.http.get("bootstrap-static/"))
