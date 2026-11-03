"""football-data.org v4 adapter.

What this API gives us for the Premier League (code "PL"):
  * standings    -> club position / points / games played       (free plan)
  * teams+squads -> player name, position, date of birth,
                    nationality and (when present) contract end  (squads are listed
                    as a paid-plan feature on their pricing page; the code copes
                    with an empty squad list and the pipeline carries on without it)

It does NOT provide market value, injuries or per-player performance stats, which
is why the pipeline also uses the FPL API and a Transfermarkt export.

Auth: header `X-Auth-Token`, taken from the FOOTBALL_DATA_API_KEY env var / .env.
"""
from __future__ import annotations

import logging
import os

import pandas as pd

from ..config import RAW_DIR
from ._http import HttpError, JsonClient, RateLimiter

log = logging.getLogger(__name__)

BASE_URL = "https://api.football-data.org/v4"


class FootballDataError(RuntimeError):
    pass


def _month_end_aware(value: object) -> pd.Timestamp:
    """Contract dates come as 'YYYY-MM' (month precision) or 'YYYY-MM-DD'."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return pd.NaT
    text = str(value)
    ts = pd.to_datetime(text, errors="coerce")
    if pd.isna(ts):
        return pd.NaT
    return ts + pd.offsets.MonthEnd(0) if len(text) == 7 else ts


def parse_standings(data: dict) -> pd.DataFrame:
    tables = data.get("standings") or []
    table = next((s["table"] for s in tables if s.get("type") == "TOTAL"), None)
    if table is None and tables:
        table = tables[0].get("table", [])
    rows = [
        {
            "fd_team_id": r["team"].get("id"),
            "club_name": r["team"].get("name"),
            "club_short_name": r["team"].get("shortName"),
            "tla": r["team"].get("tla"),
            "club_position": r.get("position"),
            "club_played": r.get("playedGames"),
            "club_points": r.get("points"),
        }
        for r in (table or [])
    ]
    return pd.DataFrame(rows, columns=[
        "fd_team_id", "club_name", "club_short_name", "tla",
        "club_position", "club_played", "club_points",
    ])


def parse_squads(data: dict) -> pd.DataFrame:
    rows = []
    for team in data.get("teams", []):
        for p in team.get("squad") or []:
            contract = p.get("contract") or {}
            rows.append({
                "name": p.get("name"),
                "club_name": team.get("name"),
                "tla": team.get("tla"),
                "birth_date": p.get("dateOfBirth"),
                "nationality": p.get("nationality"),
                "position": p.get("position"),
                "contract_end": _month_end_aware(contract.get("until")),
            })
    df = pd.DataFrame(rows, columns=[
        "name", "club_name", "tla", "birth_date", "nationality", "position", "contract_end",
    ])
    df["birth_date"] = pd.to_datetime(df["birth_date"], errors="coerce")
    return df


class FootballDataClient:
    def __init__(
        self,
        token: str | None = None,
        *,
        calls_per_minute: int = 10,
        max_age_hours: float = 12.0,
        client: JsonClient | None = None,
    ):
        token = token or os.environ.get("FOOTBALL_DATA_API_KEY")
        if client is None:
            if not token:
                raise FootballDataError(
                    "No API token. Register free at https://www.football-data.org/client/register "
                    "and set FOOTBALL_DATA_API_KEY (environment variable or .env file)."
                )
            client = JsonClient(
                BASE_URL,
                headers={"X-Auth-Token": token},
                limiter=RateLimiter(calls_per_minute, 60.0),
                cache_dir=RAW_DIR / "footballdata",
                max_age_hours=max_age_hours,
            )
        self.http = client

    def _get(self, path: str, params: dict | None = None) -> dict:
        try:
            return self.http.get(path, params)
        except HttpError as exc:
            if exc.status in (401, 403):
                raise FootballDataError(
                    f"{exc}. Check your token, and note some resources (e.g. squads) are "
                    "only available on paid plans."
                ) from exc
            raise FootballDataError(str(exc)) from exc

    def standings(self, competition: str = "PL", season: int | None = None) -> pd.DataFrame:
        params = {"season": season} if season else None
        return parse_standings(self._get(f"competitions/{competition}/standings", params))

    def squads(self, competition: str = "PL", season: int | None = None) -> pd.DataFrame:
        params = {"season": season} if season else None
        df = parse_squads(self._get(f"competitions/{competition}/teams", params))
        if df.empty:
            log.warning("football-data.org returned no squad data (plan limitation?)")
        return df
