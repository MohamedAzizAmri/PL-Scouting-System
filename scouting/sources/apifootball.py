"""API-Football (api-sports.io) adapter, used for per-player injury history.

Endpoints used (response shapes confirmed from real free-plan responses):
  GET /teams?search=<name>      -> response[{team: {id, name, national, ...}}]
  GET /players/squads?team=<id> -> response[{team, players: [{id, name, age, position}]}]
  GET /sidelined?player=<id>    -> response[{type, start, end}]   (all clubs and competitions)

Things to know:
  * Free plan: 100 requests/day in total, and seasons 2022-2024 only for the
    season-based endpoints. `sidelined` is not season-based and returned records up to
    2025 in testing. Squads looked like a late-2025 snapshot, so very recent
    transfers can be missing. The pipeline reports the newest injury date it saw.
  * The API answers HTTP 200 with a non-empty `errors` field for plan or parameter
    problems, so errors are checked in the body, not just the status code.
  * Every call that is not already cached spends the daily budget (persisted on disk
    so repeated runs in one day cannot exceed it). Responses are cached for weeks.
  * Terms: not reviewed. Keep raw responses (data/raw/, git-ignored) out of public repos.

Auth: header `x-apisports-key`, from the API_FOOTBALL_KEY env var / .env.
"""
from __future__ import annotations

import json
import logging
import os
import re
from datetime import date
from pathlib import Path

import pandas as pd

from .. import config as C
from ..matching import club_similarity, normalize_club
from ._http import HttpError, JsonClient, RateLimiter

log = logging.getLogger(__name__)

BASE_URL = "https://v3.football.api-sports.io"


class ApiFootballError(RuntimeError):
    pass


class BudgetExhausted(ApiFootballError):
    pass


class DailyBudget:
    """Counts network requests per calendar day; persisted so separate runs share it."""

    def __init__(self, path: Path | None, limit: int, today: date | None = None):
        self.path, self.limit = path, limit
        self.day = str(today or date.today())
        self.used = 0
        if path is not None and path.exists():
            try:
                saved = json.loads(path.read_text())
                if saved.get("day") == self.day:
                    self.used = int(saved.get("used", 0))
            except (OSError, ValueError):
                pass

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)

    def take(self) -> bool:
        if self.used >= self.limit:
            return False
        self.used += 1
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"day": self.day, "used": self.used}))
        return True


# --------------------------------------------------------------------------- #
# Parsing (pure functions)
# --------------------------------------------------------------------------- #
_NON_INJURY = re.compile(
    r"\b(suspen\w*|cards?|ban|banned|illness|ill|sick|covid\w*|personal|international|loan|"
    r"rest|coach\w*|inactive|lacking|fitness|health|visa|transfer)\b",
    re.I,
)


def is_injury_type(text: object) -> bool:
    """True for physical injuries; False for suspensions, illness, loans, rest, etc.

    Anything not recognised as a non-injury counts as an injury ("Hamstring",
    "Muscle Injury", "Knock", "Broken ankle", "Surgery", ...). The pipeline reports the
    type counts it saw so a wrong call is visible.
    """
    return isinstance(text, str) and bool(text.strip()) and not _NON_INJURY.search(text)


def parse_team_search(payload: dict) -> list[dict]:
    out = []
    for item in payload.get("response") or []:
        team = item.get("team") or {}
        if team.get("id") is not None:
            out.append({"id": team["id"], "name": team.get("name", ""), "national": bool(team.get("national")),
                        "country": team.get("country") or item.get("country") or ""})
    return out


_SIDE_TEAM = re.compile(r"\b(U\s?\d{2}|W|II|Women|Reserves)\s*$", re.IGNORECASE)   # "Liverpool U21", "Ipswich Town W"


def choose_team(club: str, candidates: list[dict]) -> dict | None:
    """Pick the one first-team men's side for `club`; None if there is no unambiguous answer.

    Skips national sides and youth/women's/reserve teams ("Coventry City U21", "Liverpool W"),
    prefers English clubs, then an identical raw name ("Bournemouth" over "Bournemouth FC"),
    then a unique normalised name, then a unique token-subset match ("Coventry" for "Coventry City").
    """
    clubs = [c for c in candidates if not c["national"] and not _SIDE_TEAM.search(c["name"])]
    english = [c for c in clubs if str(c.get("country", "")).lower() == "england"]
    clubs = english or clubs          # "Ipswich" also exists in Australia; we only want England
    wanted = club.strip().lower()
    same_name = [c for c in clubs if c["name"].strip().lower() == wanted]
    if len(same_name) == 1:
        return same_name[0]
    target = normalize_club(club)
    exact = [c for c in clubs if normalize_club(c["name"]) == target]
    if len(exact) == 1:
        return exact[0]
    if exact:
        return None
    loose = [c for c in clubs if club_similarity(club, c["name"]) >= 0.9]
    return loose[0] if len(loose) == 1 else None


def parse_squad(payload: dict) -> list[dict]:
    rows = []
    for entry in payload.get("response") or []:
        for p in entry.get("players") or []:
            if p.get("id") is not None:
                rows.append({"af_id": p["id"], "name": p.get("name", ""), "age": p.get("age"),
                             "position": p.get("position")})
    return rows


def parse_sidelined(payload: dict) -> list[dict]:
    rows = []
    for r in payload.get("response") or []:
        start = pd.to_datetime(r.get("start"), errors="coerce")
        if pd.isna(start):
            continue
        rows.append({"type": r.get("type"), "start": start, "end": pd.to_datetime(r.get("end"), errors="coerce")})
    return rows


def summarise_injuries(records: list[dict], *, today: pd.Timestamp, window_years: int) -> tuple[int, int]:
    """(injury count, days missed) for injuries overlapping the last `window_years` years.

    Open-ended records (no end date) are treated as lasting until `today`.
    """
    window_start = today - pd.DateOffset(years=window_years)
    count = days = 0
    for r in records:
        if not is_injury_type(r["type"]):
            continue
        end = r["end"] if pd.notna(r["end"]) else today
        lo, hi = max(r["start"], window_start), min(end, today)
        if hi >= lo:
            count += 1
            days += int((hi - lo).days)
    return count, days


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #
class ApiFootballClient:
    def __init__(
        self,
        token: str | None = None,
        *,
        daily_budget: int = C.AF_DAILY_BUDGET,
        budget: DailyBudget | None = None,
        client: JsonClient | None = None,
        max_age_days: float = 21.0,
    ):
        if client is None:
            token = token or os.environ.get("API_FOOTBALL_KEY")
            if not token:
                raise ApiFootballError(
                    "No API key. Get a free one at https://www.api-football.com/ and set "
                    "API_FOOTBALL_KEY (environment variable or .env file)."
                )
            client = JsonClient(
                BASE_URL,
                headers={"x-apisports-key": token},
                limiter=RateLimiter(10, 60.0),
                cache_dir=C.RAW_DIR / "apifootball",
                max_age_hours=max_age_days * 24,
            )
        self.http = client
        self.budget = budget or DailyBudget(C.RAW_DIR / "apifootball" / "budget.json", daily_budget)

    def _get(self, path: str, params: dict) -> dict:
        if not self.http.is_cached(path, params) and not self.budget.take():
            raise BudgetExhausted(f"daily request budget ({self.budget.limit}) used up")
        try:
            data = self.http.get(path, params)
        except HttpError as exc:
            raise ApiFootballError(str(exc)) from exc
        if data.get("errors"):                    # 200 OK with an error body
            self.http.invalidate(path, params)    # never cache an error
            raise ApiFootballError(f"{path} {params}: {data['errors']}")
        return data

    def find_team(self, club: str) -> dict | None:
        # search on the first word: the API's own names are shorter than ours ("Tottenham")
        term = normalize_club(club).split()[0]
        if len(term) < 3:
            return None
        return choose_team(club, parse_team_search(self._get("teams", {"search": term})))

    def squad(self, team_id: int) -> list[dict]:
        return parse_squad(self._get("players/squads", {"team": team_id}))

    def sidelined(self, player_id: int) -> list[dict]:
        return parse_sidelined(self._get("sidelined", {"player": player_id}))
