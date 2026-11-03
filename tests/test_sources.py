"""Source adapters, tested offline with hand-written payloads.

NOTE: the FPL and football-data.org payloads below are written by hand from their
documented shapes, NOT captured from the live APIs. They prove the parsers behave
as designed on that shape; they cannot prove the live shape is identical.
"""
import json

import numpy as np
import pandas as pd
import pytest

from scouting.sources._http import HttpError, JsonClient, RateLimiter
from scouting.sources.footballdata import (
    FootballDataClient, FootballDataError, parse_squads, parse_standings,
)
from scouting.sources.fpl import parse_bootstrap

BOOTSTRAP = {
    "events": [{"id": 1, "finished": True}, {"id": 2, "finished": True}, {"id": 3, "finished": False}],
    "teams": [{"id": 1, "name": "Arsenal", "short_name": "ARS"}, {"id": 2, "name": "Man City", "short_name": "MCI"}],
    "elements": [
        {"id": 10, "first_name": "Test", "second_name": "Striker", "web_name": "Striker", "team": 2,
         "element_type": 4, "now_cost": 135, "status": "a", "news": "", "minutes": 900,
         "goals_scored": 9, "assists": 2, "expected_goals": "7.84", "expected_assists": "1.10",
         "creativity": "120.5", "influence": "300.0", "threat": "850.0", "clean_sheets": 0,
         "selected_by_percent": "55.2", "total_points": 80},
        {"id": 11, "first_name": "Test", "second_name": "Keeper", "web_name": "Keeper", "team": 1,
         "element_type": 1, "now_cost": 55, "status": "i", "news": "Knee - back soon",
         "minutes": 810, "saves": 30, "goals_conceded": 8, "expected_goals_conceded": "9.2"},
        {"id": 12, "first_name": "Some", "second_name": "Manager", "web_name": "Manager", "team": 1,
         "element_type": 5, "now_cost": 50, "minutes": 0},
    ],
}


def test_fpl_parse_types_prices_and_filters():
    df = parse_bootstrap(BOOTSTRAP)
    assert list(df["position_group"]) == ["FWD", "GK"]              # the 'manager' row is dropped
    fwd = df.iloc[0]
    assert fwd["fpl_price_m"] == 13.5 and fwd["club"] == "Man City" and fwd["club_short"] == "MCI"
    assert fwd["xg"] == 7.84 and fwd["goals"] == 9                   # strings coerced to numbers
    assert fwd["player_id"] == "fpl:10" and fwd["name"] == "Test Striker"
    assert df.iloc[1]["status"] == "i" and df.iloc[1]["news"].startswith("Knee")
    assert (df["club_played"] == 2).all()                            # finished gameweeks


def test_fpl_parse_tolerates_missing_fields():
    df = parse_bootstrap(BOOTSTRAP)
    assert np.isnan(df.iloc[0]["defensive_contribution"]) and np.isnan(df.iloc[1]["threat"])


STANDINGS = {"standings": [
    {"type": "HOME", "table": []},
    {"type": "TOTAL", "table": [
        {"position": 1, "team": {"id": 57, "name": "Arsenal FC", "shortName": "Arsenal", "tla": "ARS"},
         "playedGames": 9, "points": 22},
        {"position": 2, "team": {"id": 65, "name": "Manchester City FC", "shortName": "Man City", "tla": "MCI"},
         "playedGames": 9, "points": 20},
    ]},
]}


def test_football_data_standings_uses_the_total_table():
    df = parse_standings(STANDINGS)
    assert df["tla"].tolist() == ["ARS", "MCI"] and df["club_position"].tolist() == [1, 2]
    assert df["club_played"].tolist() == [9, 9]


def test_football_data_squads_and_contract_month_precision():
    data = {"teams": [{"name": "Arsenal FC", "tla": "ARS", "squad": [
        {"name": "Test Player", "dateOfBirth": "2000-05-01", "position": "Offence", "nationality": "England",
         "contract": {"until": "2028-06"}},
        {"name": "No Contract", "dateOfBirth": "1999-01-01"},
    ]}]}
    df = parse_squads(data)
    assert df.loc[0, "contract_end"] == pd.Timestamp("2028-06-30")   # month-end, not the 1st
    assert pd.isna(df.loc[1, "contract_end"])


def test_football_data_empty_squads_gives_empty_frame():
    assert parse_squads({"teams": [{"name": "Arsenal FC", "squad": []}]}).empty


def test_football_data_requires_a_token(monkeypatch):
    monkeypatch.delenv("FOOTBALL_DATA_API_KEY", raising=False)
    with pytest.raises(FootballDataError, match="FOOTBALL_DATA_API_KEY"):
        FootballDataClient()


# --- HTTP client -------------------------------------------------------------- #
class _Resp:
    def __init__(self, status, payload=None, headers=None):
        self.status_code, self._payload, self.headers = status, payload, headers or {}
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


class _Session:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), 0

    def get(self, url, params=None, timeout=None):
        self.calls += 1
        return self.responses.pop(0)


def _client(responses, tmp_path, **kw):
    sleeps = []
    c = JsonClient("https://x.test/v4", session=_Session(responses), cache_dir=tmp_path,
                   sleep=sleeps.append, **kw)
    return c, sleeps


def test_http_retries_on_429_and_honours_retry_after(tmp_path):
    c, sleeps = _client([_Resp(429, {}, {"Retry-After": "7"}), _Resp(200, {"ok": 1})], tmp_path)
    assert c.get("a") == {"ok": 1} and 7.0 in sleeps


def test_http_caches_successful_responses(tmp_path):
    c, _ = _client([_Resp(200, {"v": 1})], tmp_path)
    assert c.get("a") == {"v": 1} and c.get("a") == {"v": 1}
    assert c.session.calls == 1


def test_http_cache_expires(tmp_path):
    c, _ = _client([_Resp(200, {"v": 1}), _Resp(200, {"v": 2})], tmp_path, max_age_hours=0)
    assert c.get("a") == {"v": 1}
    assert c.get("a") == {"v": 2}


def test_http_client_errors_are_not_retried(tmp_path):
    c, _ = _client([_Resp(403, {"message": "plan does not include this"})], tmp_path)
    with pytest.raises(HttpError, match="plan does not include this"):
        c.get("a")
    assert c.session.calls == 1


def test_http_gives_up_after_repeated_server_errors(tmp_path):
    c, _ = _client([_Resp(503, {})] * 4, tmp_path, retries=3)
    with pytest.raises(HttpError):
        c.get("a")
    assert c.session.calls == 4


def test_football_data_403_becomes_a_plan_hint(tmp_path):
    http, _ = _client([_Resp(403, {"message": "restricted"})], tmp_path)
    with pytest.raises(FootballDataError, match="paid plans"):
        FootballDataClient(client=http).squads()


def test_rate_limiter_blocks_the_eleventh_call_in_a_minute():
    now = [0.0]
    slept = []

    def sleep(s):
        slept.append(s)
        now[0] += s

    lim = RateLimiter(10, 60.0, clock=lambda: now[0], sleep=sleep)
    for _ in range(10):
        lim.wait()
        now[0] += 1.0
    lim.wait()                                    # 11th call: must wait for the window to clear
    assert len(slept) == 1 and 49.0 <= slept[0] <= 51.0

def test_join_squads_fills_text_nationality_and_string_birth_dates():
    import pandas as pd
    from scouting.enrich import join_squads
    from scouting.schema import ensure_columns
    from scouting.sources.footballdata import parse_squads
    players = ensure_columns(pd.DataFrame({
        "player_id": [1], "name": ["Rodri Hernandez"], "web_name": ["Rodri"], "club": ["Man City"],
        "club_short": ["MCI"], "position_group": ["MID"], "data_source": ["fpl"], "minutes": [900]}))
    squads = parse_squads({"teams": [{"name": "Manchester City FC", "tla": "MCI", "squad": [
        {"name": "Rodri Hernandez", "dateOfBirth": "1996-06-22", "nationality": "Spain", "position": "Midfield"}]}]})
    out, info, _ = join_squads(players, squads)
    assert out.loc[0, "nationality"] == "Spain"
    assert out.loc[0, "birth_date"] == pd.Timestamp("1996-06-22")
