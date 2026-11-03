"""API-Football adapter and injury-history collection, tested offline.

Shapes follow real free-plan responses captured from the live API (sidelined:
[{type,start,end}], squads: [{team, players:[{id,name,age,position}]}], teams search:
[{team:{id,name,national}}]). The four-record Hojlund history is the real one; other
players, clubs and values here are hand-written test data.
"""
import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from scouting import enrich
from scouting.injury_history import collect_injury_history
from scouting.pipeline import build_players
from scouting.schema import ensure_columns
from scouting.sources._http import JsonClient
from scouting.sources.apifootball import (
    BASE_URL, ApiFootballClient, ApiFootballError, BudgetExhausted, DailyBudget, choose_team,
    is_injury_type, parse_sidelined, parse_team_search, summarise_injuries,
)

TODAY = pd.Timestamp("2026-10-30")

REAL_HISTORY = {"response": [
    {"type": "Hamstring", "start": "2024-07-29", "end": "2024-09-19"},
    {"type": "Muscle Injury", "start": "2024-02-23", "end": "2024-03-16"},
    {"type": "Muscle Injury", "start": "2023-11-13", "end": "2023-11-28"},
    {"type": "Back Injury", "start": "2023-07-01", "end": "2023-09-02"},
]}

# every reason string seen in a real 2024 season file
INJURY_REASONS = [
    "Knee Injury", "Thigh Injury", "Ankle Injury", "Muscle Injury", "Injury", "Calf Injury",
    "Hamstring Injury", "Groin Injury", "Leg Injury", "Shoulder Injury", "Knock", "Foot Injury",
    "Achilles Tendon Injury", "Back Injury", "Broken Leg", "Toe Injury", "Broken ankle",
    "Lower Back Injury", "Head Injury", "Finger Injury", "Chest Injury", "Concussion",
    "Hip Injury", "Surgery", "Broken nose", "Contusion", "Jaw injury", "Hand Injury", "Hamstring",
]
NON_INJURY_REASONS = [
    "Red Card", "Yellow Cards", "Lacking Match Fitness", "Illness", "Coach's decision", "Suspended",
    "Inactive", "Loan agreement", "Health problems", "International duty", "Personal Reasons", "Rest",
]


def test_every_real_reason_is_classified_correctly():
    assert all(is_injury_type(r) for r in INJURY_REASONS)
    assert not any(is_injury_type(r) for r in NON_INJURY_REASONS)
    assert not is_injury_type(None) and not is_injury_type("  ")


def test_parse_sidelined_and_window_summary():
    recs = parse_sidelined(REAL_HISTORY)
    assert len(recs) == 4 and recs[0]["start"] == pd.Timestamp("2024-07-29")
    # the 2023 back injury ended before the 3-year window opened (2023-10-30)
    assert summarise_injuries(recs, today=TODAY, window_years=3) == (3, 52 + 22 + 15)


def test_summary_clips_to_window_ignores_suspensions_and_handles_open_records():
    recs = parse_sidelined({"response": [
        {"type": "Knee Injury", "start": "2023-10-20", "end": "2023-11-09"},   # 10 of 20 days in window
        {"type": "Suspended", "start": "2025-01-01", "end": "2025-01-08"},     # not an injury
        {"type": "Knock", "start": "2026-10-20", "end": None},                 # still out: until today
    ]})
    assert summarise_injuries(recs, today=TODAY, window_years=3) == (2, 10 + 10)


def test_empty_history_means_zero_injuries():
    assert summarise_injuries(parse_sidelined({"response": []}), today=TODAY, window_years=3) == (0, 0)


def test_choose_team_prefers_exact_and_skips_national_sides():
    cands = parse_team_search({"response": [
        {"team": {"id": 746, "name": "Sunderland", "national": False}},
        {"team": {"id": 900, "name": "Sunderland W", "national": False}},
    ]})
    assert choose_team("Sunderland", cands)["id"] == 746
    spurs = [{"id": 47, "name": "Tottenham", "national": False}, {"id": 1, "name": "Tottenham W", "national": False}]
    assert choose_team("Spurs", spurs)["id"] == 47               # token-subset, only one fits
    both = [{"id": 1, "name": "Everton", "national": False}, {"id": 2, "name": "Everton", "national": False}]
    assert choose_team("Everton", both) is None                   # ambiguous: no guess
    nat = [{"id": 9, "name": "England", "national": True}]
    assert choose_team("England", nat) is None


# --- budget ---------------------------------------------------------------------------- #
def test_daily_budget_persists_across_runs_and_resets_next_day(tmp_path):
    path = tmp_path / "b.json"
    b1 = DailyBudget(path, 3, today=date(2026, 10, 30))
    assert [b1.take(), b1.take()] == [True, True]
    b2 = DailyBudget(path, 3, today=date(2026, 10, 30))          # a second run the same day
    assert b2.remaining == 1 and b2.take() and not b2.take()
    assert DailyBudget(path, 3, today=date(2026, 10, 31)).remaining == 3


# --- client over a fake session --------------------------------------------------------- #
class _Resp:
    def __init__(self, status, payload):
        self.status_code, self._payload, self.headers = status, payload, {}
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def get(self, url, params=None, timeout=None):
        key = (url.removeprefix(BASE_URL + "/"), tuple(sorted((params or {}).items())))
        self.calls.append(key)
        payload = self.routes.get(key)
        return _Resp(200, payload) if payload is not None else _Resp(404, {"message": f"no route {key}"})


def _client(tmp_path, routes, limit, day=date(2026, 10, 30)):
    http = JsonClient(BASE_URL, session=FakeSession(routes), cache_dir=tmp_path / "cache",
                      max_age_hours=24 * 21, sleep=lambda s: None)
    return ApiFootballClient(client=http, budget=DailyBudget(None, limit, today=day))


def test_errors_in_a_200_response_raise_and_are_not_cached(tmp_path):
    plan_error = {"errors": {"plan": "Free plans do not have access to this season"}, "response": []}
    c = _client(tmp_path, {("sidelined", (("player", 1),)): plan_error}, limit=5)
    with pytest.raises(ApiFootballError, match="Free plans"):
        c.sidelined(1)
    assert not c.http.is_cached("sidelined", {"player": 1})


def test_budget_stops_network_calls_but_cached_calls_are_free(tmp_path):
    route = {("sidelined", (("player", 1),)): REAL_HISTORY, ("sidelined", (("player", 2),)): REAL_HISTORY}
    c = _client(tmp_path, route, limit=1)
    c.sidelined(1)
    c.sidelined(1)                                   # cached: no budget spent
    with pytest.raises(BudgetExhausted):
        c.sidelined(2)
    assert len(c.http.session.calls) == 1


def test_missing_key_is_explained(monkeypatch):
    monkeypatch.delenv("API_FOOTBALL_KEY", raising=False)
    with pytest.raises(ApiFootballError, match="API_FOOTBALL_KEY"):
        ApiFootballClient()


# --- full collection flow ------------------------------------------------------------------ #
def _sq(*players):
    return {"response": [{"team": {}, "players": [{"id": i, "name": n, "age": 25, "position": "x"} for i, n in players]}]}


ROUTES = {
    ("teams", (("search", "manchester"),)): {"response": [
        {"team": {"id": 33, "name": "Manchester United", "national": False}},
        {"team": {"id": 50, "name": "Manchester City", "national": False}}]},
    ("teams", (("search", "sunderland"),)): {"response": [
        {"team": {"id": 746, "name": "Sunderland", "national": False}},
        {"team": {"id": 900, "name": "Sunderland W", "national": False}}]},
    ("players/squads", (("team", 33),)): _sq((153434, "W. Fish"), (162511, "S. Lammens"),
                                              (18885, "K. Darlow"), (284382, "Dermot William Mee")),
    ("players/squads", (("team", 746),)): _sq((1, "A. Other")),
    ("sidelined", (("player", 162511),)): {"response": []},
    ("sidelined", (("player", 153434),)): {"response": [
        {"type": "Ankle Injury", "start": "2024-08-10", "end": "2024-10-04"},
        {"type": "Suspended", "start": "2025-01-05", "end": "2025-01-12"}]},
    ("sidelined", (("player", 18885),)): {"response": REAL_HISTORY["response"] + [
        {"type": "Hamstring", "start": "2025-08-20", "end": "2025-09-19"}]},
    ("sidelined", (("player", 284382),)): {"response": [{"type": "Knock", "start": "2026-10-20", "end": None}]},
}


def _players():
    rows = [("fpl:1", "William Fish", "Man Utd", 500), ("fpl:2", "Senne Lammens", "Man Utd", 900),
            ("fpl:3", "Karl Darlow", "Man Utd", 100), ("fpl:4", "Dermot William Mee", "Man Utd", 50),
            ("fpl:5", "Jordan Placeholder", "Sunderland", 700)]
    return ensure_columns(pd.DataFrame({
        "player_id": [r[0] for r in rows], "name": [r[1] for r in rows],
        "web_name": [r[1].split()[-1] for r in rows], "club": [r[2] for r in rows],
        "club_short": ["MUN"] * 4 + ["SUN"], "position_group": "MID", "data_source": "fpl",
        "minutes": [r[3] for r in rows]}))


def test_collection_matches_abbreviated_names_and_computes_history(tmp_path):
    c = _client(tmp_path, ROUTES, limit=50)
    history, info, unmatched = collect_injury_history(_players(), c, today=TODAY)
    assert history.loc["fpl:1"].tolist() == [1, 55]                 # suspension excluded
    assert history.loc["fpl:2"].tolist() == [0, 0]                  # covered, genuinely clean
    assert history.loc["fpl:3"].tolist() == [4, 52 + 22 + 15 + 30]
    assert history.loc["fpl:4"].tolist() == [1, 10]                 # still-open record runs to today
    assert "fpl:5" not in history.index and unmatched["player_id"].tolist() == ["fpl:5"]
    assert info["matched_to_api_football"] == 4 and info["histories_pending"] == 0
    assert info["teams_found"] == 2 and info["newest_injury_end"] == "2025-09-19"
    assert "staleness_warning" in info                              # 13 months old


def test_unloaded_players_stay_nan_never_zero(tmp_path):
    c = _client(tmp_path, ROUTES, limit=50)
    history, _, _ = collect_injury_history(_players(), c, today=TODAY)
    out = enrich.apply_injury_history(_players(), history)
    assert out.set_index("player_id").loc["fpl:2", "injury_count_3y"] == 0
    assert np.isnan(out.set_index("player_id").loc["fpl:5", "injury_count_3y"])


def test_collection_resumes_across_days_within_the_budget(tmp_path):
    day1 = _client(tmp_path, ROUTES, limit=6)                        # 4 for teams/squads, 2 left
    history, info, _ = collect_injury_history(_players(), day1, today=TODAY)
    assert history.index.tolist() == ["fpl:2", "fpl:1"]              # most-played players first
    assert info["histories_pending"] == 2 and "budget" in info["stopped_early"]

    day2 = _client(tmp_path, ROUTES, limit=6, day=date(2026, 10, 31))   # same cache, fresh budget
    history, info, _ = collect_injury_history(_players(), day2, today=TODAY)
    assert len(history) == 4 and info["histories_pending"] == 0
    assert 6 - day2.budget.remaining == 2                            # only the two missing histories cost anything


def test_repeated_api_errors_stop_the_run(tmp_path):
    bad = {k: v for k, v in ROUTES.items() if k[0] != "sidelined"}   # sidelined -> 404 each time
    c = _client(tmp_path, bad, limit=50)
    history, info, _ = collect_injury_history(_players(), c, today=TODAY)
    assert history.empty and "repeated API errors" in info["stopped_early"]


# --- pipeline wiring ------------------------------------------------------------------------- #
class _FplStub:
    def players(self):
        return _players()


def test_pipeline_uses_api_football_and_reports(tmp_path, monkeypatch):
    from scouting import config as C
    monkeypatch.setattr(C, "TRANSFERMARKT_CSV", tmp_path / "none.csv")
    monkeypatch.setattr(C, "INJURIES_CSV", tmp_path / "none2.csv")
    players, meta, report = build_players("fpl", fpl_client=_FplStub(), use_api_football=True,
                                          af_client=_client(tmp_path, ROUTES, limit=50), today=TODAY)
    assert meta["enrichment"]["api_football_injuries"]["histories_loaded"] == 4
    assert any("newest injury record" in w for w in meta["warnings"])
    assert "api_football" in report["joined_source"].tolist()
    assert not any("no injury-history file" in w for w in meta["warnings"])


def test_pipeline_survives_missing_api_football_key_unless_strict(monkeypatch, tmp_path):
    from scouting import config as C
    monkeypatch.setattr(C, "TRANSFERMARKT_CSV", tmp_path / "none.csv")
    monkeypatch.setattr(C, "INJURIES_CSV", tmp_path / "none2.csv")
    monkeypatch.delenv("API_FOOTBALL_KEY", raising=False)
    _, meta, _ = build_players("fpl", fpl_client=_FplStub(), use_api_football=True, today=TODAY)
    assert any("API-Football injuries skipped" in w for w in meta["warnings"])
    with pytest.raises(ApiFootballError):
        build_players("fpl", fpl_client=_FplStub(), use_api_football=True, strict=True, today=TODAY)

def test_choose_team_prefers_english_club_over_same_named_foreign_club():
    cands = [{"id": 1, "name": "Bournemouth", "national": False, "country": "England"},
             {"id": 2, "name": "AFC Bournemouth", "national": False, "country": "Brazil"}]
    assert choose_team("Bournemouth", cands)["id"] == 1
    twins = [dict(c, name="Bournemouth", country="") for c in cands]
    assert choose_team("Bournemouth", twins) is None          # two identical names, no country info: no guess

def _t(i, name, country="England"):
    return {"id": i, "name": name, "national": False, "country": country}


def test_choose_team_real_search_results_for_the_four_failing_clubs():
    liverpool = [_t(40, "Liverpool"), _t(1847, "Liverpool W"), _t(2358, "Liverpool Montevideo", "Uruguay"),
                _t(7196, "Liverpool U21"), _t(7630, "City of Liverpool"), _t(8669, "AFC Liverpool"),
                _t(15636, "Liverpool U23")]
    bournemouth = [_t(35, "Bournemouth"), _t(8714, "Bournemouth FC"), _t(15513, "AFC Bournemouth W"),
                    _t(17794, "AFC Bournemouth U23")]
    coventry = [_t(1346, "Coventry"), _t(8755, "Coventry United"), _t(15645, "Coventry City U23"),
                _t(20092, "Coventry City U21"), _t(23170, "Coventry City U18")]
    ipswich = [_t(57, "Ipswich"), _t(11179, "Ipswich Knights", "Australia"), _t(11996, "Ipswich Wanderers"),
                _t(14551, "Ipswich U23"), _t(15422, "Ipswich Town W"), _t(22895, "Ipswich", "Australia")]
    assert choose_team("Liverpool", liverpool)["id"] == 40
    assert choose_team("Bournemouth", bournemouth)["id"] == 35
    assert choose_team("Coventry City", coventry)["id"] == 1346
    assert choose_team("Ipswich Town", ipswich)["id"] == 57