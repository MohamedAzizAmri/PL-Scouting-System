import numpy as np
import pandas as pd
import pytest

from scouting import cli
from scouting.sources._http import HttpError
from scouting.sources.fpl_history import (
    FplHistoryClient, blank_untracked_stats, collect_history, parse_history_past, season_start_year,
)

# shape taken from a real element-summary/10/ response (history_past, last entry)
REAL_ROW = {
    "season_name": "2025/26", "element_code": 198869, "start_cost": 55, "end_cost": 51, "total_points": 45,
    "minutes": 699, "goals_scored": 0, "assists": 1, "clean_sheets": 5, "goals_conceded": 6, "own_goals": 0,
    "penalties_saved": 0, "penalties_missed": 0, "yellow_cards": 0, "red_cards": 0, "saves": 0, "bonus": 3,
    "bps": 181, "influence": "154.8", "creativity": "133.4", "threat": "54.0", "ict_index": "34.3",
    "clearances_blocks_interceptions": 38, "recoveries": 28, "tackles": 15, "defensive_contribution": 53,
    "starts": 9, "expected_goals": "0.45", "expected_assists": "0.69", "expected_goal_involvements": "1.14",
    "expected_goals_conceded": "8.68",
}
OLD_ROW = {"season_name": "2019/20", "element_code": 198869, "end_cost": 45, "minutes": 1800, "goals_scored": 1,
           "assists": 2, "total_points": 80, "bps": 400, "starts": 20}


class _Stub:
    """Stands in for FplHistoryClient; ids in `broken` raise like a failing server."""
    def __init__(self, payloads, broken=()):
        self.payloads, self.broken, self.calls = payloads, set(broken), []

    def element_summary(self, element_id):
        self.calls.append(element_id)
        if element_id in self.broken:
            raise HttpError(500, "server error", f"element-summary/{element_id}/")
        return self.payloads.get(element_id, {"history_past": []})


def _players(n=3):
    return pd.DataFrame({"player_id": [f"fpl:{i}" for i in range(1, n + 1)], "position_group": ["MID"] * n})


def test_parse_real_row_converts_strings_and_keeps_identity():
    df = parse_history_past({"history_past": [REAL_ROW]}, "fpl:10", "DEF")
    r = df.iloc[0]
    assert r["player_id"] == "fpl:10" and r["position_group"] == "DEF" and r["season_start_year"] == 2025
    assert r["xg"] == 0.45 and r["threat"] == 54.0 and r["price_m"] == 5.1 and r["element_code"] == 198869


def test_missing_fields_become_nan_not_zero():
    r = parse_history_past({"history_past": [OLD_ROW]}, "fpl:10", "DEF").iloc[0]
    assert np.isnan(r["xg"]) and np.isnan(r["defensive_contribution"]) and r["minutes"] == 1800


def test_empty_history_gives_empty_frame_with_columns():
    df = parse_history_past({"history_past": []}, "fpl:1", "MID")
    assert df.empty and "minutes" in df.columns


def test_season_start_year():
    assert season_start_year("2025/26") == 2025 and np.isnan(season_start_year("garbage"))


def test_stat_that_is_zero_for_everyone_in_a_season_is_blanked():
    df = pd.DataFrame({"season": ["2020/21"] * 2 + ["2025/26"] * 2, "xg": [0.0, 0.0, 0.0, 3.2],
                       "xa": [np.nan, 0.0, 1.0, 0.0]})
    for col in ("xgc", "influence", "creativity", "threat", "defensive_contribution",
                "clearances_blocks_interceptions", "recoveries", "tackles", "bps"):
        df[col] = 1.0
    out, blanked = blank_untracked_stats(df)
    assert out.loc[:1, "xg"].isna().all() and out.loc[2:, "xg"].notna().all()   # real zeros survive in tracked seasons
    assert "2020/21:xg" in blanked and "2025/26:xg" not in blanked


def test_collect_history_combines_players_and_reports_failures():
    stub = _Stub({1: {"history_past": [REAL_ROW]}, 2: {"history_past": [OLD_ROW, REAL_ROW]}}, broken=[3])
    df, info = collect_history(_players(3), stub)
    assert len(df) == 3 and set(df["player_id"]) == {"fpl:1", "fpl:2"}
    assert info["players_ok"] == 2 and info["players_failed"] == 1 and info["failed_ids"] == ["fpl:3"]
    assert info["seasons"] == ["2019/20", "2025/26"]


def test_collect_history_stops_after_repeated_failures_and_respects_limit():
    stub = _Stub({}, broken=range(1, 20))
    _, info = collect_history(_players(12), stub, max_consecutive_errors=3)
    assert info["stopped_early"] and len(stub.calls) == 3
    stub2 = _Stub({})
    collect_history(_players(12), stub2, limit=4)
    assert stub2.calls == [1, 2, 3, 4]


def test_cli_fetch_history_writes_csv(tmp_path, monkeypatch):
    players = tmp_path / "players.csv"
    _players(2).to_csv(players, index=False)
    stub = _Stub({1: {"history_past": [REAL_ROW]}, 2: {"history_past": [REAL_ROW]}})
    monkeypatch.setattr(cli, "FplHistoryClient", lambda **kw: stub)
    out = tmp_path / "seasons.csv"
    assert cli.main(["fetch-history", "--data", str(players), "--output", str(out)]) == 0
    assert len(pd.read_csv(out)) == 2


def test_cli_fetch_history_fails_cleanly_when_nothing_downloads(tmp_path, monkeypatch):
    players = tmp_path / "players.csv"
    _players(2).to_csv(players, index=False)
    monkeypatch.setattr(cli, "FplHistoryClient", lambda **kw: _Stub({}, broken=[1, 2]))
    out = tmp_path / "seasons.csv"
    assert cli.main(["fetch-history", "--data", str(players), "--output", str(out)]) == 3
    assert not out.exists()


def test_real_client_is_polite_by_default():
    c = FplHistoryClient()
    assert c.http.limiter.max_calls == 60 and c.http.max_age_hours == 24 * 30
