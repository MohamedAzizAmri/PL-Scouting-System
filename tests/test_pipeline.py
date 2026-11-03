import json

import pandas as pd
import pytest

from scouting import config as C
from scouting import enrich
from scouting.pipeline import build_players, refresh
from scouting.schema import ensure_columns
from scouting.sources.files import (
    import_transfermarkt, load_injuries, load_transfermarkt, normalise_transfermarkt,
)
from scouting.sources.footballdata import FootballDataError
from scouting.sources.fpl import parse_bootstrap

TODAY = pd.Timestamp("2026-10-30")

BOOTSTRAP = {
    "events": [{"id": i, "finished": True} for i in range(1, 10)],
    "teams": [{"id": 1, "name": "Arsenal", "short_name": "ARS"},
              {"id": 2, "name": "Man Utd", "short_name": "MUN"}],
    "elements": [
        {"id": 1, "first_name": "Martin", "second_name": "Odegaard", "web_name": "Odegaard", "team": 1,
         "element_type": 3, "now_cost": 85, "status": "a", "minutes": 800, "goals_scored": 3},
        {"id": 2, "first_name": "Bruno Miguel", "second_name": "Borges Fernandes", "web_name": "B.Fernandes",
         "team": 2, "element_type": 3, "now_cost": 90, "status": "a", "minutes": 780, "goals_scored": 4},
    ],
}


class _FplStub:
    def players(self):
        return parse_bootstrap(BOOTSTRAP)


class _FdBroken:
    def standings(self):
        raise FootballDataError("HTTP 403: not on your plan")


@pytest.fixture(autouse=True)
def _no_external_files(monkeypatch, tmp_path):
    monkeypatch.setattr(C, "TRANSFERMARKT_CSV", tmp_path / "absent_tm.csv")
    monkeypatch.setattr(C, "INJURIES_CSV", tmp_path / "absent_inj.csv")


def _base():
    return ensure_columns(parse_bootstrap(BOOTSTRAP))


# --- standings ------------------------------------------------------------------ #
def test_standings_join_by_tla_then_by_name():
    standings = pd.DataFrame({
        "tla": ["ARS", "XXX"], "club_name": ["Arsenal FC", "Manchester United FC"],
        "club_position": [1, 5], "club_played": [9, 9], "club_points": [22, 14],
    })
    out, info = enrich.join_standings(_base(), standings)
    assert out["club_position"].tolist() == [1, 5]       # Arsenal by TLA, Man Utd by name+alias
    assert info["clubs_matched"] == 2


def test_standings_unmatched_club_keeps_fallback_values():
    standings = pd.DataFrame({"tla": ["ARS"], "club_name": ["Arsenal FC"],
                              "club_position": [1], "club_played": [9], "club_points": [22]})
    out, info = enrich.join_standings(_base(), standings)
    assert pd.isna(out.loc[1, "club_position"]) and out.loc[1, "club_played"] == 9
    assert info["unmatched_clubs"] == ["Man Utd"]


# --- transfermarkt ---------------------------------------------------------------- #
TM_RAW = pd.DataFrame({
    "name": ["Martin Ødegaard", "Bruno Fernandes", "Someone Else"],
    "current_club_name": ["Arsenal Football Club", "Manchester United Football Club", "Elsewhere FC"],
    "date_of_birth": ["1998-12-17", "1994-09-08", "2000-01-01"],
    "contract_expiration_date": ["2028-06-30", "2027-06-30", None],
    "market_value_in_eur": [70_000_000, 45_000_000, 1_000_000],
    "current_club_domestic_competition_id": ["GB1", "GB1", "FR1"],
    "last_season": [2026, 2026, 2026],
})


def test_transfermarkt_join_fills_value_and_only_missing_dates():
    base = _base()
    base.loc[0, "birth_date"] = pd.Timestamp("1990-01-01")          # already known: must survive
    out, info, unmatched = enrich.join_transfermarkt(base, normalise_transfermarkt(TM_RAW))
    assert out["market_value_eur_m"].tolist() == [70.0, 45.0]
    assert out.loc[0, "birth_date"] == pd.Timestamp("1990-01-01")
    assert out.loc[1, "birth_date"] == pd.Timestamp("1994-09-08")
    assert info["match_rate"] == 1.0 and unmatched.empty


def test_transfermarkt_columns_are_matched_flexibly_and_errors_list_columns():
    alt = TM_RAW.rename(columns={"market_value_in_eur": "market_value", "current_club_name": "club"})
    assert normalise_transfermarkt(alt)["market_value_eur_m"].iloc[0] == 70.0
    with pytest.raises(ValueError, match="Columns found"):
        normalise_transfermarkt(TM_RAW.drop(columns=["market_value_in_eur"]))


def test_import_transfermarkt_keeps_premier_league_only(tmp_path):
    src, dst = tmp_path / "players.csv", tmp_path / "slim.csv"
    TM_RAW.to_csv(src, index=False)
    slim = import_transfermarkt(src, dst)
    assert len(slim) == 2 and "Someone Else" not in slim["name"].tolist()
    assert len(load_transfermarkt(dst)) == 2


# --- injuries ---------------------------------------------------------------------- #
def _injuries():
    return pd.DataFrame({
        "player": ["Martin Odegaard"] * 3 + ["Nobody Known"],
        "club": ["Arsenal"] * 3 + ["Elsewhere"],
        "from_date": ["2026-01-10", "2025-03-01", "2019-01-01", "2026-02-01"],
        "until_date": ["2026-01-30", "2025-03-11", "2019-02-01", "2026-02-05"],
    })


def test_injury_file_days_are_computed_from_dates(tmp_path):
    p = tmp_path / "inj.csv"
    _injuries().to_csv(p, index=False)
    df = load_injuries(p)
    assert df["days_out"].tolist() == [20.0, 10.0, 31.0, 4.0]


def test_injury_join_uses_a_three_year_window_and_zero_fills(tmp_path):
    p = tmp_path / "inj.csv"
    _injuries().to_csv(p, index=False)
    out, info, _ = enrich.join_injuries(_base(), load_injuries(p), today=TODAY)
    assert out.loc[0, "injury_count_3y"] == 2 and out.loc[0, "injury_days_3y"] == 30   # 2019 one ignored
    assert out.loc[1, "injury_count_3y"] == 0
    assert info["injury_records_unmatched"] == 1


def test_injury_file_needs_required_columns(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"who": ["x"]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="player"):
        load_injuries(p)


# --- pipeline ------------------------------------------------------------------------ #
def test_synthetic_pipeline_writes_all_outputs(tmp_path):
    path, meta = refresh("synthetic", out_dir=tmp_path, today=TODAY)
    df = pd.read_csv(path)
    assert len(df) == meta["n_players"] == 480 and "archetype" not in df.columns
    assert (df["data_source"] == "synthetic").all()
    saved = json.loads((tmp_path / "metadata.json").read_text())
    assert saved["coverage"]["market_value_eur_m"] == 1.0 and "injuries" in saved["enrichment"]
    assert (tmp_path / "unmatched.csv").exists()


def test_fpl_pipeline_degrades_gracefully_and_says_so():
    players, meta, _ = build_players("fpl", fpl_client=_FplStub(), today=TODAY)
    assert len(players) == 2
    joined = " ".join(meta["warnings"])
    assert "Transfermarkt" in joined and "injury" in joined


def test_broken_football_data_is_a_warning_unless_strict():
    players, meta, _ = build_players("fpl", fpl_client=_FplStub(), use_football_data=True,
                                     fd_client=_FdBroken(), today=TODAY)
    assert len(players) == 2 and any("football-data" in w for w in meta["warnings"])
    with pytest.raises(FootballDataError):
        build_players("fpl", fpl_client=_FplStub(), use_football_data=True,
                      fd_client=_FdBroken(), strict=True, today=TODAY)


def test_fpl_pipeline_with_transfermarkt_and_injuries(tmp_path):
    tm, inj = tmp_path / "tm.csv", tmp_path / "inj.csv"
    normalise_transfermarkt(TM_RAW).to_csv(tm, index=False)
    _injuries().to_csv(inj, index=False)
    players, meta, _ = build_players("fpl", fpl_client=_FplStub(), transfermarkt_path=tm,
                                     injuries_path=inj, today=TODAY)
    assert players["market_value_eur_m"].tolist() == [70.0, 45.0]
    assert meta["enrichment"]["transfermarkt"]["match_rate"] == 1.0
    assert players["injury_count_3y"].tolist() == [2, 0]


def test_unknown_source_is_rejected():
    with pytest.raises(ValueError, match="unknown source"):
        build_players("sofascore")

# --- transfermarkt: transfers and stale snapshots ------------------------------------- #
def _tm_all_leagues():
    return pd.DataFrame({
        "name": ["Ezri Konsa", "Tammy Abraham", "Someone Else", "Tammy Abraham"],
        "current_club_name": ["Aston Villa", "Besiktas", "Elsewhere FC", "Wrong Namesake FC"],
        "date_of_birth": ["1997-10-23", "1997-10-02", "2000-01-01", "1990-05-05"],
        "contract_expiration_date": ["2029-06-30", "2026-06-30", None, None],
        "market_value_in_eur": [30_000_000, 18_000_000, 1_000_000, 500_000],
        "current_club_domestic_competition_id": ["GB1", "TR1", "FR1", "FR1"],
        "last_season": [2025] * 4,
    })


def _movers():
    df = pd.DataFrame({
        "player_id": ["fpl:1", "fpl:2"], "name": ["Ezri Konsa Ngoyo", "Tammy Abraham"],
        "web_name": ["Konsa", "Abraham"], "club": ["Arsenal", "Aston Villa"],
        "birth_date": pd.to_datetime(["1997-10-23", "1997-10-02"]),
        "contract_end": pd.NaT, "market_value_eur_m": float("nan"),
    })
    return df


def test_transfermarkt_rescues_movers_by_birth_date_without_taking_old_contracts():
    out, info, unmatched = enrich.join_transfermarkt(_movers(), normalise_transfermarkt(_tm_all_leagues()))
    assert out["market_value_eur_m"].tolist() == [30.0, 18.0]      # Konsa changed club, Abraham came from abroad
    assert out["contract_end"].isna().all()                        # rescued players keep NO snapshot contract
    assert info["matched"] == 2 and info["matched_by_birth_date"] == 2 and unmatched.empty


def test_birth_date_rescue_refuses_wrong_dates_and_ambiguity():
    base = _movers()
    base.loc[1, "birth_date"] = pd.Timestamp("1999-01-01")          # same name, different person
    out, info, unmatched = enrich.join_transfermarkt(base, normalise_transfermarkt(_tm_all_leagues()))
    assert pd.isna(out.loc[1, "market_value_eur_m"]) and unmatched["name"].tolist() == ["Tammy Abraham"]
    twin = _tm_all_leagues()
    twin.loc[len(twin)] = ["Ezri K Konsa", "Other FC", "1997-10-23", None, 1_000_000, "ES1", 2025]
    out2, _, _ = enrich.join_transfermarkt(_movers(), normalise_transfermarkt(twin))
    assert pd.isna(out2.loc[0, "market_value_eur_m"])               # two candidates -> no guess


def test_import_transfermarkt_all_leagues_flags_premier_league(tmp_path):
    src, dst = tmp_path / "players.csv", tmp_path / "all.csv"
    _tm_all_leagues().to_csv(src, index=False)
    slim = import_transfermarkt(src, dst, "all")
    assert len(slim) == 4 and slim["in_premier_league"].tolist() == [True, False, False, False]
    assert load_transfermarkt(dst)["in_premier_league"].sum() == 1
