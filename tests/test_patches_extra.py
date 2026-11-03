import pandas as pd
from scouting.schema import ensure_columns
from scouting.sources.footballdata import parse_squads
from scouting.sources.apifootball import parse_team_search
from tests.test_engine import _eligible


def test_all_empty_text_columns_accept_strings():
    df = ensure_columns(pd.DataFrame({"name": ["A"]}))
    for col in ("nationality", "status", "news"):
        assert df[col].dtype == object
        df.at[0, col] = "text"                    # used to raise under pandas 3


def test_parse_squads_returns_real_dates():
    squads = parse_squads({"teams": [{"name": "X FC", "tla": "X", "squad": [
        {"name": "A B", "dateOfBirth": "2000-01-31", "nationality": "Spain"}, {"name": "C D"}]}]})
    assert pd.api.types.is_datetime64_any_dtype(squads["birth_date"])
    assert squads.loc[0, "birth_date"] == pd.Timestamp("2000-01-31") and pd.isna(squads.loc[1, "birth_date"])


def test_parse_team_search_keeps_country():
    out = parse_team_search({"response": [{"team": {"id": 1, "name": "Arsenal", "national": False, "country": "England"}}]})
    assert out[0]["country"] == "England"

def test_explanations_never_list_a_stat_as_both_shared_and_different(engine):
    q = _eligible(engine, "MID").iloc[0]["name"]
    for _, r in engine.recommend(q, k=15).iterrows():
        stat = r["key_difference"].split(" (")[0]
        assert stat not in r["shared_strengths"], (r["name"], r["shared_strengths"], r["key_difference"])