import numpy as np
import pandas as pd
import pytest

from scouting.engine import AmbiguousPlayer, InsufficientData, PlayerNotFound, ScoutingEngine

TODAY = pd.Timestamp("2026-10-30")


def _eligible(engine, pos):
    df = engine.df
    return df[(df.position_group == pos) & df.eligible]


def test_recommendations_exclude_the_query_and_stay_in_position(engine):
    q = _eligible(engine, "MID").iloc[0]
    recs = engine.recommend(q["name"], k=10)
    assert len(recs) == 10
    assert q["name"] not in recs["name"].tolist()
    assert (recs["position_group"] == "MID").all()
    assert recs["similarity"].is_monotonic_decreasing
    assert recs["similarity"].between(0, 100).all()
    assert list(recs.index) == list(range(1, 11))


def test_recommend_is_deterministic(engine):
    name = _eligible(engine, "FWD").iloc[0]["name"]
    pd.testing.assert_frame_equal(engine.recommend(name, k=8), engine.recommend(name, k=8))


@pytest.mark.parametrize("pos,margin", [("DEF", 0.4), ("MID", 0.4), ("FWD", 0.4), ("GK", 0.1)])
def test_neighbours_share_the_ground_truth_playing_style(engine, archetypes, pos, margin):
    """Synthetic players have a known archetype. Top-5 neighbours should share it far
    more often than chance (chance = probability two random players share one)."""
    members = _eligible(engine, pos)
    arch = members["player_id"].map(archetypes)
    chance = float((arch.value_counts(normalize=True) ** 2).sum())
    by_name = engine.df.set_index("name")["player_id"].map(archetypes)
    hits = total = 0
    for i, row in members.iterrows():
        recs = engine.recommend(int(i), k=5)
        hits += int((by_name.loc[recs["name"]].to_numpy() == archetypes[row["player_id"]]).sum())
        total += len(recs)
    assert hits / total > chance + margin


def test_most_similar_player_is_closer_than_the_5th(engine):
    recs = engine.recommend(_eligible(engine, "DEF").iloc[3]["name"], k=5)
    assert recs["similarity"].iloc[0] > recs["similarity"].iloc[4]


def test_filters_are_honoured(engine):
    name = _eligible(engine, "MID").iloc[0]["name"]
    recs = engine.recommend(name, k=50, max_age=26, price_bands=["Budget", "Mid-range"],
                            max_difficulty="Hard", injury_risk=["Low risk", "Moderate risk"])
    assert not recs.empty
    assert (recs["age"] <= 26).all()
    assert recs["price_band"].isin(["Budget", "Mid-range"]).all()
    assert recs["difficulty_label"].isin(["Easy", "Moderate", "Hard"]).all()
    assert recs["injury_risk"].isin(["Low risk", "Moderate risk"]).all()


def test_exclude_club_and_min_similarity(engine):
    row = _eligible(engine, "FWD").iloc[0]
    recs = engine.recommend(row["name"], k=50, exclude_clubs=[row["club"].lower()], min_similarity=50)
    assert (recs["club"] != row["club"]).all()
    assert (recs["similarity"] >= 50).all()


def test_low_minute_players_are_not_recommended_unless_asked(engine):
    df = engine.df
    low = df[(df.position_group == "DEF") & ~df.eligible]["name"]
    assert len(low) > 0
    name = _eligible(engine, "DEF").iloc[0]["name"]
    assert not set(engine.recommend(name, k=500)["name"]) & set(low)
    assert set(engine.recommend(name, k=500, include_low_minutes=True)["name"]) & set(low)


def test_max_difficulty_validates_input(engine):
    with pytest.raises(ValueError):
        engine.recommend(_eligible(engine, "MID").iloc[0]["name"], max_difficulty="Impossible")


def test_group_by_category(engine):
    name = _eligible(engine, "MID").iloc[0]["name"]
    groups = engine.recommend_by_category(name, by="price_band", per_group=2)
    assert groups and all(len(g) <= 2 for g in groups.values())
    for label, part in groups.items():
        assert (part["price_band"] == label).all()
    with pytest.raises(ValueError):
        engine.recommend_by_category(name, by="shoe_size")


def test_player_lookup_errors_are_helpful(engine):
    some = _eligible(engine, "FWD").iloc[0]["name"]
    with pytest.raises(PlayerNotFound) as err:
        engine.resolve(some[:-1] + "q")          # typo
    assert some in err.value.suggestions
    with pytest.raises(PlayerNotFound):
        engine.resolve("Nobody At All")
    family = some.split()[-1]
    if (engine.df["name_norm"].str.split().map(lambda t: family.lower() in t)).sum() > 1:
        with pytest.raises(AmbiguousPlayer):
            engine.resolve(family.upper()[:-1] + family[-1])  # case-insensitive family name


def test_lookup_is_case_and_accent_insensitive(engine):
    name = _eligible(engine, "MID").iloc[0]["name"]
    assert engine.resolve(name.upper()) == engine.resolve(name.lower())


def test_profile_percentiles(engine):
    prof = engine.profile(_eligible(engine, "FWD").iloc[0]["name"])
    assert prof["percentile"].between(0, 100).all() and len(prof) >= 5


def test_query_below_minutes_threshold_still_works_but_is_flagged(engine):
    low = engine.df[(engine.df.position_group == "MID") & ~engine.df.eligible].iloc[0]
    assert not engine.query_summary(low["name"])["eligible"]
    assert len(engine.recommend(low["name"], k=3)) == 3


def test_engine_refuses_to_run_without_enough_playing_time(synth):
    players = synth.players.drop(columns=["archetype"]).assign(minutes=0)
    with pytest.raises(InsufficientData):
        ScoutingEngine(players, today=TODAY)


def test_engine_skips_a_stat_the_source_does_not_have(synth):
    players = synth.players.drop(columns=["archetype", "defensive_contribution"])
    eng = ScoutingEngine(players, today=TODAY)
    assert "defensive_contribution_p90" not in eng.groups["MID"].features
    assert len(eng.recommend(_eligible(eng, "MID").iloc[0]["name"], k=5)) == 5


def test_missing_values_do_not_produce_nan_similarity(synth):
    players = synth.players.drop(columns=["archetype"]).copy()
    players.loc[players.index[::7], "creativity"] = np.nan
    eng = ScoutingEngine(players, today=TODAY)
    recs = eng.recommend(_eligible(eng, "MID").iloc[0]["name"], k=10)
    assert recs["similarity"].notna().all()

def test_departed_players_are_hidden_by_default_but_can_be_included(engine):
    mids = _eligible(engine, "MID")
    q, gone = mids.iloc[0]["name"], mids.iloc[1]["name"]
    df = engine.df.copy()
    df.loc[df["name"] == gone, ["status", "news"]] = ["u", "Has joined Al Hilal permanently"]
    eng = ScoutingEngine(df)
    assert gone not in eng.recommend(q, k=len(df))["name"].tolist()
    assert gone in eng.recommend(q, k=len(df), include_departed=True)["name"].tolist()
    found = eng.search(gone)
    assert found["left_club_note"].iloc[0] == "Has joined Al Hilal permanently"
    assert eng.query_summary(gone)["departed"]