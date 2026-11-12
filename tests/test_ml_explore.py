import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402

from scouting.cli import main  # noqa: E402
from scouting.ml.dataset import build_features, load_player_seasons  # noqa: E402
from scouting.ml.explore import evaluate_k, explore  # noqa: E402
from scouting.sources import synthetic  # noqa: E402

TOTALS = ["minutes", "starts", "goals", "assists", "xg", "xa", "xgc", "clean_sheets", "goals_conceded",
          "saves", "yellow_cards", "red_cards", "influence", "creativity", "threat", "defensive_contribution"]


@pytest.fixture(scope="module")
def seasons():
    """Three synthetic 'finished seasons': partial-season totals scaled up to look like full ones."""
    frames = []
    for year, seed in [(2022, 11), (2023, 12), (2024, 13)]:
        p = synthetic.generate(seed, today=pd.Timestamp("2026-10-30")).players.copy()
        p[TOTALS] = p[TOTALS].astype(float) * 3.5
        p["season"], p["season_start_year"] = f"{year}/{str(year + 1)[2:]}", year
        p["player_id"] = p["player_id"] + f"-{year}"
        frames.append(p)
    return pd.concat(frames, ignore_index=True)


def test_load_player_seasons_filters_minutes_and_checks_columns(tmp_path, seasons):
    f = tmp_path / "s.csv"
    seasons.to_csv(f, index=False)
    out = load_player_seasons(f, min_minutes=900)
    assert (out["minutes"] >= 900).all() and 0 < len(out) < len(seasons)
    seasons.drop(columns=["season"]).to_csv(f, index=False)
    with pytest.raises(ValueError, match="fetch-history"):
        load_player_seasons(f)


def test_features_are_percentiles_within_each_season(seasons):
    data, pct = build_features(seasons[seasons["minutes"] >= 900], "xg", ["MID"])
    assert data[pct].min().min() >= 0 and data[pct].max().max() <= 1
    assert data.groupby("season")[pct[0]].mean().between(0.4, 0.6).all()    # ~uniform in every season


def test_feature_set_respects_start_year(seasons):
    s = seasons[seasons["minutes"] >= 900].copy()
    s.loc[s["season_start_year"] == 2022, "season_start_year"] = 2015
    data, _ = build_features(s, "ict", ["MID"])
    assert 2015 not in set(data["season_start_year"])                   # before the ICT era


def test_clustering_recovers_known_midfield_roles(seasons):
    """The synthetic midfielders have four known styles; k=4 should line up with them."""
    data, pct = build_features(seasons[seasons["minutes"] >= 900], "xg", ["MID"])
    arche = seasons.set_index(["player_id", "season"])["archetype"]
    truth = arche.loc[list(zip(data["player_id"], data["season"]))].to_numpy()
    labels = KMeans(n_clusters=4, n_init=10, random_state=0).fit_predict(data[pct].to_numpy())
    assert adjusted_rand_score(truth, labels) > 0.5


def test_evaluate_k_reports_all_metrics_and_stability_is_sensible(seasons):
    data, pct = build_features(seasons[seasons["minutes"] >= 900], "xg", ["MID"])
    X = data[pct].to_numpy()
    good = evaluate_k(X, 4, n_pairs=4)
    assert set(good) == {"k", "bic", "silhouette", "stability", "min_share"}
    assert good["stability"] > 0.5
    noise = evaluate_k(np.random.default_rng(0).random((300, 6)), 6, n_pairs=4)
    assert noise["stability"] < good["stability"]                       # structureless data is less stable


def test_explore_skips_populations_that_are_too_small(seasons):
    res = explore(seasons[seasons["minutes"] >= 900].head(40), feature_sets=["xg"], populations=["by_position"],
                  ks=range(2, 4), min_rows=60)
    assert res["note"].notna().all()


def test_cli_ml_explore_prints_and_saves(tmp_path, seasons, capsys):
    f, out = tmp_path / "s.csv", tmp_path / "res.csv"
    seasons.to_csv(f, index=False)
    code = main(["ml-explore", "--seasons", str(f), "--features", "xg", "--grouping", "attackers_joint",
                 "--max-k", "4", "--n-pairs", "2", "--output", str(out)])
    assert code == 0 and "silhouette" in capsys.readouterr().out
    assert set(pd.read_csv(out)["population"]) == {"DEF", "MID+FWD"}
