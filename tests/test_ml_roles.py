import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("joblib")
import joblib  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402

from scouting.cli import main  # noqa: E402
from scouting.ml.roles import fit_roles, profile_table, training_memberships  # noqa: E402
from scouting.sources import synthetic  # noqa: E402

TOTALS = ["minutes", "starts", "goals", "assists", "xg", "xa", "xgc", "clean_sheets", "goals_conceded",
          "saves", "yellow_cards", "red_cards", "influence", "creativity", "threat", "defensive_contribution"]
TODAY = pd.Timestamp("2026-10-30")


@pytest.fixture(scope="module")
def seasons():
    frames = []
    for year, seed in [(2022, 11), (2023, 12), (2024, 13), (2025, 14)]:
        p = synthetic.generate(seed, today=TODAY).players.copy()
        p[TOTALS] = p[TOTALS].astype(float) * 3.5
        p["season"], p["season_start_year"] = f"{year}/{str(year + 1)[2:]}", year
        p["player_id"] = p["player_id"] + f"-{year}"
        frames.append(p)
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="module")
def current():
    return synthetic.generate(21, today=TODAY).players       # this season's (partial) totals


def test_fit_gives_soft_memberships_that_sum_to_one(seasons):
    model = fit_roles(seasons[seasons["minutes"] >= 900], "xg", "attackers_joint", 3)
    mem = training_memberships(model, seasons[seasons["minutes"] >= 900])
    probs = mem[[f"role_{i}" for i in range(3)]]
    assert np.allclose(probs.sum(axis=1), 1.0) and (mem["role_prob"] >= 1 / 3 - 1e-9).all()


def test_role_zero_is_the_largest_and_profile_table_is_percentiles(seasons):
    model = fit_roles(seasons[seasons["minutes"] >= 900], "xg", "attackers_joint", 3)
    for label, t in profile_table(model).items():
        assert t["share"].is_monotonic_decreasing and abs(t["share"].sum() - 1) < 0.01
        assert t.drop(columns="share").to_numpy().min() >= 0 and t.drop(columns="share").to_numpy().max() <= 100


def test_roles_line_up_with_known_midfield_styles(seasons):
    s = seasons[seasons["minutes"] >= 900]
    model = fit_roles(s, "xg", "by_position", 4)
    mem = training_memberships(model, s)
    mid = mem[mem["population"] == "MID"]
    truth = s.set_index(["player_id", "season"])["archetype"].loc[list(zip(mid["player_id"], mid["season"]))]
    assert adjusted_rand_score(truth.to_numpy(), mid["role"].to_numpy()) > 0.5


def test_assign_current_players_skips_goalkeepers_and_low_minutes(seasons, current):
    model = fit_roles(seasons[seasons["minutes"] >= 900], "xg", "outfield_joint", 4)
    out = model.assign(current, min_minutes=270)
    gk = current["position_group"] == "GK"
    assert out.loc[gk, "role"].isna().all()
    few = current["minutes"] < 270
    assert out.loc[few, "role"].isna().all()
    ok = out["role"].notna()
    assert ok.sum() > 100 and np.allclose(out.loc[ok, [f"role_{i}" for i in range(4)]].sum(axis=1), 1.0)


def test_current_assignment_matches_archetype_better_than_chance(seasons, current):
    model = fit_roles(seasons[seasons["minutes"] >= 900], "xg", "by_position", 4)
    out = model.assign(current, min_minutes=400)
    mid = (current["position_group"] == "MID") & out["role"].notna()
    assert adjusted_rand_score(current.loc[mid, "archetype"], out.loc[mid, "role"]) > 0.3


def test_too_few_rows_is_a_clear_error(seasons):
    with pytest.raises(ValueError, match="need at least"):
        fit_roles(seasons[seasons["minutes"] >= 900].head(30), "xg", "outfield_joint", 4)


def test_cli_train_roles_saves_a_reloadable_model(tmp_path, seasons, current, capsys):
    sf, pf, mf = tmp_path / "s.csv", tmp_path / "p.csv", tmp_path / "roles.joblib"
    seasons.to_csv(sf, index=False)
    current.drop(columns=["archetype"]).to_csv(pf, index=False)
    code = main(["train-roles", "--seasons", str(sf), "--data", str(pf), "--features", "xg",
                 "--grouping", "attackers_joint", "--k", "3", "--model-out", str(mf)])
    out = capsys.readouterr().out
    assert code == 0 and "bootstrap stability" in out and "current season:" in out
    assert joblib.load(mf).k == 3


# --- step 2c: roles in the data, the engine, the CLI and the pipeline ------------------------- #
from scouting import config as C  # noqa: E402
from scouting.engine import ScoutingEngine  # noqa: E402
from scouting.ml.roles import apply_roles  # noqa: E402
from scouting.pipeline import build_players  # noqa: E402

NAMES = ["Creators", "Wide defenders", "Centre-backs", "Strikers"]


@pytest.fixture(scope="module")
def named_model(seasons):
    return fit_roles(seasons[seasons["minutes"] >= 900], "xg", "outfield_joint", 4, names=NAMES)


def test_names_must_match_k(seasons):
    with pytest.raises(ValueError, match="expected 4 role names"):
        fit_roles(seasons[seasons["minutes"] >= 900], "xg", "outfield_joint", 4, names=["only", "three", "names"])


def test_apply_roles_adds_human_names_and_keeps_other_columns(named_model, current):
    out = apply_roles(current, named_model)
    assert set(out["role"].dropna()) <= set(NAMES) and out["role"].notna().sum() > 100
    assert (out.loc[out["role"].notna(), "role_prob"] > 0).all()
    assert out["name"].tolist() == current["name"].tolist()


def test_engine_filters_by_role_and_rejects_the_filter_when_no_roles_exist(named_model, current):
    base = current.drop(columns=["archetype"])
    with pytest.raises(ValueError, match="no roles in the data"):
        ScoutingEngine(base, today=TODAY).recommend(base.iloc[0]["name"], roles=["Creators"])
    eng = ScoutingEngine(apply_roles(base, named_model), today=TODAY)
    who = eng.df[eng.df["eligible"] & (eng.df["role"] == "Strikers")]["name"].iloc[0]
    recs = eng.recommend(who, k=5, roles=["Strikers"])
    assert len(recs) > 0 and (recs["role"] == "Strikers").all()
    assert "role" in eng.query_summary(who).index


def test_cli_assign_roles_rewrites_players_csv(tmp_path, named_model, current, capsys):
    pf, mf = tmp_path / "p.csv", tmp_path / "roles.joblib"
    current.drop(columns=["archetype"]).to_csv(pf, index=False)
    joblib.dump(named_model, mf)
    assert main(["assign-roles", "--data", str(pf), "--model", str(mf)]) == 0
    again = pd.read_csv(pf)
    assert again["role"].notna().sum() > 100 and set(again["role"].dropna()) <= set(NAMES)
    assert main(["assign-roles", "--data", str(pf), "--model", str(mf)]) == 0     # idempotent
    assert "Added roles" in capsys.readouterr().out


def test_pipeline_applies_a_saved_model_and_survives_a_broken_one(tmp_path, named_model, monkeypatch):
    from test_pipeline import _FplStub   # reuse the offline FPL payload
    good, bad = tmp_path / "good.joblib", tmp_path / "bad.joblib"
    joblib.dump(named_model, good)
    bad.write_text("not a model")
    monkeypatch.setattr(C, "ROLE_MODEL", good)
    players, meta, _ = build_players("fpl", fpl_client=_FplStub(), today=TODAY)
    assert "role" in players.columns and meta["enrichment"]["roles"]["k"] == 4
    monkeypatch.setattr(C, "ROLE_MODEL", bad)
    players, meta, _ = build_players("fpl", fpl_client=_FplStub(), today=TODAY)
    assert any("role model not applied" in w for w in meta["warnings"]) and len(players) > 0

def test_named_roles_print_examples_and_support_secondary_roles(tmp_path, seasons, current, capsys):
    from scouting.ml.roles import apply_roles, fit_roles, training_memberships

    names = ["Creative attackers", "Full-backs and holding midfielders", "Centre-backs", "Goal-threat forwards"]
    s = seasons[seasons["minutes"] >= 900]
    model = fit_roles(s, "ict", "outfield_joint", 4, names=names)
    mem = training_memberships(model, s)
    assert "raw_role" in mem.columns and (mem["raw_role"] == "role_0").any()
    assert model.display_key("role_0") == "role_0 (Creative attackers)"

    # Verify CLI train-roles with --names prints non-empty historical examples
    sf, pf, mf = tmp_path / "s.csv", tmp_path / "p.csv", tmp_path / "roles.joblib"
    seasons.to_csv(sf, index=False)
    current.drop(columns=["archetype"]).to_csv(pf, index=False)
    code = main([
        "train-roles", "--seasons", str(sf), "--data", str(pf),
        "--features", "ict", "--grouping", "outfield_joint", "--k", "4",
        "--names", "|".join(names), "--model-out", str(mf),
    ])
    out = capsys.readouterr().out
    assert code == 0
    assert "role_0 (Creative attackers) examples: " in out
    # Ensure examples are not blank after the colon
    for line in out.splitlines():
        if "examples:" in line:
            assert len(line.split("examples:")[1].strip()) > 0


def test_evaluate_consecutive_seasons_benchmarks_methods(seasons):
    from scouting.ml.evaluate import evaluate_consecutive_seasons

    # Strip the "-YYYY" suffix added by the `seasons` fixture so player_ids link across years
    linked = seasons.copy()
    linked["player_id"] = linked["player_id"].str.replace(r"-\d{4}$", "", regex=True)

    summary, diag = evaluate_consecutive_seasons(linked, feature_set="ict", k_roles=4, min_minutes=900)
    assert set(summary["method"]) >= {"heuristic_pos", "hybrid_gmm", "pca_space"}
    assert diag["consecutive_pairs"] > 50
    assert 0.0 <= diag["role_retention_rate"] <= 1.0