import numpy as np
import pandas as pd

from scouting.categories import UNKNOWN, add_categories
from scouting.schema import ensure_columns

TODAY = pd.Timestamp("2026-10-30")


def _df(n=10, **cols):
    base = pd.DataFrame({
        "player_id": [f"p{i}" for i in range(n)], "name": [f"P{i}" for i in range(n)],
        "club": [f"C{i % 4}" for i in range(n)], "position_group": "MID",
        "minutes": 900.0, "club_played": 10.0, "club_position": 8.0,
        "birth_date": TODAY - pd.Timedelta(days=int(27 * 365.25)),
        "contract_end": TODAY + pd.Timedelta(days=int(2 * 365.25)),
        "market_value_eur_m": 20.0, "eligible": True,
    })
    for k, v in cols.items():
        base[k] = v
    return add_categories(ensure_columns(base), TODAY)


def test_age_band_boundaries():
    ages = [20.5, 21.0, 24.9, 25.0, 29.9, 30.0]
    births = [TODAY - pd.Timedelta(days=int(a * 365.25)) for a in ages]
    out = _df(len(ages), birth_date=births)
    assert out["age_band"].tolist() == [
        "Prospect (<21)", "Developing (21-24)", "Developing (21-24)",
        "Prime (25-29)", "Prime (25-29)", "Veteran (30+)",
    ]


def test_unknown_age_is_labelled_not_dropped():
    out = _df(3, birth_date=pd.NaT)
    assert (out["age_band"] == UNKNOWN).all()


def test_market_value_bands():
    out = _df(5, market_value_eur_m=[5, 10, 29.9, 30, 60])
    assert out["price_band"].tolist() == ["Budget", "Mid-range", "Mid-range", "Premium", "Elite"]
    assert (out["price_basis"] == "market_value").all()


def test_price_falls_back_to_fpl_when_market_value_coverage_is_low():
    out = _df(10, market_value_eur_m=[np.nan] * 9 + [50.0], fpl_price_m=np.linspace(4.5, 13, 10))
    assert (out["price_basis"] == "fpl_price").all()
    assert out["price_band"].iloc[0] == "Budget" and out["price_band"].iloc[-1] == "Elite"


def test_no_price_data_is_unknown():
    out = _df(4, market_value_eur_m=np.nan, fpl_price_m=np.nan)
    assert (out["price_band"] == UNKNOWN).all() and (out["price_basis"] == "none").all()


def test_difficulty_rises_with_value_all_else_equal():
    out = _df(10, market_value_eur_m=np.arange(1, 11) * 5.0)
    assert out["difficulty_score"].is_monotonic_increasing
    assert out["difficulty_score"].iloc[-1] > out["difficulty_score"].iloc[0] + 15


def test_difficulty_rises_with_contract_length():
    ends = [TODAY + pd.Timedelta(days=int(y * 365.25)) for y in (0.3, 1, 2, 4)]
    out = _df(4, contract_end=ends)
    assert out["difficulty_score"].is_monotonic_increasing


def test_difficulty_survives_missing_components_by_reweighting():
    out = _df(10, contract_end=pd.NaT, club_position=np.nan)
    assert out["difficulty_score"].notna().all()
    assert not out["difficulty_drivers"].str.contains("contract|club").any()


def test_difficulty_unknown_when_almost_nothing_is_known():
    out = _df(4, market_value_eur_m=np.nan, fpl_price_m=np.nan, contract_end=pd.NaT,
              club_position=np.nan, club_played=np.nan, birth_date=pd.NaT)
    assert (out["difficulty_label"] == UNKNOWN).all()


def test_injury_history_thresholds():
    out = _df(5, injury_days_3y=[100, 10, 40, 10, 0], injury_count_3y=[1, 4, 1, 2, 0])
    assert out["injury_risk"].tolist() == [
        "High risk", "High risk", "Moderate risk", "Moderate risk", "Low risk"
    ]
    assert (out["injury_basis"] == "history").all()


def test_availability_proxy_only_for_players_with_a_sample():
    out = _df(3, minutes=[880.0, 600.0, 100.0], eligible=[True, True, False])
    assert out["injury_risk"].tolist() == ["Low risk", "Moderate risk", "No injury data"]
    assert (out["injury_basis"].iloc[:2] == "availability_proxy").all()
    assert out["injury_basis"].iloc[2] == "none"


def test_proxy_needs_enough_club_matches():
    out = _df(2, club_played=3.0)
    assert (out["injury_risk"] == "No injury data").all()


def test_current_status_labels():
    out = _df(4, status=["a", "i", "d", None])
    assert out["current_status"].tolist() == ["Available", "Injured", "Doubtful", UNKNOWN]
