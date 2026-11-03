import numpy as np
import pandas as pd

from scouting.features import add_per90, percentile_against, resolve_min_minutes


def _players():
    return pd.DataFrame({
        "position_group": ["FWD"] * 4,
        "minutes": [3000, 3000, 90, 0],
        "goals": [10, 20, 1, 0],
    })


def test_shrinkage_pulls_small_samples_to_the_position_average():
    out = add_per90(_players().assign(xg=[10, 20, 1, 0]), shrink_minutes=270)
    prior = 31 / (6090 / 90)               # total goals / total nineties, position-wide
    cameo = out.loc[2, "goals_p90"]
    raw_cameo = 1 / 1.0                    # 1 goal in 90 minutes = 1.0 per 90 unshrunk
    assert abs(cameo - prior) < abs(raw_cameo - prior) * 0.35
    # a full-season player barely moves from his own rate
    assert abs(out.loc[1, "goals_p90"] - 20 / (3000 / 90)) < 0.1


def test_zero_minutes_gets_the_prior_not_a_crash():
    out = add_per90(_players(), shrink_minutes=270)
    assert np.isfinite(out.loc[3, "goals_p90"])


def test_missing_stat_column_becomes_nan():
    out = add_per90(_players(), shrink_minutes=270)
    assert out["saves_p90"].isna().all()


def test_percentile_midrank():
    ref = np.array([1.0, 2.0, 3.0, 4.0])
    assert percentile_against(ref, np.array([0.0, 2.5, 5.0])).tolist() == [0.0, 0.5, 1.0]
    assert percentile_against(ref, np.array([2.0]))[0] == 0.375     # (1 below + half of 1 tie) / 4
    assert np.isnan(percentile_against(ref, np.array([np.nan]))[0])


def test_auto_min_minutes_is_clamped():
    assert resolve_min_minutes(pd.Series([90, 40]), "auto") == 180          # floor (very early season)
    assert resolve_min_minutes(pd.Series([810, 300]), "auto") == 243        # ~30% of leader
    assert resolve_min_minutes(pd.Series([3420, 100]), "auto") == 900       # cap (full season)
    assert resolve_min_minutes(pd.Series([3420]), 500) == 500               # explicit wins
