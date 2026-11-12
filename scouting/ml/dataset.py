"""Training data for the role models: qualified player-seasons turned into comparable features.

Each row of `player_seasons.csv` is one player's totals for one finished season. We keep seasons
with enough minutes, convert totals to shrunk per-90 rates (same recipe as the live engine), then
rank-normalise every feature WITHIN a season and population, so that 2017 and 2025 are comparable
and a feature's scale cannot dominate the clustering.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..features import add_per90, percentile_against

# Feature sets. ICT stats exist from 2016/17, xG/xA from 2022/23 (see sources/fpl_history.py).
FEATURE_SETS: dict[str, dict] = {
    "ict": {
        "min_year": 2016,
        "features": ["goals_p90", "assists_p90", "creativity_p90", "threat_p90",
                     "influence_p90", "yellow_cards_p90"],
    },
    "xg": {
        "min_year": 2022,
        "features": ["goals_p90", "assists_p90", "xg_p90", "xa_p90", "creativity_p90",
                     "threat_p90", "influence_p90", "yellow_cards_p90"],
    },
}

# Which positions are clustered together. GK is left out: FPL gives too little to describe them.
POPULATIONS: dict[str, dict[str, list[str]]] = {
    "by_position": {"DEF": ["DEF"], "MID": ["MID"], "FWD": ["FWD"]},
    "attackers_joint": {"DEF": ["DEF"], "MID+FWD": ["MID", "FWD"]},
    "outfield_joint": {"DEF+MID+FWD": ["DEF", "MID", "FWD"]},
}


def load_player_seasons(path, *, min_minutes: float = 900.0) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = {"player_id", "season", "season_start_year", "position_group", "minutes"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}; run `fetch-history` first")
    df = df[df["minutes"].fillna(0) >= min_minutes].copy()
    return df.reset_index(drop=True)


def build_features(
    seasons: pd.DataFrame,
    feature_set: str,
    positions: list[str],
    *,
    shrink_minutes: float = 270.0,
) -> tuple[pd.DataFrame, list[str]]:
    """Return (rows, feature columns): per-season percentile features for the given positions.

    Percentiles are computed inside each season over the rows in `positions`, so for a joint
    population a striker's threat is ranked against midfielders too, which is what lets roles
    like "winger" and "striker" separate.
    """
    spec = FEATURE_SETS[feature_set]
    df = seasons[seasons["position_group"].isin(positions) & (seasons["season_start_year"] >= spec["min_year"])]
    df = add_per90(df.reset_index(drop=True), shrink_minutes)
    feats = [f for f in spec["features"] if f in df and df[f].notna().any()]
    out = df[["player_id", "season", "season_start_year", "position_group", "minutes"] + feats].copy()
    for _, idx in out.groupby("season").groups.items():
        for f in feats:
            out.loc[idx, f"pct_{f}"] = percentile_against(out.loc[idx, f].to_numpy(float), out.loc[idx, f].to_numpy(float))
    pct = [f"pct_{f}" for f in feats]
    out = out.dropna(subset=pct).reset_index(drop=True)
    return out, pct
