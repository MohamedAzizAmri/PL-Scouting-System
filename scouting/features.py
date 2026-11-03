"""Turning season totals into comparable per-90 profiles."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RATE_STATS


def resolve_min_minutes(minutes: pd.Series, setting: str | float = "auto") -> float:
    """Minimum minutes to be a recommendation candidate.

    "auto" = 30% of the most-played player's minutes, clamped to [180, 900]: strict
    enough late in the season to avoid cameo-driven noise, lenient enough in October
    that a full squad is still in the pool.
    """
    if setting != "auto":
        return float(setting)
    top = float(np.nanmax(minutes.to_numpy(dtype=float))) if len(minutes) else 0.0
    return float(min(900.0, max(180.0, 0.3 * top)))


def add_per90(df: pd.DataFrame, shrink_minutes: float = 270.0) -> pd.DataFrame:
    """Add `<stat>_p90` columns, shrunk toward the position-group average.

        rate = (total + prior_rate * k) / (nineties + k),   k = shrink_minutes / 90

    A player with 3 full matches played is pulled strongly toward his position's
    average; one with 30 matches barely moves. This stops a 40-minute cameo with a
    goal from looking like the best striker in the league.
    """
    out = df.copy()
    nineties = out["minutes"].astype(float) / 90.0
    k = shrink_minutes / 90.0
    for stat in RATE_STATS:
        col = f"{stat}_p90"
        if stat not in out.columns or out[stat].isna().all():
            out[col] = np.nan
            continue
        totals = pd.to_numeric(out[stat], errors="coerce")
        rate = pd.Series(np.nan, index=out.index, dtype=float)
        for _, idx in out.groupby("position_group").groups.items():
            t, n = totals.loc[idx], nineties.loc[idx]
            ok = t.notna() & (n > 0)
            denom = n[ok].sum()
            prior = float(t[ok].sum() / denom) if denom > 0 else 0.0
            rate.loc[idx] = (t + prior * k) / (n + k)
        out[col] = rate
    return out


def percentile_against(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Mid-rank percentile (0..1) of each value within `reference` (NaNs ignored)."""
    ref = np.sort(reference[~np.isnan(reference)])
    values = np.asarray(values, dtype=float)
    if ref.size == 0:
        return np.full(values.shape, np.nan)
    left = np.searchsorted(ref, values, side="left")
    right = np.searchsorted(ref, values, side="right")
    pct = (left + right) / 2.0 / ref.size
    return np.where(np.isnan(values), np.nan, pct)
