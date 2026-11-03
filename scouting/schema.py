"""The unified player table every source is converted into.

Sources (FPL, football-data.org, Transfermarkt, synthetic) differ wildly in
shape; the engine only ever sees this schema.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import POSITION_GROUPS

IDENTITY_COLUMNS = [
    "player_id", "name", "web_name", "club", "club_short", "position_group", "data_source",
]

# Season totals. NaN means "this source does not provide it".
STAT_COLUMNS = [
    "minutes", "starts", "goals", "assists", "xg", "xa", "xgc", "clean_sheets",
    "goals_conceded", "saves", "yellow_cards", "red_cards", "bonus", "bps",
    "influence", "creativity", "threat", "defensive_contribution",
]

OPTIONAL_COLUMNS = [
    "nationality", "birth_date",
    "fpl_price_m",            # FPL price in GBP millions (a price *proxy*, not a transfer value)
    "market_value_eur_m",     # Transfermarkt market value in EUR millions
    "contract_end",           # contract expiry date
    "club_position", "club_played", "club_points",
    "status", "news",         # current availability, e.g. FPL status flag + text
    "injury_count_3y", "injury_days_3y",
    "total_points", "selected_by_percent",
]

DATE_COLUMNS = ["birth_date", "contract_end"]
TEXT_COLUMNS = ["nationality", "status", "news"]

def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with every schema column present (NaN where missing) and typed."""
    out = df.copy()
    for col in STAT_COLUMNS + OPTIONAL_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    for col in ("web_name", "club_short", "data_source"):
        if col not in out.columns:
            out[col] = ""
    for col in TEXT_COLUMNS:
         out[col] = out[col].astype(object).where(out[col].notna(), np.nan)
    for col in DATE_COLUMNS:
        out[col] = pd.to_datetime(out[col], errors="coerce")
    for col in STAT_COLUMNS + [
        "fpl_price_m", "market_value_eur_m", "club_position", "club_played", "club_points",
        "injury_count_3y", "injury_days_3y", "total_points", "selected_by_percent",
    ]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def validate(df: pd.DataFrame) -> None:
    missing = [c for c in IDENTITY_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"player table is missing required columns: {missing}")
    bad = set(df["position_group"].dropna().unique()) - set(POSITION_GROUPS)
    if bad:
        raise ValueError(f"unknown position groups: {sorted(bad)}")
    if df["player_id"].duplicated().any():
        raise ValueError("player_id must be unique")
    if df["minutes"].isna().all():
        raise ValueError("no minutes data: similarity needs per-player playing time")
