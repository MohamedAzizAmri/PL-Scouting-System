"""Scouting categories: age, market price, difficulty to buy, injury history.

Age and price are straightforward bands. The other two are *heuristics* built from
whatever data is available, and say so in the output:

  difficulty to buy  weighted blend of value, contract length, club status,
                     importance to the club and age (components that cannot be
                     computed are dropped and the rest re-weighted).
                     It is a transparent rule of thumb, not a validated model.

  injury history     real history if an injury file was supplied (count and days
                     missed over three years); otherwise an availability proxy
                     (minutes played / minutes available) which cannot tell an
                     injury from rotation, and is labelled `availability_proxy`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

UNKNOWN = "Unknown"
STATUS_LABELS = {
    "a": "Available", "d": "Doubtful", "i": "Injured",
    "s": "Suspended", "u": "Unavailable", "n": "Not in squad",
}
ORDERS = {
    "age_band": C.AGE_LABELS + [UNKNOWN],
    "price_band": C.PRICE_LABELS + [UNKNOWN],
    "difficulty_label": C.DIFFICULTY_LABELS + [UNKNOWN],
    "injury_risk": C.INJURY_LABELS,
}


def _band(series: pd.Series, bins: list[float], labels: list[str]) -> pd.Series:
    cut = pd.cut(series, bins=bins, labels=labels, right=False)
    return cut.astype(object).where(cut.notna(), UNKNOWN)


def add_categories(df: pd.DataFrame, today: pd.Timestamp | None = None) -> pd.DataFrame:
    out = df.copy()
    today = (today or pd.Timestamp.today()).normalize()

    # ---- age -------------------------------------------------------------- #
    out["age"] = ((today - out["birth_date"]).dt.days / 365.25).round(1)
    out["age_band"] = _band(out["age"], C.AGE_BINS, C.AGE_LABELS)

    # ---- price ------------------------------------------------------------ #
    coverage = out["market_value_eur_m"].notna().mean()
    if coverage >= C.MIN_MARKET_VALUE_COVERAGE:
        basis, value = "market_value", out["market_value_eur_m"]
    elif out["fpl_price_m"].notna().any():
        basis, value = "fpl_price", out["fpl_price_m"]
    else:
        basis, value = "none", pd.Series(np.nan, index=out.index)
    out["value_m"], out["price_basis"] = value, basis

    if basis == "market_value":
        out["price_band"] = _band(value, C.PRICE_BINS_EUR_M, C.PRICE_LABELS)
        value_component = value.rank(pct=True)
    elif basis == "fpl_price":
        pct = value.groupby(out["position_group"]).rank(pct=True)
        idx = np.digitize(pct.fillna(-1), C.PRICE_QUANTILES_FPL)
        out["price_band"] = np.where(pct.isna(), UNKNOWN, np.array(C.PRICE_LABELS, dtype=object)[idx])
        value_component = pct
    else:
        out["price_band"] = UNKNOWN
        value_component = pd.Series(np.nan, index=out.index)

    # ---- availability ------------------------------------------------------ #
    # minutes_share: how much of his club's available minutes he played (used for
    # "importance"). availability_rate: the same number, but only for players with a
    # meaningful sample. A fringe player with 60 minutes is not "injury prone", he
    # simply is not selected, so the proxy says nothing about him.
    played = out["club_played"].where(out["club_played"] >= C.MIN_CLUB_MATCHES_FOR_PROXY)
    minutes_share = (out["minutes"] / (played * 90.0)).clip(0, 1)
    eligible = out["eligible"] if "eligible" in out else pd.Series(True, index=out.index)
    out["availability_rate"] = minutes_share.where(eligible).round(3)

    # ---- difficulty to buy -------------------------------------------------- #
    years_left = ((out["contract_end"] - today).dt.days / 365.25).clip(lower=0)
    out["contract_years_left"] = years_left.round(2)
    n_clubs = max(int(out["club"].nunique()), 2)
    components = pd.DataFrame({
        "value": value_component,
        "contract": (years_left / C.CONTRACT_YEARS_FULL).clip(0, 1),
        "club": ((n_clubs - out["club_position"]) / (n_clubs - 1)).clip(0, 1),
        "importance": minutes_share,
        "age": ((C.AGE_EASIEST - out["age"]) / (C.AGE_EASIEST - C.AGE_HARDEST)).clip(0, 1),
    })
    weights = np.array([C.DIFFICULTY_WEIGHTS[c] for c in components.columns])
    present = components.notna().to_numpy()
    contrib = components.fillna(0.0).to_numpy() * weights
    avail_weight = (present * weights).sum(axis=1)
    score = np.where(avail_weight >= 0.4, contrib.sum(axis=1) / np.where(avail_weight == 0, 1, avail_weight) * 100, np.nan)
    out["difficulty_score"] = np.round(score, 1)
    out["difficulty_label"] = _band(out["difficulty_score"], [-np.inf, *C.DIFFICULTY_CUTS, np.inf], C.DIFFICULTY_LABELS)

    top2 = np.argsort(-contrib, axis=1)[:, :2]
    names = np.array(components.columns, dtype=object)
    comp_vals = components.to_numpy()
    out["difficulty_drivers"] = [
        ", ".join(names[j] for j in row if not np.isnan(comp_vals[i, j]) and comp_vals[i, j] >= 0.5)
        for i, row in enumerate(top2)
    ]

    # ---- injury history ----------------------------------------------------- #
    days, count = out["injury_days_3y"], out["injury_count_3y"]
    has_history = days.notna() & count.notna()
    history_risk = np.select(
        [(days > C.INJURY_HIGH_DAYS) | (count >= C.INJURY_HIGH_COUNT),
         (days >= C.INJURY_MODERATE_DAYS) | (count >= C.INJURY_MODERATE_COUNT)],
        ["High risk", "Moderate risk"], default="Low risk",
    )
    avail = out["availability_rate"]
    proxy_risk = np.select(
        [avail >= C.AVAILABILITY_LOW_RISK, avail >= C.AVAILABILITY_MODERATE_RISK],
        ["Low risk", "Moderate risk"], default="High risk",
    )
    out["injury_risk"] = np.where(has_history, history_risk, np.where(avail.notna(), proxy_risk, "No injury data"))
    out["injury_basis"] = np.where(has_history, "history", np.where(avail.notna(), "availability_proxy", "none"))

    out["current_status"] = [
        STATUS_LABELS.get(s, UNKNOWN) if isinstance(s, str) else UNKNOWN for s in out["status"]
    ]
    return out
