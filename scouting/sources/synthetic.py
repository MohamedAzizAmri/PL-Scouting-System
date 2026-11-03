"""Synthetic Premier League-style data with FICTIONAL clubs and players.

Used for the demo, the tests, and development when no API access is available.
Nothing here describes real people: names are assembled from made-up syllables and
every row carries data_source == "synthetic".

Each player is generated from a playing-style *archetype* (e.g. "poacher",
"destroyer"), which gives the tests a ground truth: a good similarity engine
should recommend players of the same archetype.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

CLUBS = [
    ("Alderbrook", "ALD"), ("Ashcombe Town", "ASH"), ("Brackenridge", "BRK"), ("Cardew United", "CRD"),
    ("Dunmere", "DUN"), ("Eastholt", "EAS"), ("Fenwick City", "FEN"), ("Garrowby", "GAR"),
    ("Harwick Town", "HAR"), ("Ironvale", "IRO"), ("Kestrel Park", "KES"), ("Larkhall", "LAR"),
    ("Marlowe Athletic", "MAR"), ("Northgate", "NOR"), ("Oakhurst", "OAK"), ("Pennywell", "PEN"),
    ("Quarrydale", "QUA"), ("Redmoor", "RED"), ("Stonehaven United", "STO"), ("Thornley", "THO"),
]

_STATS = ("xg", "xa", "creativity", "threat", "influence", "dc", "yc", "cs", "xgc", "saves", "gc")


def _arch(**kw: float) -> dict[str, float]:
    return {k: kw.get(k, 0.0) for k in _STATS}


# per-90 means for a league-average player of that archetype
ARCHETYPES: dict[str, dict[str, dict[str, float]]] = {
    "GK": {
        "shot-stopper": _arch(saves=3.8, xgc=1.35, gc=1.35, cs=0.28, influence=13, creativity=1, yc=0.03),
        "sweeper-keeper": _arch(saves=2.4, xgc=0.95, gc=1.0, cs=0.38, influence=20, creativity=3, yc=0.02),
    },
    "DEF": {
        "stopper": _arch(dc=10.5, cs=0.32, xgc=1.1, xg=0.05, xa=0.03, creativity=4, threat=6, influence=22, yc=0.24),
        "ball-playing": _arch(dc=7.5, cs=0.36, xgc=1.0, xg=0.06, xa=0.07, creativity=14, threat=8, influence=20, yc=0.12),
        "attacking full-back": _arch(dc=5.5, cs=0.28, xgc=1.2, xg=0.10, xa=0.17, creativity=28, threat=22, influence=17, yc=0.15),
    },
    "MID": {
        "playmaker": _arch(xg=0.10, xa=0.24, creativity=32, threat=20, influence=16, dc=4.5, yc=0.10),
        "box-to-box": _arch(xg=0.14, xa=0.11, creativity=16, threat=25, influence=24, dc=8.0, yc=0.17),
        "destroyer": _arch(xg=0.04, xa=0.05, creativity=8, threat=7, influence=20, dc=11.5, yc=0.28),
        "advanced creator": _arch(xg=0.26, xa=0.20, creativity=28, threat=52, influence=14, dc=3.0, yc=0.08),
    },
    "FWD": {
        "poacher": _arch(xg=0.58, xa=0.08, creativity=8, threat=80, influence=12, dc=1.0, yc=0.08),
        "pressing target man": _arch(xg=0.40, xa=0.13, creativity=14, threat=55, influence=22, dc=2.5, yc=0.16),
        "creative forward": _arch(xg=0.30, xa=0.27, creativity=30, threat=52, influence=17, dc=1.5, yc=0.09),
    },
}
SQUAD_SHAPE = {"GK": 3, "DEF": 8, "MID": 8, "FWD": 5}
STARTERS = {"GK": 1, "DEF": 4, "MID": 4, "FWD": 2}
_POS_VALUE_ADJ = {"GK": -0.4, "DEF": -0.1, "MID": 0.1, "FWD": 0.3}
_FPL_PRICE_RANGE = {"GK": (4.0, 5.5), "DEF": (4.0, 6.5), "MID": (4.5, 12.5), "FWD": (4.5, 12.5)}
_INJURIES = ["hamstring", "knee ligament", "ankle", "calf", "groin", "muscle strain", "illness", "knock"]

_GIVEN = ["Tal", "Ven", "Kor", "Mar", "Del", "Bran", "Jor", "Ses", "Lio", "Nik",
          "Ras", "Teo", "Ulf", "Wen", "Yar", "Zan", "Ilo", "Dov", "Fen", "Gal"]
_GIVEN_END = ["an", "ek", "os", "iv", "or", "as", "en", "ul"]
_FAMILY = ["Brak", "Hol", "Marn", "Pell", "Quen", "Rask", "Sorv", "Tren", "Varn", "Wald",
           "Zeld", "Orm", "Kest", "Lund", "Drev"]
_FAMILY_END = ["ov", "ik", "sen", "ard", "ley", "man", "ric", "et"]


@dataclass
class SyntheticData:
    players: pd.DataFrame    # unified schema + `archetype` (ground truth, never a model input)
    injuries: pd.DataFrame   # injury-history file: player, club, from_date, until_date, days_out, injury


def _names(rng: np.random.Generator, n: int) -> list[tuple[str, str]]:
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    while len(out) < n:
        given = rng.choice(_GIVEN) + rng.choice(_GIVEN_END)
        family = rng.choice(_FAMILY) + rng.choice(_FAMILY_END)
        if f"{given} {family}" in seen:
            continue
        seen.add(f"{given} {family}")
        out.append((given, family))
    return out


def generate(
    seed: int = 7,
    *,
    matches_played: int = 10,
    today: pd.Timestamp | None = None,
) -> SyntheticData:
    rng = np.random.default_rng(seed)
    today = (today or pd.Timestamp.today()).normalize()
    available = matches_played * 90.0

    # --- clubs & table ---------------------------------------------------------
    strength = np.sort(rng.normal(0, 1, len(CLUBS)))[::-1]
    points = np.round(matches_played * (1.35 + 0.5 * strength) + rng.normal(0, 1.5, len(CLUBS)))
    order = np.argsort(-points, kind="stable")
    position_of = {int(club_idx): rank + 1 for rank, club_idx in enumerate(order)}

    n_total = len(CLUBS) * sum(SQUAD_SHAPE.values())
    names = iter(_names(rng, n_total))
    rows: list[dict] = []
    injury_rows: list[dict] = []
    pid = 0

    for c_idx, (club, short) in enumerate(CLUBS):
        for pos, n_pos in SQUAD_SHAPE.items():
            archs = list(ARCHETYPES[pos])
            ability = rng.normal(0.4 * strength[c_idx], 1.0, n_pos)
            starter_rank = np.argsort(-ability)
            is_starter = np.zeros(n_pos, bool)
            is_starter[starter_rank[: STARTERS[pos]]] = True

            for j in range(n_pos):
                pid += 1
                given, family = next(names)
                arch_name = archs[int(rng.integers(len(archs)))]
                mu = ARCHETYPES[pos][arch_name]
                a = ability[j]
                q = np.exp(0.22 * a)                       # output quality factor
                noise = lambda: rng.lognormal(0, 0.12)     # noqa: E731

                proneness = rng.gamma(1.5, 0.7)
                missed = min(0.9, rng.exponential(0.07 * proneness))
                if is_starter[j]:
                    share = rng.beta(9, 1.4) * (1 - missed)
                else:
                    share = rng.beta(1.3, 5.0) if rng.random() < 0.55 else 0.0
                minutes = round(min(share, 1.0) * available)
                n90 = minutes / 90.0

                team_def = np.exp(-0.18 * strength[c_idx])  # better teams concede less
                xg_t = mu["xg"] * q * noise() * n90
                xa_t = mu["xa"] * q * noise() * n90
                gc_p90 = mu["gc"] * team_def * noise()
                age = float(np.clip(rng.normal(26.8, 4.2), 17, 38))
                age_prem = 0.12 * (26 - age) if age < 26 else (-0.10 * (age - 29) if age > 29 else 0.0)
                value = float(np.clip(
                    np.exp(2.0 + 0.7 * (a + 0.4 * strength[c_idx]) + _POS_VALUE_ADJ[pos] + age_prem
                           + rng.normal(0, 0.25)), 0.3, 180))
                years_left = float(np.clip(rng.uniform(0.2, 5.0) + 0.35 * a, 0.1, 5.5))

                status, news = "a", ""
                if rng.random() < 0.05 * proneness:
                    status, news = "i", "Injured - expected back in a few weeks"
                elif rng.random() < 0.04:
                    status, news = "d", "Late fitness test"

                rows.append({
                    "player_id": f"syn:{pid:04d}",
                    "name": f"{given} {family}", "web_name": family,
                    "club": club, "club_short": short, "position_group": pos,
                    "data_source": "synthetic", "archetype": arch_name,
                    "birth_date": today - pd.Timedelta(days=int(age * 365.25)),
                    "contract_end": today + pd.Timedelta(days=int(years_left * 365.25)),
                    "market_value_eur_m": round(value, 1),
                    "minutes": minutes, "starts": int(round(minutes / 80)),
                    "goals": rng.poisson(xg_t * rng.lognormal(0, 0.2)),
                    "assists": rng.poisson(xa_t * rng.lognormal(0, 0.2)),
                    "xg": xg_t, "xa": xa_t,
                    "xgc": mu["xgc"] * team_def * noise() * n90,
                    "clean_sheets": rng.binomial(int(n90), min(0.9, mu["cs"] / team_def)) if n90 >= 1 else 0,
                    "goals_conceded": rng.poisson(gc_p90 * n90),
                    "saves": rng.poisson(mu["saves"] * noise() * n90),
                    "yellow_cards": rng.poisson(mu["yc"] * noise() * n90),
                    "red_cards": int(rng.random() < 0.02 * n90 / 10),
                    "influence": mu["influence"] * q * noise() * n90,
                    "creativity": mu["creativity"] * q * noise() * n90,
                    "threat": mu["threat"] * q * noise() * n90,
                    "defensive_contribution": mu["dc"] * noise() * n90,
                    "status": status, "news": news,
                    "club_position": position_of[c_idx], "club_played": matches_played,
                    "club_points": float(points[c_idx]),
                })

                # injury history over the last three years
                for _ in range(rng.poisson(proneness * 1.3)):
                    days = int(np.clip(rng.lognormal(np.log(18), 0.9), 3, 200))
                    start = today - pd.Timedelta(days=int(rng.integers(10, 365 * 3)))
                    injury_rows.append({
                        "player": f"{given} {family}", "club": club,
                        "from_date": start, "until_date": start + pd.Timedelta(days=days),
                        "days_out": days, "injury": rng.choice(_INJURIES),
                    })

    players = pd.DataFrame(rows)

    # FPL-style price proxy: rank of market value within position -> price range
    prices = pd.Series(np.nan, index=players.index)
    for pos, grp in players.groupby("position_group"):
        lo, hi = _FPL_PRICE_RANGE[pos]
        pct = grp["market_value_eur_m"].rank(pct=True)
        prices.loc[grp.index] = np.round(lo + (hi - lo) * pct ** 1.6, 1)
    players["fpl_price_m"] = prices
    return SyntheticData(players, pd.DataFrame(injury_rows))
