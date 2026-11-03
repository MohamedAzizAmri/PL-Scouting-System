"""Join the secondary sources onto the base player table.

Every join is "fill, never overwrite" for fields another source already provided,
and returns a small info dict (match rates) that the pipeline records in
metadata.json, so a poor join is visible rather than silent.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .matching import club_similarity, match_by_birth_date, match_players

log = logging.getLogger(__name__)


def join_standings(players: pd.DataFrame, standings: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Attach club position / points / games played from a league table."""
    out = players.copy()
    if standings.empty:
        return out, {"clubs_matched": 0, "clubs_total": int(out["club"].nunique())}
    tla = standings["tla"].fillna("").str.upper()

    def find(club: str, short: str) -> pd.Series | None:
        hits = standings[tla == str(short).upper()] if short else standings.iloc[0:0]
        if len(hits) == 1:
            return hits.iloc[0]
        sims = standings["club_name"].map(lambda n: club_similarity(n, club))
        best = sims.idxmax()
        return standings.loc[best] if sims[best] >= 0.75 else None

    clubs = out[["club", "club_short"]].drop_duplicates()
    found = {r.club: find(r.club, r.club_short) for r in clubs.itertuples()}
    for col in ("club_position", "club_played", "club_points"):
        mapped = out["club"].map(lambda c, col=col: found[c][col] if found.get(c) is not None else np.nan)
        out[col] = mapped.where(mapped.notna(), out[col])
    matched = sum(v is not None for v in found.values())
    missing = [c for c, v in found.items() if v is None]
    if missing:
        log.warning("standings: no match for clubs %s", missing)
    return out, {"clubs_matched": matched, "clubs_total": len(found), "unmatched_clubs": missing}


def _fill_from(
    players: pd.DataFrame,
    other: pd.DataFrame,
    *,
    other_club_col: str,
    fill_cols: list[str],
    overwrite_cols: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    out = players.copy()
    res = match_players(
        out, other, right_name_col="name", right_club_col=other_club_col if other_club_col in other else None
    )
    other = other.reset_index(drop=True)
    for _, pair in res.pairs.iterrows():
        li, ri = pair["left_index"], pair["right_index"]
        for col in fill_cols:
            if col in other.columns and pd.notna(other.at[ri, col]):
                if col in overwrite_cols or pd.isna(out.at[li, col]):
                    if out[col].dtype.kind == "f" and not pd.api.types.is_number(other.at[ri, col]):
                        out[col] = out[col].astype(object)     # e.g. text landing in an all-NaN column
                    out.at[li, col] = other.at[ri, col]
    info = {"matched": int(len(res.pairs)), "total": int(len(players)), "match_rate": round(res.rate, 3)}
    unmatched = out.loc[res.unmatched_left, ["player_id", "name", "club"]]
    return out, info, unmatched


def join_squads(players: pd.DataFrame, squads: pd.DataFrame) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """football-data.org squads -> fill birth date, contract end, nationality."""
    if squads.empty:
        return players.copy(), {"matched": 0, "total": len(players), "match_rate": 0.0}, players.iloc[0:0][["player_id", "name", "club"]]
    return _fill_from(
        players, squads, other_club_col="club_name",
        fill_cols=["birth_date", "contract_end", "nationality"],
    )


def join_transfermarkt(players: pd.DataFrame, tm: pd.DataFrame) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Transfermarkt -> market value (authoritative), plus birth date / contract end as fallbacks.

    Pass 1 matches by name and club against Premier League rows. Pass 2 rescues
    players whose club differs from the snapshot (summer transfers) by requiring the
    same birth date plus a shared name token, searching ALL leagues in the file. Those
    players get a market value but never a contract end: the snapshot's contract
    belongs to their old club.
    """
    out = players.copy()
    tm = tm.reset_index(drop=True)
    in_pl = tm["in_premier_league"].astype(bool) if "in_premier_league" in tm else pd.Series(True, index=tm.index)
    pl_orig = tm.index[in_pl].to_numpy()
    pl = tm[in_pl].reset_index(drop=True)

    res = match_players(out, pl, right_name_col="name", right_club_col="club" if "club" in pl else None)
    used: set[int] = set()

    def apply(pairs, orig_of, cols):
        for _, pair in pairs.iterrows():
            li, ri = pair["left_index"], orig_of(pair["right_index"])
            used.add(int(ri))
            for col in cols:
                val = tm.at[ri, col]
                if pd.notna(val) and (col == "market_value_eur_m" or pd.isna(out.at[li, col])):
                    out.at[li, col] = val

    apply(res.pairs, lambda r: pl_orig[r], ["market_value_eur_m", "birth_date", "contract_end"])
    still = out.loc[res.unmatched_left]
    rescued = match_by_birth_date(still, tm, exclude_right=used)
    apply(rescued, lambda r: r, ["market_value_eur_m", "birth_date"])
    rescued_left = set(rescued["left_index"])
    unmatched_ids = [i for i in res.unmatched_left if i not in rescued_left]
    n_matched = len(res.pairs) + len(rescued)
    info = {
        "matched": int(n_matched), "total": int(len(players)),
        "match_rate": round(n_matched / len(players), 3) if len(players) else 0.0,
        "matched_by_birth_date": int(len(rescued)),
    }
    return out, info, out.loc[unmatched_ids, ["player_id", "name", "club"]]


def join_injuries(
    players: pd.DataFrame,
    injuries: pd.DataFrame,
    *,
    today: pd.Timestamp,
    years: int = 3,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Aggregate injuries in the last `years` years and attach count + days missed.

    Players absent from the file get 0 (no injuries *recorded*): the file must cover
    the players you care about, otherwise "no injuries" really means "not in file".
    """
    out = players.copy()
    window_start = today - pd.DateOffset(years=years)
    recent = injuries[injuries["from_date"] >= window_start]
    agg = (
        recent.groupby(["player", "club"], as_index=False)
        .agg(injury_count_3y=("from_date", "size"), injury_days_3y=("days_out", "sum"))
        .rename(columns={"player": "name"})
    )
    out["injury_count_3y"], out["injury_days_3y"] = 0.0, 0.0
    if agg.empty:
        return out, {"matched": 0, "total": len(players), "match_rate": 0.0}, out.iloc[0:0][["player_id", "name", "club"]]
    res = match_players(out, agg, right_name_col="name", right_club_col="club")
    for _, pair in res.pairs.iterrows():
        li, ri = pair["left_index"], pair["right_index"]
        out.at[li, "injury_count_3y"] = agg.at[ri, "injury_count_3y"]
        out.at[li, "injury_days_3y"] = agg.at[ri, "injury_days_3y"]
    info = {
        "players_with_injuries": int(len(res.pairs)),
        "injury_records_unmatched": int(res.unmatched_right_count),
        "total": len(players),
    }
    unmatched_players = out.iloc[0:0][["player_id", "name", "club"]]
    return out, info, unmatched_players


def apply_injury_history(players: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Set injury count/days for players present in `history` (indexed by player_id).

    Players absent from `history` keep whatever they had (NaN = "no data", never 0).
    """
    out = players.copy()
    mask = out["player_id"].isin(history.index)
    for col in ("injury_count_3y", "injury_days_3y"):
        out.loc[mask, col] = out.loc[mask, "player_id"].map(history[col]).to_numpy()
    return out
