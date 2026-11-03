"""Per-player injury history from API-Football, built within a daily request budget.

Flow (everything is cached, so a re-run only spends budget on what is still missing):
  1. find each club's API-Football team id            (1 request per club)
  2. fetch each squad                                 (1 request per club)
  3. match our players to squad entries by name       (no requests)
  4. fetch `sidelined` history per matched player     (1 request per player)

A full first pass is roughly 40 + ~600 requests, i.e. about ten days at a 60/day
budget; the nightly job just keeps going until everyone is loaded. Players not yet
loaded keep NaN injury columns and fall back to the availability proxy, which is
different from 0 injuries: "no data" is never reported as "no injuries".
"""
from __future__ import annotations

import logging
from collections import Counter

import pandas as pd

from . import config as C
from .matching import match_players
from .sources.apifootball import (
    ApiFootballClient, ApiFootballError, BudgetExhausted, is_injury_type, summarise_injuries,
)

log = logging.getLogger(__name__)
MAX_CONSECUTIVE_ERRORS = 3


def collect_injury_history(
    players: pd.DataFrame,
    client: ApiFootballClient,
    *,
    today: pd.Timestamp,
    window_years: int = C.AF_INJURY_WINDOW_YEARS,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Return (history indexed by player_id, info, unmatched players)."""
    info: dict = {"budget_remaining_at_start": client.budget.remaining, "stopped_early": None}
    clubs = list(dict.fromkeys(players["club"].dropna()))

    # 1-2: teams and squads -----------------------------------------------------------
    squad_rows: list[dict] = []
    teams_found = 0
    try:
        for club in clubs:
            team = client.find_team(club)
            if team is None:
                log.warning("API-Football: no unambiguous team for %r", club)
                continue
            teams_found += 1
            squad_rows += [{**row, "club": club} for row in client.squad(team["id"])]
    except BudgetExhausted:
        info["stopped_early"] = "budget used up while loading teams/squads; re-run tomorrow"
    except ApiFootballError as exc:
        info["stopped_early"] = f"API error while loading teams/squads: {exc}"
    info.update(teams_found=teams_found, teams_total=len(clubs), squad_players=len(squad_rows))

    # 3: match ----------------------------------------------------------------------------
    unmatched = players.iloc[0:0][["player_id", "name", "club"]]
    id_of: dict[str, int] = {}
    if squad_rows:
        squads = pd.DataFrame(squad_rows)
        res = match_players(players, squads, right_name_col="name", right_club_col="club",
                            cross_club_fallback=True)
        for _, pair in res.pairs.iterrows():
            id_of[players.at[pair["left_index"], "player_id"]] = int(squads.at[pair["right_index"], "af_id"])
        unmatched = players.loc[res.unmatched_left, ["player_id", "name", "club"]]
        info["matched_to_api_football"] = len(id_of)
        info["match_rate"] = round(res.rate, 3)
    else:
        info.update(matched_to_api_football=0, match_rate=0.0)

    # 4: histories, most-played players first -----------------------------------------------
    order = players[players["player_id"].isin(id_of)].sort_values("minutes", ascending=False)["player_id"]
    rows, type_counts, newest = [], Counter(), pd.NaT
    errors_in_a_row = 0
    for pid in order:
        try:
            records = client.sidelined(id_of[pid])
            errors_in_a_row = 0
        except BudgetExhausted:
            info["stopped_early"] = info["stopped_early"] or "daily budget used up; continues next run"
            break
        except ApiFootballError as exc:
            errors_in_a_row += 1
            log.warning("sidelined failed for %s: %s", pid, exc)
            if errors_in_a_row >= MAX_CONSECUTIVE_ERRORS:
                info["stopped_early"] = f"stopped after repeated API errors: {exc}"
                break
            continue
        count, days = summarise_injuries(records, today=today, window_years=window_years)
        rows.append({"player_id": pid, "injury_count_3y": count, "injury_days_3y": days})
        for r in records:
            type_counts[("injury" if is_injury_type(r["type"]) else "excluded", r["type"])] += 1
            if pd.notna(r["end"]) and (pd.isna(newest) or r["end"] > newest):
                newest = r["end"]

    history = pd.DataFrame(rows, columns=["player_id", "injury_count_3y", "injury_days_3y"]).set_index("player_id")
    info.update(
        histories_loaded=len(history),
        histories_pending=len(id_of) - len(history),
        newest_injury_end=None if pd.isna(newest) else str(newest.date()),
        record_types=[f"{kind}: {name} x{n}" for (kind, name), n in type_counts.most_common(25)],
        budget_remaining_at_end=client.budget.remaining,
    )
    if pd.notna(newest) and (today - newest).days > C.AF_STALE_AFTER_DAYS:
        info["staleness_warning"] = (
            f"newest injury record ends {newest.date()}, over {C.AF_STALE_AFTER_DAYS} days before "
            f"{today.date()}: the plan or source may not include recent injuries"
        )
    return history, info, unmatched
