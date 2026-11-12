from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..config import RAW_DIR
from ._http import HttpError, JsonClient, RateLimiter

log = logging.getLogger(__name__)

BASE_URL = "https://fantasy.premierleague.com/api"

# unified column -> history_past field (same names as the current-season columns)
SEASON_STATS = {
    "minutes": "minutes", "starts": "starts", "goals": "goals_scored", "assists": "assists",
    "xg": "expected_goals", "xa": "expected_assists", "xgc": "expected_goals_conceded",
    "clean_sheets": "clean_sheets", "goals_conceded": "goals_conceded", "saves": "saves",
    "yellow_cards": "yellow_cards", "red_cards": "red_cards", "bonus": "bonus", "bps": "bps",
    "influence": "influence", "creativity": "creativity", "threat": "threat",
    "defensive_contribution": "defensive_contribution",
    "clearances_blocks_interceptions": "clearances_blocks_interceptions",
    "recoveries": "recoveries", "tackles": "tackles",
    "total_points": "total_points",
}
# stats that FPL only started recording later; all-zero in a season means "not tracked"
MAYBE_UNTRACKED = [
    "xg", "xa", "xgc", "influence", "creativity", "threat", "defensive_contribution",
    "clearances_blocks_interceptions", "recoveries", "tackles", "bps",
]
COLUMNS = ["player_id", "element_code", "season", "season_start_year", "position_group",
           "price_m"] + list(SEASON_STATS)


def season_start_year(name: object) -> float:
    """'2025/26' -> 2025."""
    try:
        return float(str(name).split("/")[0])
    except ValueError:
        return np.nan


def parse_history_past(payload: dict, player_id: str, position_group: str) -> pd.DataFrame:
    rows = []
    for h in payload.get("history_past") or []:
        row = {
            "player_id": player_id,
            "element_code": h.get("element_code"),
            "season": h.get("season_name"),
            "season_start_year": season_start_year(h.get("season_name")),
            "position_group": position_group,
            "price_m": h["end_cost"] / 10 if h.get("end_cost") is not None else np.nan,
        }
        for col, key in SEASON_STATS.items():
            row[col] = h.get(key, np.nan)
        rows.append(row)
    df = pd.DataFrame(rows, columns=COLUMNS)
    for col in ["price_m", "element_code", *SEASON_STATS]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def blank_untracked_stats(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Set a stat to NaN for any season where every player has 0 or NaN for it."""
    out, blanked = df.copy(), []
    for season, idx in out.groupby("season").groups.items():
        for col in MAYBE_UNTRACKED:
            if out.loc[idx, col].fillna(0).eq(0).all():
                out.loc[idx, col] = np.nan
                blanked.append(f"{season}:{col}")
    return out, blanked


class FplHistoryClient:
    """Polite client: 60 requests a minute at most, every response cached for 30 days."""

    def __init__(self, client: JsonClient | None = None, calls_per_minute: int = 60):
        self.http = client or JsonClient(
            BASE_URL,
            headers={"User-Agent": "pl-scouting-student-project/0.1 (educational)"},
            limiter=RateLimiter(calls_per_minute, 60.0),
            cache_dir=RAW_DIR / "fpl_history",
            max_age_hours=24 * 30,
        )

    def element_summary(self, element_id: int) -> dict:
        return self.http.get(f"element-summary/{element_id}/")


def collect_history(
    players: pd.DataFrame,
    client: FplHistoryClient,
    *,
    limit: int | None = None,
    max_consecutive_errors: int = 5,
) -> tuple[pd.DataFrame, dict]:
    """Fetch past seasons for every FPL player in `players` (needs player_id 'fpl:<id>')."""
    todo = players[players["player_id"].astype(str).str.startswith("fpl:")]
    if limit:
        todo = todo.head(limit)
    frames: list[pd.DataFrame] = []
    failed: list[str] = []
    streak = 0
    stopped = None
    for pid, pos in zip(todo["player_id"], todo["position_group"]):
        try:
            payload = client.element_summary(int(str(pid).split(":")[1]))
        except HttpError as exc:
            failed.append(str(pid))
            streak += 1
            log.warning("history failed for %s: %s", pid, exc)
            if streak >= max_consecutive_errors:
                stopped = f"{streak} failures in a row, last: {exc}"
                break
            continue
        streak = 0
        frames.append(parse_history_past(payload, str(pid), str(pos)))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)
    df, blanked = blank_untracked_stats(df)
    info = {
        "players_requested": int(len(todo)), "players_ok": int(len(frames)),
        "players_failed": len(failed), "failed_ids": failed[:20], "stopped_early": stopped,
        "player_seasons": int(len(df)),
        "seasons": sorted(df["season"].dropna().unique().tolist()),
        "untracked_stats_blanked": blanked,
    }
    return df, info
