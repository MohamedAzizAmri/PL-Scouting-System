"""Readers for the two files YOU supply: a Transfermarkt export and an injury history.

Neither has a free, stable API, so they enter the pipeline as plain CSVs in
data/external/.

Transfermarkt export
    The open dataset https://github.com/dcaribou/transfermarkt-datasets (CC0, also on
    Kaggle as "davidcariboo/player-scores") ships a `players` table with market value,
    date of birth and contract expiry. Its maintainers have paused updates (current to
    6 July 2026), so values are a snapshot. Run
        python -m scouting import-transfermarkt path/to/players.csv.gz
    to slim it down to Premier League players.
    Column names are matched flexibly (see _TM_COLUMNS); if your file differs, the
    importer lists the columns it found.

Injury history (optional, any source you are allowed to use)
    CSV with columns: player, from_date, until_date  (optional: club, days_out, injury).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_TM_COLUMNS = {
    "name": ["name", "player_name"],
    "club": ["current_club_name", "club_name", "club"],
    "birth_date": ["date_of_birth", "dob", "birth_date"],
    "contract_end": ["contract_expiration_date", "contract_end", "contract_until"],
    "market_value_eur": ["market_value_in_eur", "market_value_eur", "market_value"],
    "league": ["current_club_domestic_competition_id", "league", "competition"],
}
PREMIER_LEAGUE_ID = "GB1"


def _pick(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lowered = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lowered:
            return lowered[cand]
    return None


def normalise_transfermarkt(raw: pd.DataFrame) -> pd.DataFrame:
    cols = {key: _pick(raw, cands) for key, cands in _TM_COLUMNS.items()}
    for required in ("name", "market_value_eur"):  # league is optional
        if cols[required] is None:
            raise ValueError(
                f"Transfermarkt file has no '{required}' column "
                f"(looked for {_TM_COLUMNS[required]}). Columns found: {list(raw.columns)}"
            )
    out = pd.DataFrame({
        "name": raw[cols["name"]],
        "club": raw[cols["club"]] if cols["club"] else "",
        "birth_date": pd.to_datetime(raw[cols["birth_date"]], errors="coerce") if cols["birth_date"] else pd.NaT,
        "contract_end": pd.to_datetime(raw[cols["contract_end"]], errors="coerce") if cols["contract_end"] else pd.NaT,
        "market_value_eur_m": pd.to_numeric(raw[cols["market_value_eur"]], errors="coerce") / 1e6,
        # rows from other leagues stay in the file so recent arrivals can be found by birth date
        "in_premier_league": (raw[cols["league"]] == PREMIER_LEAGUE_ID) if cols["league"] else True,
    })
    return out.dropna(subset=["name"]).reset_index(drop=True)


def load_transfermarkt(path: Path) -> pd.DataFrame:
    """Read either the slim file written by `import_transfermarkt` or a raw export."""
    raw = pd.read_csv(path)
    if "market_value_eur_m" not in raw.columns:
        return normalise_transfermarkt(raw)
    out = pd.DataFrame({
        "name": raw["name"],
        "club": raw["club"] if "club" in raw else "",
        "birth_date": pd.to_datetime(raw["birth_date"], errors="coerce") if "birth_date" in raw else pd.NaT,
        "contract_end": pd.to_datetime(raw["contract_end"], errors="coerce") if "contract_end" in raw else pd.NaT,
        "market_value_eur_m": pd.to_numeric(raw["market_value_eur_m"], errors="coerce"),
        "in_premier_league": raw["in_premier_league"].astype(bool) if "in_premier_league" in raw else True,
    })
    return out.dropna(subset=["name"]).reset_index(drop=True)


def import_transfermarkt(src: Path, dst: Path, competition_id: str = "GB1") -> pd.DataFrame:
    """Reduce the full Transfermarkt `players` table.

    competition_id="GB1" keeps current Premier League players only. "all" keeps every
    active player (flagging which are Premier League), so players who joined from
    abroad after the snapshot can be matched by birth date.
    """
    raw = pd.read_csv(src)
    comp_col = _pick(raw, ["current_club_domestic_competition_id"])
    if comp_col is None:
        log.warning("no competition column found: keeping every row, filtering is up to you")
    elif competition_id.lower() != "all":
        raw = raw[raw[comp_col] == competition_id]
    season_col = _pick(raw, ["last_season"])
    if season_col is not None and not raw.empty:
        raw = raw[raw[season_col] == raw[season_col].max()]
    slim = normalise_transfermarkt(raw)
    dst.parent.mkdir(parents=True, exist_ok=True)
    slim.to_csv(dst, index=False)
    log.info("wrote %d players to %s", len(slim), dst)
    return slim


def load_injuries(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    lowered = {c.lower(): c for c in df.columns}
    for required in ("player", "from_date"):
        if required not in lowered:
            raise ValueError(
                f"injury file needs a '{required}' column; found {list(df.columns)}"
            )
    out = pd.DataFrame({
        "player": df[lowered["player"]],
        "club": df[lowered["club"]] if "club" in lowered else "",
        "from_date": pd.to_datetime(df[lowered["from_date"]], errors="coerce"),
    })
    # days out: explicit column if present, otherwise (until_date - from_date)
    if "until_date" in lowered:
        until = pd.to_datetime(df[lowered["until_date"]], errors="coerce")
        computed = (until - out["from_date"]).dt.days.astype(float)
    else:
        computed = pd.Series(np.nan, index=df.index)
    if "days_out" in lowered:
        out["days_out"] = pd.to_numeric(df[lowered["days_out"]], errors="coerce").fillna(computed)
    else:
        out["days_out"] = computed
    return out.dropna(subset=["player", "from_date"]).reset_index(drop=True)
