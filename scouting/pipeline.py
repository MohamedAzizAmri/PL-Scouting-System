"""Source -> enrich -> save. This is what the scheduled job runs.

    base players   FPL API (per-player stats, price, status)   or   synthetic demo data
    + standings    football-data.org   (club position, games played)
    + squads       football-data.org   (birth date, contract end; plan-dependent)
    + market value Transfermarkt CSV   (data/external/transfermarkt_pl.csv)
    + injuries     injury CSV          (data/external/injuries.csv, optional)

Optional enrichments that are unavailable (no token, plan limit, file missing) are
skipped with a warning recorded in metadata.json rather than failing the run, unless
strict=True (useful in CI when you want a broken source to turn the job red).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import config as C
from . import enrich
from .schema import ensure_columns, validate
from .sources import synthetic
from .sources.files import load_injuries, load_transfermarkt
from .sources.footballdata import FootballDataClient, FootballDataError
from .injury_history import collect_injury_history
from .sources.apifootball import ApiFootballClient, ApiFootballError
from .sources.fpl import FplClient

log = logging.getLogger(__name__)


def build_players(
    source: str = "fpl",
    *,
    use_football_data: bool = False,
    transfermarkt_path: Path | None = None,
    injuries_path: Path | None = None,
    today: pd.Timestamp | None = None,
    seed: int = 7,
    strict: bool = False,
    fpl_client: FplClient | None = None,
    fd_client: FootballDataClient | None = None,
    use_api_football: bool = False,
    af_client: ApiFootballClient | None = None,
    af_budget: int = C.AF_DAILY_BUDGET,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Return (players, metadata, unmatched_report)."""
    today = (today or pd.Timestamp.today()).normalize()
    meta: dict = {"source": source, "enrichment": {}, "warnings": []}
    unmatched: list[pd.DataFrame] = []

    def warn(msg: str) -> None:
        log.warning(msg)
        meta["warnings"].append(msg)

    injuries = None
    if source == "synthetic":
        data = synthetic.generate(seed, today=today)
        players, injuries = data.players.drop(columns=["archetype"]), data.injuries
    elif source == "fpl":
        players = (fpl_client or FplClient()).players()
    else:
        raise ValueError(f"unknown source {source!r} (use 'fpl' or 'synthetic')")
    players = ensure_columns(players)

    if use_football_data:
        try:
            fd = fd_client or FootballDataClient()
            standings = fd.standings()
            players, info = enrich.join_standings(players, standings)
            meta["enrichment"]["football_data_standings"] = info
            try:
                players, info, miss = enrich.join_squads(players, fd.squads())
                meta["enrichment"]["football_data_squads"] = info
                unmatched.append(miss.assign(joined_source="football_data_squads"))
                if info["matched"] == 0:
                    warn("football-data.org returned no usable squad data (likely a plan limit)")
            except FootballDataError as exc:
                if strict:
                    raise
                warn(f"football-data squads skipped: {exc}")
        except FootballDataError as exc:
            if strict:
                raise
            warn(f"football-data.org skipped: {exc}")

    tm_path = transfermarkt_path or (C.TRANSFERMARKT_CSV if C.TRANSFERMARKT_CSV.exists() else None)
    if tm_path and Path(tm_path).exists():
        players, info, miss = enrich.join_transfermarkt(players, load_transfermarkt(Path(tm_path)))
        meta["enrichment"]["transfermarkt"] = {**info, "file": str(tm_path)}
        unmatched.append(miss.assign(joined_source="transfermarkt"))
        if info["match_rate"] < 0.8:
            warn(f"Transfermarkt match rate is only {info['match_rate']:.0%}: check data/processed/unmatched.csv")
    elif source != "synthetic":
        warn("no Transfermarkt file: price falls back to the FPL price and contract data may be missing")

    inj_path = injuries_path or (C.INJURIES_CSV if C.INJURIES_CSV.exists() else None)
    if inj_path and Path(inj_path).exists():
        injuries = load_injuries(Path(inj_path))
        meta["enrichment"]["injuries_file"] = str(inj_path)
    if injuries is not None:
        players, info, _ = enrich.join_injuries(players, injuries, today=today)
        meta["enrichment"]["injuries"] = info
    elif not use_api_football:
        warn("no injury-history file: injury category will use the availability proxy")

    if use_api_football:
        if source == "synthetic":
            warn("--api-football ignored for synthetic data")
        else:
            try:
                af = af_client or ApiFootballClient(daily_budget=af_budget)
                history, info, miss = collect_injury_history(players, af, today=today)
                players = enrich.apply_injury_history(players, history)
                meta["enrichment"]["api_football_injuries"] = info
                unmatched.append(miss.assign(joined_source="api_football"))
                for key in ("stopped_early", "staleness_warning"):
                    if info.get(key):
                        warn(f"API-Football injuries: {info[key]}")
                if info["histories_pending"]:
                    warn(f"API-Football injuries: {info['histories_pending']} matched players still "
                         "waiting for their history (they use the availability proxy until loaded)")
            except ApiFootballError as exc:
                if strict:
                    raise
                warn(f"API-Football injuries skipped: {exc}")

    validate(players)
    meta.update(
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        reference_date=str(today.date()),
        n_players=int(len(players)),
        by_position=players["position_group"].value_counts().to_dict(),
        coverage={
            col: round(float(players[col].notna().mean()), 3)
            for col in ("birth_date", "contract_end", "market_value_eur_m", "club_position", "status")
        },
    )
    report = pd.concat(unmatched, ignore_index=True) if unmatched else pd.DataFrame(
        columns=["player_id", "name", "club", "joined_source"]
    )
    return players, meta, report


def save(players: pd.DataFrame, meta: dict, report: pd.DataFrame, out_dir: Path = C.PROCESSED_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "players.csv"
    players.to_csv(csv_path, index=False)
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    report.to_csv(out_dir / "unmatched.csv", index=False)
    return csv_path


def refresh(source: str = "fpl", out_dir: Path = C.PROCESSED_DIR, **kwargs) -> tuple[Path, dict]:
    players, meta, report = build_players(source, **kwargs)
    return save(players, meta, report, out_dir), meta
