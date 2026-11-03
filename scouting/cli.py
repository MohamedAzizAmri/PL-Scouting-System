"""Command line interface.

    python -m scouting refresh --source synthetic          # demo data, no network
    python -m scouting refresh --source fpl --football-data
    python -m scouting import-transfermarkt players.csv.gz
    python -m scouting search "salah"
    python -m scouting recommend "Player Name" -k 10 --max-difficulty Hard --explain
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from . import config as C
from .categories import ORDERS
from .engine import AmbiguousPlayer, InsufficientData, PlayerNotFound, ScoutingEngine
from .pipeline import refresh
from .sources._http import HttpError
from .sources.files import import_transfermarkt
from .sources.footballdata import FootballDataError

SYNTHETIC_BANNER = "** SYNTHETIC DEMO DATA: fictional clubs and players, not real Premier League data **"


def _engine(args) -> ScoutingEngine:
    path = Path(args.data)
    if not path.exists():
        sys.exit(f"{path} not found. Run `python -m scouting refresh` first.")
    return ScoutingEngine.from_csv(path)


def _print_df(df: pd.DataFrame, fmt: str, explain: bool) -> None:
    compact = ["name", "club", "age", "similarity", "age_band", "price_band", "value_m",
               "difficulty_label", "injury_risk", "current_status"]
    cols = compact + (["shared_strengths", "key_difference"] if explain else [])
    if fmt == "csv":
        print(df.to_csv())
    elif fmt == "json":
        print(df.reset_index().to_json(orient="records", indent=2))
    else:
        with pd.option_context("display.width", 250, "display.max_columns", None, "display.max_colwidth", 80):
            print(df[cols].to_string())


def _cmd_refresh(args) -> int:
    path, meta = refresh(
        args.source, use_football_data=args.football_data,
        transfermarkt_path=Path(args.transfermarkt) if args.transfermarkt else None,
        injuries_path=Path(args.injuries) if args.injuries else None,
        strict=args.strict, seed=args.seed,
        use_api_football=args.api_football, af_budget=args.af_budget,
    )
    print(f"Wrote {meta['n_players']} players to {path}")
    af = meta["enrichment"].get("api_football_injuries")
    if af:
        print(f"  API-Football injuries: {af['histories_loaded']} loaded, {af['histories_pending']} pending, "
              f"newest record {af['newest_injury_end']}, budget left today {af['budget_remaining_at_end']}")
    for warning in meta["warnings"]:
        print(f"  warning: {warning}")
    print(json.dumps({k: meta[k] for k in ("enrichment", "coverage")}, indent=2, default=str))
    return 0


def _cmd_import_tm(args) -> int:
    dst = Path(args.output)
    slim = import_transfermarkt(Path(args.input), dst, args.competition)
    print(f"Wrote {len(slim)} players to {dst}")
    return 0


def _cmd_search(args) -> int:
    eng = _engine(args)
    print(eng.search(args.query, args.limit).to_string())
    return 0


def _cmd_recommend(args) -> int:
    eng = _engine(args)
    filters = dict(
        min_age=args.min_age, max_age=args.max_age, age_bands=args.age_band,
        max_value_m=args.max_value, price_bands=args.price_band,
        max_difficulty=args.max_difficulty, injury_risk=args.injury_risk,
        exclude_clubs=args.exclude_club, min_similarity=args.min_similarity,
        include_low_minutes=args.include_low_minutes, include_departed=args.include_departed,
    )
    if eng.is_synthetic:
        print(SYNTHETIC_BANNER)
    q = eng.query_summary(args.player)
    print(
        f"\nQuery: {q['name']} ({q['club']}, {q['position_group']}), age {q['age']} [{q['age_band']}], "
        f"value {q['value_m']} [{q['price_band']}, basis: {q['price_basis']}], "
        f"difficulty to buy: {q['difficulty_label']} ({q['difficulty_drivers'] or 'n/a'}), "
        f"injury: {q['injury_risk']} [{q['injury_basis']}], status: {q['current_status']}"
    )
    if q["departed"]:
        print(f"  note: this player has left the club ({q['news']}); recommendations are for a similar profile")
    if not q["eligible"]:
        print(f"  note: only {q['minutes']:.0f} minutes played (min {eng.min_minutes:.0f}); profile is low-confidence")
    print()
    if args.group_by:
        for label, part in eng.recommend_by_category(
            args.player, by=args.group_by, per_group=args.per_group, **filters
        ).items():
            print(f"--- {args.group_by}: {label}")
            _print_df(part, args.format, args.explain)
    else:
        _print_df(eng.recommend(args.player, args.k, **filters), args.format, args.explain)
    return 0


def _cmd_profile(args) -> int:
    eng = _engine(args)
    print(eng.profile(args.player).to_string(index=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="scouting", description="Premier League player scouting system")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("refresh", help="fetch + enrich data and write data/processed/players.csv")
    r.add_argument("--source", choices=["fpl", "synthetic"], default="fpl")
    r.add_argument("--football-data", action="store_true", help="add standings/squads from football-data.org")
    r.add_argument("--transfermarkt", help=f"Transfermarkt CSV (default {C.TRANSFERMARKT_CSV} if present)")
    r.add_argument("--injuries", help=f"injury-history CSV (default {C.INJURIES_CSV} if present)")
    r.add_argument("--api-football", action="store_true",
                   help="per-player injury history from API-Football (needs API_FOOTBALL_KEY; resumes daily)")
    r.add_argument("--af-budget", type=int, default=C.AF_DAILY_BUDGET,
                   help="max API-Football requests to spend per day (free plan allows 100)")
    r.add_argument("--strict", action="store_true", help="fail if an optional source breaks")
    r.add_argument("--seed", type=int, default=7, help="synthetic data seed")
    r.set_defaults(func=_cmd_refresh)

    t = sub.add_parser("import-transfermarkt", help="slim a Transfermarkt players export to Premier League")
    t.add_argument("input")
    t.add_argument("-o", "--output", default=str(C.TRANSFERMARKT_CSV))
    t.add_argument("--competition", default="GB1", help="Transfermarkt competition id (Premier League = GB1)")
    t.set_defaults(func=_cmd_import_tm)

    def add_data(sp):
        sp.add_argument("--data", default=str(C.PLAYERS_CSV))

    s = sub.add_parser("search", help="find a player by name")
    s.add_argument("query")
    s.add_argument("--limit", type=int, default=10)
    add_data(s)
    s.set_defaults(func=_cmd_search)

    pr = sub.add_parser("profile", help="show a player's per-90 profile and percentiles")
    pr.add_argument("player")
    add_data(pr)
    pr.set_defaults(func=_cmd_profile)

    rec = sub.add_parser("recommend", help="players with a similar profile")
    rec.add_argument("player")
    rec.add_argument("-k", type=int, default=10)
    add_data(rec)
    rec.add_argument("--min-age", type=float)
    rec.add_argument("--max-age", type=float)
    rec.add_argument("--age-band", action="append", choices=ORDERS["age_band"])
    rec.add_argument("--max-value", type=float, help="in value_m units (EUR m, or FPL GBP m without market values)")
    rec.add_argument("--price-band", action="append", choices=ORDERS["price_band"])
    rec.add_argument("--max-difficulty", choices=ORDERS["difficulty_label"][:-1])
    rec.add_argument("--injury-risk", action="append", choices=ORDERS["injury_risk"])
    rec.add_argument("--exclude-club", action="append")
    rec.add_argument("--min-similarity", type=float, help="drop weak matches (0-100); ~60+ is a close match")
    rec.add_argument("--include-low-minutes", action="store_true")
    rec.add_argument("--include-departed", action="store_true",
                     help="also recommend players who have left their club (hidden by default)")
    rec.add_argument("--group-by", choices=list(ORDERS))
    rec.add_argument("--per-group", type=int, default=3)
    rec.add_argument("--format", choices=["table", "csv", "json"], default="table")
    rec.add_argument("--explain", action="store_true", help="show shared strengths and key difference")
    rec.set_defaults(func=_cmd_recommend)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        return args.func(args)
    except (PlayerNotFound, AmbiguousPlayer, InsufficientData) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (HttpError, FootballDataError) as exc:
        print(f"error: could not fetch data: {exc}\n"
              "Existing data in data/processed was left untouched. Check your connection, "
              "or run with --source synthetic for the offline demo.", file=sys.stderr)
        return 3
    except (ValueError, FileNotFoundError) as exc:   # bad input files, unknown source, etc.
        print(f"error: {exc}", file=sys.stderr)
        return 2
