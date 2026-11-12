"""Command line interface."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .categories import ORDERS
from .engine import AmbiguousPlayer, InsufficientData, PlayerNotFound, ScoutingEngine
from .pipeline import refresh
from .sources._http import HttpError
from .sources.files import import_transfermarkt
from .sources.fpl_history import FplHistoryClient, collect_history
from .sources.footballdata import FootballDataError

SYNTHETIC_BANNER = "** SYNTHETIC DEMO DATA: fictional clubs and players, not real Premier League data **"


def _engine(args) -> ScoutingEngine:
    path = Path(args.data)
    if not path.exists():
        sys.exit(f"{path} not found. Run `python -m scouting refresh` first.")
    return ScoutingEngine.from_csv(path)


def _print_df(df: pd.DataFrame, fmt: str, explain: bool) -> None:
    compact = ["name", "club", "age", "similarity", "role", "age_band", "price_band", "value_m",
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


def _cmd_fetch_history(args) -> int:
    src = Path(args.data)
    if not src.exists():
        sys.exit(f"{src} not found. Run `python -m scouting refresh` first.")
    players = pd.read_csv(src, usecols=["player_id", "position_group"])
    df, info = collect_history(players, FplHistoryClient(calls_per_minute=args.calls_per_minute), limit=args.limit)
    if df.empty:
        print("error: no history fetched; nothing written", file=sys.stderr)
        print(json.dumps(info, indent=2, default=str), file=sys.stderr)
        return 3
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} player-seasons ({info['players_ok']} players) to {out}")
    print(json.dumps(info, indent=2, default=str))
    return 0 if not info["stopped_early"] else 3


def _cmd_ml_explore(args) -> int:
    try:
        from .ml.dataset import load_player_seasons
        from .ml.explore import explore
    except ImportError as exc:
        sys.exit(f"{exc}. Install the ML extras first: pip install -r requirements-ml.txt")
    seasons = load_player_seasons(args.seasons, min_minutes=args.min_minutes)
    print(f"{len(seasons)} player-seasons with >= {args.min_minutes:.0f} minutes")
    res = explore(seasons, feature_sets=args.features, populations=args.grouping,
                  ks=range(2, args.max_k + 1), n_pairs=args.n_pairs)
    with pd.option_context("display.width", 200, "display.max_rows", None):
        print(res.to_string(index=False))
    if args.output:
        res.to_csv(args.output, index=False)
        print(f"saved to {args.output}")
    return 0


def _cmd_train_roles(args) -> int:
    try:
        import joblib

        from .ml.dataset import build_features, load_player_seasons
        from .ml.explore import stability
        from .ml.roles import fit_roles, profile_table, training_memberships
    except ImportError as exc:
        sys.exit(f"{exc}. Install the ML extras first: pip install -r requirements-ml.txt")
    seasons = load_player_seasons(args.seasons, min_minutes=args.min_minutes)
    model = fit_roles(seasons, args.features, args.grouping, args.k, seed=args.seed,
                      names=[n.strip() for n in args.names.split("|")] if args.names else None)
    names = {}
    if Path(args.data).exists():
        players = pd.read_csv(args.data)
        names = dict(zip(players["player_id"], players["name"]))
    member = training_memberships(model, seasons)
    member["name"] = member["player_id"].map(names).fillna(member["player_id"])
    recent = member[member["season_start_year"] >= member["season_start_year"].max() - 2]
    tables = profile_table(model)
    for label, table in tables.items():
        pm = model.populations[label]
        X = build_features(seasons, args.features, pm.positions)[0].filter(like="pct_").to_numpy()
        print(f"\n=== {label}: {args.k} roles, {len(X)} player-seasons, bootstrap stability {stability(X, args.k):.2f}")
        print("mean percentile (0-100) of each stat, per role:")
        
        shown = table.copy()
        if hasattr(model, "display_key"):
            shown.index = [model.display_key(i) for i in shown.index]
        print(shown.to_string())
        
        raw_col = "raw_role" if "raw_role" in recent.columns else "role"
        for i in range(model.k):
            raw_r = f"role_{i}"
            disp_r = model.display_key(raw_r) if hasattr(model, "display_key") else (
                f"{raw_r} ({model.role_label(raw_r)})" if model.names else raw_r
            )
            top = recent[(recent["population"] == label) & (recent[raw_col] == raw_r)].nlargest(args.examples, "role_prob")
            print(f"  {disp_r} examples: " + "; ".join(f"{r['name']} {r['season']}" for _, r in top.iterrows()))
            
    if Path(args.data).exists():
        cur = model.assign(players, min_minutes=args.assign_min_minutes)
        cur["name"] = players["name"]
        print(f"\ncurrent season: {cur['role'].notna().sum()} of {len(cur)} players assigned "
              f"(>= {args.assign_min_minutes:.0f} minutes, outfield)")
        print(cur["role"].value_counts().sort_index().to_string())
        for role in sorted(cur["role"].dropna().unique()):
            disp_r = model.display_key(role) if hasattr(model, "display_key") else role
            top = cur[cur["role"] == role].nlargest(args.examples, "role_prob")
            print(f"  {disp_r} now: " + "; ".join(top["name"]))
            
    out = Path(args.model_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out)
    print(f"\nsaved model to {out}")
    return 0


def _cmd_ml_evaluate(args: argparse.Namespace) -> int:
    try:
        from .ml.dataset import load_player_seasons
        from .ml.evaluate import evaluate_consecutive_seasons
    except ImportError as exc:
        sys.exit(f"{exc}. Install the ML extras first: pip install -r requirements-ml.txt")
    seasons = load_player_seasons(args.seasons, min_minutes=args.min_minutes)
    summary, diag = evaluate_consecutive_seasons(
        seasons, feature_set=args.features, k_roles=args.k, min_minutes=args.min_minutes
    )
    print(f"=== Consecutive-season self-retrieval ({diag['consecutive_pairs']} player-season pairs)")
    print(f"Primary GMM role year-over-year retention: {diag['role_retention_rate']*100:.1f}%")
    print(f"PCA explained variance: {diag['pca_explained_variance']*100:.1f}%\n")
    print(summary.to_string(index=False))
    return 0


def _cmd_assign_roles(args) -> int:
    try:
        import joblib

        from .ml.roles import apply_roles
    except ImportError as exc:
        sys.exit(f"{exc}. Install the ML extras first: pip install -r requirements-ml.txt")
    src, model_path = Path(args.data), Path(args.model)
    if not src.exists():
        sys.exit(f"{src} not found. Run `python -m scouting refresh` first.")
    if not model_path.exists():
        sys.exit(f"{model_path} not found. Run `python -m scouting train-roles` first.")
    model = joblib.load(model_path)
    players = pd.read_csv(src).drop(
        columns=["role", "role_prob", "second_role", "second_role_prob"], errors="ignore"
    )
    out = apply_roles(players, model, min_minutes=args.assign_min_minutes)
    out.to_csv(src, index=False)
    print(f"Added roles to {int(out['role'].notna().sum())} of {len(out)} players in {src}")
    print(out["role"].value_counts().to_string())
    return 0


def _cmd_train_value(args) -> int:
    try:
        import joblib
        from .ml.value_model import train_value_model
    except ImportError as exc:
        sys.exit(f"{exc}. Install the ML extras first: pip install -r requirements-ml.txt")
    
    players = pd.read_csv(args.data)
    model, feats, cv_log, cv_eur = train_value_model(players)
    
    print(f"=== Value Model Trained ({len(feats)} features) ===")
    print(f"Features: {', '.join(feats)}")
    print(f"CV Mean Absolute Error (Log10): {cv_log:.3f}")
    print(f"CV Mean Absolute Error (Euros): €{cv_eur:.1f}m")
    
    out = Path(args.model_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": feats}, out)
    print(f"Saved value model to {out}")
    return 0


def _cmd_assign_value(args) -> int:
    try:
        import joblib
        from .ml.value_model import predict_values
    except ImportError as exc:
        sys.exit(f"{exc}. Install the ML extras first: pip install -r requirements-ml.txt")
        
    players = pd.read_csv(args.data)
    blob = joblib.load(args.model)
    
    preds = predict_values(players, blob["model"], blob["features"])
    players["expected_value_m"] = np.round(preds, 1)
    
    # Value Gap = Expected (Model) - Actual (Transfermarkt)
    # High positive gap = Undervalued
    players["value_gap_m"] = np.round(players["expected_value_m"] - players["market_value_eur_m"].fillna(0), 1)
    players.loc[players["market_value_eur_m"].isna(), "value_gap_m"] = np.nan
    
    players.to_csv(args.data, index=False)
    print(f"Assigned expected_value_m and value_gap_m to {len(players)} players in {args.data}")
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
        roles=args.role,
    )
    # Include secondary_role opt-in if passed from CLI (will wire later if needed)
    if "include_secondary_role" in eng.recommend.__code__.co_varnames:
        filters["include_secondary_role"] = getattr(args, "include_secondary_role", False)

    if eng.is_synthetic:
        print(SYNTHETIC_BANNER)
    q = eng.query_summary(args.player)
    print(
        f"\nQuery: {q['name']} ({q['club']}, {q['position_group']}), age {q['age']} [{q['age_band']}], "
        f"value {q['value_m']} [{q['price_band']}, basis: {q['price_basis']}], "
        f"difficulty to buy: {q['difficulty_label']} ({q['difficulty_drivers'] or 'n/a'}), "
        f"injury: {q['injury_risk']} [{q['injury_basis']}], status: {q['current_status']}"
        + (f", role: {q['role']}" if isinstance(q["role"], str) else "")
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

    h = sub.add_parser("fetch-history", help="download past-season totals per player (training data for ML)")
    h.add_argument("--data", default=str(C.PLAYERS_CSV))
    h.add_argument("--output", default=str(C.SEASONS_CSV))
    h.add_argument("--limit", type=int, help="only the first N players (for a quick test)")
    h.add_argument("--calls-per-minute", type=int, default=60, help="be polite to the FPL servers")
    h.set_defaults(func=_cmd_fetch_history)

    m = sub.add_parser("ml-explore", help="how many player roles does the data support? (needs fetch-history)")
    m.add_argument("--seasons", default=str(C.SEASONS_CSV))
    m.add_argument("--min-minutes", type=float, default=900.0)
    m.add_argument("--features", action="append", choices=["ict", "xg"])
    m.add_argument("--grouping", action="append", choices=["by_position", "attackers_joint", "outfield_joint"])
    m.add_argument("--max-k", type=int, default=8)
    m.add_argument("--n-pairs", type=int, default=8, help="bootstrap pairs for the stability score")
    m.add_argument("--output", help="also write the table to this CSV")
    m.set_defaults(func=_cmd_ml_explore)

    ar = sub.add_parser("assign-roles", help="add the ML role to players.csv using the saved model (no network)")
    ar.add_argument("--data", default=str(C.PLAYERS_CSV))
    ar.add_argument("--model", default=str(C.ROLE_MODEL))
    ar.add_argument("--assign-min-minutes", type=float, default=270.0)
    ar.set_defaults(func=_cmd_assign_roles)

    tr = sub.add_parser("train-roles", help="fit the role model on past seasons and print each role's profile")
    tr.add_argument("--seasons", default=str(C.SEASONS_CSV))
    tr.add_argument("--data", default=str(C.PLAYERS_CSV))
    tr.add_argument("--features", choices=["ict", "xg"], default="ict")
    tr.add_argument("--grouping", choices=["by_position", "attackers_joint", "outfield_joint"], default="outfield_joint")
    tr.add_argument("--k", type=int, default=4)
    tr.add_argument("--min-minutes", type=float, default=900.0)
    tr.add_argument("--assign-min-minutes", type=float, default=270.0, help="minutes this season to get a role")
    tr.add_argument("--examples", type=int, default=6)
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--names", help='role names in role_0..role_k-1 order, separated by "|"')
    tr.add_argument("--model-out", default=str(C.ROLE_MODEL))
    tr.set_defaults(func=_cmd_train_roles)

    p_eval = sub.add_parser("ml-evaluate", help="Benchmark ML vs heuristic similarity on consecutive seasons")
    p_eval.add_argument("--seasons", default=str(C.SEASONS_CSV))
    p_eval.add_argument("--features", choices=["ict", "xg"], default="ict")
    p_eval.add_argument("--k", type=int, default=4)
    p_eval.add_argument("--min-minutes", type=float, default=900.0)
    p_eval.set_defaults(func=_cmd_ml_evaluate)

    tv = sub.add_parser("train-value", help="Train the market value regression model")
    tv.add_argument("--data", default=str(C.PLAYERS_CSV))
    tv.add_argument("--model-out", default=str(Path(C.ROLE_MODEL).parent / "value_model.joblib"))
    tv.set_defaults(func=_cmd_train_value)

    av = sub.add_parser("assign-value", help="Calculate expected_value_m and value_gap_m")
    av.add_argument("--data", default=str(C.PLAYERS_CSV))
    av.add_argument("--model", default=str(Path(C.ROLE_MODEL).parent / "value_model.joblib"))
    av.set_defaults(func=_cmd_assign_value)

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
    rec.add_argument("--role", action="append", help="only players with this ML role (repeatable)")
    rec.add_argument("--include-low-minutes", action="store_true")
    rec.add_argument("--include-departed", action="store_true",
                     help="also recommend players who have left their club (hidden by default)")
    rec.add_argument("--include-secondary-role", action="store_true", 
                     help="allow --role filters to match players whose secondary role prob is >= 0.25")
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