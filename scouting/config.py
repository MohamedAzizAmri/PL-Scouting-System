"""Central configuration: paths, profile definitions, category thresholds.

Everything that is a judgement call (which stats define a "profile", how heavily
each counts, where the price/age bands sit) lives here so it can be tuned in one
place and is easy to explain in a report or interview.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"  # API response cache (git-ignored)
EXTERNAL_DIR = DATA_DIR / "external"  # files you download yourself (Transfermarkt, injuries)
PROCESSED_DIR = DATA_DIR / "processed"  # pipeline output (committed by the scheduled job)

PLAYERS_CSV = PROCESSED_DIR / "players.csv"
METADATA_JSON = PROCESSED_DIR / "metadata.json"
ROLE_MODEL = ROOT / "models" / "roles.joblib"   # fitted role model (rebuilt by `train-roles`)
ROLE_NAMES: dict[str, str] = {}   # e.g. {"role_0": "Ball-winning defender"}; filled in after reading the profile table
SEASONS_CSV = PROCESSED_DIR / "player_seasons.csv"   # past-season totals per player (ML training data)
TRANSFERMARKT_CSV = EXTERNAL_DIR / "transfermarkt_pl.csv"
INJURIES_CSV = EXTERNAL_DIR / "injuries.csv"

# API-Football (api-sports.io) injury history. The free plan allows 100 requests/day
# in total, so the nightly job spends at most this many and resumes the next day.
AF_DAILY_BUDGET = 60
AF_INJURY_WINDOW_YEARS = 3
AF_STALE_AFTER_DAYS = 120   # warn if the newest injury record is older than this


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Tiny .env reader so the API key can live in a git-ignored file."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


load_dotenv()

POSITION_GROUPS = ("GK", "DEF", "MID", "FWD")

# --------------------------------------------------------------------------- #
# Player "profile": which per-90 stats describe a role, per position group,
# and how much each one counts in the distance between two players.
# Realised output (goals/assists) is noisier than expected output (xG/xA), so it
# gets a lower weight. Cards are a weak style signal.
# Stats missing from the data (e.g. defensive_contribution before 2025/26) are
# dropped automatically.
# --------------------------------------------------------------------------- #
PROFILE_FEATURES: dict[str, dict[str, float]] = {
    "GK": {
        "saves_p90": 1.0,
        "goals_conceded_p90": 0.8,
        "xgc_p90": 1.0,
        "clean_sheets_p90": 0.7,
        "influence_p90": 0.5,
    },
    "DEF": {
        "defensive_contribution_p90": 1.0,
        "clean_sheets_p90": 0.6,
        "xgc_p90": 0.8,
        "xg_p90": 0.8,
        "xa_p90": 0.8,
        "creativity_p90": 1.0,
        "threat_p90": 0.8,
        "influence_p90": 0.7,
        "yellow_cards_p90": 0.4,
    },
    "MID": {
        "xg_p90": 1.0,
        "xa_p90": 1.0,
        "goals_p90": 0.6,
        "assists_p90": 0.6,
        "creativity_p90": 1.0,
        "threat_p90": 1.0,
        "influence_p90": 0.7,
        "defensive_contribution_p90": 1.0,
    },
    "FWD": {
        "xg_p90": 1.0,
        "goals_p90": 0.6,
        "xa_p90": 0.9,
        "assists_p90": 0.5,
        "threat_p90": 1.0,
        "creativity_p90": 0.8,
        "influence_p90": 0.6,
    },
}

# Raw season totals that get converted to (shrunk) per-90 rates.
RATE_STATS = [
    "goals", "assists", "xg", "xa", "xgc", "clean_sheets", "goals_conceded",
    "saves", "yellow_cards", "red_cards", "influence", "creativity", "threat",
    "defensive_contribution",
]

# For "shared strengths" text only: a *low* value of these is the good direction.
LOWER_IS_BETTER = {"goals_conceded_p90", "xgc_p90", "yellow_cards_p90", "red_cards_p90"}

FEATURE_LABELS = {
    "saves_p90": "saves/90", "goals_conceded_p90": "goals conceded/90",
    "xgc_p90": "xG conceded/90", "clean_sheets_p90": "clean sheets/90",
    "influence_p90": "influence/90", "defensive_contribution_p90": "defensive actions/90",
    "xg_p90": "xG/90", "xa_p90": "xA/90", "goals_p90": "goals/90",
    "assists_p90": "assists/90", "creativity_p90": "creativity/90",
    "threat_p90": "threat/90", "yellow_cards_p90": "yellow cards/90",
}

# --------------------------------------------------------------------------- #
# Similarity engine settings
# --------------------------------------------------------------------------- #
# Per-90 rates from few minutes are noisy; shrink them toward the position
# average as if every player had this many extra "prior" minutes.
SHRINK_MINUTES = 270.0
# Minimum minutes to be a recommendation candidate. "auto" scales with how far
# into the season we are (30% of the most-played player, between 180 and 900).
MIN_MINUTES = "auto"

# --------------------------------------------------------------------------- #
# Categories
# --------------------------------------------------------------------------- #
AGE_BINS = [-float("inf"), 21, 25, 30, float("inf")]
AGE_LABELS = ["Prospect (<21)", "Developing (21-24)", "Prime (25-29)", "Veteran (30+)"]

PRICE_LABELS = ["Budget", "Mid-range", "Premium", "Elite"]
PRICE_BINS_EUR_M = [-float("inf"), 10, 30, 60, float("inf")]  # absolute, Transfermarkt value
PRICE_QUANTILES_FPL = [0.40, 0.75, 0.93]  # relative, used when only the FPL price exists
MIN_MARKET_VALUE_COVERAGE = 0.60  # below this, fall back to FPL price for everyone

DIFFICULTY_LABELS = ["Easy", "Moderate", "Hard", "Very hard"]
DIFFICULTY_CUTS = [35, 55, 75]
DIFFICULTY_WEIGHTS = {
    "value": 0.35,       # pricier players cost more to move
    "contract": 0.25,    # long contracts = selling club holds the leverage
    "club": 0.15,        # top-of-table clubs rarely need to sell
    "importance": 0.15,  # a regular starter is harder to prise away than a fringe player
    "age": 0.10,         # young + good = high resale value, club resists
}
CONTRACT_YEARS_FULL = 4.0
AGE_EASIEST, AGE_HARDEST = 32.0, 21.0

INJURY_LABELS = ["Low risk", "Moderate risk", "High risk", "No injury data"]
# From an injury-history file (last 3 seasons)
INJURY_HIGH_DAYS, INJURY_MODERATE_DAYS = 90, 30
INJURY_HIGH_COUNT, INJURY_MODERATE_COUNT = 4, 2
# From the availability proxy (minutes played / minutes available)
AVAILABILITY_LOW_RISK, AVAILABILITY_MODERATE_RISK = 0.85, 0.60
MIN_CLUB_MATCHES_FOR_PROXY = 5
