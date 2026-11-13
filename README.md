# PL Scouting: find players with a similar profile

Give it a Premier League player. It returns players with a similar playing profile and labels each one by **age**, **market price**, **expected market value (ML)**, **tactical role (ML)**, **difficulty to buy**, and **injury history**, so you can filter for, say, "a cheaper, younger, available creative attacker with a similar profile to this midfielder".

```bash
python -m scouting recommend "Player Name" -k 10 --max-age 25 --max-difficulty Hard --explain
```

## Status: read this first

| Part | State |
|---|---|
| Similarity engine, categories, CLI, pipeline, ML models | Built and tested (**139 tests passing**) |
| Machine Learning (`scouting/ml/`) | Unsupervised tactical role clustering (`role`, `role_prob`) and expected market value / value-gap modeling (`expected_value_m`, `value_gap_m`) integrated into the pipeline and CLI |
| Streamlit Web App (`app/streamlit_app.py`) | Built and deployed to **Streamlit Community Cloud**, reading directly from `data/processed/players.csv` |
| API-Football / football-data.org / FPL loaders | Connected and running live via daily **GitHub Actions** (`refresh.yml`) using encrypted repository secrets |
| Shipped `data/processed/players.csv` | **Live Premier League data**, refreshed automatically every day by `scouting-bot` |

## Quick start

```bash
pip install -r requirements.txt -r requirements-ml.txt -r requirements-dev.txt
python -m pytest -q                              # 139 tests, ~10s, no network

# 1. Launch the interactive web dashboard locally
streamlit run app/streamlit_app.py

# 2. Demo via CLI, no network, no key (fictional players)
python -m scouting refresh --source synthetic
python -m scouting search "quenik"
python -m scouting recommend "Lioen Quenik" -k 5 --explain

# 3. Real data refresh via CLI
cp .env.example .env                             # put FOOTBALL_DATA_API_KEY and API_FOOTBALL_KEY in it
python -m scouting refresh --source fpl --football-data --api-football
python -m scouting recommend "Saka" -k 10 --role "Creative attackers"
```

Sample output (synthetic data):

```text
Query: Lioen Quenik (Cardew United, MID), age 29.8 [Prime (25-29)], value 12.9 [Mid-range], difficulty to buy: Hard (value, importance), injury: Low risk [history]

               name               club   age  similarity       age_band price_band  value_m difficulty_label    injury_risk current_status
rank
1      Maran Rasket  Stonehaven United  31.6        92.9  Veteran (30+)     Budget      4.4             Easy       Low risk      Available
2     Maras Varnric           Ironvale  31.2        87.3  Veteran (30+)  Mid-range     10.4         Moderate       Low risk      Available
```

Useful options: `--min-age/--max-age`, `--age-band`, `--price-band`, `--max-value`, `--max-difficulty {Easy,Moderate,Hard,Very hard}`, `--injury-risk`, `--role`, `--exclude-club`, `--min-similarity 60`, `--group-by price_band --per-group 3`, `--format csv|json`, `--explain`.

## Where the data comes from, and what each source can't do

| Source | Provides | Cannot provide |
|---|---|---|
| **football-data.org** (free key, 10 calls/min) | Standings (club position, games played). Squads with birth date and contract end *if your plan includes squads* (their [pricing page](https://www.football-data.org/pricing) lists squads as paid) | Market value, injuries, per-player performance stats. Without per-player stats it **cannot build a play-style profile on its own** |
| **FPL API** (no key, unofficial) | Per-player season stats (minutes, goals, assists, xG, xA, ICT components), FPL price, current injury status/news, multi-season historical summaries | Real transfer values, detailed historical injury logs. Positions are coarse: GK/DEF/MID/FWD, wingers count as MID |
| **Transfermarkt open dataset** ([dcaribou](https://github.com/dcaribou/transfermarkt-datasets), CC0) | Market value, birth date, contract end | Injuries. **Updates are paused** (current to 6 July 2026), so values are a snapshot |
| **API-Football** (free key, 100 requests/day) | **Per-player injury history**: type, start and end date, across all clubs. Confirmed on a real free-plan key | Free plan blocks season-based endpoints after 2024. The squad lists looked like a late-2025 snapshot, so recent transfers may be missing, and the newest injury records we saw were from 2025. The nightly job reports the newest record it finds. Terms not reviewed |
| **Injury file (`data/external/injuries.csv`)** | Pre-compiled or custom 3-year injury records (`player`, `from_date`, `days`) merged directly into the pipeline | Must be updated if new external injury sources are added |

Using Transfermarkt values:

```bash
# download players.csv(.gz) from the dataset's README / Kaggle (davidcariboo/player-scores)
python -m scouting import-transfermarkt players.csv.gz     # -> data/external/transfermarkt_pl.csv
python -m scouting refresh --source fpl --football-data    # picks it up automatically
```

`refresh` records match rates in `data/processed/metadata.json` and lists players it could not join in `data/processed/unmatched.csv`. Names are the only join key, so check these after the first run.

Injury history from API-Football:

```bash
# put API_FOOTBALL_KEY in .env, then:
python -m scouting refresh --source fpl --football-data --api-football
```
It finds each club's squad (about 40 requests once), matches players by name, then fetches each player's history, **most-played players first, at most 60 requests a day** (`--af-budget`). Everything is cached in `data/raw/`, so each run resumes where the last stopped. Players not loaded yet fall back to `data/external/injuries.csv` or the availability proxy until they are. The refresh output shows loaded/pending counts, the newest injury date seen, and which record types were counted as injuries vs excluded (suspensions, illness, loans, international duty).

Terms: raw API responses stay in `data/raw/` (git-ignored). Check API-Football's terms before publishing anything beyond derived per-player counts.

## How it works

**Similarity** (`scouting/engine.py`)
1. Season totals become per-90 rates, **shrunk toward the position average** so a 40-minute cameo cannot look like the league's best striker.
2. Within each position group, each profile stat becomes a percentile rank among players with enough minutes.
3. Distance = weighted Euclidean distance between percentile vectors. Which stats count for each position, and how much, is in `scouting/config.py`.
4. `similarity = 100 * exp(-(d/s)^2)`, where `s` is the median pairwise distance in that position. Around 60+ is a close match.
5. Each recommendation automatically computes `shared_strengths` and `key_difference` so scouts immediately see *why* two players matched.

On the synthetic data, where each player has a known playing style, the top-5 neighbours of outfield players share it 91-99% of the time (chance is 26-36%). Goalkeepers do worse (77-87% vs 52-64% by chance): their FPL stats mostly reflect the team in front of them. **This validates the maths, not real-world scouting quality.**

**Machine Learning Roles & Valuation** (`scouting/ml/`)
* **Tactical Role Clustering (`scouting/ml/roles.py`):** Fits unsupervised clustering models across historical multi-season player data (`train-roles`) using per-90 feature sets (`ict`, `xg`) and configurable population groupings (`outfield_joint` or per-position). Assigns each player a primary `role` (e.g., *Creative attackers*, *Full-backs and holding midfielders*, *Centre-backs*, *Goal-threat forwards*) and soft assignment confidence (`role_prob`). Saved models in `models/` are automatically applied during `refresh`.
* **Expected Value & Value Gap:** Estimates fair market value (`expected_value_m`) based on age, statistical output, and role, comparing it against actual market price (`value_m`) to compute `value_gap_m` and highlight undervalued transfer targets.

```bash
# Train and save a named tactical role model from historical seasons
python -m scouting train-roles \
  --seasons data/external/seasons.csv \
  --data data/processed/players.csv \
  --features ict \
  --grouping outfield_joint \
  --k 4 \
  --names "Creative attackers|Full-backs and holding midfielders|Centre-backs|Goal-threat forwards" \
  --model-out models/roles.joblib
```

**Categories** (`scouting/categories.py`; thresholds in `config.py`)

| Category | Rule |
|---|---|
| Age band | Prospect (<21), Developing (21-24), Prime (25-29), Veteran (30+) |
| Price band | Transfermarkt value: Budget <€10m, Mid-range <€30m, Premium <€60m, Elite ≥€60m. If under 60% of players have a market value, falls back to the **FPL price** with relative bands inside each position (`price_basis` says which was used) |
| Difficulty to buy | Weighted blend of value (35%), contract years left (25%), club league position (15%), share of club minutes played (15%), age (10%). Missing components are dropped and the rest re-weighted. Labels: Easy <35, Moderate <55, Hard <75, Very hard. **A transparent heuristic, not a validated model**; `difficulty_drivers` shows what pushed it up |
| Injury risk | With injury history (API-Football or `injuries.csv`): High if >90 days missed or ≥4 injuries in 3 years, Moderate if ≥30 days or ≥2, else Low (`injury_basis = history`). Without: an **availability proxy** (minutes played ÷ minutes available) for players with a meaningful sample (`availability_proxy`). Players with too few minutes get "No injury data" |

Current availability (Available / Doubtful / Injured / Suspended) comes from FPL's status flag and is shown separately.

## Interactive Web Dashboard (`app/streamlit_app.py`)

The project includes a full Streamlit web application deployed on **Streamlit Community Cloud**:
* **Player Search & Similarity Filtering:** Pick any Premier League player and filter replacements by age range, price band, acquisition difficulty, injury risk, and ML tactical role.
* **Visualizations & Comparison:** Inspect per-90 percentile profiles, market value vs. expected value gaps (`value_gap_m`), and side-by-side stat breakdowns.
* **Read-Only Cloud Architecture:** The Streamlit app reads the pre-computed `data/processed/players.csv` committed by GitHub Actions, requiring zero API keys in the frontend environment.

## Automation

`.github/workflows/refresh.yml` runs daily (and on demand via `workflow_dispatch`): runs the full `pytest` suite (139 tests), executes `refresh` against the live APIs, applies saved ML models from `models/`, and commits updated files in `data/processed/` back to `main` via `scouting-bot`.

Configured repository secrets (Settings → Secrets and variables → Actions):
1. `FOOTBALL_DATA_API_KEY`
2. `API_FOOTBALL_KEY`

Whenever `scouting-bot` pushes a new commit to `main`, Streamlit Community Cloud automatically reloads the live web dashboard with the latest data.

## Python API

The engine is a plain Python class, so scripts, notebooks, or dashboards can query it directly:

```python
from scouting import ScoutingEngine

eng = ScoutingEngine.from_csv("data/processed/players.csv")
eng.recommend("Player Name", k=10, max_age=25, price_bands=["Budget", "Mid-range"])
eng.recommend_by_category("Player Name", by="price_band", per_group=3)
eng.profile("Player Name")        # per-90 values and percentiles, e.g. for a radar chart
```

## Layout

```text
app/
  streamlit_app.py   interactive Streamlit web dashboard
scouting/
  config.py          thresholds, profile definitions, paths   <- tune the judgement calls here
  engine.py          similarity + recommend + filters
  categories.py      age / price / difficulty / injury
  features.py        per-90 + shrinkage + percentiles
  matching.py        name/club joins across sources
  enrich.py          joins: standings, squads, Transfermarkt, injuries
  pipeline.py        source -> enrich -> ML apply -> save (what the job runs)
  cli.py             command-line interface (refresh, search, profile, recommend, train-roles)
  injury_history.py  API-Football backfill within a daily budget
  ml/                unsupervised role clustering (roles.py) & valuation models
  sources/           footballdata.py, fpl.py, fpl_history.py, apifootball.py, files.py, synthetic.py, _http.py
models/              serialized .joblib ML models applied during pipeline refresh
data/external/       Transfermarkt + injury files + multi-season history
data/processed/      players.csv, metadata.json, unmatched.csv
tests/               139 unit and integration tests
```