# PL Scouting: find players with a similar profile

Give it a Premier League player. It returns players with a similar playing profile and labels each one by **age**, **market price**, **difficulty to buy** and **injury history**, so you can filter for, say, "a cheaper, younger, available version of this midfielder".

```
python -m scouting recommend "Player Name" -k 10 --max-age 25 --max-difficulty Hard --explain
```

## Status: read this first

| Part | State |
|---|---|
| Similarity engine, categories, CLI, pipeline | Built and tested (98 tests) |
| Tested on | **Synthetic players** (fictional names) and hand-written API payloads |
| API-Football injury loader | Response shapes confirmed from **real free-plan responses**; the full collection flow (team search, squads, per-player history) has only run against a fake session, not live |
| football-data.org / FPL loaders | Written from documentation, **never run against the live APIs** (the build environment blocked network access to them). Expect to fix small field-name issues on first run; the parsers return NaN for missing fields rather than crashing |
| Shipped `data/processed/players.csv` | **Synthetic demo data.** The first real `refresh` overwrites it |

## Quick start

```bash
pip install -r requirements-dev.txt
python -m pytest -q                              # 98 tests, ~5s, no network

# 1. Demo, no network, no key (fictional players)
python -m scouting refresh --source synthetic
python -m scouting search "quenik"
python -m scouting recommend "Lioen Quenik" -k 5 --explain

# 2. Real data
cp .env.example .env                             # put your football-data.org key in it
python -m scouting refresh --source fpl --football-data
python -m scouting recommend "Saka" -k 10
```

Sample output (synthetic data):

```
Query: Lioen Quenik (Cardew United, MID), age 29.8 [Prime (25-29)], value 12.9 [Mid-range], difficulty to buy: Hard (value, importance), injury: Low risk [history]

               name               club   age  similarity       age_band price_band  value_m difficulty_label    injury_risk current_status
rank
1      Maran Rasket  Stonehaven United  31.6        92.9  Veteran (30+)     Budget      4.4             Easy       Low risk      Available
2     Maras Varnric           Ironvale  31.2        87.3  Veteran (30+)  Mid-range     10.4         Moderate       Low risk      Available
```

Useful options: `--min-age/--max-age`, `--age-band`, `--price-band`, `--max-value`, `--max-difficulty {Easy,Moderate,Hard,Very hard}`, `--injury-risk`, `--exclude-club`, `--min-similarity 60`, `--group-by price_band --per-group 3`, `--format csv|json`, `--explain`.

## Where the data comes from, and what each source can't do

| Source | Provides | Cannot provide |
|---|---|---|
| **football-data.org** (free key, 10 calls/min) | Standings (club position, games played). Squads with birth date and contract end *if your plan includes squads* (their [pricing page](https://www.football-data.org/pricing) lists squads as paid) | Market value, injuries, per-player performance stats. Without per-player stats it **cannot build a play-style profile on its own** |
| **FPL API** (no key, unofficial) | Per-player season stats (minutes, goals, assists, xG, xA, ICT components), FPL price, current injury status/news | History (current season only; the job's daily runs build it up), real injury history, transfer values. Positions are coarse: GK/DEF/MID/FWD, wingers count as MID |
| **Transfermarkt open dataset** ([dcaribou](https://github.com/dcaribou/transfermarkt-datasets), CC0) | Market value, birth date, contract end | Injuries. **Updates are paused** (current to 6 July 2026), so values are a snapshot |
| **API-Football** (free key, 100 requests/day) | **Per-player injury history**: type, start and end date, across all clubs. Confirmed on a real free-plan key | Free plan blocks season-based endpoints after 2024. The squad lists looked like a late-2025 snapshot, so recent transfers may be missing, and the newest injury records we saw were from 2025. The nightly job reports the newest record it finds. Terms not reviewed |
| **Injury file (yours, optional)** | Real injury history from any source | You have to supply it |

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
It finds each club's squad (about 40 requests once), matches players by name, then fetches each player's history, **most-played players first, at most 60 requests a day** (`--af-budget`). A full first pass takes about ten days; everything is cached in `data/raw/`, so each run resumes where the last stopped. Players not loaded yet keep *no data* (never "0 injuries") and use the availability proxy until they are. The refresh output shows loaded/pending counts, the newest injury date seen, and which record types were counted as injuries vs excluded (suspensions, illness, loans, international duty).

Terms: raw API responses stay in `data/raw/` (git-ignored). Check API-Football's terms before publishing anything beyond derived per-player counts.

Injury file (alternative or extra source): copy the layout of `data/external/injuries.example.csv` to `data/external/injuries.csv`. It must cover every player you care about: anyone absent is treated as "no injuries recorded".

## How it works

**Similarity** (`scouting/engine.py`)
1. Season totals become per-90 rates, **shrunk toward the position average** so a 40-minute cameo cannot look like the league's best striker.
2. Within each position group, each profile stat becomes a percentile rank among players with enough minutes.
3. Distance = weighted Euclidean distance between percentile vectors. Which stats count for each position, and how much, is in `scouting/config.py`.
4. `similarity = 100 * exp(-(d/s)^2)`, where `s` is the median pairwise distance in that position. Around 60+ is a close match.

On the synthetic data, where each player has a known playing style, the top-5 neighbours of outfield players share it 91-99% of the time (chance is 26-36%). Goalkeepers do worse (77-87% vs 52-64% by chance): their FPL stats mostly reflect the team in front of them. **This validates the maths, not real-world scouting quality.**

**Categories** (`scouting/categories.py`; thresholds in `config.py`)

| Category | Rule |
|---|---|
| Age band | Prospect (<21), Developing (21-24), Prime (25-29), Veteran (30+) |
| Price band | Transfermarkt value: Budget <€10m, Mid-range <€30m, Premium <€60m, Elite ≥€60m. If under 60% of players have a market value, falls back to the **FPL price** with relative bands inside each position (`price_basis` says which was used) |
| Difficulty to buy | Weighted blend of value (35%), contract years left (25%), club league position (15%), share of club minutes played (15%), age (10%). Missing components are dropped and the rest re-weighted. Labels: Easy <35, Moderate <55, Hard <75, Very hard. **A transparent heuristic, not a validated model**; `difficulty_drivers` shows what pushed it up |
| Injury risk | With injury history (API-Football or your file): High if >90 days missed or ≥4 injuries in 3 years, Moderate if ≥30 days or ≥2, else Low (`injury_basis = history`). Without: an **availability proxy** (minutes played ÷ minutes available) for players with a meaningful sample (`availability_proxy`). The proxy cannot tell an injury from rotation or a manager's choice. Players with too few minutes get "No injury data" |

Current availability (Available / Doubtful / Injured / Suspended) comes from FPL's status flag and is shown separately.

## Automation

`.github/workflows/refresh.yml` runs daily (and on demand): tests, then `refresh`, then commits `data/processed/`. To enable it:
1. Push this folder to GitHub.
2. Repo Settings → Secrets and variables → Actions → new secret `FOOTBALL_DATA_API_KEY`.
3. Add a second secret `API_FOOTBALL_KEY` for injury history. The job caches API responses between runs (not committed), so the backfill carries on across nights.
4. Commit your `data/external/transfermarkt_pl.csv` (and `injuries.csv` if you have one) so the job can use them.

Without the secret the job still runs; football-data.org is skipped with a warning. Add `--strict` to the refresh step if you would rather a broken source turn the run red. The workflow is syntactically valid but has **not been run on GitHub**.

## Scaling up

The engine is a plain Python class, so a dashboard or demo app is a thin layer over it:

```python
from scouting import ScoutingEngine
eng = ScoutingEngine.from_csv("data/processed/players.csv")
eng.recommend("Player Name", k=10, max_age=25, price_bands=["Budget", "Mid-range"])
eng.recommend_by_category("Player Name", by="price_band", per_group=3)
eng.profile("Player Name")        # per-90 values and percentiles, e.g. for a radar chart
```

Natural next steps: a Streamlit app on top of this; archive daily snapshots to build season history; blend in last season's stats (FPL `element-summary`) to stabilise early-season profiles; richer event data for real positional roles.

## Layout

```
scouting/
  config.py        thresholds, profile definitions, paths   <- tune the judgement calls here
  engine.py        similarity + recommend + filters
  categories.py    age / price / difficulty / injury
  features.py      per-90 + shrinkage + percentiles
  matching.py      name/club joins across sources
  enrich.py        joins: standings, squads, Transfermarkt, injuries
  pipeline.py      source -> enrich -> save (what the job runs)
  cli.py
  injury_history.py  API-Football backfill within a daily budget
  sources/         footballdata.py, fpl.py, apifootball.py, files.py, synthetic.py, _http.py (cache, retries, rate limit)
data/external/     your Transfermarkt + injury files
data/processed/    players.csv, metadata.json, unmatched.csv
tests/
```
