"""Streamlit front end for the scouting engine.

    streamlit run app/streamlit_app.py

Read-only: it only loads data/processed/players.csv. It never calls an API and needs no
keys, so it is safe to host publicly. Data refreshes happen in `python -m scouting refresh`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scouting import ScoutingEngine  # noqa: E402
from scouting import config as C  # noqa: E402
from scouting.categories import ORDERS  # noqa: E402

PLAYERS_CSV = ROOT / "data" / "processed" / "players.csv"
META_JSON = ROOT / "data" / "processed" / "metadata.json"

st.set_page_config(page_title="PL Scouting", page_icon="⚽", layout="wide")


@st.cache_resource(show_spinner="Building player profiles...")
def load_engine(path: str, mtime: float) -> ScoutingEngine:   # mtime busts the cache after a refresh
    return ScoutingEngine.from_csv(path)


def load_meta() -> dict:
    try:
        return json.loads(META_JSON.read_text())
    except (OSError, ValueError):
        return {}


if not PLAYERS_CSV.exists():
    st.error("data/processed/players.csv not found. Run `python -m scouting refresh` first.")
    st.stop()

eng = load_engine(str(PLAYERS_CSV), PLAYERS_CSV.stat().st_mtime)
meta = load_meta()
df = eng.df

st.title("Premier League scouting: find players with a similar profile")
if eng.is_synthetic:
    st.warning("Synthetic demo data: fictional clubs and players, not real Premier League data.")
st.caption(
    f"{len(df)} players · data from {meta.get('source', 'unknown')} · "
    f"refreshed {str(meta.get('generated_at', 'unknown'))[:16].replace('T', ' ')} UTC"
)

# ------------------------------------------------------------------ sidebar #
with st.sidebar:
    st.header("Player")
    pool = df[df["eligible"]] if "eligible" in df else df
    names = sorted(pool["name"].unique())
    player = st.selectbox("Search and pick a player", names, index=None, placeholder="Type a name...")
    st.header("Filters")
    k = st.slider("How many results", 5, 25, 10)
    age_lo, age_hi = st.slider("Age", 15, 40, (15, 40))
    price = st.multiselect("Price band", ORDERS["price_band"])
    diff = st.selectbox("Hardest acceptable to buy", ["Any"] + ORDERS["difficulty_label"][:-1], index=0)
    injury = st.multiselect("Injury risk", ORDERS["injury_risk"])
    ages = st.multiselect("Age band", ORDERS["age_band"])
    exclude_own = st.checkbox("Exclude the player's own club", value=True)
    include_departed = st.checkbox("Include players who left their club", value=False,
                                   help="Sold, loaned out or released: FPL flags them as unavailable")
    min_sim = st.slider("Minimum similarity", 0, 90, 0, help="Around 60+ is a close match")

if player is None:
    st.info("Pick a player in the sidebar to see similar profiles.")
    st.stop()

q = eng.query_summary(player)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Player", q["name"], f"{q['club']} · {q['position_group']}")
c2.metric("Age", f"{q['age']:.1f}", q["age_band"])
c3.metric("Value", "n/a" if pd.isna(q["value_m"]) else f"{q['value_m']:.1f}m", q["price_band"])
c4.metric("Difficulty to buy", q["difficulty_label"])
c5.metric("Injury risk", q["injury_risk"], f"{q['injury_basis']} · {q['current_status']}", delta_color="off")
if q["departed"]:
    st.warning(f"{q['name']} has left the club: {q['news']}")
if not q["eligible"]:
    st.caption(f"Only {q['minutes']:.0f} minutes played: this profile is low-confidence.")

filters = dict(
    min_age=age_lo if age_lo > 15 else None, max_age=age_hi if age_hi < 40 else None,
    age_bands=ages or None, price_bands=price or None,
    max_difficulty=None if diff == "Any" else diff,
    injury_risk=injury or None,
    exclude_clubs=[q["club"]] if exclude_own else None,
    min_similarity=min_sim or None, include_departed=include_departed,
)
recs = eng.recommend(player, k, **filters)
if recs.empty:
    st.warning("No players match these filters. Loosen them in the sidebar.")
    st.stop()

tab_list, tab_compare, tab_groups, tab_about = st.tabs(["Similar players", "Compare profiles", "By category", "About the data"])

SHOW = ["name", "club", "age", "similarity", "age_band", "price_band", "value_m",
        "difficulty_label", "injury_risk", "current_status"]
with tab_list:
    st.dataframe(
        recs[SHOW], width="stretch",
        column_config={
            "similarity": st.column_config.ProgressColumn("similarity", min_value=0, max_value=100, format="%.0f"),
            "age": st.column_config.NumberColumn(format="%.1f"),
            "value_m": st.column_config.NumberColumn("value (m)", format="%.1f"),
        },
    )
    with st.expander("Why these players? (shared strengths and key difference)"):
        st.dataframe(recs[["name", "shared_strengths", "key_difference"]], width="stretch")
    st.download_button("Download as CSV", recs.reset_index().to_csv(index=False), f"similar_to_{q['name']}.csv")

with tab_compare:
    other = st.selectbox("Compare with", recs["name"].tolist())
    a, b = eng.profile(player), eng.profile(other)
    long = pd.concat([a.assign(player=q["name"]), b.assign(player=other)])
    chart = (
        alt.Chart(long).mark_bar().encode(
            y=alt.Y("feature:N", title=None, sort=list(a["feature"])),
            x=alt.X("percentile:Q", scale=alt.Scale(domain=[0, 100]), title="percentile among same-position players"),
            color=alt.Color("player:N", title=None),
            yOffset="player:N",
            tooltip=["player", "feature", alt.Tooltip("per90:Q", format=".2f"), "percentile"],
        ).properties(height=40 * len(a) + 40)
    )
    st.altair_chart(chart, width="stretch")

with tab_groups:
    by = st.radio("Group by", list(ORDERS), horizontal=True, format_func=lambda s: s.replace("_", " "))
    per = st.slider("Per group", 1, 6, 3)
    for label, part in eng.recommend_by_category(player, by=by, per_group=per, **filters).items():
        st.markdown(f"**{label}**")
        st.dataframe(part[SHOW], width="stretch")

with tab_about:
    st.markdown(
        """
**How similarity works.** Per-90 stats are shrunk toward the position average, turned into percentiles
within each position group, and compared with a weighted distance. Around 60+ is a close match.

**Difficulty to buy** is a transparent heuristic (value, contract, club position, minutes share, age),
not a validated model. **Injury risk** uses real history where loaded, otherwise an availability proxy
that cannot tell injury from rotation. Positions are FPL's coarse GK/DEF/MID/FWD.

Early in the season the sample is small, so profiles are noisy.
"""
    )
    st.json({k: meta.get(k) for k in ("source", "n_players", "coverage", "warnings")}, expanded=False)
