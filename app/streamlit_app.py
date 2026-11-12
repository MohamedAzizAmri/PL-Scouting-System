"""Premier League Scouting Dashboard."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scouting import ScoutingEngine
from scouting import config as C
from scouting.categories import ORDERS

PLAYERS_CSV = ROOT / "data" / "processed" / "players.csv"
META_JSON = ROOT / "data" / "processed" / "metadata.json"

st.set_page_config(page_title="PL Scouting System", page_icon="⚽", layout="wide")


@st.cache_resource(show_spinner="Building player profiles...")
def load_engine(path: str, mtime: float) -> ScoutingEngine:
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

st.title("⚽ Premier League Scouting System")
st.markdown("Find similar player profiles using a hybrid rule-based and ML clustering engine.")

if eng.is_synthetic:
    st.warning("Synthetic demo data: fictional clubs and players, not real Premier League data.")
st.caption(
    f"{len(df)} players · data from {meta.get('source', 'unknown')} · "
    f"refreshed {str(meta.get('generated_at', 'unknown'))[:16].replace('T', ' ')} UTC"
)

# ------------------------------------------------------------------ sidebar #
with st.sidebar:
    st.header("Search & Filters")
    pool = df[df["eligible"]] if "eligible" in df else df
    names = sorted(pool["name"].unique())
    def_idx = names.index("Bukayo Saka") if "Bukayo Saka" in names else None
    player = st.selectbox("Search and pick a player", names, index=def_idx, placeholder="Type a name...")
    
    st.divider()
    st.subheader("Filter Recommendations")
    k = st.slider("How many results", 5, 25, 10)
    age_lo, age_hi = st.slider("Age", 15, 40, (15, 40))
    price = st.multiselect("Price band", ORDERS["price_band"])
    diff = st.selectbox("Hardest acceptable to buy", ["Any"] + ORDERS["difficulty_label"][:-1], index=0)
    injury = st.multiselect("Injury risk", ORDERS["injury_risk"])
    ages = st.multiselect("Age band", ORDERS["age_band"])
    role_options = sorted([r for r in df["role"].dropna().unique() if isinstance(r, str)]) if "role" in df else []
    roles = st.multiselect("Role (ML)", role_options, help="Only shown after `train-roles` and `assign-roles`") if role_options else []
    include_secondary = st.checkbox("Match secondary roles (≥25% prob)", value=True) if roles else False

    exclude_own = st.checkbox("Exclude the player's own club", value=True)
    include_departed = st.checkbox("Include players who left their club", value=False,
                                   help="Sold, loaned out or released: FPL flags them as unavailable")
    min_sim = st.slider("Minimum similarity", 0, 90, 0, help="Around 60+ is a close match")

if player is None:
    st.info("Pick a player in the sidebar to see similar profiles.")
    st.stop()

# ------------------------------------------------------------------ summary #
q = eng.query_summary(player)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Player", q["name"], f"{q['club']} · {q['position_group']}")
c2.metric("Age", f"{q['age']:.1f}", q["age_band"])
c3.metric("Value", "n/a" if pd.isna(q["value_m"]) else f"€{q['value_m']:.1f}m", q["price_band"])
c4.metric("Difficulty to buy", q["difficulty_label"])
c5.metric("Injury risk", q["injury_risk"], f"{q['injury_basis']} · {q['current_status']}", delta_color="off")
if isinstance(q["role"], str):
    st.caption(f"ML role: **{q['role']}**")
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
    min_similarity=min_sim or None, include_departed=include_departed, roles=roles or None,
)
if "include_secondary_role" in eng.recommend.__code__.co_varnames:
    filters["include_secondary_role"] = include_secondary

recs = eng.recommend(player, k, **filters)
if recs.empty:
    st.warning("No players match these filters. Loosen them in the sidebar.")
    st.stop()

# ------------------------------------------------------------------ main tabs #
tab_list, tab_compare, tab_map, tab_gems, tab_about = st.tabs([
    "🔍 Similar Players", "📊 Compare Profiles", "🗺️ ML Role Map", "💎 Undervalued Gems", "ℹ️ About Data"
])

SHOW = ["name", "club", "age", "similarity", "role", "age_band", "price_band", "value_m",
        "difficulty_label", "injury_risk", "current_status"]

with tab_list:
    # Fix 1: Replaced width="stretch" with use_container_width=True
    st.dataframe(
        recs[SHOW], use_container_width=True,
        column_config={
            "similarity": st.column_config.ProgressColumn("similarity", min_value=0, max_value=100, format="%.0f"),
            "age": st.column_config.NumberColumn(format="%.1f"),
            "value_m": st.column_config.NumberColumn("value (€m)", format="%.1f"),
        },
    )
    with st.expander("Why these players? (shared strengths and key difference)"):
        st.dataframe(recs[["name", "shared_strengths", "key_difference"]], use_container_width=True)
    st.download_button("Download as CSV", recs.reset_index().to_csv(index=False), f"similar_to_{q['name']}.csv")

with tab_compare:
    other = st.selectbox("Compare with", recs["name"].tolist())
    a, b = eng.profile(player), eng.profile(other)
    long = pd.concat([a.assign(player=q["name"]), b.assign(player=other)])
    
    # Fix 2: Changed to a facet chart (row="feature:N") which works on all Altair versions
    chart = (
        alt.Chart(long).mark_bar().encode(
            x=alt.X("percentile:Q", scale=alt.Scale(domain=[0, 100]), title="Percentile among same-position players"),
            y=alt.Y("player:N", title=None, axis=alt.Axis(labels=False, ticks=False)),
            color=alt.Color("player:N", title="Player", legend=alt.Legend(orient="bottom")),
            row=alt.Row("feature:N", title=None, sort=list(a["feature"]), header=alt.Header(labelAngle=0, labelAlign="left")),
            tooltip=["player", "feature", alt.Tooltip("per90:Q", format=".2f"), "percentile"],
        ).properties(height=30)
    )
    st.altair_chart(chart, use_container_width=True)

with tab_map:
    st.subheader("Tactical Landscape (PCA Projection)")
    st.markdown("Outfield players projected into a 2D space based on per-90 performance metrics. Colors represent GMM clustered roles.")
    
    outfield = df[(df["position_group"] != "GK") & df["eligible"] & df["role"].notna()].copy()
    feature_cols = [c for c in outfield.columns if c.endswith("_p90")]
    
    if feature_cols:
        X = outfield[feature_cols].fillna(0)
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X)
        outfield["PCA_1"] = coords[:, 0]
        outfield["PCA_2"] = coords[:, 1]
        
        # Highlight queried player and recommended players
        rec_names = recs["name"].tolist()
        def get_size(name):
            if name == q["name"]: return 250
            if name in rec_names: return 120
            return 40
            
        def get_opacity(name):
            if name == q["name"]: return 1.0
            if name in rec_names: return 0.9
            return 0.3

        outfield["Size"] = outfield["name"].apply(get_size)
        outfield["Opacity"] = outfield["name"].apply(get_opacity)
        
        scatter = alt.Chart(outfield).mark_circle().encode(
            x=alt.X("PCA_1", title="Attacking Output / Threat \u2192"),
            y=alt.Y("PCA_2", title="Defensive Action / Deep Influence \u2192"),
            color=alt.Color("role:N", title="Primary Role", scale=alt.Scale(scheme="category10")),
            size=alt.Size("Size:Q", legend=None),
            opacity=alt.Opacity("Opacity:Q", legend=None),
            tooltip=["name", "club", "role", "minutes", "market_value_eur_m"]
        ).properties(height=500).interactive()
        
        st.altair_chart(scatter, use_container_width=True)
    else:
        st.info("Performance features not available for projection.")

with tab_gems:
    st.subheader("Undervalued Targets (Model Residuals)")
    st.markdown("Players under 24 whose expected market value (based on stats, age, and role) exceeds their actual Transfermarkt valuation.")
    
    if "expected_value_m" in df.columns and "value_gap_m" in df.columns:
        gems = df[(df["age"] <= 24) & (df["minutes"] >= 400)].sort_values("value_gap_m", ascending=False)
        gems_disp = gems[["name", "club", "age", "role", "market_value_eur_m", "expected_value_m", "value_gap_m"]].copy()
        gems_disp = gems_disp.dropna(subset=["market_value_eur_m"]).head(20)
        
        # Fix 3: Removed hide_index=True entirely and caught any Pandas styling errors
        try:
            styled_gems = gems_disp.style.format({
                "age": "{:.1f}", "market_value_eur_m": "€{:.1f}m", 
                "expected_value_m": "€{:.1f}m", "value_gap_m": "€{:.1f}m"
            }).background_gradient(subset=["value_gap_m"], cmap="Blues")
            st.dataframe(styled_gems, use_container_width=True)
        except Exception:
            st.dataframe(gems_disp, use_container_width=True)
    else:
        st.warning("Value model data missing. Run `python -m scouting train-value` and `assign-value` first.")

with tab_about:
    st.markdown("""
**How similarity works.** Per-90 stats are shrunk toward the position average, turned into percentiles
within each position group, and compared with a weighted distance. Around 60+ is a close match.

**Difficulty to buy** is a transparent heuristic (value, contract, club position, minutes share, age),
not a validated model. **Injury risk** uses real history where loaded, otherwise an availability proxy
that cannot tell injury from rotation. Positions are FPL's coarse GK/DEF/MID/FWD.

### Machine Learning Notes
- **Roles (GMM Clustering):** FPL lacks progressive passing/carrying data, which is why unsupervised clustering cleanly separates 4 macro-roles (0.77 stability) rather than 10+ micro-roles.
- **Valuation (Gradient Boosting):** Predicts expected transfer value based on age and on-pitch output. (Note: historical FPL stats have survivorship bias as they are only fetched for active 2026/27 players).
- **Injury API Limitation:** API-Football's free tier caps injury records around July 2026. Therefore, 3-year injury proneness comes from API-Football, while live injury status comes from FPL.
    """)
    st.json({k: meta.get(k) for k in ("source", "n_players", "coverage", "warnings")}, expanded=False)