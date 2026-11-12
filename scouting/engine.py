"""The scouting engine: given a player, find players with a similar profile.

How similarity works
--------------------
1. Convert season totals to per-90 rates, shrunk toward the position average so
   small samples do not look like stars (features.add_per90).
2. Within each position group, turn each profile stat into a percentile rank among
   eligible players (enough minutes). Percentiles are robust to outliers and put
   xG/90 and creativity/90 on the same 0-1 scale.
3. Distance between two players = weighted Euclidean distance between their
   percentile vectors (weights in config.PROFILE_FEATURES).
4. similarity = 100 * exp(-(d / s)^2), where s is the median pairwise distance
   within the position group. d == s scores ~37; half that, ~78.

Players are only compared within their own position group (FPL's coarse GK / DEF /
MID / FWD), because percentiles are computed per group.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from difflib import get_close_matches
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .categories import ORDERS, add_categories
from .features import add_per90, percentile_against, resolve_min_minutes
from .matching import normalize_name
from .schema import ensure_columns, validate

log = logging.getLogger(__name__)

OUTPUT_COLUMNS = [
    "name", "club", "position_group", "age", "similarity", "age_band", "price_band", "value_m", "expected_value_m", "value_gap_m",
    "difficulty_label", "difficulty_score", "injury_risk", "injury_basis", "availability_rate",
    "current_status", "role", "role_prob", "shared_strengths", "key_difference",
]


class PlayerNotFound(LookupError):
    def __init__(self, query: str, suggestions: list[str]):
        self.query, self.suggestions = query, suggestions
        hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        super().__init__(f"No player matching {query!r}.{hint}")


class AmbiguousPlayer(LookupError):
    def __init__(self, query: str, candidates: pd.DataFrame):
        self.query, self.candidates = query, candidates
        opts = "; ".join(f"{r.name} ({r.club})" for r in candidates.itertuples())
        super().__init__(f"{query!r} matches several players: {opts}. Use the full name.")


class InsufficientData(ValueError):
    pass


@dataclass
class _Group:
    idx: np.ndarray          # row positions in engine.df
    eligible: np.ndarray     # bool per member
    features: list[str]
    weights: np.ndarray
    pct: np.ndarray          # (members, features) percentile ranks, NaN -> 0.5
    scale: float             # median pairwise distance among eligible members


def _wdist(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> np.ndarray:
    diff = a[:, None, :] - b[None, :, :]
    return np.sqrt(((diff ** 2) * w).sum(axis=-1) / w.sum())


class ScoutingEngine:
    def __init__(
        self,
        players: pd.DataFrame,
        *,
        min_minutes: str | float = C.MIN_MINUTES,
        shrink_minutes: float = C.SHRINK_MINUTES,
        today: pd.Timestamp | None = None,
    ):
        df = ensure_columns(players).reset_index(drop=True)
        validate(df)
        df = add_per90(df, shrink_minutes)
        self.min_minutes = resolve_min_minutes(df["minutes"], min_minutes)
        df["eligible"] = df["minutes"].fillna(0) >= self.min_minutes
        # FPL flags players who have left their club (sold, loaned out, released) with status "u".
        # They stay in the percentile pool (their stats are real) but are not recommended by default.
        df["departed"] = df["status"].astype(str).str.lower().eq("u")
        df = add_categories(df, today)
        df["name_norm"] = df["name"].map(normalize_name)
        df["web_norm"] = df["web_name"].map(normalize_name)
        self.df = df
        self.groups: dict[str, _Group] = {}
        for pos in C.POSITION_GROUPS:
            group = self._build_group(pos)
            if group is None:
                log.warning("not enough eligible %s players (min %.0f minutes): skipped", pos, self.min_minutes)
            else:
                self.groups[pos] = group
        if not self.groups:
            raise InsufficientData(
                f"No position group has enough players with >= {self.min_minutes:.0f} minutes. "
                "Is the season too early, or is the minutes column empty?"
            )

    @classmethod
    def from_csv(cls, path: Path | str, **kwargs) -> "ScoutingEngine":
        return cls(pd.read_csv(path), **kwargs)

    @property
    def is_synthetic(self) -> bool:
        return bool((self.df["data_source"] == "synthetic").any())

    # ------------------------------------------------------------------ build #
    def _build_group(self, pos: str) -> _Group | None:
        df = self.df
        idx = df.index[df["position_group"] == pos].to_numpy()
        if idx.size == 0:
            return None
        elig = df.loc[idx, "eligible"].to_numpy(bool)
        if elig.sum() < 5:
            return None
        feats = [
            f for f in C.PROFILE_FEATURES[pos]
            if f in df.columns and df.loc[idx[elig], f].notna().sum() >= 5
        ]
        if not feats:
            return None
        pct = np.column_stack([
            percentile_against(df.loc[idx[elig], f].to_numpy(float), df.loc[idx, f].to_numpy(float))
            for f in feats
        ])
        pct = np.where(np.isnan(pct), 0.5, pct)
        weights = np.array([C.PROFILE_FEATURES[pos][f] for f in feats], dtype=float)
        pe = pct[elig]
        dist = _wdist(pe, pe, weights)
        scale = max(float(np.median(dist[np.triu_indices(len(pe), 1)])), 1e-6)
        return _Group(idx, elig, feats, weights, pct, scale)

    # ----------------------------------------------------------------- lookup #
    def _candidates(self, query: str) -> tuple[str, list[int]]:
        q, df = normalize_name(query), self.df
        if not q:
            return "none", []
        exact = df.index[(df["name_norm"] == q) | (df["web_norm"] == q)].tolist()
        if exact:
            return "exact", exact
        toks = q.split()
        has_all = lambda s: all(t in s.split() for t in toks)  # noqa: E731
        contains = df.index[df["name_norm"].map(has_all) | df["web_norm"].map(has_all)].tolist()
        if contains:
            return "contains", contains
        pool = {**{n: i for i, n in df["name_norm"].items()}, **{n: i for i, n in df["web_norm"].items() if n}}
        close = get_close_matches(q, list(pool), n=5, cutoff=0.7)
        return "fuzzy", list(dict.fromkeys(pool[c] for c in close))

    def search(self, query: str, limit: int = 10) -> pd.DataFrame:
        _, idx = self._candidates(query)
        out = self.df.loc[idx[:limit], ["name", "web_name", "club", "position_group", "minutes", "current_status", "news"]].copy()
        out["news"] = out["news"].where(self.df.loc[out.index, "departed"], "")   # only shown for players who left
        return out.rename(columns={"news": "left_club_note"})

    def resolve(self, player: str | int) -> int:
        if isinstance(player, (int, np.integer)):
            if player in self.df.index:
                return int(player)
            raise PlayerNotFound(str(player), [])
        kind, idx = self._candidates(player)
        if kind in ("exact", "contains"):
            if len(idx) == 1:
                return idx[0]
            raise AmbiguousPlayer(player, self.df.loc[idx[:8], ["name", "club"]])
        raise PlayerNotFound(player, [self.df.at[i, "name"] for i in idx])

    # ---------------------------------------------------------------- profile #
    def profile(self, player: str | int) -> pd.DataFrame:
        """Per-90 values and percentile rank (vs. same-position eligible players)."""
        i = self.resolve(player)
        g = self.groups.get(self.df.at[i, "position_group"])
        if g is None:
            raise InsufficientData("No profile available for this position group.")
        pos_in_group = int(np.where(g.idx == i)[0][0])
        return pd.DataFrame({
            "feature": [C.FEATURE_LABELS.get(f, f) for f in g.features],
            "per90": [self.df.at[i, f] for f in g.features],
            "percentile": np.round(g.pct[pos_in_group] * 100).astype(int),
        })

    # -------------------------------------------------------------- recommend #
    def _explain(self, g: _Group, q: int, c: int) -> tuple[str, str]:
        pq, pc = g.pct[q].copy(), g.pct[c].copy()
        for j, f in enumerate(g.features):  # orient so higher is always "better"
            if f in C.LOWER_IS_BETTER:
                pq[j], pc[j] = 1 - pq[j], 1 - pc[j]
        both = np.minimum(pq, pc)
        close = np.abs(pq - pc) <= 0.2          # a "shared" strength must be similar in BOTH players
        order = [j for j in np.argsort(-(g.weights * both)) if both[j] >= 0.6 and close[j]][:3]
        label = lambda j: C.FEATURE_LABELS.get(g.features[j], g.features[j])  # noqa: E731
        shared = ", ".join(f"{label(j)} (P{pq[j]*100:.0f}/P{pc[j]*100:.0f})" for j in order) or "no standout shared strength"
        gap = g.weights * np.abs(pq - pc)
        gap[order] = -1                          # never report a stat as both shared and different
        j = int(np.argmax(gap))
        diff = f"{label(j)} (P{pq[j]*100:.0f} vs P{pc[j]*100:.0f})" if gap[j] > 0 else "no notable difference"
        return shared, diff

    def recommend(
        self,
        player: str | int,
        k: int = 10,
        *,
        min_age: float | None = None,
        max_age: float | None = None,
        age_bands: list[str] | None = None,
        max_value_m: float | None = None,
        price_bands: list[str] | None = None,
        max_difficulty: str | None = None,
        injury_risk: list[str] | None = None,
        exclude_clubs: list[str] | None = None,
        min_similarity: float | None = None,
        include_low_minutes: bool = False,
        include_departed: bool = False,
        roles: list[str] | None = None,
        include_secondary_role: bool = False,
    ) -> pd.DataFrame:
        """Players most similar to `player`, optionally filtered by category.

        `max_value_m` applies to `value_m`, i.e. EUR millions when market values are
        loaded, otherwise the FPL price in GBP millions (see the `price_basis` column).
        Players who have left their club (FPL status "u") are skipped unless `include_departed`.
        """
        i = self.resolve(player)
        pos = self.df.at[i, "position_group"]
        g = self.groups.get(pos)
        if g is None:
            raise InsufficientData(f"Not enough {pos} players to build profiles.")
        q = int(np.where(g.idx == i)[0][0])

        dist = _wdist(g.pct[q][None, :], g.pct, g.weights)[0]
        cand = self.df.iloc[g.idx].copy()
        cand["distance"] = dist
        cand["similarity"] = np.round(100 * np.exp(-((dist / g.scale) ** 2)), 1)
        pos_lookup = {row_i: n for n, row_i in enumerate(g.idx)}

        keep = cand.index != i
        if not include_low_minutes:
            keep &= cand["eligible"].to_numpy()
        if not include_departed:
            keep &= ~cand["departed"].to_numpy()
        if roles:
            if self.df["role"].isna().all():
                raise ValueError("no roles in the data: run `python -m scouting train-roles` then `assign-roles`")
            in_primary = cand["role"].isin(roles).to_numpy()
            if include_secondary_role and "second_role" in cand.columns and "second_role_prob" in cand.columns:
                in_secondary = (
                    cand["second_role"].isin(roles).to_numpy()
                    & (cand["second_role_prob"].fillna(0).to_numpy() >= 0.25)
                )
                keep &= (in_primary | in_secondary)
            else:
                keep &= in_primary
            
        if min_age is not None:
            keep &= (cand["age"] >= min_age).to_numpy()
        if max_age is not None:
            keep &= (cand["age"] <= max_age).to_numpy()
        if age_bands:
            keep &= cand["age_band"].isin(age_bands).to_numpy()
        if max_value_m is not None:
            keep &= (cand["value_m"] <= max_value_m).to_numpy()
        if price_bands:
            keep &= cand["price_band"].isin(price_bands).to_numpy()
        if max_difficulty is not None:
            if max_difficulty not in C.DIFFICULTY_LABELS:
                raise ValueError(f"max_difficulty must be one of {C.DIFFICULTY_LABELS}")
            allowed = C.DIFFICULTY_LABELS[: C.DIFFICULTY_LABELS.index(max_difficulty) + 1]
            keep &= cand["difficulty_label"].isin(allowed).to_numpy()
        if injury_risk:
            keep &= cand["injury_risk"].isin(injury_risk).to_numpy()
        if exclude_clubs:
            excluded = {normalize_name(c) for c in exclude_clubs}
            keep &= ~cand["club"].map(normalize_name).isin(excluded).to_numpy()
        if min_similarity is not None:
            keep &= (cand["similarity"] >= min_similarity).to_numpy()

        top = cand[keep].sort_values("similarity", ascending=False).head(k).copy()
        expl = [self._explain(g, q, pos_lookup[idx]) for idx in top.index]
        top["shared_strengths"] = [e[0] for e in expl]
        top["key_difference"] = [e[1] for e in expl]
        top = top.reset_index(drop=True)
        top.index = top.index + 1
        top.index.name = "rank"
        present_cols = [c for c in OUTPUT_COLUMNS if c in top.columns]
        return top[present_cols]

    def recommend_by_category(
        self, player: str | int, by: str = "price_band", per_group: int = 3, **filters
    ) -> dict[str, pd.DataFrame]:
        """Top `per_group` similar players inside each category value, e.g. each price band."""
        if by not in ORDERS:
            raise ValueError(f"`by` must be one of {list(ORDERS)}")
        ranked = self.recommend(player, k=len(self.df), **filters)
        out = {}
        for label in ORDERS[by]:
            part = ranked[ranked[by] == label].head(per_group)
            if not part.empty:
                out[label] = part
        return out

    def query_summary(self, player: str | int) -> pd.Series:
        i = self.resolve(player)
        cols = ["name", "club", "position_group", "age", "age_band", "value_m", "price_band",
                "price_basis", "difficulty_label", "difficulty_score", "difficulty_drivers",
                "injury_risk", "injury_basis", "current_status", "minutes", "eligible", "departed", "news", "role"]
        return self.df.loc[i, cols]
