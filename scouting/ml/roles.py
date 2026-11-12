"""Role model: soft clusters of player profiles, fitted on past seasons, applied to today's players.

Training: percentile features within each finished season (see dataset.py), k-means to find the
cluster centres, then a Gaussian mixture started from those centres so every player gets a
probability for each role instead of one hard label (a hybrid player is split between roles).

Applying to the current season: a player's per-90 rates are ranked against the MOST RECENT
finished season's distribution (not against this season's thin sample), then scored by the mixture.

Roles are numbered by size here ("role_0" is the largest). Naming them is a human step: read the
profile table that `train-roles` prints and pass `--names` or set config.ROLE_NAMES.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from ..features import add_per90, percentile_against
from .dataset import FEATURE_SETS, POPULATIONS, build_features


@dataclass
class PopulationModel:
    label: str
    positions: list[str]
    raw_features: list[str]            # per-90 column names, e.g. "threat_p90"
    reference: dict[str, np.ndarray]   # sorted per-90 values of the latest training season
    gmm: GaussianMixture
    order: np.ndarray                  # mixture component -> role number (0 = largest)
    sizes: np.ndarray                  # training share per role (by role number)
    centroids: np.ndarray              # (k, n_features) mean percentile per role, role-number order


@dataclass
class RoleModel:
    feature_set: str
    grouping: str
    k: int
    populations: dict[str, PopulationModel] = field(default_factory=dict)
    names: list[str] = field(default_factory=list)   # human names, in role_0..role_{k-1} order

    def role_label(self, role: str) -> str:
        """'role_2' -> its human name if one was given, else 'role_2'."""
        if not isinstance(role, str) or not role.startswith("role_"):
            return str(role)
        i = int(role.split("_")[1])
        return self.names[i] if i < len(self.names) and self.names[i] else role

    def display_key(self, role: str) -> str:
        """'role_0' -> 'role_0 (Creative attackers)' when named, else 'role_0'."""
        lbl = self.role_label(role)
        return f"{role} ({lbl})" if lbl != role else role

    def role_columns(self) -> list[str]:
        return [f"role_{i}" for i in range(self.k)]

    def assign(self, players: pd.DataFrame, *, min_minutes: float = 270.0) -> pd.DataFrame:
        """Role probabilities for current players (NaN where no population applies or too few minutes)."""
        df = add_per90(players.reset_index(drop=True))
        cols = self.role_columns() + ["role", "role_prob", "second_role", "second_role_prob"]
        out = pd.DataFrame(np.nan, index=players.index, columns=cols)
        out["role"] = pd.Series([None] * len(out), index=out.index, dtype=object)
        out["second_role"] = pd.Series([None] * len(out), index=out.index, dtype=object)

        for pm in self.populations.values():
            mask = (df["position_group"].isin(pm.positions) & (df["minutes"].fillna(0) >= min_minutes)).to_numpy()
            if not mask.any():
                continue
            feat_cols = [
                percentile_against(pm.reference[f], df.loc[mask, f].to_numpy(float))
                for f in pm.raw_features
            ]
            X = np.column_stack(feat_cols)
            ok = ~np.isnan(X).any(axis=1)
            if not ok.any():
                continue
            comp = pm.gmm.predict_proba(X[ok])
            probs = np.zeros_like(comp)
            probs[:, pm.order] = comp                      # reorder mixture components -> role numbers
            rows = out.index[mask][ok]
            out.loc[rows, self.role_columns()[: probs.shape[1]]] = probs
            top1 = probs.argmax(axis=1)
            out.loc[rows, "role"] = [f"role_{i}" for i in top1]
            out.loc[rows, "role_prob"] = probs.max(axis=1)
            if probs.shape[1] >= 2:
                order2 = np.argsort(-probs, axis=1)[:, 1]
                out.loc[rows, "second_role"] = [f"role_{i}" for i in order2]
                out.loc[rows, "second_role_prob"] = probs[np.arange(len(probs)), order2]
        return out


def fit_roles(
    seasons: pd.DataFrame, feature_set: str, grouping: str, k: int, *, seed: int = 0, names: list[str] | None = None,
) -> RoleModel:
    clean_names = [n.strip() for n in (names or []) if n.strip()]
    if clean_names and len(clean_names) != k:
        raise ValueError(f"expected {k} role names, got {len(clean_names)}")
    model = RoleModel(feature_set, grouping, k, names=clean_names)
    for label, positions in POPULATIONS[grouping].items():
        data, pct = build_features(seasons, feature_set, positions)
        if len(data) < 10 * k:
            raise ValueError(f"{label}: only {len(data)} player-seasons for k={k}; need at least {10 * k}")
        X = data[pct].to_numpy()
        km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(X)
        gmm = GaussianMixture(n_components=k, covariance_type="diag", means_init=km.cluster_centers_,
                              random_state=seed).fit(X)
        hard = gmm.predict(X)
        counts = np.bincount(hard, minlength=k)
        order_desc = np.argsort(-counts)                   # components sorted largest first
        order = np.empty(k, dtype=int)
        order[order_desc] = np.arange(k)                   # component -> role number
        raw = [c.removeprefix("pct_") for c in pct]
        latest = data["season_start_year"].max()
        ref = {f: np.sort(data.loc[data["season_start_year"] == latest, f].dropna().to_numpy(float)) for f in raw}
        cent = np.vstack([X[hard == comp].mean(axis=0) if (hard == comp).any() else gmm.means_[comp]
                          for comp in order_desc])
        model.populations[label] = PopulationModel(
            label, positions, raw, ref, gmm, order, counts[order_desc] / counts.sum(), cent)
    return model


def profile_table(model: RoleModel) -> dict[str, pd.DataFrame]:
    """Per population: share of training rows and mean percentile (0-100) of every feature, per role."""
    out = {}
    for label, pm in model.populations.items():
        t = pd.DataFrame(np.round(pm.centroids * 100).astype(int), columns=[f.removesuffix("_p90") for f in pm.raw_features])
        t.insert(0, "share", np.round(pm.sizes, 3))
        t.index = [model.display_key(f"role_{i}") for i in range(len(t))]
        out[label] = t
    return out


def training_memberships(model: RoleModel, seasons: pd.DataFrame) -> pd.DataFrame:
    """Role probabilities for every qualified past player-season (used for examples and diagnostics)."""
    frames = []
    for label, pm in model.populations.items():
        data, pct = build_features(seasons, model.feature_set, pm.positions)
        comp = pm.gmm.predict_proba(data[pct].to_numpy())
        probs = np.zeros_like(comp)
        probs[:, pm.order] = comp
        keep_cols = [c for c in ["player_id", "name", "season", "season_start_year", "position_group", "minutes"] if c in data.columns]
        part = data[keep_cols].copy()
        part["population"] = label
        part[[f"role_{i}" for i in range(model.k)]] = probs
        raw_roles = [f"role_{i}" for i in probs.argmax(axis=1)]
        part["raw_role"] = raw_roles
        part["role"] = raw_roles
        part["role_name"] = [model.role_label(r) for r in raw_roles]
        part["role_display"] = [model.display_key(r) for r in raw_roles]
        part["role_prob"] = probs.max(axis=1)
        frames.append(part)
    return pd.concat(frames, ignore_index=True)


def apply_roles(players: pd.DataFrame, model: RoleModel, *, min_minutes: float = 270.0) -> pd.DataFrame:
    """Return a copy of `players` with `role` (human name), `role_prob`, and secondary role columns."""
    assigned = model.assign(players, min_minutes=min_minutes)
    out = players.copy()
    out["role"] = pd.Series(
        [model.role_label(r) if isinstance(r, str) else np.nan for r in assigned["role"]],
        index=out.index, dtype=object,
    )
    out["role_prob"] = assigned["role_prob"].to_numpy()
    if "second_role" in assigned.columns:
        out["second_role"] = pd.Series(
            [model.role_label(r) if isinstance(r, str) else np.nan for r in assigned["second_role"]],
            index=out.index, dtype=object,
        )
        out["second_role_prob"] = assigned["second_role_prob"].to_numpy()
    return out 