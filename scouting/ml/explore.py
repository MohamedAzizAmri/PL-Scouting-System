"""Experiment harness: how many roles does the data support, and are they stable?

For each combination of feature set and grouping of positions it fits clusterings for a range of
k and reports, per k:
  bic        Gaussian-mixture BIC (lower is better; flattening out = enough components)
  silhouette how separated the k-means clusters are (higher is better, >0.25 is decent)
  stability  mean adjusted Rand index between clusterings fitted on different bootstrap
             resamples and applied to the same players (1 = identical every time)
  min_share  size of the smallest cluster (tiny clusters are usually noise)
No ground truth exists for "a role", so stability and interpretability are the honest tests.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture

from .dataset import FEATURE_SETS, POPULATIONS, build_features


def stability(X: np.ndarray, k: int, *, n_pairs: int = 8, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n_pairs):
        labels = []
        for _ in range(2):
            idx = rng.integers(0, len(X), len(X))
            km = KMeans(n_clusters=k, n_init=3, random_state=int(rng.integers(1 << 30))).fit(X[idx])
            labels.append(km.predict(X))
        scores.append(adjusted_rand_score(labels[0], labels[1]))
    return float(np.mean(scores))


def evaluate_k(X: np.ndarray, k: int, *, seed: int = 0, n_pairs: int = 8) -> dict:
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
    gmm = GaussianMixture(n_components=k, covariance_type="diag", n_init=3, random_state=seed).fit(X)
    return {
        "k": k,
        "bic": round(float(gmm.bic(X)), 1),
        "silhouette": round(float(silhouette_score(X, km.labels_)), 3),
        "stability": round(stability(X, k, n_pairs=n_pairs, seed=seed), 3),
        "min_share": round(float(np.bincount(km.labels_).min() / len(X)), 3),
    }


def explore(
    seasons: pd.DataFrame,
    *,
    feature_sets: list[str] | None = None,
    populations: list[str] | None = None,
    ks: range = range(2, 9),
    n_pairs: int = 8,
    min_rows: int = 60,
) -> pd.DataFrame:
    rows = []
    for fs in feature_sets or list(FEATURE_SETS):
        for pop in populations or list(POPULATIONS):
            for label, positions in POPULATIONS[pop].items():
                data, pct = build_features(seasons, fs, positions)
                if len(data) < min_rows:
                    rows.append({"features": fs, "grouping": pop, "population": label, "n": len(data),
                                 "k": np.nan, "note": f"only {len(data)} rows (< {min_rows}): skipped"})
                    continue
                X = data[pct].to_numpy()
                for k in ks:
                    if k >= len(data) // 10:
                        break
                    rows.append({"features": fs, "grouping": pop, "population": label, "n": len(data),
                                 **evaluate_k(X, k, n_pairs=n_pairs)})
    return pd.DataFrame(rows)
