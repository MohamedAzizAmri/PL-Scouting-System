"""Step 2d: Evaluate similarity methods using consecutive-season self-retrieval.

Since there is no ground-truth label for 'similar player', we test temporal consistency:
if a player plays >= 900 minutes in season t and season t+1, how close is his season-(t+1)
profile to his own season-t profile compared to everyone else in season t+1?

Metrics reported across all consecutive (player, t -> t+1) pairs:
- median_rank: median rank of the player's future self (1 = best possible, lower is better)
- top5_rate: share of players whose future self ranks in the top 5 most similar
- top10_rate: share of players whose future self ranks in the top 10 most similar
- mrr: Mean Reciprocal Rank, mean(1 / rank)
- role_retention: how often a player keeps the same primary GMM role from season t to t+1
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

from .. import config as C
from .dataset import POPULATIONS, build_features


def _pairwise_sqdist(A: np.ndarray, B: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    diff = A[:, None, :] - B[None, :, :]
    if weights is None:
        return np.sqrt((diff ** 2).mean(axis=-1))
    w = weights / weights.sum()
    return np.sqrt(((diff ** 2) * w).sum(axis=-1))


def evaluate_consecutive_seasons(
    seasons: pd.DataFrame,
    *,
    feature_set: str = "ict",
    k_roles: int = 4,
    pca_components: int = 4,
    min_minutes: float = 900.0,
    seed: int = 0,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Compare heuristic baseline vs PCA and GMM-hybrid similarity on consecutive player-seasons."""
    positions = POPULATIONS["outfield_joint"]["DEF+MID+FWD"]
    
    # 1. Filter for minutes BEFORE passing to build_features
    qualified = seasons[seasons["minutes"].fillna(0) >= min_minutes]
    
    # 2. Call build_features WITHOUT the min_minutes argument
    data, pct_cols = build_features(qualified, feature_set, positions)
    
    if data.empty or len(pct_cols) == 0:
        raise ValueError("Not enough player-seasons to evaluate.")

    X = data[pct_cols].to_numpy(float)
    n_comp = min(pca_components, X.shape[1], max(1, len(X) - 1))
    pca = PCA(n_components=n_comp, random_state=seed).fit(X)
    Z = pca.transform(X)

    km = KMeans(n_clusters=k_roles, n_init=20, random_state=seed).fit(X)
    gmm = GaussianMixture(
        n_components=k_roles, covariance_type="diag", means_init=km.cluster_centers_, random_state=seed
    ).fit(X)
    probs = gmm.predict_proba(X)
    sqrt_probs = np.sqrt(np.clip(probs, 1e-9, 1.0))
    hard_role = probs.argmax(axis=1)

    # Heuristic feature weights (average of outfield weights in config.PROFILE_FEATURES)
    raw_names = [c.removeprefix("pct_") for c in pct_cols]
    heur_w = np.array([
        np.mean([C.PROFILE_FEATURES[p].get(f, 1.0) for p in ("DEF", "MID", "FWD")])
        for f in raw_names
    ], dtype=float)

    data = data.reset_index(drop=True).copy()
    data["row_idx"] = np.arange(len(data))
    data["role_id"] = hard_role

    years = sorted(data["season_start_year"].unique())
    records: list[dict] = []
    same_role_hits: list[int] = []

    for y in years:
        if (y + 1) not in years:
            continue
        cur = data[data["season_start_year"] == y]
        nxt = data[data["season_start_year"] == (y + 1)]
        common = set(cur["player_id"]).intersection(set(nxt["player_id"]))
        if len(common) < 10:
            continue

        idx_t = cur["row_idx"].to_numpy()
        idx_n = nxt["row_idx"].to_numpy()
        nxt_pids = nxt["player_id"].to_numpy()
        nxt_pos = nxt["position_group"].to_numpy()

        # Precompute distance matrices (len(cur) x len(nxt))
        d_equal = _pairwise_sqdist(X[idx_t], X[idx_n])
        d_heur = _pairwise_sqdist(X[idx_t], X[idx_n], heur_w)
        d_pca = _pairwise_sqdist(Z[idx_t], Z[idx_n])
        d_role = _pairwise_sqdist(sqrt_probs[idx_t], sqrt_probs[idx_n])
        d_hybrid = 0.7 * d_heur + 0.3 * d_role

        for local_i, r in enumerate(cur.itertuples()):
            if r.player_id not in common:
                continue
            target_mask = (nxt_pids == r.player_id)
            if not target_mask.any():
                continue
            target_j = int(np.where(target_mask)[0][0])
            same_role_hits.append(int(r.role_id == nxt.iloc[target_j]["role_id"]))

            # Same-position mask for strict positional heuristic baseline
            pos_mask = (nxt_pos == r.position_group)

            for method, d_vec, restrict_pos in [
                ("heuristic_pos", d_heur[local_i], True),
                ("heuristic_all", d_heur[local_i], False),
                ("equal_pct", d_equal[local_i], False),
                ("pca_space", d_pca[local_i], False),
                ("hybrid_gmm", d_hybrid[local_i], False),
                ("hybrid_gmm_pos", d_hybrid[local_i], True),
            ]:
                scores = d_vec.copy()
                if restrict_pos:
                    scores = np.where(pos_mask, scores, np.inf)
                target_d = scores[target_j]
                rank = int(1 + np.sum(scores < target_d))
                pool_size = int(np.sum(pos_mask)) if restrict_pos else len(nxt)
                records.append({
                    "method": method,
                    "player_id": r.player_id,
                    "season_t": r.season_start_year,
                    "rank": rank,
                    "pct_rank": rank / max(1, pool_size),
                    "top5": int(rank <= 5),
                    "top10": int(rank <= 10),
                    "rr": 1.0 / rank,
                })

    df_rec = pd.DataFrame(records)
    summary = (
        df_rec.groupby("method", as_index=False)
        .agg(
            pairs=("rank", "size"),
            median_rank=("rank", "median"),
            median_pct_rank=("pct_rank", lambda s: round(float(np.median(s) * 100), 1)),
            top5_rate=("top5", lambda s: round(float(s.mean()), 3)),
            top10_rate=("top10", lambda s: round(float(s.mean()), 3)),
            mrr=("rr", lambda s: round(float(s.mean()), 3)),
        )
        .sort_values("mrr", ascending=False)
        .reset_index(drop=True)
    )
    diagnostics = {
        "consecutive_pairs": int(len(same_role_hits)),
        "role_retention_rate": round(float(np.mean(same_role_hits)), 3) if same_role_hits else 0.0,
        "pca_explained_variance": round(float(pca.explained_variance_ratio_.sum()), 3),
    }
    return summary, diagnostics