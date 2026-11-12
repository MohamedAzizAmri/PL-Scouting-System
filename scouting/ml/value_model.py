"""Step 3: Market Value Regression Model.

Predicts log10(market_value_eur_m) using age, minutes, club standing, 
learned GMM role probabilities, and per-90 performance stats.
Uses HistGradientBoostingRegressor which natively handles missing values.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error

from ..features import add_per90

def prepare_value_data(players: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str]]:
    df = add_per90(players.copy())
    
    # Calculate age dynamically if it's missing but birth_date is present
    if "age" not in df.columns and "birth_date" in df.columns:
        df["age"] = (pd.Timestamp("2026-11-12") - pd.to_datetime(df["birth_date"], errors="coerce")).dt.days / 365.25
        
    # Only train on players who have a known Transfermarkt value and played >= 270 mins
    mask = df["market_value_eur_m"].notna() & (df["minutes"] >= 270)
    df = df[mask].reset_index(drop=True)
    
    # Core features
    features = [
        "age", "minutes", "club_position",
        "role_0", "role_1", "role_2", "role_3",
        "xg_p90", "xa_p90", "threat_p90", "creativity_p90",
        "influence_p90", "defensive_contribution_p90"
    ]
    # Ensure features exist in the dataframe
    use_feats = [f for f in features if f in df.columns]
    
    X = df[use_feats]
    # Predict Log10 of the value to handle the massive exponential skew of football transfers
    y = np.log10(df["market_value_eur_m"].astype(float))
    groups = df["club"]
    
    return df, X, y, groups, use_feats

def train_value_model(players: pd.DataFrame, seed: int = 42):
    df, X, y, groups, feats = prepare_value_data(players)
    
    # HistGradientBoosting handles NaNs natively and prevents overfitting on small datasets
    model = HistGradientBoostingRegressor(
        max_iter=150, max_depth=5, min_samples_leaf=5, l2_regularization=0.2, random_state=seed
    )
    
    # GroupKFold ensures players from the same club aren't split across train/test sets
    cv = GroupKFold(n_splits=5)
    maes_log = []
    maes_eur = []
    
    for train_idx, test_idx in cv.split(X, y, groups):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[test_idx])
        
        maes_log.append(mean_absolute_error(y.iloc[test_idx], preds))
        # Convert back from Log10 to actual Millions for an interpretable error metric
        maes_eur.append(mean_absolute_error(10**y.iloc[test_idx], 10**preds))
        
    cv_log = float(np.mean(maes_log))
    cv_eur = float(np.mean(maes_eur))
    
    # Train final model on 100% of the data
    model.fit(X, y)
    
    return model, feats, cv_log, cv_eur
    
def predict_values(players: pd.DataFrame, model, feats: list[str]) -> np.ndarray:
    df = add_per90(players.copy())
    
    # Impute missing feature columns with 0.0 just in case
    for f in feats:
        if f not in df.columns:
            df[f] = 0.0
            
    X = df[feats]
    log_preds = model.predict(X)
    return 10**log_preds  # Return to EUR Millions