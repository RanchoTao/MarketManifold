from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def off_diagonal_values(matrix: pd.DataFrame) -> np.ndarray:
    values = matrix.values
    mask = ~np.eye(values.shape[0], dtype=bool)
    return values[mask]


def window_metrics(window_returns: pd.DataFrame, corr: pd.DataFrame, distance: pd.DataFrame, coords: pd.DataFrame) -> dict:
    off_corr = off_diagonal_values(corr)
    off_dist = off_diagonal_values(distance)
    market_return = window_returns.mean(axis=1)
    pca = PCA(n_components=1)
    standardized = ((window_returns - window_returns.mean()) / window_returns.std(ddof=1).replace(0, np.nan)).fillna(0)
    pca.fit(standardized.values)
    coord_values = coords[["x", "y"]].values
    pairwise = []
    for i in range(len(coord_values)):
        for j in range(i + 1, len(coord_values)):
            pairwise.append(np.linalg.norm(coord_values[i] - coord_values[j]))
    return {
        "mean_correlation": float(np.nanmean(off_corr)),
        "median_correlation": float(np.nanmedian(off_corr)),
        "market_volatility": float(market_return.std(ddof=1) * np.sqrt(252)),
        "cross_sectional_dispersion": float(window_returns.std(axis=1, ddof=1).mean()),
        "pca_first_component_ratio": float(pca.explained_variance_ratio_[0]),
        "mean_pairwise_distance": float(np.nanmean(pairwise) if pairwise else np.nan),
        "mean_correlation_distance": float(np.nanmean(off_dist)),
    }

