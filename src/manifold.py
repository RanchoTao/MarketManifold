from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA
from sklearn.manifold import MDS


def standardize_window(window_returns: pd.DataFrame) -> pd.DataFrame:
    centered = window_returns - window_returns.mean(axis=0)
    std = window_returns.std(axis=0, ddof=1).replace(0, np.nan)
    return (centered / std).fillna(0.0)


def correlation_distance(corr: pd.DataFrame) -> pd.DataFrame:
    clipped = corr.clip(-1, 1)
    dist = np.sqrt(np.maximum(0, 2 * (1 - clipped)))
    np.fill_diagonal(dist.values, 0.0)
    return dist


def mds_coordinates(distance: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    model = MDS(n_components=2, dissimilarity="precomputed", random_state=random_state, n_init=4, max_iter=300)
    coords = model.fit_transform(distance.values)
    return pd.DataFrame(coords, index=distance.index, columns=["x_raw", "y_raw"])


def pca_coordinates(window_returns: pd.DataFrame, random_state: int = 42) -> tuple[pd.DataFrame, float]:
    standardized = standardize_window(window_returns)
    model = PCA(n_components=2, random_state=random_state)
    coords = model.fit_transform(standardized.T.values)
    ratio = float(model.explained_variance_ratio_[0]) if len(model.explained_variance_ratio_) else np.nan
    return pd.DataFrame(coords, index=window_returns.columns, columns=["x_pca", "y_pca"]), ratio


def align_to_previous(previous: pd.DataFrame | None, current: pd.DataFrame) -> pd.DataFrame:
    if previous is None:
        aligned = current.copy()
        aligned.columns = ["x", "y"]
        return aligned
    common = previous.index.intersection(current.index)
    prev = previous.loc[common, ["x", "y"]].to_numpy()
    curr = current.loc[common, ["x_raw", "y_raw"]].to_numpy()
    prev_center = prev.mean(axis=0)
    curr_center = curr.mean(axis=0)
    curr0 = curr - curr_center
    prev0 = prev - prev_center
    rotation, _ = orthogonal_procrustes(curr0, prev0)
    all_curr = current[["x_raw", "y_raw"]].to_numpy()
    aligned_values = (all_curr - curr_center) @ rotation + prev_center
    return pd.DataFrame(aligned_values, index=current.index, columns=["x", "y"])
