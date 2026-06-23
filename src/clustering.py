from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score


def agglomerative_labels(distance: pd.DataFrame, n_clusters: int = 6) -> np.ndarray:
    kwargs = {"n_clusters": n_clusters, "linkage": "average"}
    try:
        model = AgglomerativeClustering(metric="precomputed", **kwargs)
    except TypeError:
        model = AgglomerativeClustering(affinity="precomputed", **kwargs)
    return model.fit_predict(distance.values)


def kmeans_labels(features: pd.DataFrame, n_clusters: int = 6, random_state: int = 42) -> np.ndarray:
    return KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10).fit_predict(features.values)


def safe_silhouette(distance: pd.DataFrame, labels: np.ndarray) -> float:
    if len(set(labels)) < 2 or len(set(labels)) >= len(labels):
        return float("nan")
    return float(silhouette_score(distance.values, labels, metric="precomputed"))


def ari(labels_a: np.ndarray | None, labels_b: np.ndarray | None) -> float:
    if labels_a is None or labels_b is None:
        return float("nan")
    return float(adjusted_rand_score(labels_a, labels_b))


def sector_scores(sectors: list[str], labels: np.ndarray) -> tuple[float, float]:
    return float(adjusted_rand_score(sectors, labels)), float(normalized_mutual_info_score(sectors, labels))

