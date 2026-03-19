"""Bad-channel labeling (outlier detection) helpers.

This module holds the core logic so the Snakemake `scripts/badlabel.py` entrypoint
can remain a thin orchestrator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.ndimage as nd
from kneed import KneeLocator
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class DbscanParams:
    knn: int
    min_samples: int
    eps_optimize_k: int
    sigma: float
    eps: float | None
    eps_fallback: float = 1e-6


def _finite_row_mask(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError("Expected 2D feature matrix (n_samples, n_features)")
    return np.isfinite(x).all(axis=1)


def _estimate_eps_knee(x_scaled: np.ndarray, *, eps_optimize_k: int, sigma: float) -> float | None:
    if x_scaled.shape[0] <= eps_optimize_k + 1:
        return None
    nbrs = NearestNeighbors(n_neighbors=eps_optimize_k + 1).fit(x_scaled)
    distances, _ = nbrs.kneighbors(x_scaled)
    k_distances = np.sort(distances[:, eps_optimize_k], axis=0)
    k_distances = nd.gaussian_filter1d(k_distances, sigma=float(sigma))

    kn = KneeLocator(
        np.arange(1, len(k_distances) + 1),
        k_distances,
        curve="convex",
        direction="increasing",
    )
    return None if kn.knee_y is None else float(kn.knee_y)


def dbscan_outliers(x: np.ndarray, *, params: DbscanParams) -> tuple[np.ndarray, float]:
    """Return (is_outlier, eps_used) for a feature matrix."""
    x = np.asarray(x, dtype=np.float64)
    finite = _finite_row_mask(x)
    is_outlier = np.zeros((x.shape[0],), dtype=bool)

    if finite.sum() == 0:
        return is_outlier, float(params.eps_fallback)

    x_scaled = StandardScaler().fit_transform(x[finite])

    eps_used: float
    if params.eps is not None:
        eps_used = float(params.eps)
    else:
        eps_est = _estimate_eps_knee(x_scaled, eps_optimize_k=params.eps_optimize_k, sigma=params.sigma)
        eps_used = float(params.eps_fallback if eps_est is None else eps_est)

    labels_raw = DBSCAN(eps=eps_used, min_samples=int(params.min_samples)).fit_predict(x_scaled)
    is_outlier[finite] = labels_raw == -1
    return is_outlier, eps_used


def grouped_dbscan_outliers(
    x: np.ndarray,
    *,
    group_labels: np.ndarray | None,
    params: DbscanParams,
) -> tuple[np.ndarray, dict[Any, float]]:
    """DBSCAN outliers computed independently per group label (topological isolation)."""
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError("Expected 2D feature matrix (n_samples, n_features)")
    if group_labels is None:
        out, eps_used = dbscan_outliers(x, params=params)
        return out, {"__all__": eps_used}

    g = np.asarray(group_labels)
    if g.shape[0] != x.shape[0]:
        raise ValueError("group_labels must align with x rows")

    out = np.zeros((x.shape[0],), dtype=bool)
    eps_by_group: dict[Any, float] = {}
    for label in np.unique(g):
        mask = g == label
        out_mask, eps_used = dbscan_outliers(x[mask], params=params)
        out[mask] = out_mask
        eps_by_group[label] = eps_used
    return out, eps_by_group

