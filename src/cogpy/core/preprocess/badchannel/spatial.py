"""Spatial neighborhood operations and feature normalization."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import scipy.ndimage as nd

try:  # optional at runtime for pure-grid use
    import scipy.sparse as sp
except Exception:  # pragma: no cover
    sp = None  # type: ignore

EPS = 1e-12
MAD_SCALE_NORMAL = 0.6744897501960817


def _as_csr(adj: Any):
    if sp is None:
        return None
    if sp.issparse(adj):
        return adj.tocsr()
    return None


def neighbors_from_adjacency(adj: Any, n_nodes: int | None = None) -> list[np.ndarray]:
    csr = _as_csr(adj)
    if csr is not None:
        out: list[np.ndarray] = []
        indptr = csr.indptr
        indices = csr.indices
        n = csr.shape[0]
        for i in range(n):
            out.append(indices[indptr[i] : indptr[i + 1]])
        return out

    if isinstance(adj, tuple) and len(adj) == 2:
        src, dst = adj
        if n_nodes is None:
            n_nodes = int(np.max(src)) + 1 if len(src) else 0
        buckets: list[list[int]] = [[] for _ in range(n_nodes)]
        for s, d in zip(np.asarray(src).tolist(), np.asarray(dst).tolist(), strict=False):
            buckets[int(s)].append(int(d))
        return [np.asarray(b, dtype=np.int64) for b in buckets]

    adj_arr = np.asarray(adj, dtype=bool)
    if adj_arr.ndim != 2 or adj_arr.shape[0] != adj_arr.shape[1]:
        raise ValueError("adj must be square (dense), sparse, or (src,dst) edges")
    return [np.where(adj_arr[i])[0].astype(np.int64) for i in range(adj_arr.shape[0])]


def neighborhood_median(values: np.ndarray, *, neighbors: list[np.ndarray]) -> np.ndarray:
    x = np.asarray(values)
    if x.ndim == 1:
        out = np.full((len(neighbors),), np.nan, dtype=np.float64)
        for i, nb in enumerate(neighbors):
            out[i] = np.nanmedian(x[nb]) if len(nb) else np.nan
        return out

    if x.ndim == 2:
        out = np.full((len(neighbors), x.shape[1]), np.nan, dtype=np.float64)
        for i, nb in enumerate(neighbors):
            out[i] = np.nanmedian(x[nb, :], axis=0) if len(nb) else np.nan
        return out

    raise ValueError("values must be 1D or 2D")


def neighborhood_mad(values: np.ndarray, *, neighbors: list[np.ndarray]) -> np.ndarray:
    x = np.asarray(values)
    if x.ndim == 1:
        out = np.full((len(neighbors),), np.nan, dtype=np.float64)
        for i, nb in enumerate(neighbors):
            if len(nb) == 0:
                continue
            med = np.nanmedian(x[nb])
            out[i] = np.nanmedian(np.abs(x[nb] - med))
        return out

    if x.ndim == 2:
        out = np.full((len(neighbors), x.shape[1]), np.nan, dtype=np.float64)
        for i, nb in enumerate(neighbors):
            if len(nb) == 0:
                continue
            med = np.nanmedian(x[nb, :], axis=0)
            out[i] = np.nanmedian(np.abs(x[nb, :] - med), axis=0)
        return out

    raise ValueError("values must be 1D or 2D")


def normalize_ratio(x: np.ndarray, neigh_med: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64) / (np.asarray(neigh_med, dtype=np.float64) + EPS)


def normalize_difference(x: np.ndarray, neigh_med: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64) - np.asarray(neigh_med, dtype=np.float64)


def normalize_robust_z(x: np.ndarray, neigh_med: np.ndarray, neigh_mad: np.ndarray) -> np.ndarray:
    num = np.asarray(x, dtype=np.float64) - np.asarray(neigh_med, dtype=np.float64)
    mad = np.asarray(neigh_mad, dtype=np.float64)
    denom = np.where(mad > 0, mad, np.nan)
    return 0.6744897501960817 * num / (denom + EPS)


def anticorrelation(arr: np.ndarray, *, neighbors: list[np.ndarray]) -> np.ndarray:
    grid_shape = arr.shape[:2]
    x = np.reshape(arr, (-1, arr.shape[-1]))
    if x.shape[0] == 1:
        return np.zeros(grid_shape, dtype=np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.corrcoef(x)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    med_corr = np.full((len(neighbors),), np.nan, dtype=np.float64)
    for i, nb in enumerate(neighbors):
        med_corr[i] = np.nanmedian(corr[i, nb]) if len(nb) else np.nan

    return (1.0 - med_corr).reshape(grid_shape)


def local_robust_zscore_grid(input_arr: np.ndarray, *, footprint: np.ndarray) -> np.ndarray:
    """Local robust z-score on a 2D grid using a footprint neighborhood.

    This matches the behavior used in the current preprocess `feature.py`:
    local center = nanmedian; local scale = MAD scaled to normal.
    """
    x = np.asarray(input_arr, dtype=np.float64)
    fp = np.asarray(footprint, dtype=bool)
    if x.ndim != 2:
        raise ValueError("input_arr must be 2D (AP, ML)")
    if fp.ndim != 2:
        raise ValueError("footprint must be 2D")

    filter_kwargs = dict(footprint=fp, mode="constant", cval=np.nan)

    def _scaled_mad(values: np.ndarray) -> float:
        med = np.nanmedian(values)
        mad = np.nanmedian(np.abs(values - med))
        return float(mad / MAD_SCALE_NORMAL)

    local_med = nd.generic_filter(x, function=np.nanmedian, **filter_kwargs)
    local_mad = nd.generic_filter(x, function=_scaled_mad, **filter_kwargs)
    denom = np.where(local_mad > 0, local_mad, np.nan)
    return (x - local_med) / (denom + EPS)
