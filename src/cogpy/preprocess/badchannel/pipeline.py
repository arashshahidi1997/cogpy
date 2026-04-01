"""Bad-channel feature pipeline helpers (Split/Apply/Combine building blocks).

This module intentionally focuses on in-memory operations used by Snakemake
scripts. It does not do file I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from cogpy.utils.sliding_core import window_centers_idx, window_onsets
from . import channel_features as cf
from .spatial import (
    anticorrelation as spatial_anticorrelation,
    neighbors_from_adjacency,
    neighborhood_mad,
    neighborhood_median,
    local_robust_zscore_grid,
    normalize_difference,
    normalize_ratio,
    normalize_robust_z,
)


Norm = Literal["identity", "ratio", "difference", "robust_z"]


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    norm: Norm


DEFAULT_FEATURE_SPECS: list[FeatureSpec] = [
    FeatureSpec("anticorrelation", "identity"),
    FeatureSpec("relative_variance", "ratio"),
    FeatureSpec("deviation", "difference"),
    FeatureSpec("amplitude", "ratio"),
    FeatureSpec("time_derivative", "ratio"),
    FeatureSpec("hurst_exponent", "difference"),
    FeatureSpec("kurtosis", "robust_z"),
]


def window_centers(
    *, n_time: int, window_size: int, window_step: int
) -> tuple[np.ndarray, np.ndarray]:
    starts = window_onsets(int(n_time), int(window_size), int(window_step)).astype(
        np.int64
    )
    # Keep existing pipeline semantics for even window sizes: start + window_size//2.
    centers = window_centers_idx(
        int(n_time),
        int(window_size),
        int(window_step),
        method="upper",
    ).astype(np.int64)
    return starts, centers


def _raw_feature(block: np.ndarray, name: str) -> np.ndarray:
    if name == "relative_variance":
        return cf.relative_variance(block)
    if name == "deviation":
        return cf.deviation(block)
    if name == "standard_deviation":
        return cf.standard_deviation(block)
    if name == "amplitude":
        return cf.amplitude(block)
    if name == "time_derivative":
        return cf.time_derivative(block)
    if name == "hurst_exponent":
        return cf.hurst_exponent(block)
    if name == "kurtosis":
        return cf.kurtosis(block)
    if name == "temporal_mean_laplacian":
        return cf.temporal_mean_laplacian(block)
    raise KeyError(f"Unknown raw feature: {name}")


def compute_feature_maps_for_window(
    block: np.ndarray,
    *,
    specs: list[FeatureSpec] = DEFAULT_FEATURE_SPECS,
    adjacency: Any,
) -> dict[str, np.ndarray]:
    neighbors = neighbors_from_adjacency(
        adjacency, n_nodes=int(np.prod(block.shape[:2]))
    )
    out: dict[str, np.ndarray] = {}

    for spec in specs:
        if spec.name == "anticorrelation":
            out[spec.name] = spatial_anticorrelation(block, neighbors=neighbors)
            continue

        raw = _raw_feature(block, spec.name).astype(np.float64, copy=False)
        flat = raw.reshape(-1)

        neigh_med = neighborhood_median(flat, neighbors=neighbors)

        if spec.norm == "identity":
            val = flat
        elif spec.norm == "ratio":
            val = normalize_ratio(flat, neigh_med)
        elif spec.norm == "difference":
            val = normalize_difference(flat, neigh_med)
        elif spec.norm == "robust_z":
            neigh_mad = neighborhood_mad(flat, neighbors=neighbors)
            val = normalize_robust_z(flat, neigh_med, neigh_mad)
        else:
            raise ValueError(f"Unknown norm: {spec.norm}")

        out[spec.name] = val.reshape(raw.shape)

    return out


def compute_raw_feature_maps_for_window(
    block: np.ndarray,
    *,
    specs: list[FeatureSpec] = DEFAULT_FEATURE_SPECS,
    adjacency: Any | None = None,
) -> dict[str, np.ndarray]:
    need_neighbors = any(spec.name == "anticorrelation" for spec in specs)
    neighbors = None
    if need_neighbors:
        if adjacency is None:
            raise ValueError(
                "adjacency is required when specs include 'anticorrelation'"
            )
        neighbors = neighbors_from_adjacency(
            adjacency, n_nodes=int(np.prod(block.shape[:2]))
        )
    out: dict[str, np.ndarray] = {}

    for spec in specs:
        if spec.name == "anticorrelation":
            assert neighbors is not None
            out[spec.name] = spatial_anticorrelation(block, neighbors=neighbors)
            continue

        raw = _raw_feature(block, spec.name).astype(np.float64, copy=False)
        out[spec.name] = raw

    return out


def compute_features_sliding(
    x: np.ndarray,
    *,
    window_size: int,
    window_step: int,
    specs: list[FeatureSpec] = DEFAULT_FEATURE_SPECS,
    adjacency: Any,
    raw=False,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    if x.ndim != 3:
        raise ValueError("Expected x shaped (AP, ML, time)")
    ap, ml, t = x.shape
    _, centers = window_centers(
        n_time=t, window_size=window_size, window_step=window_step
    )
    n_windows = int(centers.shape[0])

    feature_names = [s.name for s in specs]
    feat = np.full((len(feature_names), ap, ml, n_windows), np.nan, dtype=np.float32)

    iterator = range(n_windows)
    try:
        from tqdm.auto import tqdm  # type: ignore

        iterator = tqdm(iterator, total=n_windows, desc="feature windows", unit="win")
    except Exception:  # pragma: no cover
        pass

    for widx in iterator:
        start = int(widx * window_step)
        block = x[:, :, start : start + window_size]
        if raw:
            maps = compute_raw_feature_maps_for_window(
                block, specs=specs, adjacency=adjacency
            )
        else:
            maps = compute_feature_maps_for_window(
                block, specs=specs, adjacency=adjacency
            )
        for fidx, name in enumerate(feature_names):
            feat[fidx, :, :, widx] = maps[name].astype(np.float32, copy=False)

    return feat, feature_names, centers


def normalize_features_from_raw(
    raw_feat: np.ndarray,
    feature_names: list[str],
    *,
    specs: list[FeatureSpec] = DEFAULT_FEATURE_SPECS,
    adjacency: Any,
) -> np.ndarray:
    """Apply spatial neighborhood normalization to precomputed raw feature maps.

    Parameters
    ----------
    raw_feat
        Array shaped ``(n_features, AP, ML, n_windows)`` from
        ``compute_features_sliding(..., raw=True)``.
    feature_names
        Names aligned with ``raw_feat`` first axis.
    specs
        Feature normalization specifications.
    adjacency
        Spatial adjacency used to build neighborhood lists.
    """
    x = np.asarray(raw_feat, dtype=np.float64)
    if x.ndim != 4:
        raise ValueError("raw_feat must be shaped (n_features, AP, ML, n_windows)")
    if x.shape[0] != len(feature_names):
        raise ValueError("feature_names length must match raw_feat first axis")

    name_to_idx = {name: i for i, name in enumerate(feature_names)}
    neighbors = neighbors_from_adjacency(adjacency, n_nodes=int(np.prod(x.shape[1:3])))
    out = np.full_like(x, np.nan, dtype=np.float64)

    for spec in specs:
        if spec.name not in name_to_idx:
            continue
        fidx = name_to_idx[spec.name]
        vals = x[fidx]  # (AP, ML, W)
        flat = vals.reshape(-1, vals.shape[-1])  # (nodes, W)

        if spec.name == "anticorrelation" or spec.norm == "identity":
            norm_flat = flat
        else:
            neigh_med = neighborhood_median(flat, neighbors=neighbors)
            if spec.norm == "ratio":
                norm_flat = normalize_ratio(flat, neigh_med)
            elif spec.norm == "difference":
                norm_flat = normalize_difference(flat, neigh_med)
            elif spec.norm == "robust_z":
                neigh_mad = neighborhood_mad(flat, neighbors=neighbors)
                norm_flat = normalize_robust_z(flat, neigh_med, neigh_mad)
            else:
                raise ValueError(f"Unknown norm: {spec.norm}")

        out[fidx] = norm_flat.reshape(vals.shape)

    return out.astype(np.float32, copy=False)


LEGACY_FEATURE_NAMES: list[str] = [
    "anticorrelation",
    "relative_variance",
    "deviation",
    "amplitude",
    "time_derivative",
    "hurst_exponent",
    "temporal_mean_laplacian",
]


def compute_features_sliding_legacy(
    x: np.ndarray,
    *,
    window_size: int,
    window_step: int,
    adjacency: Any,
    footprint: np.ndarray,
    zscore: bool,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Compute the current (as-is) feature set for Phase C compatibility.

    This preserves:
    - windowing semantics (left-aligned windows, center-index time labeling)
    - feature set / names used by `scripts/feature.py`
    - optional local robust z-scoring applied to *every* feature map
    """
    if x.ndim != 3:
        raise ValueError("Expected x shaped (AP, ML, time)")
    ap, ml, t = x.shape
    _, centers = window_centers(
        n_time=t, window_size=window_size, window_step=window_step
    )
    n_windows = int(centers.shape[0])

    neighbors = neighbors_from_adjacency(adjacency, n_nodes=int(ap * ml))

    feat = np.full(
        (len(LEGACY_FEATURE_NAMES), ap, ml, n_windows), np.nan, dtype=np.float32
    )

    iterator = range(n_windows)
    try:
        from tqdm.auto import tqdm  # type: ignore

        iterator = tqdm(iterator, total=n_windows, desc="feature windows", unit="win")
    except Exception:  # pragma: no cover
        pass

    for widx in iterator:
        start = int(widx * window_step)
        block = x[:, :, start : start + window_size]

        for fidx, name in enumerate(LEGACY_FEATURE_NAMES):
            if name == "anticorrelation":
                fmap = spatial_anticorrelation(block, neighbors=neighbors)
            else:
                fmap = _raw_feature(block, name)

            if zscore:
                fmap = local_robust_zscore_grid(fmap, footprint=footprint)

            feat[fidx, :, :, widx] = np.asarray(fmap, dtype=np.float32)

    return feat, list(LEGACY_FEATURE_NAMES), centers
