"""Bad-channel preprocessing utilities."""

from .badlabel import DbscanParams, dbscan_outliers, grouped_dbscan_outliers
from .grid import (
    GridAdjacency,
    grid_adjacency,
    grid_edges,
    make_footprint,
    remove_center,
)
from .pipeline import (
    DEFAULT_FEATURE_SPECS,
    LEGACY_FEATURE_NAMES,
    FeatureSpec,
    compute_feature_maps_for_window,
    compute_features_sliding,
    compute_features_sliding_legacy,
    normalize_features_from_raw,
    window_centers,
)
from .feature_normalization import (
    normalize_windowed_features,
    smooth_windowed_features,
    summarize_windowed_features,
)

__all__ = [
    "DbscanParams",
    "dbscan_outliers",
    "grouped_dbscan_outliers",
    "GridAdjacency",
    "grid_adjacency",
    "grid_edges",
    "make_footprint",
    "remove_center",
    "DEFAULT_FEATURE_SPECS",
    "LEGACY_FEATURE_NAMES",
    "FeatureSpec",
    "compute_feature_maps_for_window",
    "compute_features_sliding",
    "compute_features_sliding_legacy",
    "normalize_features_from_raw",
    "window_centers",
    "normalize_windowed_features",
    "smooth_windowed_features",
    "summarize_windowed_features",
]
