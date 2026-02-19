"""Bad-channel preprocessing utilities."""

from .badlabel import DbscanParams, dbscan_outliers, grouped_dbscan_outliers
from .grid import GridAdjacency, grid_adjacency, grid_edges, make_footprint, remove_center
from .pipeline import (
    DEFAULT_FEATURE_SPECS,
    LEGACY_FEATURE_NAMES,
    FeatureSpec,
    compute_feature_maps_for_window,
    compute_features_sliding,
    compute_features_sliding_legacy,
    window_centers,
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
    "window_centers",
]

