"""Spatiotemporal filtering for xarray signals.

All public functions accept ``xarray.DataArray`` inputs with an ``fs``
coordinate/attribute and preserve metadata through transforms. Dask-backed
arrays are supported via ``xr.apply_ufunc(..., dask="parallelized")``.

Submodules
----------
temporal : Butterworth bandpass/lowpass/highpass, notch, decimation
spatial  : Gaussian, median, median-highpass over (AP, ML) grid
reference : Common median reference
normalization : Z-score normalization
"""

from ._utils import bandpass_filt_params, get_coord_fs
from .temporal import (
    bandpassx,
    butterworth_bandpass_shoulder,
    decimatex,
    highpassx,
    lowpassx,
    notchesx,
    notchx,
)
from .spatial import (
    gaussian_spatialx,
    median_highpassx,
    median_spatialx,
    median_subtractx,
)
from .reference import cmrx
from .normalization import zscorex

# Private utilities re-exported for backward compat (used by other cogpy modules)
from ._utils import _apply_full_array, _fs_scalar

__all__ = [
    # Utility
    "bandpass_filt_params",
    "get_coord_fs",
    # Temporal
    "bandpassx",
    "butterworth_bandpass_shoulder",
    "decimatex",
    "highpassx",
    "lowpassx",
    "notchx",
    "notchesx",
    # Spatial
    "gaussian_spatialx",
    "median_highpassx",
    "median_spatialx",
    "median_subtractx",
    # Reference
    "cmrx",
    # Normalization
    "zscorex",
]
