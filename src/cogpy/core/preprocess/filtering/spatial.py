"""Spatial filters for xarray grid signals (Gaussian, median, highpass).

Operate over (AP, ML) dimensions, leaving other dims untouched.
"""

import numpy as np
import xarray as xr
import scipy.ndimage as nd

from ._utils import _apply_full_array, _fs_scalar


def gaussian_spatialx(
    sigx: xr.DataArray,
    *,
    sigma: float | tuple[float, float] = 1.0,
    ap_dim: str = "AP",
    ml_dim: str = "ML",
    mode: str = "reflect",
) -> xr.DataArray:
    """Spatial Gaussian lowpass over (AP, ML), leaving other dims untouched."""
    if ap_dim not in sigx.dims or ml_dim not in sigx.dims:
        raise ValueError(f"Expected dims '{ap_dim}' and '{ml_dim}' in sigx.dims={tuple(sigx.dims)}")

    if isinstance(sigma, (list, tuple, np.ndarray)):
        sigma_ap, sigma_ml = float(sigma[0]), float(sigma[1])
    else:
        sigma_ap = sigma_ml = float(sigma)

    sigma_by_axis = []
    for d in sigx.dims:
        if d == ap_dim:
            sigma_by_axis.append(sigma_ap)
        elif d == ml_dim:
            sigma_by_axis.append(sigma_ml)
        else:
            sigma_by_axis.append(0.0)

    out = _apply_full_array(sigx, nd.gaussian_filter, sigma=tuple(sigma_by_axis), mode=str(mode))
    out.name = (sigx.name + "_gauss_spatial") if sigx.name else "gaussian_spatial"
    out.attrs.update({"filter_type": "gaussian_spatial", "sigma": (sigma_ap, sigma_ml)})
    return out


def median_spatialx(
    sigx: xr.DataArray,
    *,
    size: int | tuple[int, int] = 3,
    ap_dim: str = "AP",
    ml_dim: str = "ML",
) -> xr.DataArray:
    """Spatial median lowpass over (AP, ML), leaving other dims untouched."""
    if ap_dim not in sigx.dims or ml_dim not in sigx.dims:
        raise ValueError(f"Expected dims '{ap_dim}' and '{ml_dim}' in sigx.dims={tuple(sigx.dims)}")

    if isinstance(size, (list, tuple, np.ndarray)):
        size_ap, size_ml = int(size[0]), int(size[1])
    else:
        size_ap = size_ml = int(size)

    size_by_axis = []
    for d in sigx.dims:
        if d == ap_dim:
            size_by_axis.append(size_ap)
        elif d == ml_dim:
            size_by_axis.append(size_ml)
        else:
            size_by_axis.append(1)

    out = _apply_full_array(sigx, nd.median_filter, size=tuple(size_by_axis))
    out.name = (sigx.name + "_median_spatial") if sigx.name else "median_spatial"
    out.attrs.update({"filter_type": "median_spatial", "size": (size_ap, size_ml)})
    return out


def median_subtractx(
    sigx: xr.DataArray,
    *,
    dims: tuple[str, ...] = ("AP", "ML"),
    skipna: bool = True,
) -> xr.DataArray:
    """Subtract the median across spatial dims (common average/median reference)."""
    for d in dims:
        if d not in sigx.dims:
            raise ValueError(f"median_subtractx expected dim {d!r} in sigx.dims={tuple(sigx.dims)}")
    axes = tuple(sigx.get_axis_num(d) for d in dims)

    def _medsub_full(x: np.ndarray) -> np.ndarray:
        if bool(skipna):
            med = np.nanmedian(x, axis=axes, keepdims=True)
        else:
            med = np.median(x, axis=axes, keepdims=True)
        return x - med

    out = _apply_full_array(sigx, _medsub_full, output_dtype=sigx.dtype)
    out.name = (sigx.name + "_medsub") if sigx.name else "median_subtract"
    out.attrs.update({"filter_type": "median_subtract", "dims": tuple(dims)})
    return out


def median_highpassx(
    sigx: xr.DataArray,
    *,
    size: int | tuple[int, int, int] = (7, 7, 101),
    time_dim: str = "time",
    ap_dim: str = "AP",
    ml_dim: str = "ML",
) -> xr.DataArray:
    """Highpass-like filter via subtraction of a spatiotemporal median.

    Parameters
    ----------
    sigx
        Input signal DataArray.
    size
        Median filter window size. If an int, applies that spatial size to AP/ML
        and a default temporal size of 101 samples. If a 3-tuple, interpreted as
        ``(AP, ML, time)`` window sizes in samples.
    time_dim, ap_dim, ml_dim
        Dimension names.
    """
    if ap_dim not in sigx.dims or ml_dim not in sigx.dims or time_dim not in sigx.dims:
        raise ValueError(
            f"Expected dims '{time_dim}', '{ap_dim}', '{ml_dim}' in sigx.dims={tuple(sigx.dims)}"
        )

    if isinstance(size, (list, tuple, np.ndarray)):
        if len(size) != 3:
            raise ValueError("size must be an int or a 3-tuple (AP, ML, time).")
        size_ap, size_ml, size_t = int(size[0]), int(size[1]), int(size[2])
    else:
        size_ap = size_ml = int(size)
        size_t = 101

    if size_ap < 1 or size_ml < 1 or size_t < 1:
        raise ValueError("All median filter sizes must be >= 1.")

    size_by_axis = []
    for d in sigx.dims:
        if d == ap_dim:
            size_by_axis.append(size_ap)
        elif d == ml_dim:
            size_by_axis.append(size_ml)
        elif d == time_dim:
            size_by_axis.append(size_t)
        else:
            size_by_axis.append(1)

    med = _apply_full_array(sigx, nd.median_filter, size=tuple(size_by_axis))
    out = sigx - med
    out.attrs = dict(sigx.attrs)
    out.name = (sigx.name + "_medhp") if sigx.name else "median_highpass"
    out.attrs.update(
        {"filter_type": "median_highpass", "size": (size_ap, size_ml, size_t), "dims": (ap_dim, ml_dim, time_dim)}
    )
    return out
