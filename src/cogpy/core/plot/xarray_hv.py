"""HoloViews helpers for xarray objects.

This module contains small utilities that make it easier to create quick
"viewer-like" HoloViews objects from xarray tensors.

The functions are intentionally lightweight and avoid any repo-specific
assumptions (no pipeline context, no derivative paths).
"""

from __future__ import annotations

import numpy as np
import xarray as xr


def normalize_coords_to_index(sigx: xr.DataArray, coos: list[str] | tuple[str, ...] = ("AP", "ML")) -> xr.DataArray:
    """
    Replace selected coordinate values with 0..N-1 indices.

    This is useful when coordinates are non-uniform/unsorted (or when you just
    want a consistent grid indexing for quick visualization).
    """
    out = sigx
    for coo in coos:
        if coo in out.dims:
            out = out.assign_coords(**{coo: np.arange(out.sizes[coo])})
    return out


def grid_movie(
    sigx: xr.DataArray,
    *,
    x_dim: str = "ML",
    y_dim: str = "AP",
    normalize_index: bool = True,
    title: str | None = None,
    width: int = 400,
    height: int = 350,
    cmap: str = "RdBu_r",
    symmetric: bool = True,
    colorbar: bool = True,
):
    """
    Create a dynamic HoloViews Image "movie" over remaining dimensions.

    Parameters
    ----------
    sigx
        Input DataArray with at least ``(x_dim, y_dim)`` dimensions and one or
        more additional dimensions (e.g., ``time``) to animate over.
    x_dim, y_dim
        Spatial dimensions to plot as an image.
    normalize_index
        If True, replaces ``x_dim`` and ``y_dim`` coordinate values with
        integer indices 0..N-1 before plotting.
    title
        Plot title. Defaults to ``sigx.name``.

    Returns
    -------
    hv.DynamicMap
        A dynamic image that can be scrubbed/animated over remaining dims.
    """
    try:
        import holoviews as hv
        from holoviews import opts
    except Exception as e:  # pragma: no cover
        raise ImportError("grid_movie requires holoviews to be installed.") from e

    if x_dim not in sigx.dims or y_dim not in sigx.dims:
        raise ValueError(f"sigx must have dims {x_dim!r} and {y_dim!r}, got {list(sigx.dims)}")

    data = normalize_coords_to_index(sigx, (y_dim, x_dim)) if normalize_index else sigx

    hm = hv.Dataset(data).to(hv.Image, [x_dim, y_dim], dynamic=True)
    hm = hm.opts(
        opts.Image(
            cmap=cmap,
            symmetric=bool(symmetric),
            colorbar=bool(colorbar),
            width=int(width),
            height=int(height),
            title=(data.name if title is None else str(title)),
        )
    )
    return hm

