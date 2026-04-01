"""Vector-field analysis: divergence, curl, critical points, classification.

Provides tools for analysing velocity fields produced by optical-flow
or phase-gradient methods.  Critical-point detection and pattern
classification follow the NeuroPattToolbox approach [1]_.

References
----------
.. [1] Townsend & Gong, "Detection and analysis of spatiotemporal
   patterns in brain activity", PLOS Comp Biol, 2018.
   DOI: 10.1371/journal.pcbi.1006643
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import xarray as xr

from ._types import Geometry, PatternType

__all__ = [
    "divergence",
    "curl",
    "critical_points",
    "classify_pattern",
]


# ---------------------------------------------------------------------------
# Differential operators
# ---------------------------------------------------------------------------


def divergence(u: xr.DataArray, v: xr.DataArray, geometry: Geometry) -> xr.DataArray:
    """Compute the divergence of a 2-D velocity field.

    Parameters
    ----------
    u, v : DataArray
        Velocity components with spatial dims ``(…, AP, ML)``.
    geometry : Geometry
        Regular grid geometry.

    Returns
    -------
    DataArray
        Divergence ∂u/∂x + ∂v/∂y.
    """
    du_dx = np.gradient(u.values, geometry.dx, axis=-2)
    dv_dy = np.gradient(v.values, geometry.dy, axis=-1)
    div = du_dx + dv_dy
    return xr.DataArray(div, dims=u.dims, coords=u.coords)


def curl(u: xr.DataArray, v: xr.DataArray, geometry: Geometry) -> xr.DataArray:
    """Compute the curl (vorticity) of a 2-D velocity field.

    Parameters
    ----------
    u, v : DataArray
        Velocity components with spatial dims ``(…, AP, ML)``.
    geometry : Geometry
        Regular grid geometry.

    Returns
    -------
    DataArray
        Curl ∂v/∂x − ∂u/∂y.
    """
    dv_dx = np.gradient(v.values, geometry.dx, axis=-2)
    du_dy = np.gradient(u.values, geometry.dy, axis=-1)
    c = dv_dx - du_dy
    return xr.DataArray(c, dims=u.dims, coords=u.coords)


# ---------------------------------------------------------------------------
# Critical-point detection
# ---------------------------------------------------------------------------


@dataclass
class CriticalPoint:
    """A zero-velocity point in a vector field.

    Parameters
    ----------
    location : tuple[int, int]
        Grid indices (AP, ML).
    type : str
        One of ``"source"``, ``"sink"``, ``"center"``, ``"saddle"``.
    """

    location: tuple[int, int]
    type: Literal["source", "sink", "center", "saddle"]


def critical_points(
    u: xr.DataArray,
    v: xr.DataArray,
    geometry: Geometry,
) -> list[CriticalPoint]:
    """Detect and classify critical points in a 2-D velocity field.

    Critical points are found where the velocity magnitude is a local
    minimum and classified by the Jacobian eigenvalues, following [1]_.

    For 3-D inputs ``(time, AP, ML)`` the first time frame is used.

    Parameters
    ----------
    u, v : DataArray
        Velocity components with dims ``(AP, ML)`` or ``(time, AP, ML)``.
    geometry : Geometry
        Regular grid geometry.

    Returns
    -------
    list of CriticalPoint

    References
    ----------
    .. [1] Townsend & Gong, PLOS Comp Biol, 2018.
       DOI: 10.1371/journal.pcbi.1006643
    """
    uu = u.values
    vv = v.values
    if uu.ndim == 3:
        uu = uu[0]
        vv = vv[0]

    speed = np.sqrt(uu**2 + vv**2)
    n_ap, n_ml = speed.shape
    dx, dy = geometry.dx, geometry.dy

    du_dx = np.gradient(uu, dx, axis=0)
    du_dy = np.gradient(uu, dy, axis=1)
    dv_dx = np.gradient(vv, dx, axis=0)
    dv_dy = np.gradient(vv, dy, axis=1)

    pts: list[CriticalPoint] = []
    for i in range(1, n_ap - 1):
        for j in range(1, n_ml - 1):
            # Simple local minimum of speed.
            patch = speed[i - 1 : i + 2, j - 1 : j + 2]
            if speed[i, j] != patch.min():
                continue

            # Jacobian at this point.
            J = np.array([[du_dx[i, j], du_dy[i, j]], [dv_dx[i, j], dv_dy[i, j]]])
            eigs = np.linalg.eigvals(J)
            re = eigs.real
            im = eigs.imag

            if np.all(re > 0) and np.allclose(im, 0, atol=1e-8):
                cp_type = "source"
            elif np.all(re < 0) and np.allclose(im, 0, atol=1e-8):
                cp_type = "sink"
            elif not np.allclose(im, 0, atol=1e-8) and np.allclose(re, 0, atol=1e-6):
                cp_type = "center"
            elif re[0] * re[1] < 0:
                cp_type = "saddle"
            else:
                continue
            pts.append(CriticalPoint(location=(i, j), type=cp_type))

    return pts


# ---------------------------------------------------------------------------
# Pattern classification
# ---------------------------------------------------------------------------


def classify_pattern(
    u: xr.DataArray,
    v: xr.DataArray,
    geometry: Geometry,
) -> PatternType:
    """Classify the dominant spatial pattern of a velocity field.

    Uses the relative magnitude of divergence, curl, and directional
    coherence to categorise the field, following [1]_.

    For 3-D inputs ``(time, AP, ML)`` the first time frame is used.

    Parameters
    ----------
    u, v : DataArray
        Velocity components.
    geometry : Geometry
        Regular grid geometry.

    Returns
    -------
    PatternType

    References
    ----------
    .. [1] Townsend & Gong, PLOS Comp Biol, 2018.
       DOI: 10.1371/journal.pcbi.1006643
    """
    div_field = divergence(u, v, geometry).values
    curl_field = curl(u, v, geometry).values

    if div_field.ndim == 3:
        div_field = div_field[0]
        curl_field = curl_field[0]

    uu = u.values
    vv = v.values
    if uu.ndim == 3:
        uu = uu[0]
        vv = vv[0]

    mean_div = np.mean(np.abs(div_field))
    mean_curl = np.mean(np.abs(curl_field))
    speed = np.sqrt(uu**2 + vv**2)
    mean_speed = np.mean(speed)

    if mean_speed < 1e-12:
        return PatternType.uncertain

    # Directional coherence (like PGD for velocity).
    mag = np.where(speed == 0, 1.0, speed)
    ux = uu / mag
    uy = vv / mag
    coherence = np.sqrt(np.mean(ux) ** 2 + np.mean(uy) ** 2)

    # Thresholds (heuristic, similar to NeuroPattToolbox).
    div_ratio = mean_div / mean_speed
    curl_ratio = mean_curl / mean_speed

    if coherence > 0.7 and curl_ratio < 0.3:
        return PatternType.planar
    if curl_ratio > 0.3 and div_ratio < 0.15:
        return PatternType.rotating
    if div_ratio > 0.5 and np.mean(div_field) > 0:
        return PatternType.source
    if div_ratio > 0.5 and np.mean(div_field) < 0:
        return PatternType.sink
    if curl_ratio > 0.2 and div_ratio > 0.2:
        return PatternType.spiral

    return PatternType.mixed
