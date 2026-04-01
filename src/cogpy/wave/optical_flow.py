"""Dense velocity-field estimation via optical flow.

Wraps scikit-image solvers to compute frame-to-frame motion on spatial
maps (amplitude, phase, or LFP frames).

Adapted from NeuroPattToolbox [1]_ using scikit-image solvers.

References
----------
.. [1] Townsend & Gong, "Detection and analysis of spatiotemporal
   patterns in brain activity", PLOS Comp Biol, 2018.
   DOI: 10.1371/journal.pcbi.1006643
.. [2] Afrashteh et al., "Optical-flow analysis toolbox for
   characterization of spatiotemporal dynamics in mesoscale optical
   imaging of brain activity", NeuroImage, 2017.
   DOI: 10.1016/j.neuroimage.2017.03.034
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from cogpy.utils.imports import import_optional

__all__ = ["compute_flow", "flow_to_speed_direction"]


def compute_flow(
    frames: xr.DataArray,
    method: str = "tvl1",
    **kwargs,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Estimate dense velocity field between consecutive frames.

    Parameters
    ----------
    frames : DataArray
        Spatial maps with dims ``(time, AP, ML)``.
    method : {"tvl1", "ilk"}
        ``"tvl1"`` uses TV-L1 variational flow (default).
        ``"ilk"`` uses iterative Lucas–Kanade.
    **kwargs
        Forwarded to the scikit-image solver.

    Returns
    -------
    u, v : DataArray
        Velocity components along AP and ML respectively, with dims
        ``(time, AP, ML)``.  Length is ``n_time - 1``.

    References
    ----------
    .. [1] Townsend & Gong (2018), DOI: 10.1371/journal.pcbi.1006643
    .. [2] Afrashteh et al. (2017), DOI: 10.1016/j.neuroimage.2017.03.034
    """
    import_optional("skimage", extra="signal")
    from skimage.registration import optical_flow_tvl1, optical_flow_ilk

    vals = frames.values.astype(np.float64)  # (time, AP, ML)
    n_t = vals.shape[0]

    solver = {"tvl1": optical_flow_tvl1, "ilk": optical_flow_ilk}
    if method not in solver:
        raise ValueError(f"Unknown method {method!r}; choose 'tvl1' or 'ilk'")
    flow_fn = solver[method]

    us, vs = [], []
    for i in range(n_t - 1):
        flow = flow_fn(vals[i], vals[i + 1], **kwargs)
        # scikit-image returns (row_flow, col_flow) == (AP, ML).
        us.append(flow[0])
        vs.append(flow[1])

    u_arr = np.stack(us)
    v_arr = np.stack(vs)

    t_coords = frames.coords["time"].values[:-1]
    coords = {
        "time": t_coords,
        "AP": frames.coords["AP"],
        "ML": frames.coords["ML"],
    }
    if "fs" in frames.coords:
        coords["fs"] = frames.coords["fs"]

    u = xr.DataArray(u_arr, dims=("time", "AP", "ML"), coords=coords)
    v = xr.DataArray(v_arr, dims=("time", "AP", "ML"), coords=coords)
    return u, v


def flow_to_speed_direction(
    u: xr.DataArray,
    v: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Convert velocity components to speed and direction.

    Parameters
    ----------
    u, v : DataArray
        Velocity components (AP, ML).

    Returns
    -------
    speed, direction : DataArray
        Speed magnitude and direction in radians.
    """
    speed = np.sqrt(u**2 + v**2)
    direction = np.arctan2(v.values, u.values)
    direction_da = xr.DataArray(direction, dims=u.dims, coords=u.coords)
    return speed, direction_da
