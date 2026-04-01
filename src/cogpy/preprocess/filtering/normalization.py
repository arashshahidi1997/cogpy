"""Signal normalization for xarray (z-score)."""

import numpy as np
import xarray as xr


def zscorex(
    sigx: xr.DataArray,
    *,
    dim: str = "time",
    robust: bool = False,
    eps: float = 1e-12,
) -> xr.DataArray:
    """Z-score normalization along dim.

    robust=False: (x - mean) / std
    robust=True:  (x - median) / (MAD * 1.4826)

    Intended for display normalization (z-score per window before rendering).
    Not a filter — does not modify frequency content.
    """
    if dim not in sigx.dims:
        raise ValueError(
            f"zscorex expected dim {dim!r} in sigx.dims={tuple(sigx.dims)}"
        )

    if bool(robust):
        center = sigx.median(dim=dim)
        mad = (np.abs(sigx - center)).median(dim=dim)
        scale = mad * 1.4826
    else:
        center = sigx.mean(dim=dim)
        scale = sigx.std(dim=dim)

    out = (sigx - center) / (scale + float(eps))
    out.attrs = dict(sigx.attrs)
    out.name = (sigx.name + "_zscore") if sigx.name else "zscore"
    out.attrs.update(
        {"normalization": "zscore", "dim": str(dim), "robust": bool(robust)}
    )
    return out
