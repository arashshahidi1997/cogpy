from __future__ import annotations

import numpy as np
import xarray as xr

__all__ = [
    "normalize_windowed_features",
    "smooth_windowed_features",
    "summarize_windowed_features",
]


def normalize_windowed_features(
    ds: xr.Dataset,
    *,
    dim: str = "time_win",
    robust: bool = True,
    eps: float = 1e-12,
) -> xr.Dataset:
    """
    Normalize each feature variable across the window axis.

    For each variable in ds that contains `dim`:
        robust=True:  (x - median(dim)) / (MAD(dim) * 1.4826 + eps)
        robust=False: (x - mean(dim))   / (std(dim) + eps)

    Variables that do not contain `dim` are passed through unchanged.

    Parameters
    ----------
    ds : xr.Dataset
        Windowed feature dataset from extract_channel_features_xr.
    dim : str
        Window dimension name. Default "time_win".
    robust : bool
        Use median/MAD instead of mean/std. Default True.
    eps : float
        Small constant added to denominator to avoid division by zero.

    Returns
    -------
    xr.Dataset
        Same structure as input. Each normalized variable has attrs updated:
            normalized=True
            normalization="robust_zscore" or "zscore"
            normalization_dim=dim

    Notes
    -----
    Output values are dimensionless z-scores. Original units are lost.
    The output dataset attrs contain all input attrs plus
    normalization_applied=True.
    """
    dim = str(dim)
    eps = float(eps)

    out_vars: dict[str, xr.DataArray] = {}
    for name, da in ds.data_vars.items():
        if dim not in da.dims:
            out_vars[name] = da
            continue

        if robust:
            center = da.median(dim=dim)
            scale = (np.abs(da - center)).median(dim=dim) * 1.4826 + eps
            normalization = "robust_zscore"
        else:
            center = da.mean(dim=dim)
            scale = da.std(dim=dim) + eps
            normalization = "zscore"

        out_da = (da - center) / scale
        out_attrs = dict(da.attrs)
        out_attrs.update(
            {
                "normalized": True,
                "normalization": normalization,
                "normalization_dim": dim,
            }
        )
        out_vars[name] = out_da.assign_attrs(out_attrs)

    out = xr.Dataset(data_vars=out_vars, coords=ds.coords, attrs=dict(ds.attrs))
    out.attrs.update(
        {
            "normalization_applied": True,
            "normalization_dim": dim,
            "normalization_robust": bool(robust),
        }
    )
    return out


def smooth_windowed_features(
    ds: xr.Dataset,
    *,
    window_s: float,
    dim: str = "time_win",
    method: str = "median",
    min_periods: int = 1,
) -> xr.Dataset:
    """
    Smooth feature variables along the window axis using a duration in seconds.

    Converts `window_s` (seconds) to a window count using `ds.attrs["fs_win"]`
    or `ds.attrs["window_step_s"]`. Applies a rolling reduction along `dim`.

    Parameters
    ----------
    ds : xr.Dataset
        Windowed feature dataset. Must have attrs["fs_win"] or
        attrs["window_step_s"] to convert window_s to window count.
    window_s : float
        Smoothing window duration in seconds.
    dim : str
        Window dimension name. Default "time_win".
    method : str
        Reduction method: "median", "mean". Default "median".
    min_periods : int
        Minimum number of observations for a valid result. Default 1.

    Returns
    -------
    xr.Dataset
        Smoothed dataset. Attrs updated with smoothing_window_s and
        smoothing_method.

    Raises
    ------
    ValueError
        If neither "fs_win" nor "window_step_s" is present in ds.attrs.
    ValueError
        If method is not one of {"median","mean"}.
    """
    dim = str(dim)
    window_s = float(window_s)
    min_periods = int(min_periods)

    if window_s <= 0:
        raise ValueError(f"window_s must be > 0, got {window_s!r}")
    if min_periods < 1:
        raise ValueError(f"min_periods must be >= 1, got {min_periods!r}")

    if "fs_win" in ds.attrs:
        fs_win = float(ds.attrs["fs_win"])
    elif "window_step_s" in ds.attrs:
        fs_win = 1.0 / float(ds.attrs["window_step_s"])
    elif "fs" in ds.attrs and "window_step" in ds.attrs:
        fs_win = float(ds.attrs["fs"]) / float(ds.attrs["window_step"])
    else:
        raise ValueError(
            "Dataset missing 'fs_win' or 'window_step_s'. "
            "These are set by extract_channel_features_xr when fs is present."
        )

    if not np.isfinite(fs_win) or fs_win <= 0:
        raise ValueError(f"fs_win must be positive and finite, got {fs_win!r}")

    n_win = max(1, int(round(window_s * fs_win)))
    method = str(method)
    if method not in {"median", "mean"}:
        raise ValueError(f"method must be 'median' or 'mean', got {method!r}")

    out_vars: dict[str, xr.DataArray] = {}
    for name, da in ds.data_vars.items():
        if dim not in da.dims:
            out_vars[name] = da
            continue

        rolled = da.rolling({dim: n_win}, center=True, min_periods=min_periods)
        out_da = rolled.median() if method == "median" else rolled.mean()
        out_vars[name] = out_da.assign_attrs(dict(da.attrs))

    out = xr.Dataset(data_vars=out_vars, coords=ds.coords, attrs=dict(ds.attrs))
    out.attrs.update(
        {
            "smoothing_window_s": window_s,
            "smoothing_n_win": int(n_win),
            "smoothing_method": method,
        }
    )
    return out


def summarize_windowed_features(
    ds: xr.Dataset,
    *,
    dim: str = "time_win",
    stats: tuple[str, ...] = ("median", "mad", "max"),
) -> xr.Dataset:
    """
    Collapse the window axis to summary statistics per electrode/channel.

    For each variable in ds that contains `dim`, computes the requested
    stats across `dim`, returning a variable per stat named
    "{feature}_{stat}" (e.g. "variance_median", "variance_mad").

    Parameters
    ----------
    stats : tuple of str
        Any subset of {"median","mean","std","mad","max","min","p95","p05"}.
        "mad" = median absolute deviation * 1.4826 (normalized).
        "p95"/"p05" = 95th/5th percentile.

    Returns
    -------
    xr.Dataset
        No window axis. All variables are per-electrode/channel summaries.
        Original feature attrs preserved on each output variable, plus
        stat="..." attr.
    """
    dim = str(dim)
    stats = tuple(str(s) for s in stats)
    allowed = {"median", "mean", "std", "mad", "max", "min", "p95", "p05"}
    unknown = sorted(set(stats) - allowed)
    if unknown:
        raise ValueError(f"Unknown stats {unknown}. Allowed: {sorted(allowed)}")

    out_vars: dict[str, xr.DataArray] = {}

    # Pass through non-windowed variables unchanged.
    for name, da in ds.data_vars.items():
        if dim not in da.dims:
            out_vars[name] = da

    for feat, da in ds.data_vars.items():
        if dim not in da.dims:
            continue

        for stat in stats:
            if stat == "median":
                out_da = da.median(dim=dim)
            elif stat == "mean":
                out_da = da.mean(dim=dim)
            elif stat == "std":
                out_da = da.std(dim=dim)
            elif stat == "max":
                out_da = da.max(dim=dim)
            elif stat == "min":
                out_da = da.min(dim=dim)
            elif stat == "mad":
                med = da.median(dim=dim)
                out_da = (np.abs(da - med)).median(dim=dim) * 1.4826
            elif stat == "p95":
                out_da = da.quantile(0.95, dim=dim).squeeze("quantile", drop=True)
            elif stat == "p05":
                out_da = da.quantile(0.05, dim=dim).squeeze("quantile", drop=True)
            else:  # pragma: no cover
                raise AssertionError(f"Unhandled stat {stat!r}")

            out_name = f"{feat}_{stat}"
            out_attrs = dict(da.attrs)
            out_attrs.update({"stat": stat, "stat_dim": dim})
            out_vars[out_name] = out_da.assign_attrs(out_attrs)

    out = xr.Dataset(data_vars=out_vars, coords=ds.coords, attrs=dict(ds.attrs))
    if dim in out.dims:
        out = out.drop_dims(dim)
    out.attrs.update({"summary_dim": dim, "summary_stats": list(stats)})
    return out

