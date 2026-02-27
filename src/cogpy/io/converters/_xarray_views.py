from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import xarray as xr


@dataclass(frozen=True)
class TimeChannelView:
    x_tc: np.ndarray  # (n_time, n_ch)
    time_s: np.ndarray  # (n_time,)
    fs: float
    ch_names: list[str]
    ap: np.ndarray | None  # (n_ch,)
    ml: np.ndarray | None  # (n_ch,)


def as_ieeg_time_channel(da: xr.DataArray) -> xr.DataArray:
    """
    Normalize diverse continuous-signal DataArrays into IEEGTimeChannel form.

    Accepted inputs include:
      - grid: ("time","ML","AP") (permutation allowed)
      - time×channel: ("time","channel") (permutation allowed)
      - legacy time×ch: ("time","ch")
      - multichannel: ("channel","time") (permutation allowed)
    """
    from cogpy.datasets.schemas import coerce_ieeg_grid, coerce_ieeg_time_channel, coerce_multichannel

    if not isinstance(da, xr.DataArray):
        raise TypeError("da must be an xarray.DataArray")

    da2 = da
    if "ch" in da2.dims:
        da2 = da2.rename({"ch": "channel"})

    dims = set(da2.dims)

    # Grid-like: coerce to ("time","ML","AP") then stack ML-major -> channel.
    if {"ML", "AP"}.issubset(dims):
        da_grid = coerce_ieeg_grid(da2)
        da_tc = (
            da_grid.stack(channel=("ML", "AP"))
            .reset_index("channel")
            .transpose("time", "channel")
        )
        return coerce_ieeg_time_channel(da_tc)

    # Multichannel computational view.
    if dims == {"channel", "time"}:
        da_mc = coerce_multichannel(da2)
        da_tc = da_mc.transpose("time", "channel")
        return coerce_ieeg_time_channel(da_tc)

    # time×channel view (or permutation).
    if dims == {"time", "channel"}:
        # Ensure coords exist before schema coercion/validation.
        if "channel" not in da2.coords:
            da2 = da2.assign_coords(channel=np.arange(int(da2.sizes["channel"])))

        if "time" not in da2.coords:
            n_time = int(da2.sizes["time"])
            fs = _maybe_extract_fs(da2.attrs)
            if fs is not None and fs > 0:
                t = np.arange(n_time, dtype=float) / float(fs)
            else:
                t = np.arange(n_time, dtype=float)
            da2 = da2.assign_coords(time=t)

        return coerce_ieeg_time_channel(da2)

    raise ValueError(
        f"Unsupported dims {tuple(da2.dims)}. Expected grid ('time','ML','AP') or "
        "time×channel ('time','channel' or legacy 'time','ch') or multichannel ('channel','time')."
    )


def extract_time_channel_view(
    da_tc: xr.DataArray,
    *,
    ch_name_style: Literal["coord", "index"] = "coord",
) -> TimeChannelView:
    """
    Extract a small, typed bundle from an IEEGTimeChannel-like DataArray.
    """
    from cogpy.datasets.schemas import coerce_ieeg_time_channel

    da2 = coerce_ieeg_time_channel(da_tc)

    x_tc = np.asarray(da2.data)
    if x_tc.ndim != 2:
        raise ValueError(f"Expected 2D time×channel data, got array with shape {x_tc.shape}")

    n_time = int(da2.sizes["time"])
    if "time" in da2.coords:
        time_s = np.asarray(da2["time"].values, dtype=float)
        if time_s.shape != (n_time,):
            raise ValueError(f"Unexpected time coord shape {time_s.shape}; expected ({n_time},)")
    else:
        time_s = np.arange(n_time, dtype=float)

    fs = _maybe_extract_fs(da2.attrs)
    if fs is None:
        fs = _infer_fs_from_time(time_s)

    if "channel" in da2.coords:
        ch_coord = np.asarray(da2["channel"].values)
    else:
        ch_coord = np.arange(int(da2.sizes["channel"]))

    ap = None
    ml = None
    if "AP" in da2.coords and da2["AP"].dims == ("channel",):
        ap = np.asarray(da2["AP"].values)
    if "ML" in da2.coords and da2["ML"].dims == ("channel",):
        ml = np.asarray(da2["ML"].values)

    if ap is not None and ml is not None:
        ch_names = [f"ML{mlv}_AP{apv}" for mlv, apv in zip(ml, ap, strict=True)]
    elif ch_name_style == "coord" and "channel" in da2.coords:
        ch_names = [str(v) for v in ch_coord.tolist()]
    else:
        ch_names = [f"ch{ii:03d}" for ii in range(int(da2.sizes["channel"]))]

    return TimeChannelView(
        x_tc=x_tc,
        time_s=time_s,
        fs=float(fs),
        ch_names=ch_names,
        ap=ap,
        ml=ml,
    )


def _infer_fs_from_time(time_s: np.ndarray) -> float:
    if time_s.shape[0] < 2:
        raise ValueError("Cannot infer fs from time coordinate with length < 2; provide da.attrs['fs'].")
    dt = np.diff(time_s)
    if not np.all(np.isfinite(dt)):
        raise ValueError("Cannot infer fs from non-finite time coordinate.")
    dt0 = float(dt[0])
    if dt0 <= 0:
        raise ValueError("Cannot infer fs: time coordinate is not strictly increasing.")
    if not np.allclose(dt, dt0, rtol=1e-5, atol=1e-12):
        raise ValueError("Cannot infer fs: time coordinate is not uniformly spaced; provide da.attrs['fs'].")
    return 1.0 / dt0


def _maybe_extract_fs(attrs: dict) -> float | None:
    candidates = ("fs", "Fs", "sampling_rate", "SamplingRate", "sampling_frequency", "SamplingFrequency")
    for k in candidates:
        if k in attrs:
            try:
                return float(attrs[k])
            except (TypeError, ValueError):
                continue
    for container_key in ("meta", "metadata"):
        meta = attrs.get(container_key, None)
        if isinstance(meta, dict):
            for k in candidates:
                if k in meta:
                    try:
                        return float(meta[k])
                    except (TypeError, ValueError):
                        continue
    return None

