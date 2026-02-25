from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import gaussian_filter1d

from .schemas import validate_ieeg_grid

__all__ = [
    "example_ieeg_grid",
    "example_multichannel",
    "example_spectrogram4d",
    "example_bursts_table",
]


Mode = Literal["small", "large"]


def _mode_defaults_ieeg_grid(mode: Mode) -> tuple[int, int, int]:
    if mode == "small":
        return 8, 8, 20_000
    if mode == "large":
        return 16, 16, 200_000
    raise ValueError(f"mode must be 'small' or 'large', got {mode!r}")


def example_ieeg_grid(
    *,
    mode: Mode = "small",
    seed: int = 0,
    dtype: str | np.dtype = "float32",
    fs: float = 1000.0,
    n_ap: int | None = None,
    n_ml: int | None = None,
    n_time: int | None = None,
    ap_range: tuple[float, float] = (-4.0, 1.0),
    ml_range: tuple[float, float] = (-4.0, 4.0),
    smooth_sigma: float = 4.0,
    shared_scale: float = 0.35,
    noise_scale: float = 1.0,
) -> xr.DataArray:
    """
    Deterministic grid-shaped iEEG-like signal for GUI development.

    Returns
    -------
    xr.DataArray
        Dims: ("time","ML","AP").
        Coords: time (s), ML (physical), AP (physical).
        Attrs: fs (Hz), units ("a.u.").
    """
    if fs <= 0:
        raise ValueError("fs must be > 0")

    d_ap, d_ml, d_t = _mode_defaults_ieeg_grid(mode)
    n_ap = d_ap if n_ap is None else int(n_ap)
    n_ml = d_ml if n_ml is None else int(n_ml)
    n_time = d_t if n_time is None else int(n_time)
    if n_ap < 1 or n_ml < 1 or n_time < 2:
        raise ValueError("n_ap/n_ml must be >= 1 and n_time >= 2")

    rng = np.random.default_rng(int(seed))
    dtype_np = np.dtype(dtype)

    time = (np.arange(n_time, dtype=np.float64) / float(fs)).astype(np.float64)
    ap = np.linspace(float(ap_range[0]), float(ap_range[1]), n_ap, dtype=np.float64)
    ml = np.linspace(float(ml_range[0]), float(ml_range[1]), n_ml, dtype=np.float64)

    # Shared low-frequency component plus channel-specific noise.
    shared = rng.standard_normal(size=(n_time,)).astype(np.float64)
    shared = gaussian_filter1d(shared, sigma=float(smooth_sigma), mode="nearest")
    shared = shared / (np.std(shared) if np.std(shared) != 0 else 1.0)

    noise = rng.standard_normal(size=(n_time, n_ml, n_ap)).astype(np.float64)
    noise = gaussian_filter1d(noise, sigma=float(smooth_sigma), axis=0, mode="nearest")

    arr = (shared_scale * shared[:, None, None]) + (noise_scale * noise)
    arr = arr.astype(dtype_np, copy=False)

    sig = xr.DataArray(
        arr,
        dims=("time", "ML", "AP"),
        coords={"time": time, "ML": ml, "AP": ap},
        name="ieeg",
        attrs={"fs": float(fs), "units": "a.u."},
    )

    validate_ieeg_grid(sig)
    return sig


def example_multichannel(
    *,
    mode: Mode = "small",
    seed: int = 0,
    dtype: str | np.dtype = "float32",
    fs: float = 1000.0,
    n_channel: int | None = None,
    n_time: int | None = None,
    smooth_sigma: float = 4.0,
) -> xr.DataArray:
    """
    Deterministic multichannel (non-grid) time series.

    Returns dims ("channel","time") with string channel labels.
    """
    if fs <= 0:
        raise ValueError("fs must be > 0")
    if mode == "small":
        d_ch, d_t = 16, 20_000
    elif mode == "large":
        d_ch, d_t = 64, 1_000_000
    else:
        raise ValueError(f"mode must be 'small' or 'large', got {mode!r}")

    n_channel = d_ch if n_channel is None else int(n_channel)
    n_time = d_t if n_time is None else int(n_time)
    if n_channel < 1 or n_time < 2:
        raise ValueError("n_channel must be >= 1 and n_time >= 2")

    rng = np.random.default_rng(int(seed))
    dtype_np = np.dtype(dtype)

    time = (np.arange(n_time, dtype=np.float64) / float(fs)).astype(np.float64)

    sig = rng.standard_normal(size=(n_channel, n_time)).astype(np.float64)
    sig = gaussian_filter1d(sig, sigma=float(smooth_sigma), axis=1, mode="nearest")
    sig = sig.astype(dtype_np, copy=False)

    return xr.DataArray(
        sig,
        dims=("channel", "time"),
        coords={"channel": [f"ch{i}" for i in range(n_channel)], "time": time},
        name="multichannel",
        attrs={"fs": float(fs), "units": "a.u."},
    )


def example_spectrogram4d(
    *,
    mode: Mode = "small",
    seed: int = 0,
    dtype: str | np.dtype = "float32",
) -> xr.DataArray:
    """
    4D orthoslicer-friendly tensor: dims ("ml","ap","time","freq").

    Uses the existing deterministic toy generator in `datasets.tensor.make_dataset`.
    """
    from .tensor import make_dataset

    if mode == "small":
        kwargs = dict(duration=2.0, nt=80, nf=60, nml=10, nap=10)
    elif mode == "large":
        kwargs = dict(duration=8.0, nt=300, nf=150, nml=16, nap=16)
    else:
        raise ValueError(f"mode must be 'small' or 'large', got {mode!r}")

    da = make_dataset(**kwargs, seed=int(seed))
    da = da.astype(np.dtype(dtype), copy=False)
    return da


def example_bursts_table(
    spec: xr.DataArray,
    *,
    h_quantile: float = 0.99,
    h: float | None = None,
    footprint=None,
) -> pd.DataFrame:
    """
    Compute a BurstPeaksTable for a spectrogram-like tensor using h-maxima.
    """
    from .tensor import detect_bursts_hmaxima

    return detect_bursts_hmaxima(spec, h_quantile=h_quantile, h=h, footprint=footprint)
