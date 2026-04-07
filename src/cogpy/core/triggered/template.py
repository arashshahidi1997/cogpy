"""
Template estimation, fitting, and subtraction primitives.

These operate on epoch arrays and continuous signals to support
template-based artifact removal or signal extraction.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

__all__ = [
    "estimate_template",
    "fit_scaling",
    "subtract_template",
]


def estimate_template(
    epochs: xr.DataArray | np.ndarray,
    *,
    method: str = "mean",
    event_dim: str = "event",
) -> xr.DataArray | np.ndarray:
    """
    Estimate a template waveform from stacked epochs.

    Parameters
    ----------
    epochs : xr.DataArray or ndarray
        Epoch array with shape ``(n_events, ..., n_lag)``.
    method : {"mean", "median", "trimmean"}
        Aggregation method.

        - ``"mean"``: arithmetic mean (default).
        - ``"median"``: robust to outliers.
        - ``"trimmean"``: 20% trimmed mean (requires scipy).
    event_dim : str
        Name of event dimension (xarray only).

    Returns
    -------
    xr.DataArray or ndarray
        Template with shape ``(..., n_lag)`` — one fewer axis than input.
    """
    method = str(method)

    if isinstance(epochs, xr.DataArray):
        if method == "mean":
            return epochs.mean(dim=event_dim)
        elif method == "median":
            return epochs.median(dim=event_dim)
        elif method == "trimmean":
            from scipy.stats import trim_mean

            # apply_ufunc moves core dims to the end, so axis=-1 inside.
            result = xr.apply_ufunc(
                lambda x: trim_mean(x, proportiontocut=0.1, axis=-1),
                epochs,
                input_core_dims=[[event_dim]],
                vectorize=False,
            )
            return result
        else:
            raise ValueError(f"Unknown method {method!r}")
    else:
        arr = np.asarray(epochs, dtype=float)
        if method == "mean":
            return np.nanmean(arr, axis=0)
        elif method == "median":
            return np.nanmedian(arr, axis=0)
        elif method == "trimmean":
            from scipy.stats import trim_mean

            return trim_mean(arr, proportiontocut=0.1, axis=0)
        else:
            raise ValueError(f"Unknown method {method!r}")


def fit_scaling(
    epochs: np.ndarray,
    template: np.ndarray,
) -> np.ndarray:
    """
    Fit per-event scaling coefficients via least-squares projection.

    For each event *i*, finds scalar ``alpha_i`` minimizing
    ``||epochs[i] - alpha_i * template||^2``.

    Parameters
    ----------
    epochs : (n_events, ..., n_lag) float
        Epoch array.
    template : (..., n_lag) float
        Template waveform (same shape as a single epoch).

    Returns
    -------
    alpha : (n_events,) float
        Per-event scaling coefficients.

    Notes
    -----
    This is the standard dot-product projection:
    ``alpha_i = <epochs_i, template> / <template, template>``.
    """
    epochs = np.asarray(epochs, dtype=float)
    template = np.asarray(template, dtype=float)

    # Flatten non-event dims for dot product
    n_events = epochs.shape[0]
    ep_flat = epochs.reshape(n_events, -1)
    t_flat = template.ravel()

    denom = np.dot(t_flat, t_flat)
    if denom < 1e-30:
        return np.zeros(n_events)

    alpha = ep_flat @ t_flat / denom
    return alpha


def subtract_template(
    signal: np.ndarray | xr.DataArray,
    event_samples: np.ndarray,
    template: np.ndarray,
    *,
    scaling: np.ndarray | None = None,
) -> np.ndarray | xr.DataArray:
    """
    Subtract a template waveform at each event location in a continuous signal.

    Parameters
    ----------
    signal : (n_channels, n_time) or (n_time,) float
        Continuous signal.  If xr.DataArray, operates on ``.values``
        and returns an xr.DataArray with same metadata.
    event_samples : (n_events,) int
        Sample indices where template onset aligns.
    template : (n_channels, n_lag) or (n_lag,) float
        Template to subtract.  Must be broadcastable with
        the channel dimension of *signal*.
    scaling : (n_events,) float, optional
        Per-event amplitude scaling.  If None, uses 1.0 for all events.

    Returns
    -------
    cleaned : same type and shape as *signal*
        Signal with template subtracted at each event.

    Notes
    -----
    Out-of-bounds events (where template would extend beyond signal
    boundaries) are silently skipped.
    """
    is_xr = isinstance(signal, xr.DataArray)
    if is_xr:
        arr = signal.values.copy()
        coords = signal.coords
        dims = signal.dims
        attrs = signal.attrs
        name = signal.name
    else:
        arr = np.array(signal, dtype=float)

    template = np.asarray(template, dtype=float)
    event_samples = np.asarray(event_samples, dtype=int).ravel()

    if scaling is None:
        scaling = np.ones(len(event_samples))
    else:
        scaling = np.asarray(scaling, dtype=float).ravel()

    n_lag = template.shape[-1]
    n_time = arr.shape[-1]

    for i, s0 in enumerate(event_samples):
        s1 = s0 + n_lag
        if s0 < 0 or s1 > n_time:
            continue
        arr[..., s0:s1] -= scaling[i] * template

    if is_xr:
        return xr.DataArray(arr, dims=dims, coords=coords, attrs=attrs, name=name)
    return arr
