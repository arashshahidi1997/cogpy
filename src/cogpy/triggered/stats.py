"""
Triggered statistics on event-locked epochs.

All functions expect an *epochs* array with shape ``(n_events, ..., n_lag)``
where the first axis indexes events and the last axis is the peri-event
time (lag).  Additional axes (channels, spatial, etc.) are preserved.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

__all__ = [
    "triggered_average",
    "triggered_std",
    "triggered_median",
    "triggered_snr",
]


def triggered_average(
    epochs: xr.DataArray | np.ndarray,
    *,
    event_dim: str = "event",
) -> xr.DataArray | np.ndarray:
    """
    Mean across events (event-triggered average / ETA).

    Parameters
    ----------
    epochs : xr.DataArray or ndarray
        Epoch array with an event axis.  If ndarray, the first axis
        is assumed to be events.
    event_dim : str
        Name of event dimension (xarray only, default ``"event"``).

    Returns
    -------
    xr.DataArray or ndarray
        Mean across events, preserving all other dimensions.
    """
    if isinstance(epochs, xr.DataArray):
        return epochs.mean(dim=event_dim)
    return np.nanmean(np.asarray(epochs, dtype=float), axis=0)


def triggered_std(
    epochs: xr.DataArray | np.ndarray,
    *,
    event_dim: str = "event",
    ddof: int = 1,
) -> xr.DataArray | np.ndarray:
    """
    Standard deviation across events.

    Parameters
    ----------
    epochs : xr.DataArray or ndarray
        Epoch array with event axis.
    event_dim : str
        Name of event dimension (xarray only).
    ddof : int
        Delta degrees of freedom (default 1 = unbiased).

    Returns
    -------
    xr.DataArray or ndarray
        Std across events, preserving all other dimensions.
    """
    if isinstance(epochs, xr.DataArray):
        return epochs.std(dim=event_dim, ddof=ddof)
    arr = np.asarray(epochs, dtype=float)
    return np.nanstd(arr, axis=0, ddof=ddof)


def triggered_median(
    epochs: xr.DataArray | np.ndarray,
    *,
    event_dim: str = "event",
) -> xr.DataArray | np.ndarray:
    """
    Median across events (robust alternative to mean).

    Parameters
    ----------
    epochs : xr.DataArray or ndarray
        Epoch array with event axis.
    event_dim : str
        Name of event dimension (xarray only).

    Returns
    -------
    xr.DataArray or ndarray
        Median across events.
    """
    if isinstance(epochs, xr.DataArray):
        return epochs.median(dim=event_dim)
    return np.nanmedian(np.asarray(epochs, dtype=float), axis=0)


def triggered_snr(
    epochs: xr.DataArray | np.ndarray,
    *,
    event_dim: str = "event",
) -> xr.DataArray | np.ndarray:
    """
    Signal-to-noise ratio of the triggered average.

    Defined as ``mean / (std / sqrt(n_events))``, i.e. the ratio of the
    average to its standard error.  High values indicate a consistent
    event-locked component.

    Parameters
    ----------
    epochs : xr.DataArray or ndarray
        Epoch array with event axis.
    event_dim : str
        Name of event dimension (xarray only).

    Returns
    -------
    xr.DataArray or ndarray
        SNR array, same shape as a single epoch.
    """
    avg = triggered_average(epochs, event_dim=event_dim)
    std = triggered_std(epochs, event_dim=event_dim, ddof=1)

    if isinstance(epochs, xr.DataArray):
        n = epochs.sizes[event_dim]
    else:
        n = np.asarray(epochs).shape[0]

    se = std / np.sqrt(max(n, 1))
    eps = 1e-12
    if isinstance(se, xr.DataArray):
        return avg / se.where(se > eps, eps)
    se_arr = np.asarray(se, dtype=float)
    se_arr = np.where(se_arr > eps, se_arr, eps)
    return np.asarray(avg, dtype=float) / se_arr
