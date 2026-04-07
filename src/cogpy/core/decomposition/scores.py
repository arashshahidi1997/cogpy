"""Score processing for decomposition factor time series.

Standalone functions for smoothing, envelope removal, thresholding,
and spike extraction on factor score DataArrays.
"""

from __future__ import annotations

import numpy as np
import scipy.ndimage as nd
import scipy.signal as scisig
import xarray as xr

from ..utils import xarr as xut, time_series as ts
from ..utils._functools import untuple


# ---------------------------------------------------------------------------
# Bandpass / serialisation (formerly erppca/scores.py)
# ---------------------------------------------------------------------------


def scx_bandpass(scx: xr.DataArray, wl, wh, order, axis=0):
    """Bandpass-filter factor scores."""
    fs = scx.fs
    b, a = scisig.butter(order, [wl, min(wh, fs / 2 - 0.1)], btype="bandpass", fs=fs)
    bp_filt = lambda x: scisig.filtfilt(b, a, x, axis=axis)
    return xut.xarr_wrap(bp_filt)(scx)


def get_score_dict(scx: xr.DataArray):
    """Serialise factor scores to a plain dict."""
    return {
        "scores": scx.data,
        "time": scx.time.values,
        "factor": scx.factor.values,
        "fs": scx.fs,
    }


# ---------------------------------------------------------------------------
# Score processing pipeline (extracted from SpatSpecDecomposition)
# ---------------------------------------------------------------------------


def scx_gaussian(scx: xr.DataArray, sigma: float = 0.25) -> xr.DataArray:
    """Smooth factor scores with a Gaussian filter.

    Parameters
    ----------
    scx : xr.DataArray
        Factor scores ``(time, factor)`` with ``fs`` attribute.
    sigma : float
        Gaussian width in seconds.
    """
    return xut.xarr_wrap(nd.gaussian_filter)(scx, sigma=(sigma * scx.fs, 0))


def scx_lowerenv(scx: xr.DataArray) -> dict[str, xr.DataArray]:
    """Remove lower envelope from factor scores."""
    scx_lower = ts.lower_envelope(scx, axis=0)
    scx_noenv = xr.zeros_like(scx)
    scx_noenv.data[:] = scx - scx_lower
    return {"scx_lower": scx_lower, "scx_noenv": scx_noenv}


def scx_threshold(scx: xr.DataArray, quantile: float = 0.25) -> xr.DataArray:
    """Threshold factor scores at a quantile."""
    return ts.threshold(scx, quantile=quantile, axis=0)


def scx_process(
    scx: xr.DataArray,
    return_all: bool = False,
    sigma: float = 0.25,
    quantile: float = 0.25,
) -> xr.DataArray | dict[str, xr.DataArray]:
    """Full score processing pipeline: smooth → remove envelope → threshold.

    Parameters
    ----------
    scx : xr.DataArray
        Raw factor scores.
    return_all : bool
        If True, return dict with intermediate results.
    """
    print("scx processing ...")
    print("\t smoothing ...")
    scx_smooth = scx_gaussian(scx, sigma=sigma)
    scx_noenv_dict = scx_lowerenv(scx_smooth)
    print("\t thresholding ...")
    scx_thresh = scx_threshold(scx_noenv_dict["scx_noenv"], quantile=quantile)
    print("scx processing done!")
    if return_all:
        return (
            {"scx": scx, "scx_smooth": scx_smooth}
            | scx_noenv_dict
            | {"scx_thresh": scx_thresh}
        )
    return scx_thresh


def scx_spikes(scx: xr.DataArray, max_halfdur: float = 5.0, return_trace: bool = False):
    """Extract spike-like events from factor scores.

    Parameters
    ----------
    scx : xr.DataArray
        Factor scores ``(time, factor)`` with ``fs`` attribute.
    max_halfdur : float
        Maximum half-duration in seconds for peri-peak window.
    return_trace : bool
        If True, include peri-peak waveforms in output.

    Returns
    -------
    dict
        Per-factor point-process DataArrays keyed ``"factor0"``, ``"factor1"``, etc.
    """
    fs = scx.fs
    nfactor = scx.factor.size
    max_halfdur_samp = max([ts.seconds_to_samples(max_halfdur, fs), 5])
    periwin = np.arange(-max_halfdur_samp, max_halfdur_samp + 1)
    facpp = {}
    for ifactor in range(nfactor):
        scx_ = scx.data[:, ifactor]
        scx_peak_time = scisig.argrelmax(scx_)[0]
        scx_peak_amp = scx_[scx_peak_time]
        npeak = scx_peak_time.shape[0]
        peripeak_wins = scx_peak_time.reshape(-1, 1) + periwin.reshape(1, -1)
        peripeak_wins = np.clip(peripeak_wins, 0, scx_.size - 1)
        scx_peripeak_slc = scx_[peripeak_wins]
        scx_halfamp_dur = np.sum(
            scx_peripeak_slc > np.expand_dims(scx_peak_amp / 2, axis=1), axis=1
        )

        fppx = xr.DataArray(
            np.array([scx_peak_time, scx_peak_amp, scx_halfamp_dur]),
            dims=["feature", "peak"],
            coords={"feature": ["time", "amp", "dur"], "peak": np.arange(npeak)},
        )

        res = (fppx,)
        if return_trace:
            scx_peripeak_slc_xr = xr.DataArray(
                scx_peripeak_slc,
                dims=["time", "win"],
                coords={"time": scx_peak_time, "win": periwin},
            )
            res += (scx_peripeak_slc_xr,)
        facpp["factor" + str(ifactor)] = untuple(res)
    return facpp
