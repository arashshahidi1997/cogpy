"""
Bivariate spectral measures: cross-spectrum, coherence, PLV.

Status
------
STATUS: ACTIVE
Reason: Multitaper cross-spectral measures for iEEG connectivity analysis.
Superseded by: n/a
Safe to remove: no

Design
------
All functions operate on pre-computed tapered FFTs (mtfft) rather than
raw signals. The caller computes mtfft once via multitaper_fft() and
passes it to whichever measures are needed — avoiding redundant FFT
computation when multiple measures are derived from the same data.

Tapered FFT convention (from multitaper_fft):
    mtfft : (..., taper, freq)  complex128

Connectivity functions take two mtfft arrays (x and y) with matching
shapes except possibly in leading batch dims. All measures average
across the taper axis (-2) to produce estimates with K degrees of
freedom, where K = number of tapers.

Output shapes:
    cross_spectrum   : (..., freq)  complex
    coherence        : (..., freq)  real [0, 1]
    plv              : (..., freq)  real [0, 1]
    mtm_fstat        : (...,)       real  (scalar per channel)
"""

from __future__ import annotations

import numpy as np


EPS = 1e-12

__all__ = [
    "cross_spectrum",
    "coherence",
    "plv",
    "mtm_fstat",
    "cross_corr_peak",
    "cross_corr_lag",
    "pac_modulation_index",
    "spectral_granger",
]


def _check_mtfft_shapes(mtfft_x: np.ndarray, mtfft_y: np.ndarray) -> None:
    if mtfft_x.shape[-2:] != mtfft_y.shape[-2:]:
        raise ValueError(
            "mtfft_x and mtfft_y must match on taper and freq axes: "
            f"mtfft_x.shape[-2:]={mtfft_x.shape[-2:]}, mtfft_y.shape[-2:]={mtfft_y.shape[-2:]}"
        )


def cross_spectrum(mtfft_x, mtfft_y):
    """
    Multitaper cross-spectral density estimate.

    Computes the average conjugate product across tapers:
        S_xy(f) = mean_k( mtfft_x_k(f) * conj(mtfft_y_k(f)) )

    Parameters
    ----------
    mtfft_x : (..., taper, freq)  complex — tapered FFTs for signal x
    mtfft_y : (..., taper, freq)  complex — tapered FFTs for signal y
        Must have same shape as mtfft_x.

    Returns
    -------
    csd : (..., freq)  complex
        Cross-spectral density. Take np.abs(csd) for amplitude,
        np.angle(csd) for phase.

    Examples
    --------
    >>> mtfft_x, freqs = multitaper_fft(x, NW=4)
    >>> mtfft_y, freqs = multitaper_fft(y, NW=4)
    >>> csd = cross_spectrum(mtfft_x, mtfft_y)
    >>> csd.shape  # (..., freq)
    """
    mtfft_x = np.asarray(mtfft_x)
    mtfft_y = np.asarray(mtfft_y)
    _check_mtfft_shapes(mtfft_x, mtfft_y)
    return np.mean(mtfft_x * np.conj(mtfft_y), axis=-2)


def coherence(mtfft_x, mtfft_y):
    """
    Multitaper magnitude squared coherence (MSC).

    MSC(f) = |S_xy(f)|^2 / (S_xx(f) * S_yy(f))

    where S_xy is the cross-spectrum and S_xx, S_yy are auto-spectra,
    all averaged across tapers. Values in [0, 1].

    Parameters
    ----------
    mtfft_x : (..., taper, freq)  complex
    mtfft_y : (..., taper, freq)  complex

    Returns
    -------
    coh : (..., freq)  real, values in [0, 1]

    Notes
    -----
    Multitaper coherence has better statistical properties than Welch
    coherence: K tapers provide K independent estimates, enabling
    jackknife confidence intervals. Under H0 (no coherence), the
    bias-corrected coherence follows a known Beta distribution.

    Examples
    --------
    >>> coh = coherence(mtfft_x, mtfft_y)
    >>> coh.shape  # (..., freq)
    >>> assert np.all((coh >= 0) & (coh <= 1))
    """
    mtfft_x = np.asarray(mtfft_x)
    mtfft_y = np.asarray(mtfft_y)
    _check_mtfft_shapes(mtfft_x, mtfft_y)

    s_xx = np.mean(np.abs(mtfft_x) ** 2, axis=-2)
    s_yy = np.mean(np.abs(mtfft_y) ** 2, axis=-2)
    s_xy = cross_spectrum(mtfft_x, mtfft_y)
    coh = (np.abs(s_xy) ** 2) / (s_xx * s_yy + EPS)
    return np.clip(coh, 0.0, 1.0)


def plv(mtfft_x, mtfft_y):
    """
    Phase Locking Value (PLV) via multitaper estimates.

    PLV(f) = |mean_k( exp(i * (phi_x_k(f) - phi_y_k(f))) )|

    where phi_x_k, phi_y_k are the instantaneous phases of the k-th
    tapered FFT. Averages phase differences across tapers.

    Values in [0, 1]. PLV = 1: perfect phase locking.
    PLV = 0: uniform phase distribution (no coupling).

    Parameters
    ----------
    mtfft_x : (..., taper, freq)  complex
    mtfft_y : (..., taper, freq)  complex

    Returns
    -------
    plv_out : (..., freq)  real, values in [0, 1]

    Notes
    -----
    PLV isolates phase coupling independently of amplitude, unlike MSC
    which conflates amplitude and phase contributions. Prefer PLV when
    amplitude varies across trials/windows (e.g. burst activity).

    Examples
    --------
    >>> plv_out = plv(mtfft_x, mtfft_y)
    >>> plv_out.shape  # (..., freq)
    """
    mtfft_x = np.asarray(mtfft_x)
    mtfft_y = np.asarray(mtfft_y)
    _check_mtfft_shapes(mtfft_x, mtfft_y)

    x_unit = mtfft_x / (np.abs(mtfft_x) + EPS)
    y_unit = mtfft_y / (np.abs(mtfft_y) + EPS)
    phase_diff = x_unit * np.conj(y_unit)
    plv_out = np.abs(np.mean(phase_diff, axis=-2))
    return np.clip(plv_out, 0.0, 1.0)


def mtm_fstat(mtfft, fs, N, *, f0):
    """
    Multitaper F-statistic test for a sinusoidal component at f0.

    Tests whether a spectral line is present at frequency f0 above
    the broadband background. Under H0 (no line), the F-statistic
    follows an F(2, 2K-2) distribution.

    F(f) = (K-1) * |mu(f)|^2 / (S(f) - |mu(f)|^2 / K)

    where mu(f) = mean of tapered FFTs at f (proportional to sinusoidal
    amplitude), S(f) = mean power across tapers, and K = number of tapers.

    Parameters
    ----------
    mtfft : (..., taper, freq)  complex — from multitaper_fft()
    fs : float — sampling rate in Hz
    N : int — number of time samples in original signal
    f0 : float — target frequency in Hz (e.g. 40.0 for artifact)

    Returns
    -------
    fstat : (...,)  real
        F-statistic at the frequency bin nearest to f0.
        Compare against scipy.stats.f.ppf(0.95, 2, 2*K-2) for
        significance at p=0.05.

    Notes
    -----
    More sensitive than band-power ratio for narrow sinusoidal lines.
    Directly applicable to 40 Hz artifact detection and 50/60 Hz
    line noise even after notch filtering if residual remains.

    Examples
    --------
    >>> mtfft, freqs = multitaper_fft(arr, NW=4)
    >>> fstat = mtm_fstat(mtfft, fs=1000, N=arr.shape[-1], f0=40.0)
    >>> fstat.shape  # (...,)
    """
    mtfft = np.asarray(mtfft)
    freqs = np.fft.rfftfreq(int(N), d=1.0 / float(fs))
    f0_idx = int(np.argmin(np.abs(freqs - float(f0))))
    mtfft_f0 = mtfft[..., :, f0_idx]  # (..., taper)
    mu = np.mean(mtfft_f0, axis=-1)
    S = np.mean(np.abs(mtfft_f0) ** 2, axis=-1)
    K = int(mtfft.shape[-2])
    mu2 = np.abs(mu) ** 2
    return (K - 1) * mu2 / (S - mu2 / float(K) + EPS)


def cross_corr_peak(
    x: np.ndarray, y: np.ndarray, *, axis: int = -1, normalize: bool = True
) -> np.ndarray:
    """
    Peak cross-correlation between signals x and y.

    Returns the maximum absolute value of the cross-correlation
    sequence across all lags.

    Wraps nitime.utils.crosscorr with all_lags=True.

    Parameters
    ----------
    x : (..., time)
    y : (..., time) — same shape as x
    axis : int — time axis (default -1)
    normalize : bool — normalize by sqrt(var(x) * var(y)) (default True)

    Returns
    -------
    peak : (...,) — peak cross-correlation in [0, 1] if normalized
    """
    from nitime.utils import crosscorr

    x = np.moveaxis(x, axis, -1)
    y = np.moveaxis(y, axis, -1)
    N = x.shape[-1]

    def _peak(xy):
        x_, y_ = xy[:N], xy[N:]
        r = crosscorr(x_, y_, all_lags=True)
        if normalize:
            norm = np.sqrt(np.var(x_) * np.var(y_)) + EPS
            r = r / norm
        return np.max(np.abs(r))

    xy = np.concatenate([x, y], axis=-1)
    return np.apply_along_axis(_peak, -1, xy)


def cross_corr_lag(
    x: np.ndarray, y: np.ndarray, *, axis: int = -1, fs: float = 1.0
) -> np.ndarray:
    """
    Lag at peak cross-correlation between signals x and y.

    Positive lag: x leads y. Negative lag: y leads x.

    Wraps nitime.utils.crosscorr with all_lags=True.

    Parameters
    ----------
    x : (..., time)
    y : (..., time) — same shape as x
    axis : int — time axis (default -1)
    fs : float — sampling rate Hz; converts lag from samples to seconds

    Returns
    -------
    lag : (...,) — lag in seconds at peak absolute cross-correlation
    """
    from nitime.utils import crosscorr

    x = np.moveaxis(x, axis, -1)
    y = np.moveaxis(y, axis, -1)
    N = x.shape[-1]
    lags = np.arange(-(N - 1), N) / fs

    def _lag(xy):
        x_, y_ = xy[:N], xy[N:]
        r = crosscorr(x_, y_, all_lags=True)
        return lags[np.argmax(np.abs(r))]

    xy = np.concatenate([x, y], axis=-1)
    return np.apply_along_axis(_lag, -1, xy)


def pac_modulation_index(
    pha: np.ndarray, amp: np.ndarray, *, n_bins: int = 18
) -> np.ndarray:
    """
    Phase-Amplitude Coupling (PAC) via Modulation Index (Tort et al. 2010).

    Measures whether the amplitude of a fast oscillation is modulated
    by the phase of a slow oscillation. Classic examples:
    theta-gamma coupling in hippocampus, spindle-ripple coupling
    during NREM sleep.

    Wraps tensorpac.methods.meth_pac.modulation_index.

    Parameters
    ----------
    pha : (..., time) — instantaneous phase in radians
        Typically: np.angle(hilbert(bandpass(x, f_low, f_high)))
    amp : (..., time) — instantaneous amplitude (envelope)
        Typically: np.abs(hilbert(bandpass(x, f_low, f_high)))
    n_bins : int — number of phase bins (default 18 = 20° bins)

    Returns
    -------
    mi : (...,) — modulation index, higher = stronger PAC

    Notes
    -----
    MI based on KL divergence between observed amplitude distribution
    over phase bins and uniform distribution.
    MI = 0: no coupling. MI > 0: phase-amplitude coupling present.

    For a full comodulogram (DIMS_COMODULOGRAM), call across a grid
    of (freq_phase, freq_amp) pairs and stack results.

    Examples
    --------
    >>> from scipy.signal import hilbert
    >>> pha = np.angle(hilbert(bandpass(x, 4, 8, fs)))
    >>> amp = np.abs(hilbert(bandpass(x, 30, 80, fs)))
    >>> mi = pac_modulation_index(pha, amp)
    """
    from tensorpac.methods.meth_pac import modulation_index as _mi

    pha = np.asarray(pha)
    amp = np.asarray(amp)
    if pha.shape != amp.shape:
        raise ValueError(
            f"pha and amp must have identical shapes, got {pha.shape} and {amp.shape}."
        )

    # tensorpac expects:
    #   pha: (n_pha, ..., n_times)
    #   amp: (n_amp, ..., n_times)
    #
    # We accept (..., time) and interpret it as one phase series and one
    # amplitude series per batch element.
    pha = pha[np.newaxis, ...]
    amp = amp[np.newaxis, ...]
    out = _mi(pha, amp, n_bins=int(n_bins))
    return out[0, 0, ...]


def spectral_granger(
    x: np.ndarray,
    y: np.ndarray,
    fs: float,
    *,
    axis: int = -1,
    direction: str = "both",
) -> tuple:
    """
    Spectral Granger causality between signals x and y.

    Tests directional influence at each frequency.
    Use case: cortical → hippocampal vs hippocampal → cortical
    communication during sleep consolidation.

    Wraps elephant.causality.granger.pairwise_spectral_granger.

    Parameters
    ----------
    x : (time,) — source signal (1D only, elephant limitation)
    y : (time,) — target signal
    fs : float — sampling rate in Hz
    axis : int — time axis (default -1)
    direction : str — "x_to_y" | "y_to_x" | "both" (default "both")

    Returns
    -------
    If direction == "both":
        (freqs, gc_x_to_y, gc_y_to_x)
    If direction == "x_to_y" or "y_to_x":
        (freqs, gc)

    Notes
    -----
    Granger causality is not symmetric: gc_x_to_y ≠ gc_y_to_x.
    Values > 0 indicate directional influence at that frequency.
    Interpret relative to surrogate distribution for significance.
    1D input only — elephant does not support batched input natively.
    For multichannel use, call in a loop over channel pairs.
    """
    from elephant.causality.granger import pairwise_spectral_granger

    x = np.asarray(x)
    y = np.asarray(y)
    x = np.moveaxis(x, axis, -1)
    y = np.moveaxis(y, axis, -1)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError(
            "spectral_granger only supports 1D inputs after moving time axis. "
            f"Got x.shape={x.shape}, y.shape={y.shape}."
        )
    if x.shape != y.shape:
        raise ValueError(
            f"x and y must have the same shape, got {x.shape} and {y.shape}."
        )

    freqs, causality = pairwise_spectral_granger(x, y, fs=float(fs))
    freqs = np.asarray(freqs, dtype=float)
    gc_x_to_y = np.asarray(causality.directional_causality_x_y, dtype=float)
    gc_y_to_x = np.asarray(causality.directional_causality_y_x, dtype=float)

    if direction == "both":
        return freqs, gc_x_to_y, gc_y_to_x
    if direction == "x_to_y":
        return freqs, gc_x_to_y
    if direction == "y_to_x":
        return freqs, gc_y_to_x
    raise ValueError('direction must be one of: "x_to_y", "y_to_x", "both".')
