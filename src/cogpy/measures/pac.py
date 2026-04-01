"""
Phase-amplitude coupling (PAC) measures.

STATUS: ACTIVE
Reason: PAC modulation index, preferred phase, comodulogram, and surrogate
        testing for pixecog coupling analysis (H3, H6).
Superseded by: n/a
Safe to remove: no

All functions are pure — numpy in, numpy out. No file I/O.
Accepts either raw continuous signals (phase/amplitude extracted internally
via Hilbert transform) or pre-computed phase/amplitude arrays.

Methods
-------
- Tort modulation index (Tort et al., 2010)
- Ozkurt modulation index (Ozkurt & Schnitzler, 2011)
- Canolty mean vector length (Canolty et al., 2006)

References
----------
Tort et al. (2010) J Neurophysiol, Ozkurt & Schnitzler (2011) J Neurosci Methods,
Canolty et al. (2006) Science, Cairney et al. (2018), Stokes et al. (2023).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert, butter, filtfilt

__all__ = [
    "modulation_index",
    "preferred_phase",
    "comodulogram",
    "surrogate_pac",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_phase(
    signal: np.ndarray, freq_band: tuple[float, float], fs: float
) -> np.ndarray:
    """Bandpass filter then extract instantaneous phase via Hilbert transform."""
    b, a = butter(3, [freq_band[0] / (fs / 2), freq_band[1] / (fs / 2)], btype="band")
    filtered = filtfilt(b, a, signal)
    return np.angle(hilbert(filtered))


def _extract_amplitude(
    signal: np.ndarray, freq_band: tuple[float, float], fs: float
) -> np.ndarray:
    """Bandpass filter then extract amplitude envelope via Hilbert transform."""
    b, a = butter(3, [freq_band[0] / (fs / 2), freq_band[1] / (fs / 2)], btype="band")
    filtered = filtfilt(b, a, signal)
    return np.abs(hilbert(filtered))


def _tort_mi(phase: np.ndarray, amplitude: np.ndarray, n_bins: int) -> float:
    """Tort modulation index: KL divergence of amplitude distribution over phase bins."""
    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    mean_amp = np.zeros(n_bins)
    for k in range(n_bins):
        mask = (phase >= bin_edges[k]) & (phase < bin_edges[k + 1])
        mean_amp[k] = amplitude[mask].mean() if mask.any() else 0.0

    # Normalize to probability distribution
    total = mean_amp.sum()
    if total == 0:
        return 0.0
    p = mean_amp / total

    # KL divergence from uniform
    uniform = np.ones(n_bins) / n_bins
    # Avoid log(0)
    p_safe = np.where(p > 0, p, 1e-12)
    kl = np.sum(p_safe * np.log(p_safe / uniform))
    mi = kl / np.log(n_bins)
    return float(mi)


def _ozkurt_mi(phase: np.ndarray, amplitude: np.ndarray) -> float:
    """Ozkurt normalized modulation index: |mean(amp * exp(i*phase))| / sqrt(mean(amp^2))."""
    z = np.abs(np.mean(amplitude * np.exp(1j * phase)))
    norm = np.sqrt(np.mean(amplitude**2))
    if norm == 0:
        return 0.0
    return float(z / norm)


def _canolty_mi(phase: np.ndarray, amplitude: np.ndarray) -> float:
    """Canolty mean vector length: |mean(amp * exp(i*phase))|."""
    return float(np.abs(np.mean(amplitude * np.exp(1j * phase))))


_MI_METHODS = {
    "tort": lambda ph, amp, n_bins: _tort_mi(ph, amp, n_bins),
    "ozkurt": lambda ph, amp, n_bins: _ozkurt_mi(ph, amp),
    "canolty": lambda ph, amp, n_bins: _canolty_mi(ph, amp),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def modulation_index(
    phase_signal: np.ndarray,
    amp_signal: np.ndarray,
    *,
    n_bins: int = 18,
    method: str = "tort",
) -> float:
    """
    Compute phase-amplitude coupling modulation index.

    Parameters
    ----------
    phase_signal : ndarray, shape (time,)
        Instantaneous phase of the low-frequency signal (radians, range [-pi, pi]).
    amp_signal : ndarray, shape (time,)
        Amplitude envelope of the high-frequency signal.
    n_bins : int
        Number of phase bins (default 18, matching pactools convention).
    method : str
        MI method: ``"tort"`` (KL-divergence), ``"ozkurt"`` (normalized MVL),
        or ``"canolty"`` (raw mean vector length).

    Returns
    -------
    mi : float
        Modulation index value. Higher values indicate stronger PAC.

    Notes
    -----
    Inputs should be pre-computed phase and amplitude arrays. Use
    `comodulogram` for automatic bandpass filtering and frequency sweeps.
    """
    phase_signal = np.asarray(phase_signal, dtype=np.float64).ravel()
    amp_signal = np.asarray(amp_signal, dtype=np.float64).ravel()
    if phase_signal.shape != amp_signal.shape:
        raise ValueError(
            f"phase_signal and amp_signal must have the same length; "
            f"got {len(phase_signal)} vs {len(amp_signal)}"
        )
    if method not in _MI_METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {list(_MI_METHODS)}")
    return _MI_METHODS[method](phase_signal, amp_signal, n_bins)


def preferred_phase(
    phase_signal: np.ndarray,
    amp_signal: np.ndarray,
) -> dict:
    """
    Preferred coupling phase and mean vector length.

    Computes the circular mean of the low-frequency phase weighted by
    high-frequency amplitude, indicating at which phase of the slow
    oscillation the fast activity is maximal.

    Parameters
    ----------
    phase_signal : ndarray, shape (time,)
        Instantaneous phase (radians).
    amp_signal : ndarray, shape (time,)
        Amplitude envelope.

    Returns
    -------
    result : dict with keys
        ``angle``  — float, preferred phase angle in radians [-pi, pi]
        ``mvl``    — float, mean vector length (0 = no coupling, 1 = perfect)
        ``pvalue`` — float, Rayleigh test p-value for non-uniformity
    """
    phase_signal = np.asarray(phase_signal, dtype=np.float64).ravel()
    amp_signal = np.asarray(amp_signal, dtype=np.float64).ravel()
    if phase_signal.shape != amp_signal.shape:
        raise ValueError(
            f"phase_signal and amp_signal must have the same length; "
            f"got {len(phase_signal)} vs {len(amp_signal)}"
        )

    # Amplitude-weighted mean resultant vector
    weights = amp_signal / (amp_signal.sum() + 1e-12)
    z = np.sum(weights * np.exp(1j * phase_signal))
    angle = float(np.angle(z))
    mvl = float(np.abs(z))

    # Rayleigh test approximation for weighted circular data
    n = len(phase_signal)
    R = mvl * n
    # Rayleigh test: p ≈ exp(-R^2 / n) for large n
    pvalue = float(np.exp(-(R**2) / n)) if n > 0 else 1.0
    pvalue = min(pvalue, 1.0)

    return {"angle": angle, "mvl": mvl, "pvalue": pvalue}


def comodulogram(
    signal: np.ndarray,
    freq_phase_range: tuple[float, float],
    freq_amp_range: tuple[float, float],
    fs: float,
    *,
    freq_phase_step: float | None = None,
    freq_amp_step: float | None = None,
    freq_phase_bw: float | None = None,
    freq_amp_bw: float | None = None,
    method: str = "tort",
    n_bins: int = 18,
) -> dict:
    """
    Frequency x frequency PAC comodulogram.

    Sweeps phase and amplitude frequency bands, computes modulation index
    for each pair, and returns a 2-D map.

    Parameters
    ----------
    signal : ndarray, shape (time,)
        Raw continuous signal.
    freq_phase_range : (lo, hi)
        Phase frequency range in Hz (e.g. ``(0.5, 4.0)`` for SO/delta).
    freq_amp_range : (lo, hi)
        Amplitude frequency range in Hz (e.g. ``(10.0, 100.0)``).
    fs : float
        Sampling rate in Hz.
    freq_phase_step : float or None
        Step size for phase frequency centers. Default: half of ``freq_phase_bw``.
    freq_amp_step : float or None
        Step size for amplitude frequency centers. Default: half of ``freq_amp_bw``.
    freq_phase_bw : float or None
        Bandwidth of phase band. Default: 1.0 Hz.
    freq_amp_bw : float or None
        Bandwidth of amplitude band. Default: 4.0 Hz for frequencies < 30 Hz,
        else ``freq_amp_bw = center * 0.2`` (20% fractional bandwidth).
    method : str
        MI method (see `modulation_index`).
    n_bins : int
        Number of phase bins for Tort method.

    Returns
    -------
    result : dict with keys
        ``mi``         — (n_phase, n_amp) modulation index matrix
        ``freq_phase`` — (n_phase,) phase frequency centers
        ``freq_amp``   — (n_amp,) amplitude frequency centers
    """
    signal = np.asarray(signal, dtype=np.float64).ravel()

    # Defaults
    if freq_phase_bw is None:
        freq_phase_bw = 1.0
    if freq_phase_step is None:
        freq_phase_step = freq_phase_bw / 2
    if freq_amp_step is None:
        freq_amp_step = 2.0

    freq_phase_centers = np.arange(
        freq_phase_range[0] + freq_phase_bw / 2,
        freq_phase_range[1] - freq_phase_bw / 2 + 1e-9,
        freq_phase_step,
    )
    freq_amp_centers = np.arange(
        freq_amp_range[0] + 2.0,  # minimum half-bw
        freq_amp_range[1] - 2.0 + 1e-9,
        freq_amp_step,
    )

    mi_map = np.zeros((len(freq_phase_centers), len(freq_amp_centers)))

    for i, fp in enumerate(freq_phase_centers):
        ph_lo = fp - freq_phase_bw / 2
        ph_hi = fp + freq_phase_bw / 2
        phase = _extract_phase(signal, (ph_lo, ph_hi), fs)

        for j, fa in enumerate(freq_amp_centers):
            bw = (
                freq_amp_bw
                if freq_amp_bw is not None
                else (4.0 if fa < 30 else fa * 0.2)
            )
            am_lo = fa - bw / 2
            am_hi = fa + bw / 2
            amp = _extract_amplitude(signal, (am_lo, am_hi), fs)
            mi_map[i, j] = modulation_index(phase, amp, n_bins=n_bins, method=method)

    return {
        "mi": mi_map,
        "freq_phase": freq_phase_centers,
        "freq_amp": freq_amp_centers,
    }


def surrogate_pac(
    phase_signal: np.ndarray,
    amp_signal: np.ndarray,
    *,
    n_surrogates: int = 200,
    method: str = "tort",
    n_bins: int = 18,
    surrogate_method: str = "circular_shift",
    seed: int | None = None,
) -> dict:
    """
    Surrogate testing for phase-amplitude coupling significance.

    Generates a null distribution of MI values by disrupting the
    phase-amplitude temporal relationship, then computes a z-score
    of the observed MI relative to this distribution.

    Parameters
    ----------
    phase_signal : ndarray, shape (time,)
        Instantaneous phase (radians).
    amp_signal : ndarray, shape (time,)
        Amplitude envelope.
    n_surrogates : int
        Number of surrogates (default 200).
    method : str
        MI method (see `modulation_index`).
    n_bins : int
        Number of phase bins.
    surrogate_method : str
        ``"circular_shift"`` — circularly shift amplitude relative to phase.
        ``"swap_phase"``     — randomly permute phase blocks.
        ``"time_block"``     — cut amplitude into blocks and shuffle.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    result : dict with keys
        ``observed``    — float, observed MI
        ``zscore``      — float, z-score relative to surrogate distribution
        ``pvalue``      — float, proportion of surrogates >= observed MI
        ``surrogates``  — (n_surrogates,) surrogate MI values
    """
    phase_signal = np.asarray(phase_signal, dtype=np.float64).ravel()
    amp_signal = np.asarray(amp_signal, dtype=np.float64).ravel()
    if phase_signal.shape != amp_signal.shape:
        raise ValueError(
            f"phase_signal and amp_signal must have the same length; "
            f"got {len(phase_signal)} vs {len(amp_signal)}"
        )

    rng = np.random.default_rng(seed)
    n = len(amp_signal)

    observed = modulation_index(phase_signal, amp_signal, n_bins=n_bins, method=method)

    surr_mis = np.empty(n_surrogates)
    for k in range(n_surrogates):
        if surrogate_method == "circular_shift":
            shift = rng.integers(int(0.1 * n), int(0.9 * n))
            amp_surr = np.roll(amp_signal, shift)
        elif surrogate_method == "swap_phase":
            # Randomly permute phase in blocks of ~1 cycle
            block_size = max(n // 20, 1)
            n_blocks = n // block_size
            idx = np.arange(n_blocks)
            rng.shuffle(idx)
            blocks = [phase_signal[i * block_size : (i + 1) * block_size] for i in idx]
            remainder = phase_signal[n_blocks * block_size :]
            phase_surr = np.concatenate(blocks + [remainder])
            surr_mis[k] = modulation_index(
                phase_surr, amp_signal, n_bins=n_bins, method=method
            )
            continue
        elif surrogate_method == "time_block":
            block_size = max(n // 20, 1)
            n_blocks = n // block_size
            idx = np.arange(n_blocks)
            rng.shuffle(idx)
            blocks = [amp_signal[i * block_size : (i + 1) * block_size] for i in idx]
            remainder = amp_signal[n_blocks * block_size :]
            amp_surr = np.concatenate(blocks + [remainder])
        else:
            raise ValueError(
                f"Unknown surrogate_method {surrogate_method!r}; "
                f"choose from 'circular_shift', 'swap_phase', 'time_block'"
            )
        surr_mis[k] = modulation_index(
            phase_signal, amp_surr, n_bins=n_bins, method=method
        )

    surr_mean = surr_mis.mean()
    surr_std = surr_mis.std()
    zscore = (observed - surr_mean) / (surr_std + 1e-12)
    pvalue = float(np.mean(surr_mis >= observed))

    return {
        "observed": observed,
        "zscore": zscore,
        "pvalue": pvalue,
        "surrogates": surr_mis,
    }
