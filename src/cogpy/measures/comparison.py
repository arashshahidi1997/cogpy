"""
Before/after signal comparison metrics.

Pure functions for quantifying the effect of a signal processing step
(e.g., artifact removal) by comparing signals or their spectra.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "snr_improvement",
    "residual_energy_ratio",
    "bandpower_change",
    "waveform_residual_rms",
]


def snr_improvement(
    psd_before: np.ndarray,
    psd_after: np.ndarray,
    freqs: np.ndarray,
    *,
    signal_band: tuple[float, float],
    noise_band: tuple[float, float],
) -> float:
    """
    Change in signal-to-noise ratio after processing.

    Computes ``SNR_after - SNR_before`` where
    ``SNR = 10 * log10(P_signal / P_noise)`` and power is integrated
    over the respective bands.

    Parameters
    ----------
    psd_before, psd_after : (..., n_freq) float
        Power spectral densities before and after processing.
    freqs : (n_freq,) float
        Frequency axis (Hz).
    signal_band : (f_low, f_high)
        Frequency band considered "signal" (Hz).
    noise_band : (f_low, f_high)
        Frequency band considered "noise" (Hz).

    Returns
    -------
    float
        SNR improvement in dB.  Positive means improvement.
    """
    from cogpy.spectral.features import band_power

    freqs = np.asarray(freqs, dtype=float)

    def _snr(psd):
        p_sig = band_power(psd, freqs, signal_band)
        p_noise = band_power(psd, freqs, noise_band)
        eps = 1e-30
        return 10.0 * np.log10((p_sig + eps) / (p_noise + eps))

    return float(_snr(psd_after) - _snr(psd_before))


def residual_energy_ratio(
    original: np.ndarray,
    cleaned: np.ndarray,
    *,
    axis: int = -1,
) -> np.ndarray:
    """
    Ratio of residual energy to original energy.

    ``ratio = sum((original - cleaned)^2) / sum(original^2)``

    Small values (< 0.01) mean little was removed.
    Values near 1.0 mean most energy was removed.

    Parameters
    ----------
    original, cleaned : array-like
        Signals before and after processing.
    axis : int
        Axis along which to compute energy (default: last).

    Returns
    -------
    ndarray
        Energy ratio.  Scalar if inputs are 1-D.
    """
    o = np.asarray(original, dtype=float)
    c = np.asarray(cleaned, dtype=float)
    residual = o - c
    eps = 1e-30
    return np.sum(residual**2, axis=axis) / (np.sum(o**2, axis=axis) + eps)


def bandpower_change(
    psd_before: np.ndarray,
    psd_after: np.ndarray,
    freqs: np.ndarray,
    *,
    band: tuple[float, float],
) -> np.ndarray:
    """
    Fractional change in band power: ``(P_after - P_before) / P_before``.

    Parameters
    ----------
    psd_before, psd_after : (..., n_freq) float
        Power spectral densities.
    freqs : (n_freq,) float
        Frequency axis (Hz).
    band : (f_low, f_high)
        Frequency band (Hz).

    Returns
    -------
    ndarray
        Fractional change.  Negative means power decreased.
    """
    from cogpy.spectral.features import band_power

    freqs = np.asarray(freqs, dtype=float)
    p_before = band_power(psd_before, freqs, band)
    p_after = band_power(psd_after, freqs, band)
    eps = 1e-30
    return (p_after - p_before) / (p_before + eps)


def waveform_residual_rms(
    template_before: np.ndarray,
    template_after: np.ndarray,
) -> float:
    """
    RMS of the difference between two waveforms (e.g. triggered averages).

    Useful for verifying that an artifact template has been removed:
    compare the event-triggered average before and after cleaning.

    Parameters
    ----------
    template_before, template_after : (..., n_lag) float
        Waveforms to compare (e.g. triggered averages).

    Returns
    -------
    float
        RMS of the difference.
    """
    diff = np.asarray(template_before, dtype=float) - np.asarray(template_after, dtype=float)
    return float(np.sqrt(np.mean(diff**2)))
