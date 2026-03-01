"""
Spectral features derived from pre-computed PSD estimates.

Status
------
STATUS: ACTIVE
Reason: PSD-first spectral feature extraction for iEEG channel characterization.
Superseded by: n/a
Safe to remove: no

Convention:
    psd :   (..., freq)  as returned by psd_welch or psd_multitaper
    freqs : (freq,)      Hz, strictly increasing
    scalar output : (...,)     freq axis reduced
    vector output : (..., freq) same freq axis as input
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12

__all__ = [
    "band_power",
    "relative_band_power",
    "spectral_entropy",
    "spectral_edge",
    "broadband_snr",
    "line_noise_ratio",
    "am_artifact_score",
    "am_depth",
    "aperiodic_exponent",
    "fooof_periodic",
]


def _band_mask(freqs: np.ndarray, band):
    fmin, fmax = float(band[0]), float(band[1])
    if fmin >= fmax:
        raise ValueError(f"Invalid band: fmin={fmin} >= fmax={fmax}")
    return (freqs >= fmin) & (freqs <= fmax)


def band_power(psd, freqs, band):
    """
    Integrate PSD over band using trapezoidal rule.

    Parameters
    ----------
    psd : (..., freq)
    freqs : (freq,)
    band : (fmin_hz, fmax_hz) tuple

    Returns
    -------
    power : (...,)

    Raises
    ------
    ValueError if band falls entirely outside freqs range.

    Examples
    --------
    >>> bp = band_power(psd, freqs, band=(4.0, 8.0))
    >>> bp.shape  # (...,)
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    psd = np.asarray(psd, dtype=np.float64)
    mask = _band_mask(freqs, band)
    if not np.any(mask):
        raise ValueError(
            f"Band {band} falls outside freqs range [{float(freqs[0])}, {float(freqs[-1])}]"
        )
    return np.trapz(psd[..., mask], x=freqs[mask], axis=-1)


def relative_band_power(psd, freqs, band, *, norm_range=None):
    """
    band_power(band) / band_power(norm_range).

    Parameters
    ----------
    norm_range : (fmin, fmax) | None
        Normalization band. None = full freqs range.

    Returns
    -------
    rel_power : (...,)
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    if norm_range is None:
        norm_range = (float(freqs[0]), float(freqs[-1]))
    num = band_power(psd, freqs, band)
    den = band_power(psd, freqs, norm_range)
    return num / (den + EPS)


def spectral_entropy(psd, freqs):
    """
    Shannon entropy of normalized PSD.

    p = psd / sum(psd * df); H = -sum(p * log(p + EPS) * df)
    Normalized PSD integrates to 1 (probability distribution over freq).

    Returns
    -------
    entropy : (...,)
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    psd = np.asarray(psd, dtype=np.float64)
    df = np.gradient(freqs).astype(np.float64, copy=False)
    total = np.sum(psd * df, axis=-1, keepdims=True)
    p = psd / (total + EPS)
    return -np.sum(p * np.log(p + EPS) * df, axis=-1)


def _cumulative_trapz(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    dx = np.diff(x)
    area = (y[..., 1:] + y[..., :-1]) * dx / 2.0
    cum = np.cumsum(area, axis=-1)
    return np.concatenate([np.zeros_like(cum[..., :1]), cum], axis=-1)


def spectral_edge(psd, freqs, *, p=0.95):
    """
    Frequency below which fraction p of total power lies (SEF).

    Uses cumulative trapezoidal integration.

    Returns
    -------
    sef : (...,) in Hz
    """
    if not (0.0 < float(p) < 1.0):
        raise ValueError("p must be between 0 and 1 (exclusive)")
    freqs = np.asarray(freqs, dtype=np.float64)
    psd = np.asarray(psd, dtype=np.float64)
    cum = _cumulative_trapz(psd, freqs)
    total = cum[..., -1]
    target = float(p) * total

    out = np.full(total.shape, np.nan, dtype=np.float64)
    valid = total > 0
    if not np.any(valid):
        return out

    cum_v = cum[valid]
    target_v = target[valid]

    idx = np.argmax(cum_v >= target_v[..., None], axis=-1)
    idx = np.clip(idx, 1, freqs.size - 1)
    x0 = freqs[idx - 1]
    x1 = freqs[idx]
    y0 = cum_v[np.arange(cum_v.shape[0]), idx - 1]
    y1 = cum_v[np.arange(cum_v.shape[0]), idx]
    w = (target_v - y0) / (y1 - y0 + EPS)
    out[valid] = x0 + w * (x1 - x0)
    return out


def broadband_snr(
    psd,
    freqs,
    *,
    sig_band=(1.0, 40.0),
    noise_band=(80.0, 200.0),
):
    """
    log10(P_signal / P_noise).

    Returns
    -------
    snr : (...,)  negative = noise-dominated
    """
    p_sig = band_power(psd, freqs, sig_band)
    p_noise = band_power(psd, freqs, noise_band)
    return np.log10((p_sig + EPS) / (p_noise + EPS))


def line_noise_ratio(psd, freqs, *, f_line=50.0, bw=2.0):
    """
    P_line / P_flanks.

    Line band  : (f_line - bw, f_line + bw)
    Flank bands: (f_line - 4*bw, f_line - 2*bw)
                 (f_line + 2*bw, f_line + 4*bw)
    P_flanks = mean of both flank bands.

    Returns
    -------
    lnr : (...,)
    """
    f_line = float(f_line)
    bw = float(bw)
    p_line = band_power(psd, freqs, (f_line - bw, f_line + bw))
    p_left = band_power(psd, freqs, (f_line - 4 * bw, f_line - 2 * bw))
    p_right = band_power(psd, freqs, (f_line + 2 * bw, f_line + 4 * bw))
    p_flanks = 0.5 * (p_left + p_right)
    return (p_line + EPS) / (p_flanks + EPS)


def am_artifact_score(psd, freqs, *, fc, fm, carrier_bw=2.0, sideband_bw=2.0):
    """
    log10(P_sideband / P_background).

    Sidebands : (fc ± fm ± sideband_bw)
    Background: flanks at fc ± 3*fm (outside carrier and sidebands)

    Parameters
    ----------
    fc : float — carrier frequency Hz (e.g. 40.0)
    fm : float — modulation frequency Hz (e.g. 10.0)

    Returns
    -------
    score : (...,)
    """
    fc = float(fc)
    fm = float(fm)
    sideband_bw = float(sideband_bw)

    p_sb1 = band_power(psd, freqs, (fc - fm - sideband_bw, fc - fm + sideband_bw))
    p_sb2 = band_power(psd, freqs, (fc + fm - sideband_bw, fc + fm + sideband_bw))
    p_side = p_sb1 + p_sb2

    p_bg1 = band_power(psd, freqs, (fc - 3 * fm - sideband_bw, fc - 3 * fm + sideband_bw))
    p_bg2 = band_power(psd, freqs, (fc + 3 * fm - sideband_bw, fc + 3 * fm + sideband_bw))
    p_bg = 0.5 * (p_bg1 + p_bg2)

    return np.log10((p_side + EPS) / (p_bg + EPS))


def am_depth(psd, freqs, *, fc, fm, carrier_bw=2.0, sideband_bw=2.0):
    """
    log10(P_sideband / P_carrier).

    Carrier band: (fc - carrier_bw, fc + carrier_bw)
    Shares sideband computation with am_artifact_score.

    Returns
    -------
    depth : (...,)
    """
    fc = float(fc)
    fm = float(fm)
    carrier_bw = float(carrier_bw)
    sideband_bw = float(sideband_bw)

    p_sb1 = band_power(psd, freqs, (fc - fm - sideband_bw, fc - fm + sideband_bw))
    p_sb2 = band_power(psd, freqs, (fc + fm - sideband_bw, fc + fm + sideband_bw))
    p_side = p_sb1 + p_sb2
    p_car = band_power(psd, freqs, (fc - carrier_bw, fc + carrier_bw))
    return np.log10((p_side + EPS) / (p_car + EPS))


def aperiodic_exponent(psd, freqs, *, freq_range=(30.0, 45.0)):
    """
    Fit 1/f aperiodic exponent via specparam (fooof).

    Parameters
    ----------
    freq_range : (fmin, fmax) — Hz range for fitting,
        should be free of oscillatory peaks.

    Returns
    -------
    exponent : (...,)  NaN where fit fails.

    Notes
    -----
    Operates per-spectrum via np.apply_along_axis.
    Tries specparam.SpectralModel first, falls back to fooof.FOOOF.
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    psd = np.asarray(psd, dtype=np.float64)

    try:
        from specparam import SpectralModel  # type: ignore

        def _fit_one(pxx):
            sm = SpectralModel()
            sm.fit(freqs, pxx, freq_range=freq_range)
            ap = getattr(sm, "aperiodic_params_", None)
            if ap is None or len(ap) < 2:
                return np.nan
            return float(ap[1])

    except ImportError:
        try:
            from fooof import FOOOF  # type: ignore

            def _fit_one(pxx):
                fm = FOOOF(peak_width_limits=(1, 12), verbose=False)
                fm.fit(freqs, pxx, freq_range=freq_range)
                ap = getattr(fm, "aperiodic_params_", None)
                if ap is None or len(ap) < 2:
                    return np.nan
                return float(ap[1])

        except ImportError as e:
            raise ImportError(
                "aperiodic_exponent requires specparam or fooof: pip install fooof"
            ) from e

    return np.apply_along_axis(_fit_one, -1, psd)


def fooof_periodic(psd, freqs, *, freq_range=None):
    """
    Periodic component above 1/f background via specparam (fooof).

    Returns
    -------
    periodic : (..., freq)  same shape as psd. NaN where fit fails.

    Notes
    -----
    freq_range: None = full freqs range.
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    psd = np.asarray(psd, dtype=np.float64)
    if freq_range is None:
        freq_range = (float(freqs[0]), float(freqs[-1]))

    try:
        from specparam import SpectralModel  # type: ignore

        def _fit_one(pxx):
            sm = SpectralModel()
            sm.fit(freqs, pxx, freq_range=freq_range)
            model = getattr(sm, "fooofed_spectrum_", None)
            ap = getattr(sm, "_ap_fit", None)
            if model is None or ap is None:
                return np.full_like(pxx, np.nan, dtype=np.float64)
            return np.asarray(model, dtype=np.float64) - np.asarray(ap, dtype=np.float64)

    except ImportError:
        try:
            from fooof import FOOOF  # type: ignore

            def _fit_one(pxx):
                fm = FOOOF(peak_width_limits=(1, 12), verbose=False)
                fm.fit(freqs, pxx, freq_range=freq_range)
                model = getattr(fm, "fooofed_spectrum_", None)
                ap = getattr(fm, "_ap_fit", None)
                if model is None or ap is None:
                    return np.full_like(pxx, np.nan, dtype=np.float64)
                return np.asarray(model, dtype=np.float64) - np.asarray(ap, dtype=np.float64)

        except ImportError as e:
            raise ImportError(
                "fooof_periodic requires specparam or fooof: pip install fooof"
            ) from e

    return np.apply_along_axis(_fit_one, -1, psd)

