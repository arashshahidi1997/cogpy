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
import xarray as xr

EPS = 1e-12

__all__ = [
    "band_power",
    "relative_band_power",
    "spectral_entropy",
    "spectral_flatness",
    "spectral_edge",
    "broadband_snr",
    "line_noise_ratio",
    "narrowband_ratio",
    "spectral_peak_freqs",
    "ftest_line_scan",
    "am_artifact_score",
    "am_depth",
    "aperiodic_exponent",
    "fooof_periodic",
    "reduce_tf_bands",
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

    p_bg1 = band_power(
        psd, freqs, (fc - 3 * fm - sideband_bw, fc - 3 * fm + sideband_bw)
    )
    p_bg2 = band_power(
        psd, freqs, (fc + 3 * fm - sideband_bw, fc + 3 * fm + sideband_bw)
    )
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


def spectral_flatness(psd, freqs):
    """
    Spectral flatness (Wiener entropy).

    Ratio of geometric mean to arithmetic mean of the PSD.
    1.0 = white noise (maximally flat), 0.0 = pure tone.

    More robust to bandwidth than Shannon spectral_entropy.

    Parameters
    ----------
    psd : (..., freq)
    freqs : (freq,) — unused, kept for API consistency

    Returns
    -------
    flatness : (...,) — in [0, 1]
    """
    psd = np.asarray(psd, dtype=np.float64)
    log_mean = np.mean(np.log(psd + EPS), axis=-1)
    arith_mean = np.mean(psd, axis=-1)
    return np.exp(log_mean) / (arith_mean + EPS)


def narrowband_ratio(psd, freqs, *, flank_hz=5.0):
    """
    Narrowband prominence ratio per frequency bin.

    For each bin, ratio of its power to the median power of flanking
    bins within ±flank_hz. Values >> 1 indicate narrowband peaks
    (line noise, artifact). Values near 1 indicate broadband.

    Parameters
    ----------
    psd : (..., freq)
    freqs : (freq,)  — Hz, strictly increasing
    flank_hz : float — half-width of flanking window in Hz (default 5.0)

    Returns
    -------
    ratio : (..., freq) — power / median(flanking power).
        NaN where fewer than 2 flank bins exist.

    Examples
    --------
    >>> ratio = narrowband_ratio(psd, freqs, flank_hz=5.0)
    >>> line_mask = ratio > 5.0  # strong narrowband peaks
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    psd = np.asarray(psd, dtype=np.float64)
    flank_hz = float(flank_hz)
    nf = freqs.shape[0]

    flank_median = np.full(psd.shape, np.nan, dtype=np.float64)
    for i in range(nf):
        mask = (np.abs(freqs - freqs[i]) <= flank_hz) & (np.arange(nf) != i)
        if np.sum(mask) < 2:
            continue
        flank_median[..., i] = np.median(psd[..., mask], axis=-1)

    return psd / (flank_median + EPS)


def spectral_peak_freqs(psd, freqs, *, prominence=2.0, min_distance_hz=2.0):
    """
    Detect discrete peak frequencies in a PSD.

    Thin wrapper around ``scipy.signal.find_peaks`` applied to the last
    (frequency) axis.  Useful for enumerating narrowband peaks found by
    ``narrowband_ratio`` or for identifying line-noise harmonics.

    Parameters
    ----------
    psd : (..., freq)
    freqs : (freq,)  — Hz, strictly increasing
    prominence : float
        Minimum prominence (in PSD units) for a peak to be reported.
        Default 2.0.
    min_distance_hz : float
        Minimum separation between peaks in Hz.  Converted to bins
        internally.  Default 2.0.

    Returns
    -------
    peak_freqs : ndarray (1D input) or list[ndarray] (batched input)
        Frequency values at detected peaks.  For 1D ``psd``, a single
        1D array.  For batched ``psd``, a list with one array per
        batch element (peak count may vary across elements).
    """
    from scipy.signal import find_peaks as _find_peaks

    freqs = np.asarray(freqs, dtype=np.float64)
    psd = np.asarray(psd, dtype=np.float64)

    df = float(np.median(np.diff(freqs))) if freqs.size > 1 else 1.0
    distance = max(1, int(round(float(min_distance_hz) / df)))

    if psd.ndim == 1:
        idx, _ = _find_peaks(psd, prominence=float(prominence), distance=distance)
        return freqs[idx]

    batch_shape = psd.shape[:-1]
    flat = psd.reshape(-1, psd.shape[-1])
    results = []
    for i in range(flat.shape[0]):
        idx, _ = _find_peaks(flat[i], prominence=float(prominence), distance=distance)
        results.append(freqs[idx])

    return results


def ftest_line_scan(signal, fs, *, NW=4.0, p_threshold=0.05):
    """
    F-test line scan across all frequency bins.

    Computes the Thomson (1982) F-statistic at every frequency bin
    in a single pass, using eigenvalue-weighted DPSS tapers. Identifies
    narrowband sinusoidal components against the broadband background.

    Parameters
    ----------
    signal : (..., time) — signal window(s).  Arbitrary leading batch dims
        are supported; the last axis is treated as time.
    fs : float — sampling rate Hz
    NW : float — time-bandwidth product (default 4.0)
    p_threshold : float — significance level for boolean mask (default 0.05)

    Returns
    -------
    fstat : (..., freq) — F-statistic at each frequency bin
    freqs : (freq,) — frequency axis in Hz
    sig_mask : (..., freq) — boolean, True where F exceeds critical value

    Notes
    -----
    Uses the proper eigenvalue-weighted Thomson F-test:
        mu_hat(f) = sum_k(lambda_k * H_k(0) * Y_k(f)) / sum_k(lambda_k * H_k(0)^2)
    where H_k(0) is the DC value of the k-th DPSS taper and lambda_k is its
    eigenvalue. Under H0, F ~ F(2, 2K-2).

    DPSS tapers and eigenvalues are shared across batch elements (same N,
    NW, K), so the tapered FFT is computed as a single batched matrix
    multiply.
    """
    from scipy.signal.windows import dpss
    from scipy.stats import f as f_dist

    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim < 1:
        raise ValueError(
            f"signal must have at least 1 dimension, got shape {signal.shape}"
        )

    N = signal.shape[-1]
    NW = float(NW)
    K = int(2 * NW - 1)
    batch_shape = signal.shape[:-1]

    # Get DPSS tapers and eigenvalues for weighting
    tapers, eigenvalues = dpss(N, NW, Kmax=K, return_ratios=True)
    # tapers: (K, N), eigenvalues: (K,)
    # H_k(0) = sum of k-th taper (DC response)
    H0 = np.sum(tapers, axis=1)  # (K,)

    # Batched tapered FFT: (..., time) @ (N, K) → (..., K, nfft)
    # Detrend: subtract mean along time axis
    signal = signal - np.mean(signal, axis=-1, keepdims=True)
    # tapered: (..., K, N) via broadcast
    tapered = signal[..., np.newaxis, :] * tapers  # (..., K, N)
    nfft = N
    mtfft = np.fft.rfft(tapered, n=nfft, axis=-1)  # (..., K, freq)
    freqs = np.fft.rfftfreq(N, d=1.0 / float(fs))

    # Eigenvalue-weighted line amplitude estimate (Thomson 1982)
    weights = eigenvalues * H0  # (K,)
    w_norm = np.sum(eigenvalues * H0**2)  # scalar
    # mu_hat(f) = sum_k(w_k * Y_k(f)) / w_norm
    # weights[:, None] broadcasts over freq; sum over K axis
    mu_hat = np.sum(weights[..., :, np.newaxis] * mtfft, axis=-2) / (
        w_norm + EPS
    )  # (..., freq)

    # Total power per frequency
    S = np.mean(np.abs(mtfft) ** 2, axis=-2)  # (..., freq)
    mu2 = np.abs(mu_hat) ** 2

    # Residual (broadband) power
    residual = S - mu2 * w_norm / K + EPS
    fstat = (K - 1) * mu2 * w_norm / (K * residual)

    f_crit = f_dist.ppf(1.0 - float(p_threshold), 2, 2 * K - 2)
    sig_mask = fstat > f_crit

    return fstat, freqs, sig_mask


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
            return np.asarray(model, dtype=np.float64) - np.asarray(
                ap, dtype=np.float64
            )

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
                return np.asarray(model, dtype=np.float64) - np.asarray(
                    ap, dtype=np.float64
                )

        except ImportError as e:
            raise ImportError(
                "fooof_periodic requires specparam or fooof: pip install fooof"
            ) from e

    return np.apply_along_axis(_fit_one, -1, psd)


_REDUCE_METHODS = {
    "mean": lambda da, dim: da.mean(dim=dim),
    "median": lambda da, dim: da.median(dim=dim),
    "max": lambda da, dim: da.max(dim=dim),
    "sum": lambda da, dim: da.sum(dim=dim),
}


def reduce_tf_bands(
    score: xr.DataArray,
    bands: dict[str, tuple[float, float]],
    *,
    freq_dim: str = "freq",
    method: str = "mean",
) -> xr.Dataset:
    """
    Reduce a frequency axis to per-band scalars.

    Parameters
    ----------
    score : xr.DataArray
        Input with a *freq_dim* dimension and associated coordinate in Hz.
    bands : dict
        Mapping of ``band_name → (fmin_hz, fmax_hz)``.
    freq_dim : str
        Name of the frequency dimension (default ``"freq"``).
    method : {"mean", "median", "max", "sum"}
        Reduction applied within each band.

    Returns
    -------
    xr.Dataset
        One variable per band, *freq_dim* removed.
    """
    if freq_dim not in score.dims:
        raise ValueError(
            f"Dimension {freq_dim!r} not in score.dims={tuple(score.dims)}"
        )
    if freq_dim not in score.coords:
        raise ValueError(f"Coordinate {freq_dim!r} required for frequency selection")
    reduce_fn = _REDUCE_METHODS.get(method)
    if reduce_fn is None:
        raise ValueError(
            f"Unknown method {method!r}. Use one of {sorted(_REDUCE_METHODS)}"
        )

    freqs = score[freq_dim].values
    variables: dict[str, xr.DataArray] = {}
    for name, (fmin, fmax) in bands.items():
        mask = (freqs >= fmin) & (freqs <= fmax)
        if not np.any(mask):
            raise ValueError(
                f"Band {name!r} ({fmin}, {fmax}) has no overlap with freq range "
                f"[{freqs.min():.1f}, {freqs.max():.1f}]"
            )
        band_slice = score.isel({freq_dim: mask})
        variables[name] = reduce_fn(band_slice, freq_dim)

    return xr.Dataset(variables)
