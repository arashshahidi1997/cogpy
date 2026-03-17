"""
Spatial signal measures for 2D electrode grids.

Status
------
STATUS: ACTIVE
Reason: Spatial characterization measures for iEEG grid recordings.
Superseded by: n/a
Safe to remove: no

Grid convention:
    grid : (..., AP, ML)  — AP rows, ML columns
    Batch dims (time, freq, etc.) are leading; spatial axes are always last.
    Scalar measures reduce (AP, ML) → scalar, so output shape is (...).
    2D input returns a Python float for backward compatibility.
"""

from __future__ import annotations

import functools

import numpy as np

EPS = 1e-12

__all__ = [
    "moran_i",
    "csd_power",
    "spatial_coherence_profile",
    "marginal_energy_outlier",
    "gradient_anisotropy",
    "spatial_kurtosis",
    "spatial_noise_concentration",
    "spatial_summary_xr",
]


@functools.lru_cache(maxsize=16)
def _build_adjacency(ap, ml, adjacency):
    """Build binary adjacency matrix for an (AP, ML) grid.

    Returns (N, N) array where N = ap * ml.  Symmetric.
    """
    _valid_adj = {"queen", "rook", "ap_only", "ml_only"}
    if adjacency not in _valid_adj:
        raise ValueError(f"adjacency must be one of {_valid_adj}, got {adjacency!r}.")

    n = int(ap * ml)
    W = np.zeros((n, n), dtype=float)

    if adjacency == "ap_only":
        offsets = [(-1, 0), (1, 0)]
    elif adjacency == "ml_only":
        offsets = [(0, -1), (0, 1)]
    elif adjacency == "rook":
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:  # queen
        offsets = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]

    for i in range(ap):
        for j in range(ml):
            idx = i * ml + j
            for di, dj in offsets:
                ii, jj = i + di, j + dj
                if 0 <= ii < ap and 0 <= jj < ml:
                    W[idx, ii * ml + jj] = 1.0
    return W


def moran_i(grid: np.ndarray, *, adjacency: str = "queen"):
    """
    Moran's I spatial autocorrelation for a scalar grid map.

    I = (N / W) * (sum_ij w_ij (x_i-xmean)(x_j-xmean)) /
                   sum_i (x_i - xmean)^2

    Values:
        I ~ +1 : spatially smooth (biological signal)
        I ~  0 : spatially random (independent noise)
        I ~ -1 : spatially anti-correlated (referencing artifact)

    Parameters
    ----------
    grid : (..., AP, ML) — scalar value per electrode.
        Batch dimensions are leading; spatial axes are always last two.
        NaN entries excluded from computation (single mask across batch —
        bad electrodes are assumed constant across time/freq slices).
    adjacency : str
        "queen" (8-connected) | "rook" (4-connected) |
        "ap_only" (vertical neighbors only) |
        "ml_only" (horizontal neighbors only)

        Directional modes enable stripe vs checkerboard discrimination:
        - Row-striped artifact: high I for ml_only, low/negative for ap_only
        - Column-striped artifact: high I for ap_only, low/negative for ml_only
        - Checkerboard: negative I for both

    Returns
    -------
    I : (...) float array (or scalar float for 2D input)
    """
    g = np.asarray(grid, dtype=float)
    if g.ndim < 2:
        raise ValueError(f"grid must have shape (..., AP, ML), got {g.shape}.")
    ap, ml = g.shape[-2], g.shape[-1]
    batch_shape = g.shape[:-2]
    scalar_output = g.ndim == 2
    n = ap * ml

    W = _build_adjacency(ap, ml, adjacency)

    # Flatten spatial dims: (..., N)
    x = g.reshape(batch_shape + (n,))

    # Single NaN mask: electrode is valid only if finite across ALL batch slices.
    # This is the practical case — bad electrodes are constant per recording.
    finite = np.isfinite(x)
    batch_axes = tuple(range(len(batch_shape)))
    valid = np.all(finite, axis=batch_axes) if batch_axes else finite.reshape(-1)

    if not np.any(valid):
        return float("nan") if scalar_output else np.full(batch_shape, np.nan)

    # Subset to valid electrodes
    x_v = x[..., valid]              # (..., n_valid)
    W_v = W[np.ix_(valid, valid)]    # (n_valid, n_valid)

    Wsum = float(np.sum(W_v))
    if Wsum <= 0:
        return float("nan") if scalar_output else np.full(batch_shape, np.nan)

    n_valid = x_v.shape[-1]
    xmean = np.mean(x_v, axis=-1, keepdims=True)   # (..., 1)
    xc = x_v - xmean                                # (..., n_valid)

    denom = np.sum(xc ** 2, axis=-1)                # (...)

    # Numerator: xc^T @ W @ xc  per batch element
    # W is symmetric, so xc @ W gives the same as (W @ xc^T)^T
    Wxc = xc @ W_v                                   # (..., n_valid)
    num = np.sum(xc * Wxc, axis=-1)                  # (...)

    result = np.where(
        denom > EPS,
        (n_valid / Wsum) * (num / (denom + EPS)),
        np.nan,
    )

    return float(result) if scalar_output else result


def csd_power(
    grid_signal: np.ndarray, *, spacing_mm: float = 1.0, axis: int = -1
) -> np.ndarray:
    """
    Current Source Density (CSD) via 2D Laplacian of surface potential.

    Sharpens spatial specificity by removing volume-conducted components.
    CSD = -sigma * Laplacian(V); sigma omitted (constant scaling).

    Parameters
    ----------
    grid_signal : (AP, ML, time)
    spacing_mm : float — electrode spacing in mm (default 1.0)
    axis : int — time axis (must be last axis, default -1)

    Returns
    -------
    csd : (AP, ML, time)
        Interior: 5-point finite-difference Laplacian.
        Border electrodes: NaN.

    Implementation
    --------------
    5-point stencil:
        L[i,j] = (V[i-1,j] + V[i+1,j] + V[i,j-1] + V[i,j+1]
                  - 4*V[i,j]) / spacing_mm^2
    Use manual stencil (not scipy.ndimage.laplace) for explicit
    NaN border control. Apply along spatial axes, broadcast over time.
    """
    x = np.asarray(grid_signal, dtype=float)
    if x.ndim != 3:
        raise ValueError(f"grid_signal must have shape (AP, ML, time), got {x.shape}.")
    if axis not in (-1, 2):
        raise ValueError(f"axis must refer to the last dimension (time), got axis={axis}.")
    if spacing_mm <= 0:
        raise ValueError(f"spacing_mm must be positive, got {spacing_mm}.")

    ap, ml, _t = x.shape
    out = np.full_like(x, np.nan, dtype=float)
    if ap < 3 or ml < 3:
        return out

    inv_h2 = 1.0 / (float(spacing_mm) ** 2)
    interior = (
        x[:-2, 1:-1, :]
        + x[2:, 1:-1, :]
        + x[1:-1, :-2, :]
        + x[1:-1, 2:, :]
        - 4.0 * x[1:-1, 1:-1, :]
    ) * inv_h2

    out[1:-1, 1:-1, :] = -interior
    return out


def spatial_coherence_profile(
    grid_signal: np.ndarray,
    fs: float,
    *,
    spacing_mm: float = 1.0,
    NW: float = 4,
    n_distance_bins: int = 10,
    fmin: float = 0.0,
    fmax: float = None,
) -> tuple:
    """
    Mean coherence as a function of inter-electrode distance.

    Computes pairwise coherence for all electrode pairs, bins by
    physical distance, returns mean coherence per bin.
    Output matches DIMS_SPATIAL_COHERENCE_PROFILE = (distance_bin, freq).

    Biological: coherence decreases with distance.
    Artifact: flat distance-independent profile.

    Parameters
    ----------
    grid_signal : (AP, ML, time)
    fs : float — sampling rate in Hz
    spacing_mm : float — electrode spacing in mm (default 1.0)
    NW : float — multitaper time-bandwidth product (default 4)
    n_distance_bins : int — number of distance bins (default 10)
    fmin, fmax : float — frequency range for output

    Returns
    -------
    coh_profile  : (n_distance_bins, freq) — mean coherence per bin
    distance_bins: (n_distance_bins,) — bin centers in mm
    freqs        : (freq,) — Hz

    Implementation
    --------------
    1. Reshape grid_signal (AP, ML, time) → (n_channels, time)
    2. Compute mtfft for all channels via multitaper_fft
    3. Build electrode position array from grid indices * spacing_mm
    4. Enumerate all unique pairs (upper triangle)
    5. For each pair: compute coherence via bivariate.coherence
    6. Compute pairwise distances
    7. Bin pairs by distance, average coherence within each bin
    8. Clip output to [fmin, fmax]

    Use cogpy.core.spectral.multitaper.multitaper_fft and
    cogpy.core.spectral.bivariate.coherence internally.
    Do NOT call sys.path — all imports are from the cogpy env.
    """
    from cogpy.core.spectral.multitaper import multitaper_fft
    from cogpy.core.spectral.bivariate import coherence as _coherence

    x = np.asarray(grid_signal, dtype=float)
    if x.ndim != 3:
        raise ValueError(f"grid_signal must have shape (AP, ML, time), got {x.shape}.")
    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}.")
    if spacing_mm <= 0:
        raise ValueError(f"spacing_mm must be positive, got {spacing_mm}.")
    if n_distance_bins <= 0:
        raise ValueError(f"n_distance_bins must be positive, got {n_distance_bins}.")

    ap, ml, n_time = x.shape
    n_ch = int(ap * ml)
    y = x.reshape(n_ch, n_time)

    mtfft = multitaper_fft(y, axis=-1, NW=float(NW))
    freqs = np.fft.rfftfreq(int(n_time), d=1.0 / float(fs))

    if fmax is None:
        fmax_ = float(freqs[-1])
    else:
        fmax_ = float(fmax)
    fmin_ = float(fmin)
    if fmax_ < fmin_:
        raise ValueError(f"fmax must be >= fmin, got fmin={fmin_}, fmax={fmax_}.")

    fmask = (freqs >= fmin_) & (freqs <= fmax_)
    freqs = freqs[fmask]
    mtfft = mtfft[..., fmask]

    ap_idx, ml_idx = np.divmod(np.arange(n_ch), ml)
    pos_ap = ap_idx.astype(float) * float(spacing_mm)
    pos_ml = ml_idx.astype(float) * float(spacing_mm)

    iu, ju = np.triu_indices(n_ch, k=1)
    d_ap = pos_ap[iu] - pos_ap[ju]
    d_ml = pos_ml[iu] - pos_ml[ju]
    distances = np.sqrt(d_ap**2 + d_ml**2)

    dmin = float(np.min(distances))
    dmax = float(np.max(distances))
    if not np.isfinite(dmin) or not np.isfinite(dmax) or dmax <= 0:
        raise ValueError("Invalid distance range computed from grid geometry.")

    edges = np.linspace(dmin, dmax, int(n_distance_bins) + 1)
    distance_bins = (edges[:-1] + edges[1:]) / 2.0

    coh_sum = np.zeros((int(n_distance_bins), freqs.shape[0]), dtype=float)
    coh_count = np.zeros((int(n_distance_bins),), dtype=int)

    bin_idx = np.digitize(distances, edges, right=False) - 1
    bin_idx = np.clip(bin_idx, 0, int(n_distance_bins) - 1)

    for pair_k, (i, j) in enumerate(zip(iu, ju)):
        b = int(bin_idx[pair_k])
        coh_ij = _coherence(mtfft[i], mtfft[j])
        coh_sum[b] += coh_ij
        coh_count[b] += 1

    coh_profile = np.full_like(coh_sum, np.nan, dtype=float)
    nonempty = coh_count > 0
    coh_profile[nonempty] = coh_sum[nonempty] / coh_count[nonempty, None]
    coh_profile = np.clip(coh_profile, 0.0, 1.0)

    return coh_profile, distance_bins, freqs


def marginal_energy_outlier(grid, *, robust=True, threshold=3.0):
    """
    Row and column energy outlier scores for a spatial grid.

    Marginalizes power along each grid axis, then z-scores the
    resulting row/column energy profiles to flag outliers.
    Directly detects striped high-power artifacts.

    Parameters
    ----------
    grid : (..., AP, ML) — scalar per electrode (e.g. band power, variance).
        Batch dimensions are leading; spatial axes are always last two.
    robust : bool — use median/MAD instead of mean/std (default True)
    threshold : float — |z-score| threshold for outlier flag (default 3.0)

    Returns
    -------
    dict with keys:
        'row_energy'  : (..., AP) — total energy per row
        'col_energy'  : (..., ML) — total energy per column
        'row_zscore'  : (..., AP) — z-score of row energy
        'col_zscore'  : (..., ML) — z-score of column energy
        'row_outlier' : (..., AP) — boolean, |z| > threshold
        'col_outlier' : (..., ML) — boolean, |z| > threshold
    """
    g = np.asarray(grid, dtype=float)
    if g.ndim < 2:
        raise ValueError(f"grid must have shape (..., AP, ML), got {g.shape}.")

    g_sq = g ** 2
    row_energy = np.nansum(g_sq, axis=-1)   # (..., AP)
    col_energy = np.nansum(g_sq, axis=-2)   # (..., ML)

    def _zscore(x):
        # z-score along the last axis (the row/col dimension)
        if robust:
            center = np.nanmedian(x, axis=-1, keepdims=True)
            mad = np.nanmedian(np.abs(x - center), axis=-1, keepdims=True)
            scale = 1.4826 * mad + EPS
        else:
            center = np.nanmean(x, axis=-1, keepdims=True)
            scale = np.nanstd(x, axis=-1, keepdims=True) + EPS
        return (x - center) / scale

    row_z = _zscore(row_energy)
    col_z = _zscore(col_energy)

    return {
        "row_energy": row_energy,
        "col_energy": col_energy,
        "row_zscore": row_z,
        "col_zscore": col_z,
        "row_outlier": np.abs(row_z) > float(threshold),
        "col_outlier": np.abs(col_z) > float(threshold),
    }


def gradient_anisotropy(grid):
    """
    Gradient anisotropy ratio for a spatial grid.

    Measures directional imbalance of spatial gradients.
    Striped artifacts produce strong gradients along one axis
    but not the other.

    Parameters
    ----------
    grid : (..., AP, ML) — scalar per electrode.
        Batch dimensions are leading; spatial axes are always last two.

    Returns
    -------
    anisotropy : (...) float array (or scalar float for 2D input)
        log2(mean|dV/dAP| / mean|dV/dML|)
        0.0 = isotropic (balanced gradients)
        positive = AP-dominant gradient (column-striped pattern)
        negative = ML-dominant gradient (row-striped pattern)
    """
    g = np.asarray(grid, dtype=float)
    if g.ndim < 2:
        raise ValueError(f"grid must have shape (..., AP, ML), got {g.shape}.")
    if g.shape[-2] < 2 or g.shape[-1] < 2:
        result = np.full(g.shape[:-2], np.nan)
        return float(result) if g.ndim == 2 else result

    grad_ap = np.abs(np.diff(g, axis=-2))  # (..., AP-1, ML)
    grad_ml = np.abs(np.diff(g, axis=-1))  # (..., AP, ML-1)

    mean_ap = np.nanmean(grad_ap, axis=(-2, -1))  # (...)
    mean_ml = np.nanmean(grad_ml, axis=(-2, -1))  # (...)

    result = np.log2((mean_ap + EPS) / (mean_ml + EPS))
    return float(result) if g.ndim == 2 else result


def spatial_kurtosis(grid):
    """
    Excess kurtosis of the spatial amplitude distribution.

    Flattens the (AP, ML) grid and computes excess kurtosis. High
    kurtosis indicates energy concentrated in a few electrodes (hot
    spots); low kurtosis indicates spatially diffuse energy.

    Parameters
    ----------
    grid : (..., AP, ML) — scalar per electrode.
        Batch dimensions are leading; spatial axes are always last two.

    Returns
    -------
    kurt : (...) float array (or scalar float for 2D input)
        Excess kurtosis (Fisher definition: normal = 0).
    """
    from scipy.stats import kurtosis as _scipy_kurtosis

    g = np.asarray(grid, dtype=float)
    if g.ndim < 2:
        raise ValueError(f"grid must have shape (..., AP, ML), got {g.shape}.")
    scalar_output = g.ndim == 2
    batch_shape = g.shape[:-2]
    flat = g.reshape(batch_shape + (-1,))  # (..., AP*ML)
    result = _scipy_kurtosis(flat, axis=-1, nan_policy="omit", fisher=True)
    result = np.asarray(result, dtype=float)
    return float(result) if scalar_output else result


def spatial_noise_concentration(grid, *, k=3):
    """
    Fraction of total grid energy in the top-k electrodes.

    Values near 1.0 indicate energy concentrated in a few channels
    (likely artifact or bad channels). Values near k/(AP*ML) indicate
    spatially uniform energy.

    Parameters
    ----------
    grid : (..., AP, ML) — scalar per electrode (e.g. power, variance).
        Batch dimensions are leading; spatial axes are always last two.
    k : int
        Number of top electrodes to sum (default 3).

    Returns
    -------
    concentration : (...) float array (or scalar float for 2D input)
        Fraction of total energy in top-k, in [0, 1].
    """
    g = np.asarray(grid, dtype=float)
    if g.ndim < 2:
        raise ValueError(f"grid must have shape (..., AP, ML), got {g.shape}.")
    scalar_output = g.ndim == 2
    batch_shape = g.shape[:-2]
    n = g.shape[-2] * g.shape[-1]
    k = min(int(k), n)

    energy = g ** 2
    flat = energy.reshape(batch_shape + (n,))

    # Partition to find top-k without full sort
    # np.partition: k-th smallest, so we want (n-k)-th for top-k
    if k >= n:
        top_k_sum = np.nansum(flat, axis=-1)
    else:
        partitioned = np.partition(flat, n - k, axis=-1)
        top_k_sum = np.nansum(partitioned[..., n - k :], axis=-1)

    total = np.nansum(flat, axis=-1)
    result = top_k_sum / (total + EPS)
    return float(result) if scalar_output else result


# ---------------------------------------------------------------------------
# xarray wrapper
# ---------------------------------------------------------------------------

# Registry of scalar spatial measures: name → (func, extra_kwargs)
_SCALAR_MEASURES: dict[str, tuple] = {
    "moran_i": (moran_i, {"adjacency": "queen"}),
    "moran_ap": (moran_i, {"adjacency": "ap_only"}),
    "moran_ml": (moran_i, {"adjacency": "ml_only"}),
    "gradient_anisotropy": (gradient_anisotropy, {}),
    "spatial_kurtosis": (spatial_kurtosis, {}),
    "spatial_noise_concentration": (spatial_noise_concentration, {}),
}


def spatial_summary_xr(
    da,
    *,
    measures: tuple[str, ...] | list[str] = (
        "moran_i",
        "gradient_anisotropy",
        "spatial_kurtosis",
    ),
    ap_dim: str = "AP",
    ml_dim: str = "ML",
    moran_adjacency: str = "queen",
):
    """
    Compute scalar spatial summaries preserving non-spatial coords.

    Applies each named measure to the ``(AP, ML)`` spatial dims of *da*,
    returning an ``xr.Dataset`` with one variable per measure.  All
    non-spatial dims and coords are preserved.

    Parameters
    ----------
    da : xr.DataArray
        Input with named *ap_dim* and *ml_dim* dims (any position).
    measures : sequence of str
        Measure names.  Supported: ``"moran_i"``, ``"moran_ap"``,
        ``"moran_ml"``, ``"gradient_anisotropy"``, ``"spatial_kurtosis"``,
        ``"spatial_noise_concentration"``.
    ap_dim, ml_dim : str
        Names of the AP and ML dimensions (default ``"AP"``, ``"ML"``).
    moran_adjacency : str
        Default adjacency for ``"moran_i"`` (overridden for ``"moran_ap"``
        and ``"moran_ml"``).

    Returns
    -------
    xr.Dataset
        One variable per measure.  Spatial dims removed; all other dims
        and coords preserved.
    """
    import xarray as xr

    if not isinstance(da, xr.DataArray):
        raise TypeError("Expected xr.DataArray")
    if ap_dim not in da.dims:
        raise ValueError(f"ap_dim={ap_dim!r} not in da.dims={tuple(da.dims)}")
    if ml_dim not in da.dims:
        raise ValueError(f"ml_dim={ml_dim!r} not in da.dims={tuple(da.dims)}")

    # Ensure spatial axes are last two: (..., AP, ML)
    other_dims = [d for d in da.dims if d not in (ap_dim, ml_dim)]
    ordered = da.transpose(*other_dims, ap_dim, ml_dim)
    arr = np.asarray(ordered.values, dtype=float)

    # Non-spatial coords for the output
    batch_dims = tuple(other_dims)
    batch_coords = {
        name: coord
        for name, coord in da.coords.items()
        if ap_dim not in getattr(coord, "dims", ())
        and ml_dim not in getattr(coord, "dims", ())
    }

    variables: dict[str, xr.DataArray] = {}
    for name in measures:
        if name not in _SCALAR_MEASURES:
            raise ValueError(
                f"Unknown measure {name!r}. Supported: {sorted(_SCALAR_MEASURES)}"
            )
        func, defaults = _SCALAR_MEASURES[name]
        kwargs = dict(defaults)
        # Allow overriding moran adjacency for the generic "moran_i" entry
        if name == "moran_i":
            kwargs["adjacency"] = moran_adjacency

        result = func(arr, **kwargs)
        result = np.asarray(result, dtype=float)

        variables[name] = xr.DataArray(
            result,
            dims=batch_dims,
            coords=batch_coords,
            name=name,
        )

    return xr.Dataset(variables)
