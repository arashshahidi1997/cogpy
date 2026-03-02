"""
Spatial signal measures for 2D electrode grids.

Status
------
STATUS: ACTIVE
Reason: Spatial characterization measures for iEEG grid recordings.
Superseded by: n/a
Safe to remove: no

Grid convention:
    grid : (AP, ML, ...)  — AP rows, ML columns
    All functions expect the grid as the first two axes unless noted.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12

__all__ = [
    "moran_i",
    "csd_power",
    "spatial_coherence_profile",
]


def moran_i(grid: np.ndarray, *, adjacency: str = "queen") -> float:
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
    grid : (AP, ML) — scalar value per electrode.
        NaN entries excluded from computation.
    adjacency : str — "queen" (8-connected) | "rook" (4-connected)

    Returns
    -------
    I : float

    Implementation
    --------------
    1. Build binary adjacency matrix W (N x N) where N = AP*ML
       queen: includes diagonal neighbors
       rook:  only cardinal neighbors
    2. Flatten grid, mask NaN electrodes (zero their W rows/cols)
    3. Apply Moran's I formula
    W = sum of all weights
    Use pure numpy — no scipy.sparse needed for grids up to 256 electrodes
    """
    g = np.asarray(grid, dtype=float)
    if g.ndim != 2:
        raise ValueError(f"grid must have shape (AP, ML), got {g.shape}.")
    ap, ml = g.shape
    if adjacency not in {"queen", "rook"}:
        raise ValueError('adjacency must be "queen" or "rook".')

    n = int(ap * ml)
    W = np.zeros((n, n), dtype=float)

    if adjacency == "rook":
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:  # queen
        offsets = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

    for i in range(ap):
        for j in range(ml):
            idx = i * ml + j
            for di, dj in offsets:
                ii = i + di
                jj = j + dj
                if 0 <= ii < ap and 0 <= jj < ml:
                    jdx = ii * ml + jj
                    W[idx, jdx] = 1.0

    x = g.reshape(-1)
    valid = np.isfinite(x)
    if not np.any(valid):
        return float("nan")

    x = x[valid]
    W = W[valid][:, valid]

    Wsum = float(np.sum(W))
    if Wsum <= 0:
        return float("nan")

    xmean = float(np.mean(x))
    xc = x - xmean
    denom = float(np.sum(xc**2))
    if denom <= EPS:
        return float("nan")

    num = float(xc @ (W @ xc))
    n_valid = int(x.shape[0])
    return float((n_valid / Wsum) * (num / denom))


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
