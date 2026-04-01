"""Null-model surrogate generation for travelling-wave statistics.

Provides phase-randomisation and spatial-shuffle surrogates for
testing whether observed wave properties exceed chance levels.

Ported from the surrogate logic in Muller et al. and the
EvolutionaryNeuralCodingLab travelling-waves repository [1]_.

References
----------
.. [1] Bhattacharya et al., "Traveling waves in the prefrontal cortex
   during working memory", PLOS Comp Biol, 2022.
   DOI: 10.1371/journal.pcbi.1009827
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import xarray as xr

__all__ = ["phase_randomize", "spatial_shuffle", "surrogate_test"]


def phase_randomize(
    data: xr.DataArray,
    axis: str = "time",
    rng: np.random.Generator | int | None = None,
) -> xr.DataArray:
    """Generate a phase-randomised surrogate preserving the power spectrum.

    For each spatial location the Fourier phases are replaced with
    uniform-random values while amplitudes are kept, destroying
    inter-channel phase coherence.

    Parameters
    ----------
    data : DataArray
        Signal with a *time* dimension.
    axis : str
        Dimension to randomise along.
    rng : Generator or int or None
        Random state.

    Returns
    -------
    DataArray
        Surrogate with identical power spectrum per channel.
    """
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)

    ax = data.dims.index(axis)
    vals = data.values
    n = vals.shape[ax]
    F = np.fft.rfft(vals, axis=ax)
    amp = np.abs(F)
    rand_phase = rng.uniform(0, 2 * np.pi, size=amp.shape)
    F_rand = amp * np.exp(1j * rand_phase)

    # DC and Nyquist bins must stay real for irfft to preserve their
    # power (applying a random phase would scale them by cos(θ)).
    slices_dc = [slice(None)] * F_rand.ndim
    slices_dc[ax] = 0
    F_rand[tuple(slices_dc)] = F[tuple(slices_dc)].real
    if n % 2 == 0:
        slices_nyq = [slice(None)] * F_rand.ndim
        slices_nyq[ax] = -1
        F_rand[tuple(slices_nyq)] = F[tuple(slices_nyq)].real

    surr = np.fft.irfft(F_rand, n=n, axis=ax)
    return xr.DataArray(surr, dims=data.dims, coords=data.coords)


def spatial_shuffle(
    data: xr.DataArray,
    rng: np.random.Generator | int | None = None,
) -> xr.DataArray:
    """Permute spatial positions at each time step.

    For grid data ``(time, AP, ML)`` the spatial map is flattened,
    shuffled, and reshaped.

    Parameters
    ----------
    data : DataArray
        Signal with spatial dims.
    rng : Generator or int or None
        Random state.

    Returns
    -------
    DataArray
        Surrogate with spatial coherence destroyed.
    """
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)

    vals = data.values.copy()
    if vals.ndim == 3:
        # (time, AP, ML) -> shuffle spatial at each t.
        n_t, n_ap, n_ml = vals.shape
        flat = vals.reshape(n_t, -1)
        for t in range(n_t):
            rng.shuffle(flat[t])
        vals = flat.reshape(n_t, n_ap, n_ml)
    elif vals.ndim == 2:
        # (time, ch) -> shuffle ch at each t.
        for t in range(vals.shape[0]):
            rng.shuffle(vals[t])
    else:
        raise ValueError("Expected 2-D or 3-D data")
    return xr.DataArray(vals, dims=data.dims, coords=data.coords)


def surrogate_test(
    data: xr.DataArray,
    estimator_fn: Callable[[xr.DataArray], float],
    n_surrogates: int = 200,
    seed: int | None = None,
    surrogate_type: str = "phase",
) -> tuple[float, float, np.ndarray]:
    """Non-parametric significance test against surrogate distribution.

    Parameters
    ----------
    data : DataArray
        Observed signal.
    estimator_fn : callable
        ``estimator_fn(data) -> float`` returning the statistic of interest.
    n_surrogates : int
        Number of surrogates.
    seed : int or None
        Random seed.
    surrogate_type : {"phase", "spatial"}
        Surrogate generation method.

    Returns
    -------
    p_value : float
        Fraction of surrogates ≥ the observed value.
    observed : float
        The statistic computed on the real data.
    null_dist : ndarray
        Surrogate statistic values.
    """
    rng = np.random.default_rng(seed)
    observed = estimator_fn(data)

    gen_fn = phase_randomize if surrogate_type == "phase" else spatial_shuffle

    null_dist = np.empty(n_surrogates)
    for i in range(n_surrogates):
        surr = gen_fn(data, rng=rng)
        null_dist[i] = estimator_fn(surr)

    p_value = float(np.mean(null_dist >= observed))
    return p_value, float(observed), null_dist
