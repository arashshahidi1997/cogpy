"""
Design matrix construction primitives.

Build predictor matrices from reference signals or event trains,
suitable for least-squares regression.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "lagged_design_matrix",
    "event_design_matrix",
]


def lagged_design_matrix(
    reference: np.ndarray,
    lags: np.ndarray | range,
    *,
    intercept: bool = True,
) -> np.ndarray:
    """
    Build a Toeplitz-like design matrix from a reference signal and lag set.

    Each column is a time-shifted copy of *reference*.  Useful for
    FIR-model regression where the response depends on recent values
    of a predictor.

    Parameters
    ----------
    reference : (n_time,) float
        1-D reference signal.
    lags : array-like of int
        Sample lags to include.  Lag ``k`` means the predictor at time
        ``t`` is ``reference[t - k]``.  Negative lags look forward.
    intercept : bool
        If True (default), prepend a column of ones.

    Returns
    -------
    X : (n_time, n_predictors) float
        Design matrix.  ``n_predictors = len(lags) + int(intercept)``.
        Out-of-bounds samples are zero-padded.

    Examples
    --------
    Build a design matrix with lags 0..5 for a TTL reference signal,
    then regress each channel against it to remove the artifact::

        X = lagged_design_matrix(ttl_ref, range(6))
        beta = ols_fit(X, channel_data)
        cleaned = ols_residual(X, channel_data, beta)
    """
    ref = np.asarray(reference, dtype=float).ravel()
    lags = np.asarray(lags, dtype=int).ravel()
    n = ref.size

    n_cols = len(lags) + int(intercept)
    X = np.zeros((n, n_cols), dtype=float)
    col = 0

    if intercept:
        X[:, 0] = 1.0
        col = 1

    for k in lags:
        if k >= 0:
            X[k:, col] = ref[: n - k]
        else:
            X[: n + k, col] = ref[-k:]
        col += 1

    return X


def event_design_matrix(
    n_time: int,
    event_samples: np.ndarray,
    template: np.ndarray,
    *,
    intercept: bool = True,
) -> np.ndarray:
    """
    Build a design matrix from an event train and a template waveform.

    Each column corresponds to one event and contains the template
    placed at the event onset.  Useful for template-regression removal.

    Parameters
    ----------
    n_time : int
        Total number of time samples.
    event_samples : (n_events,) int
        Sample indices of event onsets.
    template : (n_lag,) float
        Template waveform (1-D).
    intercept : bool
        If True (default), prepend a column of ones.

    Returns
    -------
    X : (n_time, n_events + int(intercept)) float
        Design matrix.  Sparse in practice but returned dense.
        Columns beyond the intercept contain the placed template.
    """
    event_samples = np.asarray(event_samples, dtype=int).ravel()
    template = np.asarray(template, dtype=float).ravel()
    n_lag = template.size
    n_events = event_samples.size

    n_cols = n_events + int(intercept)
    X = np.zeros((n_time, n_cols), dtype=float)
    col = 0

    if intercept:
        X[:, 0] = 1.0
        col = 1

    for i, s0 in enumerate(event_samples):
        s1 = s0 + n_lag
        if s0 < 0:
            # Partial overlap at start
            t_start = -s0
            X[0 : min(s1, n_time), col + i] = template[t_start : t_start + min(s1, n_time)]
        elif s1 > n_time:
            # Partial overlap at end
            X[s0:n_time, col + i] = template[: n_time - s0]
        else:
            X[s0:s1, col + i] = template
    return X
