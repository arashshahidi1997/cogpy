"""
Ordinary least-squares regression primitives.

Thin wrappers around ``numpy.linalg.lstsq`` for fitting, predicting,
and computing residuals.  Designed to work with design matrices from
:mod:`cogpy.regression.design`.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "ols_fit",
    "ols_predict",
    "ols_residual",
]


def ols_fit(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    rcond: float | None = None,
) -> np.ndarray:
    """
    Fit ordinary least-squares: find beta minimizing ``||Y - X @ beta||^2``.

    Parameters
    ----------
    X : (n_time, n_predictors) float
        Design matrix.
    Y : (n_time,) or (n_time, n_channels) float
        Response signal(s).  Multiple channels are fit independently.
    rcond : float, optional
        Cutoff for small singular values (passed to ``np.linalg.lstsq``).

    Returns
    -------
    beta : (n_predictors,) or (n_predictors, n_channels) float
        Regression coefficients.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=rcond)
    return beta


def ols_predict(
    X: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray:
    """
    Compute predicted signal: ``Y_hat = X @ beta``.

    Parameters
    ----------
    X : (n_time, n_predictors) float
        Design matrix.
    beta : (n_predictors,) or (n_predictors, n_channels) float
        Regression coefficients from :func:`ols_fit`.

    Returns
    -------
    Y_hat : (n_time,) or (n_time, n_channels) float
        Predicted signal.
    """
    return np.asarray(X, dtype=float) @ np.asarray(beta, dtype=float)


def ols_residual(
    X: np.ndarray,
    Y: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray:
    """
    Compute residual signal: ``Y - X @ beta``.

    Parameters
    ----------
    X : (n_time, n_predictors) float
        Design matrix.
    Y : (n_time,) or (n_time, n_channels) float
        Original signal.
    beta : (n_predictors,) or (n_predictors, n_channels) float
        Regression coefficients.

    Returns
    -------
    residual : same shape as Y
        Signal with predicted component removed.
    """
    return np.asarray(Y, dtype=float) - ols_predict(X, beta)
