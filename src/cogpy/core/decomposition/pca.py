"""
ERP-PCA decomposition of neural signals.

Provides varimax-rotated PCA for spatio-spectral decomposition of ECoG data.
The ``erpPCA`` class follows the scikit-learn estimator API (fit / transform).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import zscore
from sklearn.base import BaseEstimator, TransformerMixin


# ---------------------------------------------------------------------------
# erpPCA estimator
# ---------------------------------------------------------------------------


class erpPCA(BaseEstimator, TransformerMixin):
    """Varimax-rotated PCA estimator for ECoG spatio-spectral data.

    Parameters
    ----------
    nfac : int or None
        Number of factors to retain.  ``None`` keeps all.
    max_it : int
        Maximum varimax iterations.
    tol : float
        Convergence tolerance for varimax.
    verbose : bool
        Print progress.
    """

    def __init__(self, nfac=None, max_it=100, tol=1e-3, verbose=True):
        self.nfac = nfac
        self.max_it = max_it
        self.tol = tol
        self.verbose = verbose

    def fit(self, X, y=None):
        erppca_dict = erppca(
            X,
            nFac=self.nfac,
            maxIt=self.max_it,
            Tol=self.tol,
            IfVerbose=self.verbose,
            return_attrs=True,
        )
        self.__dict__ |= erppca_dict
        return self

    def transform(self, X, zscore_=False, normalize_=False):
        """Project *X* onto the rotated loadings.

        Parameters
        ----------
        X : np.ndarray or xr.DataArray
            Shape ``(cases, vars)``.
        """
        if isinstance(X, xr.DataArray):
            FSr = self.transform(X.data, zscore_, normalize_)
            return erpPCA_scx(FSr, X.time)
        else:
            inv_proj = self.unmixing_
            if normalize_:
                lmbda = self.cov_diag()
                inv_proj = self.unmixing_ / lmbda
            if zscore_:
                X = zscore(X, axis=0, ddof=1)
            return X.dot(inv_proj)

    def cov_diag(self):
        return np.sqrt(np.diag(self.cov_))

    def LR_norm(self):
        return np.linalg.norm(self.LR, axis=0)


# ---------------------------------------------------------------------------
# Linear algebra helpers
# ---------------------------------------------------------------------------


def pseudo_inverse_scaled(LR, cov):
    FSCFr = LR.dot(np.linalg.inv(LR.T.dot(LR)))
    if cov.ndim == 1:
        FSCFr = FSCFr * np.sqrt(cov)[np.newaxis]
    else:
        FSCFr = FSCFr * np.sqrt(np.diag(cov))[:, np.newaxis]
    return FSCFr


def pseudo_inverse(LR, cov):
    return LR.dot(np.linalg.inv(LR.T.dot(LR)))


# ---------------------------------------------------------------------------
# Varimax rotation
# ---------------------------------------------------------------------------


def simplicity_criterion(Y):
    """Simplicity criterion for varimax columns."""
    A = Y**2
    B = A.T.dot(A)
    G1 = np.sum(B) - np.sum(np.diag(B))
    C = np.sum(B, axis=0)
    G2 = 1 / Y.shape[0] * (np.sum(C) ** 2 - np.sum(C**2))
    return G1 - G2


def kaiser_normalization(X):
    h = np.sqrt(np.sum(X**2, axis=1))
    H = np.repeat(h[:, np.newaxis], X.shape[1], axis=1)
    Y = X / H
    Y[np.isnan(Y)] = X[np.isnan(Y)]
    return Y, H


def varimax_iter(Yo, Y):
    L, D, M = np.linalg.svd(
        Yo.T @ (Y.shape[0] * Y**3 - Y @ np.diag(np.sum(Y**2, axis=0)))
    )
    T = L @ M
    D = np.sum(D)
    Y = Yo @ T
    return Y, D


def varimax_rotation(X, maxit=100, tol=1e-4, norm=True, IfVerbose=True):
    """Varimax rotation (4M algorithm).

    Parameters
    ----------
    X : np.ndarray
        Loading matrix ``(p, m)``.
    maxit : int
        Maximum iterations.
    tol : float
        Convergence tolerance on simplicity criterion.
    norm : bool
        Apply Kaiser normalization.
    IfVerbose : bool
        Print iteration log.

    Returns
    -------
    Y : np.ndarray
        Rotated loading matrix.
    G : np.ndarray
        Convergence history ``(iteration, simplicity, delta, D_ratio)``.
    """
    if IfVerbose:
        print("-------- Varimax Rotation (4M) ------------")

    p, m = X.shape
    if IfVerbose:
        print(f"Matrix rows:          {p}")
        print(f"Matrix columns:       {m}")

    if norm:
        if IfVerbose:
            print("Kaiser's normalization:   YES")
        Y, H = kaiser_normalization(X)
    else:
        if IfVerbose:
            print("Kaiser's normalization:   NO")
        Y = X

    g = simplicity_criterion(Y)
    G = [[0, g, tol, 0]]
    Gold = g
    YY = Y.copy()

    if IfVerbose:
        print("     #         Simplicity G     Convergence")
        print(f"{0:6d}   {g:18.8f}  {tol:14.8f}")

    D = 0
    Dold = 0
    Yo = Y.copy()

    for it in range(1, maxit + 1):
        Dold = D
        Y, D = varimax_iter(Yo, Y)
        g = simplicity_criterion(Y)
        if IfVerbose:
            print(f"{it:6d}   {g:18.8f}  {Gold - g:14.8f}  {abs(D - Dold) / D:14.8f}")

        if Gold - g < tol:
            if Gold < g:
                Y = YY
            else:
                G.append([it, g, Gold - g, abs(D - Dold) / D])
            break

        YY = Y.copy()
        G.append([it, g, Gold - g, abs(D - Dold) / D])
        Gold = g

    if norm:
        Y = Y * H

    if IfVerbose:
        print("-------------------------------------------")

    return Y, np.array(G)


# ---------------------------------------------------------------------------
# Core erppca function
# ---------------------------------------------------------------------------


def redirect_loadings(L):
    s_ = np.ones(L.shape[1])
    s_[np.abs(np.max(L, axis=0)) < np.abs(np.min(L, axis=0))] = -1
    return L * s_[np.newaxis]


def sort_by_eigv(L, ev):
    ux = np.argsort(ev)[::-1]
    return L[:, ux], ev[ux]


def erppca(X, nFac=None, maxIt=100, Tol=1e-3, IfVerbose=1, return_attrs=False):
    """Run ERP-PCA: covariance → eigen → varimax → sort.

    Parameters
    ----------
    X : np.ndarray
        Data matrix ``(cases, vars)``.
    nFac : int or None
        Number of factors to retain.

    Returns
    -------
    dict
        Keys: ``LU``, ``LR``, ``FSr``, ``VT`` (and optionally fit attrs).
    """
    cases, nvars = X.shape
    print("computing covariance matrix ...")
    D = np.cov(X, rowvar=False)
    print("eigendecomposition ...")
    EV, EM = np.linalg.eigh(D)
    EV = EV[::-1]
    EM = EM[:, ::-1]
    LU = EM * np.sqrt(EV)
    rk = np.linalg.matrix_rank(D, tol=1e-4)
    LU = LU[:, :rk]
    u_ = EV[:rk]
    LU = redirect_loadings(LU)
    print("Varimax ...")
    RL, G = varimax_rotation(LU, maxIt, Tol, True, IfVerbose)
    EVr = np.sum(RL * RL, axis=0)
    LR, r_ = sort_by_eigv(RL, EVr)
    LR = redirect_loadings(LR)
    tv = np.sum(EV)
    VT = [u_, 100 * u_ / tv, r_, 100 * r_ / tv]
    FSCFr = pseudo_inverse_scaled(LR, D)
    if nFac is None:
        nFac = FSCFr.shape[1]
    FSr = zscore(X, ddof=1).dot(FSCFr[:, :nFac])
    LR = LR[:, :nFac]
    LU = LU[:, :nFac]
    out = dict(LU=LU, LR=LR, FSr=FSr, VT=VT)
    if return_attrs:
        out |= dict(nfac=nFac, convergence_=G, unmixing_=FSCFr[:, :nFac], cov_=D)
    return out


# ---------------------------------------------------------------------------
# Similarity / cross-recording comparison
# ---------------------------------------------------------------------------


def similarity_metric(ss1, ss2, fac1, fac2, freq_threshold=5):
    """Correlation-based similarity between two SpatSpec factors."""
    freq_diff = abs(ss1.ldx_df.freqmax.iloc[fac1] - ss2.ldx_df.freqmax.iloc[fac2])
    if freq_threshold is not None and freq_diff > freq_threshold:
        return 0
    correlation_val = np.corrcoef(
        ss1.ldx_slc_maxfreq[fac1].data.flatten(),
        ss2.ldx_slc_maxfreq[fac2].data.flatten(),
    )[0, 1]
    return correlation_val


def compute_similarity_matrix(ss1, ss2, freq_threshold=5):
    """Pairwise similarity matrix between factors of two SpatSpec instances."""
    nfac = ss1.ldx.shape[0]
    similarity_matrix = np.zeros((nfac, nfac))
    for fac1 in range(nfac):
        for fac2 in range(nfac):
            similarity_matrix[fac1, fac2] = similarity_metric(
                ss1, ss2, fac1, fac2, freq_threshold
            )
    return similarity_matrix


def get_similarity(ss_series: pd.Series, freq_threshold: float):
    """Build an (nrec, nrec) array of (nfac, nfac) similarity matrices."""
    nrec = len(ss_series)
    simil_arr = np.zeros((nrec, nrec), dtype=object)
    for i in np.ndindex((nrec, nrec)):
        ss1, ss2 = ss_series.iloc[i[0]], ss_series.iloc[i[1]]
        simil_arr[i] = compute_similarity_matrix(ss1, ss2, freq_threshold=freq_threshold)
    return simil_arr


# ---------------------------------------------------------------------------
# xarray helpers for factor scores / loadings
# ---------------------------------------------------------------------------


def erpPCA_ldx(LR, ld_coords, ld_shape):
    """Reshape rotated loadings into an xr.DataArray ``(factor, h, w, freq)``."""
    nFac = LR.shape[-1]
    return xr.DataArray(
        (LR.T).reshape(-1, *ld_shape),
        coords={"factor": np.arange(nFac), **ld_coords},
        dims=["factor", "h", "w", "freq"],
    )


def erpPCA_scx(FSr, times):
    """Wrap factor scores into an xr.DataArray ``(time, factor)``."""
    nFac = FSr.shape[-1]
    scx = xr.DataArray(
        FSr,
        coords={"time": times, "factor": np.arange(nFac)},
        dims=["time", "factor"],
    )
    scx.attrs["fs"] = 1 / np.min(np.diff(times))
    return scx


def erpPCA2factors(erp, mtx_sws):
    """Convert fitted erpPCA + spectrogram into (ldx, scx) DataArrays."""
    ld_coords = dict(mtx_sws.coords)
    times = ld_coords.pop("time")
    ld_shape = mtx_sws.isel(time=0).shape
    factors_ldx = erpPCA_ldx(erp.LR, ld_coords, ld_shape)
    factors_scx = erpPCA_scx(erp.FSr, times)
    return factors_ldx, factors_scx


def spatspec2erpPCA(ss):
    """Create an erpPCA estimator from a SpatSpecDecomposition."""
    from .spatspec import get_norm

    nFac = ss.ldx.shape[0]
    LR = ss.ldx.data.reshape(nFac, -1).T
    erp = erpPCA(nfac=nFac)
    erp.LR = LR
    erp.unmixing_ = pseudo_inverse_scaled(LR, ss.ldx_df.norm.values**2)
    return erp


def project_to_loadings(X, inv_proj):
    """Project data onto pre-computed inverse projection."""
    if isinstance(X, xr.DataArray):
        FSr = project_to_loadings(X.data, inv_proj)
        return erpPCA_scx(FSr, X.time)
    return X.dot(inv_proj)


def get_invproj(ldx):
    """Compute inverse projection from loading DataArray ``(factors, h, w, freq)``."""
    from .spatspec import get_norm

    nFac = ldx.shape[0]
    LR = ldx.data.reshape(nFac, -1).T
    norm = get_norm(ldx)
    return pseudo_inverse_scaled(LR, norm**2)
