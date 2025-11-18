import numpy as np
import xarray as xr
import pandas as pd
from scipy.stats import zscore
from sklearn.base import BaseEstimator, TransformerMixin

from .spatspec import SpatSpecDecomposition, get_norm


class erpPCA(BaseEstimator, TransformerMixin):
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

    def transform(self, X, zscore_=False, normalize_=False):
        """
        X: cases, vars
        """
        if isinstance(X, xr.DataArray):
            FSr = self.transform(X.data, zscore_, normalize_)
            scx = erpPCA_scx(FSr, X.time)

            return scx
        else:
            inv_proj = self.unmixing_
            if normalize_:
                lmbda = self.cov_diag()
                inv_proj = self.unmixing_ / lmbda

            if zscore_:
                X = zscore(X, axis=0, ddof=1)

            FSr = X.dot(inv_proj)
            return FSr

    def cov_diag(self):
        return np.sqrt(np.diag(self.cov_))

    def LR_norm(self):
        return np.linalg.norm(self.LR, axis=0)


def pseudo_inverse_scaled(LR, cov):
    FSCFr = LR.dot(np.linalg.inv(LR.T.dot(LR)))
    if cov.ndim == 1:
        FSCFr = FSCFr * np.sqrt(cov)[np.newaxis]
    else:
        FSCFr = FSCFr * np.sqrt(np.diag(cov))[:, np.newaxis]
    return FSCFr


def pseudo_inverse(LR, cov):
    return LR.dot(np.linalg.inv(LR.T.dot(LR)))


def simplicity_criterion(Y):
    """
    how simple the columns of Y are
    """
    A = Y**2  # (m, n)
    B = A.T.dot(A)  # (n, n) dot product of columns of Y**2
    G1 = np.sum(B) - np.sum(
        np.diag(B)
    )  # (1) , sum of off-diagonal elements: how orthogonal the columns are - for fully orthogonal Y**2 columns the off diagonal sum would be zero

    C = np.sum(B, axis=0)  # sum across rows (n,),
    G2 = 1 / Y.shape[0] * (np.sum(C) ** 2 - np.sum(C**2))

    G = G1 - G2
    return G


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
    T = L @ M  # no transpose
    D = np.sum(D)
    Y = Yo @ T
    return Y, D


def varimax_rotation(X, maxit=100, tol=1e-4, norm=True, IfVerbose=True):
    if IfVerbose:
        print("-------- Varimax Rotation (4M) ------------")

    p, m = X.shape
    if IfVerbose:
        print(f"Matrix rows:          {p}")
        print(f"Matrix columns:       {m}")

    if norm:
        if IfVerbose:
            print("Kaiser" "s normalization:   YES")
        Y, H = kaiser_normalization(X)
    else:
        if IfVerbose:
            print("Kaiser" "s normalization:   NO")
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
        # Add your chosen gVarimaxMethod implementation here
        Y, D = varimax_iter(Yo, Y)
        g = simplicity_criterion(Y)
        if IfVerbose:
            print(f"{it:6d}   {g:18.8f}  {Gold - g:14.8f}  {abs(D - Dold) / D:14.8f}")

        # if change in simplicity is smaller than tolerance break
        if Gold - g < tol:
            # if previous step was simpler, keep that, if not report the new results
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


def redirect_loadings(L):
    s_ = np.ones(L.shape[1])
    s_[np.abs(np.max(L, axis=0)) < np.abs(np.min(L, axis=0))] = -1
    L = L * s_[np.newaxis]
    return L


def sort_by_eigv(L, ev):
    ux = np.argsort(ev)[::-1]
    ev = ev[ux]
    return L[:, ux], ev


def erppca(X, nFac=None, maxIt=100, Tol=1e-3, IfVerbose=1, return_attrs=False):
    cases, nvars = X.shape
    print("computing covariance matrix ...")
    D = np.cov(X, rowvar=False)
    print("eigendecomposition ...")
    EV, EM = np.linalg.eigh(D)
    EV = EV[::-1]
    EM = EM[:, ::-1]
    LU = EM * np.sqrt(EV)
    # LU, u_ = sort_by_eigv(UL, EV)
    rk = np.linalg.matrix_rank(D, tol=1e-4)
    LU = LU[:, :rk]
    u_ = EV[:rk]
    # ux = ux[:rk]
    LU = redirect_loadings(LU)
    print("Varimax ...")
    RL, G = varimax_rotation(LU, maxIt, Tol, True, IfVerbose)
    EVr = np.sum(RL * RL, axis=0)
    LR, r_ = sort_by_eigv(RL, EVr)
    LR = redirect_loadings(LR)
    tv = np.sum(EV)
    VT = [u_, 100 * u_ / tv, r_, 100 * r_ / tv]
    # FSCFr = LR.dot(np.linalg.inv(LR.T.dot(LR)))
    # FSCFr = FSCFr * np.sqrt(np.diag(D))[:, np.newaxis]
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


def similarity_metric(
    ss1: SpatSpecDecomposition, ss2: SpatSpecDecomposition, fac1, fac2, freq_threshold=5
):
    """Compute the distance metric between a factor of this instance and a factor of another instance."""
    freq_diff = abs(ss1.ldx_df.freqmax.iloc[fac1] - ss2.ldx_df.freqmax.iloc[fac2])

    # Check frequency threshold
    if freq_threshold is not None and freq_diff > freq_threshold:
        return 0

    correlation_val = np.corrcoef(
        ss1.ldx_slc_maxfreq[fac1].data.flatten(),
        ss2.ldx_slc_maxfreq[fac2].data.flatten(),
    )[0, 1]
    similarity = correlation_val
    return similarity


def compute_similarity_matrix(
    ss1: SpatSpecDecomposition, ss2: SpatSpecDecomposition, freq_threshold=5
):
    """Compute a distance matrix between the factors of this instance and another instance."""
    nfac = ss1.ldx.shape[0]
    similarity_matrix = np.zeros((nfac, nfac))

    for fac1 in range(nfac):
        for fac2 in range(nfac):
            similarity_matrix[fac1, fac2] = similarity_metric(
                ss1, ss2, fac1, fac2, freq_threshold
            )

    return similarity_matrix


def get_similarity(ss_series: pd.Series, freq_threshold: float):
    """
    Parameters:
    - ss_series: Series of SpatSpec objects
    - freq_threshold: Threshold for frequency

    Returns:
    - simil_arr: (nrec, nrec)x(nfac, nfac) Array of similarity matrices
    """
    nrec = len(ss_series)
    simil_arr = np.zeros((nrec, nrec), dtype=object)
    for i in np.ndindex((nrec, nrec)):
        ss1, ss2 = ss_series.iloc[i[0]], ss_series.iloc[i[1]]
        simil_arr[i] = compute_similarity_matrix(
            ss1, ss2, freq_threshold=freq_threshold
        )
    return simil_arr


def erpPCA_ldx(LR, ld_coords, ld_shape):
    # print(LR.shape)
    nFac = LR.shape[-1]
    coords_ = ld_coords
    factors_ldx = xr.DataArray(
        (LR.T).reshape(-1, *ld_shape),
        coords={"factor": np.arange(nFac), **coords_},
        dims=["factor", "h", "w", "freq"],
    )
    return factors_ldx


def spatspec2erpPCA(ss: SpatSpecDecomposition):
    nFac = ss.ldx.shape[0]
    LR = ss.ldx.data.reshape(nFac, -1).T
    erp = erpPCA(nFac=nFac)
    erp.LR = LR
    erp.unmixing_ = pseudo_inverse_scaled(LR, ss.ldx_df.norm.values**2)
    return erp


def erpPCA_scx(FSr, times):
    nFac = FSr.shape[-1]
    factors_scx = xr.DataArray(
        FSr, coords={"time": times, "factor": np.arange(nFac)}, dims=["time", "factor"]
    )
    # add fs attribute
    factors_scx.attrs["fs"] = 1 / np.min(np.diff(times))
    return factors_scx


def erpPCA2factors(erp, mtx_sws):
    ld_coords = dict(mtx_sws.coords)
    times = ld_coords.pop("time")
    ld_shape = mtx_sws.isel(time=0).shape
    factors_ldx = erpPCA_ldx(erp.LR, ld_coords, ld_shape)
    factors_scx = erpPCA_scx(erp.FSr, times)
    return factors_ldx, factors_scx


def project_to_loadings(X, inv_proj):
    """
    X: cases, vars
    """
    if isinstance(X, xr.DataArray):
        FSr = project_to_loadings(X.data, inv_proj)
        scx = erpPCA_scx(FSr, X.time)
        return scx
    else:
        FSr = X.dot(inv_proj)
        return FSr


def get_invproj(ldx):
    """
    ldx: factors, h, w, freq
    """
    nFac = ldx.shape[0]
    LR = ldx.data.reshape(nFac, -1).T
    norm = get_norm(ldx)
    unmixing_ = pseudo_inverse_scaled(LR, norm**2)
    return unmixing_
