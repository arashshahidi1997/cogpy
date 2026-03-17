"""
Spatio-spectral decomposition container.

``SpatSpecDecomposition`` manages loadings, design matrices, and loading
metadata for varimax-rotated PCA applied to spatio-spectral data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from ..utils import xarr as xut, time_series as ts


# ---------------------------------------------------------------------------
# SpatSpecDecomposition
# ---------------------------------------------------------------------------


class DesignMatrixReshaper:
    """Stack / unstack xr.DataArray dimensions for PCA design matrices."""

    def __init__(self):
        self.original_dims = None
        self.stacked_coords = None

    def stack_except(self, data_array, exclude_dim):
        if exclude_dim not in data_array.dims:
            raise ValueError(
                f"Dimension '{exclude_dim}' not found in data array dimensions: {data_array.dims}"
            )
        self.original_dims = data_array.dims
        stack_dims = [dim for dim in data_array.dims if dim != exclude_dim]
        stacked_data = data_array.stack(all_dims=stack_dims).transpose(
            exclude_dim, "all_dims"
        )
        self.stacked_coords = stacked_data.indexes.get("all_dims")
        return stacked_data

    def unstack_to_original(self, stacked_data):
        if "all_dims" not in stacked_data.dims:
            raise ValueError(
                "The stacked data does not contain 'all_dims' dimension for unstacking."
            )
        if self.original_dims is None:
            raise ValueError(
                "Original dimensions have not been set. "
                "Ensure 'stack_except' is called before 'unstack_to_original'."
            )
        return stacked_data.unstack("all_dims").transpose(*self.original_dims)


class SpatSpecDecomposition:
    """Container for spatio-spectral PCA loadings and metadata.

    Parameters
    ----------
    spatspec_coords : dict or xr.DataArray
        Coordinates ``{h, w, freq}`` or an existing spectrogram DataArray
        from which coordinates are inferred.
    ld : np.ndarray or None
        Initial loading matrix ``(h*w*freq, nFac)``.
    """

    def __init__(self, spatspec_coords, ld=None):
        expected_dims = ["h", "w", "freq"]
        if isinstance(spatspec_coords, xr.DataArray):
            ld_coords = dict(spatspec_coords.coords)
            _ = ld_coords.pop("time")
            self.spatspec_coords = ld_coords
            self.spatspec_shape = spatspec_coords.transpose(
                *expected_dims, "time"
            ).shape[:-1]
        else:
            self.spatspec_coords = {k: spatspec_coords[k] for k in expected_dims}
            self.spatspec_shape = tuple(spatspec_coords[k].size for k in expected_dims)
        if ld is not None:
            self.ldx_set(ld)

    # -- Loading management -------------------------------------------------

    def ldx_set(self, ld):
        """Set loadings from a flat matrix ``(h*w*freq, nFac)``."""
        self.nFac = ld.shape[-1]
        self.ldx = xr.DataArray(
            (ld.T).reshape(-1, *self.spatspec_shape),
            coords={"factor": np.arange(self.nFac), **self.spatspec_coords},
            dims=["factor", "h", "w", "freq"],
        )

    def ldx_set_direct(self, ldx):
        """Set loadings from a pre-shaped DataArray."""
        self.ldx = ldx
        self.nFac = ldx.shape[0]

    def scx_from_FSr(self, FSr, times):
        """Wrap raw factor scores into an xr.DataArray."""
        self.nFac = FSr.shape[-1]
        scx = xr.DataArray(
            FSr,
            coords={"time": times, "factor": np.arange(self.nFac)},
            dims=["time", "factor"],
        )
        scx.attrs["fs"] = 1 / (times[1] - times[0])
        return scx

    def reconstruct(self, scx):
        """Reconstruct spectrogram from loadings and scores."""
        return self.ldx.dot(scx).transpose("h", "w", "freq", "time")

    # -- Loading processing -------------------------------------------------

    def ldx_process(self):
        """Compute loading metadata: max-freq, spatial peak, summary df."""
        self.ldx_fch, self.ldx_maxfreq, self.ldx_slc_maxfreq, self.ldx_df = ldx_process(
            self.ldx
        )
        self._set_ldx_slc_maxch()

    def _set_ldx_slc_maxch(self):
        ldx_slc_maxch = []
        for ifac in self.ldx.factor.values:
            slc_maxch = self.ldx.sel(
                factor=ifac, h=self.ldx_df.AP.loc[ifac], w=self.ldx_df.ML.loc[ifac]
            )
            assert slc_maxch.ndim == 1
            slc_maxch = xr.DataArray(
                slc_maxch.data, dims=["freq"], coords={"freq": slc_maxch.freq}
            )
            ldx_slc_maxch.append(slc_maxch)
        self.ldx_slc_maxch = xr.concat(ldx_slc_maxch, dim="factor")

    def ldx_df_mat(self):
        """Convert loading summary to a dict (for MATLAB / serialisation)."""
        columns_rename_dict = {
            "hmax": "peak_iAP",
            "wmax": "peak_iML",
            "ifreqmax": "peak_ifreq",
            "AP": "peak_AP",
            "ML": "peak_ML",
            "freqmax": "peak_freq",
        }
        return (
            self.ldx_df.reset_index()
            .rename(columns=columns_rename_dict)
            .to_dict("list")
        )

    def mark_freq_band(self, name: str, lo: float, hi: float):
        """Mark factors whose peak frequency falls in ``[lo, hi)``."""
        facs = self.ldx_df[
            (self.ldx_df.freqmax >= lo) & (self.ldx_df.freqmax < hi)
        ].index.to_list()
        self.ldx_df.loc[:, f"is_{name}"] = self.ldx_df.index.isin(facs)

    # -- Design matrix construction -----------------------------------------

    def designmat(self, mtx, log=True):
        """Flatten spectrogram into a PCA design matrix ``(time, hwf)``."""
        mtx = mtx.transpose("h", "w", "freq", "time")
        X = mtx.stack(hwf=("h", "w", "freq")).transpose("time", "hwf")
        if log:
            X.data[np.where(X.data == 0)] = 1e-10
            X = np.log10(X).reset_index("hwf")
        X.attrs = {
            "hwf_shape": self.spatspec_shape,
            "fs": mtx.fs,
            "step": mtx.window_step / mtx.fs,
        }
        return X

    def designmat_preprocess(self, X, win=10):
        """Rolling z-score preprocessing for design matrix.

        Parameters
        ----------
        X : xr.DataArray
            Design matrix with ``step`` attribute.
        win : float
            Window size in seconds.
        """
        window_size = int(win / 2 / X.step)
        Xz = xut.xarr_wrap(ts.rolling_zscore)(X, window_size=window_size)
        Xz.data[np.isnan(Xz.data)] = np.nanmin(Xz.data)
        return Xz

    def mtx_from_designmat(self, X, mtx):
        """Reshape design matrix back to spectrogram shape."""
        mtx_ = mtx.copy()
        if isinstance(X, xr.DataArray):
            X = X.set_index(hwf=["h", "w", "freq"])
            mtx_ = X.unstack("hwf")
            mtx_ = mtx_.transpose("h", "w", "freq", "time")
        else:
            mtx_.data[:] = X.reshape(X.shape[0], *self.spatspec_shape)
        return mtx_

    def get_loadings_dict(self):
        """Serialise loadings + metadata to a dict."""
        loadings_dict = {
            "loading": self.ldx.data,
            "coo_AP": self.ldx.h.values,
            "coo_ML": self.ldx.w.values,
            "coo_Freq": self.ldx.freq.values,
        }
        loadings_dict |= self.ldx_df_mat()
        return loadings_dict


# ---------------------------------------------------------------------------
# Standalone loading processing functions
# ---------------------------------------------------------------------------


def stack_and_reset_index(ldx):
    return ldx.stack(ch=("h", "w")).reset_index("ch")


def compute_ifreqmax_and_maxfreq(ldx_fch):
    factor_ld_mean = ldx_fch.mean(dim=("ch"))
    ifreqmax = factor_ld_mean.argmax(dim="freq")
    ldx_maxfreq = ldx_fch.freq[ifreqmax]
    return ifreqmax, ldx_maxfreq


def compute_spatial_ldx(ldx, ifreqmax, ldx_maxfreq):
    spat_ldx_list = [
        ldx.sel(factor=ifactor, freq=ldx_maxfreq[ifactor], method="nearest")
        for ifactor in ldx.factor
    ]
    spat_ldx_peak = np.array(
        [
            np.unravel_index(spat_ldx.argmax(), spat_ldx.shape)
            for spat_ldx in spat_ldx_list
        ]
    )
    ldx_slc_maxfreq = xr.concat(spat_ldx_list, "factor")
    return spat_ldx_peak, ldx_slc_maxfreq


def create_ldx_df(ldx, ifreqmax, ldx_maxfreq, spat_ldx_peak):
    ldx_df = pd.DataFrame(ldx_maxfreq.factor.data, columns=["factor"])
    ldx_df.loc[:, ["hmax", "wmax"]] = spat_ldx_peak
    ldx_df.loc[:, "AP"] = ldx_df.apply(lambda x: ldx.h.values[int(x.hmax)], axis=1)
    ldx_df.loc[:, "ML"] = ldx_df.apply(lambda x: ldx.w.values[int(x.wmax)], axis=1)
    ldx_df.loc[:, "freqmax"] = ldx_maxfreq.data
    ldx_df.loc[:, "ifreqmax"] = ifreqmax
    ldx_df.loc[:, "norm"] = get_norm(ldx)
    ldx_df = ldx_df.set_index("factor")
    return ldx_df[["AP", "ML", "freqmax", "hmax", "wmax", "ifreqmax", "norm"]]


def ldx_process(ldx):
    """Process loading DataArray into summary metadata.

    Returns
    -------
    ldx_fch, ldx_maxfreq, ldx_slc_maxfreq, ldx_df
    """
    ldx_fch = stack_and_reset_index(ldx)
    ifreqmax, ldx_maxfreq = compute_ifreqmax_and_maxfreq(ldx_fch)
    spat_ldx_peak, ldx_slc_maxfreq = compute_spatial_ldx(ldx, ifreqmax, ldx_maxfreq)
    ldx_df = create_ldx_df(ldx, ifreqmax, ldx_maxfreq, spat_ldx_peak)
    return ldx_fch, ldx_maxfreq, ldx_slc_maxfreq, ldx_df


def set_ldx_slc_maxch(ldx, ldx_df):
    ldx_slc_maxch = []
    for ifac in ldx.factor.values:
        slc_maxch = ldx.sel(factor=ifac, h=ldx_df.AP.loc[ifac], w=ldx_df.ML.loc[ifac])
        assert slc_maxch.ndim == 1
        slc_maxch = xr.DataArray(
            slc_maxch.data, dims=["freq"], coords={"freq": slc_maxch.freq}
        )
        ldx_slc_maxch.append(slc_maxch)
    return xr.concat(ldx_slc_maxch, dim="factor")


def get_norm(ldx):
    """L2 norm of each factor's loading vector."""
    return np.linalg.norm(ldx.data.reshape(ldx.shape[0], -1), axis=1)


def spatspec_designmat(mtx):
    """Flatten spectrogram to log-space design matrix (standalone)."""
    mtx = mtx.transpose("h", "w", "freq", "time")
    hwf_shape = mtx.shape[:-1]
    X = mtx.stack(hwf=("h", "w", "freq")).transpose("time", "hwf")
    Xlog = np.log10(X).reset_index("hwf")
    Xlog.attrs = {"hwf_shape": hwf_shape}
    return Xlog
