"""
Cross-recording factor matching via Hungarian algorithm.

Matches factors across multiple SpatSpecDecomposition instances by
maximising spatial-spectral similarity, using ``scipy.optimize.linear_sum_assignment``.
"""

from __future__ import annotations

import itertools
from copy import deepcopy

import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import linear_sum_assignment

from .pca import get_similarity
from .spatspec import SpatSpecDecomposition


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def match_factors(
    spatspec_series,
    nrec,
    nfac,
    freq_threshold=3,
    simil_score_threshold=0.7,
    optimal_ref=True,
):
    """Match factors across recordings and return matched + centroid results.

    Parameters
    ----------
    spatspec_series : pd.Series
        Series of ``SpatSpecDecomposition`` instances (one per recording).
    nrec : int
        Number of recordings.
    nfac : int
        Number of factors per recording.
    freq_threshold : float
        Max frequency difference for similarity (Hz).
    simil_score_threshold : float
        Minimum similarity to retain a match.
    optimal_ref : bool, optional
        If True (default), automatically select the reference recording
        whose factors are most similar to all others. If False, use the
        first recording (index 0) as the reference.

    Returns
    -------
    dict
        ``ldx_pkl``: centroid SpatSpecDecomposition,
        ``ldx_mat``: centroid loadings dict,
        ``ldx_all_pkl``: concatenated matched decompositions,
        ``match_df_csv``: compressed match DataFrame,
        ``metadata_csv``: matching metadata.
    """
    match_df, opt_ref = get_fac_match_df(
        spatspec_series, freq_threshold, nrec, nfac, simil_score_threshold,
        optimal_ref=optimal_ref,
    )
    match_df_compressed = compress_fac_match_df(match_df, nrec, nfac)

    ss_match_series = spatspec_series.copy()
    for irec in range(nrec):
        fac_remap = match_df_compressed.iloc[:, irec].values
        ss_match_series.iloc[irec] = remap_ss(spatspec_series.iloc[irec], fac_remap)

    ss_match_cc = concat_ss_series(ss_match_series)

    # Normalize and compute centroid
    ldx_ss = ss_match_cc.ldx.stack(ss=("h", "w", "freq"))
    ldx_ss_normalized = ldx_ss / np.expand_dims(np.linalg.norm(ldx_ss, axis=-1), -1)
    ldx_cent = ldx_ss_normalized.mean("rec").unstack("ss")
    spatspec_coords = spatspec_series.iloc[0].ldx.coords
    ss_cent = SpatSpecDecomposition(spatspec_coords)
    ss_cent.ldx_set_direct(ldx_cent)
    ss_cent.ldx_process()

    return {
        "ldx_pkl": ss_cent,
        "ldx_mat": ss_cent.get_loadings_dict(),
        "ldx_all_pkl": ss_match_cc,
        "match_df_csv": match_df_compressed,
        "metadata_csv": pd.Series(
            {
                "opt_ref": opt_ref,
                "freq_threshold": freq_threshold,
                "simil_score_threshold": simil_score_threshold,
            }
        ),
    }


# ---------------------------------------------------------------------------
# Remapping
# ---------------------------------------------------------------------------


def get_remapping(simil_arr, nrec):
    """Compute optimal factor remapping via linear sum assignment."""
    optimal_remapping = np.zeros((nrec, nrec), dtype=object)
    optimal_similarity = np.zeros((nrec, nrec), dtype=object)
    mean_optimal_similarity = np.zeros((nrec, nrec), dtype=float)

    for i in np.ndindex((nrec, nrec)):
        row_ind, col_ind = linear_sum_assignment(-simil_arr[i])
        mean_optimal_similarity[i] = simil_arr[i][row_ind, col_ind].mean()
        optimal_similarity[i] = simil_arr[i][row_ind, col_ind]
        optimal_remapping[i] = (row_ind, col_ind)

    return optimal_remapping, optimal_similarity, mean_optimal_similarity


def set_offdiag_elements(a, val):
    a[np.where(~np.eye(a.shape[0], dtype=bool))] = val


def get_similx_flat(simil_arr, nrec):
    """Flatten similarity array into a stacked xr.DataArray."""
    for i in range(nrec):
        set_offdiag_elements(simil_arr[i, i], -2)
    simil_arr_flat = np.stack([np.stack(simil_arr[i]) for i in range(nrec)])
    similx = xr.DataArray(simil_arr_flat, dims=["rec0", "rec1", "fac0", "fac1"])
    return similx.stack(r0=("rec0", "fac0"), r1=("rec1", "fac1"))


def get_match_fac_ref(
    simil_arr, optimal_remapping, refrec, nrec, nfac, simil_score_threshold=0.7
):
    """Get factor match DataFrame for a given reference recording."""
    similx_flat = get_similx_flat(simil_arr, nrec)
    match_fac_ref = pd.DataFrame(
        np.vstack(optimal_remapping[refrec])[1::2].T
    ).sort_values(refrec)
    for i, j in np.ndindex((nrec, nrec)):
        match_fac_ref.loc[:, f"s{i}_{j}"] = match_fac_ref.apply(
            lambda x: similx_flat.sel(r0=(i, x.loc[i]), r1=(j, x.loc[j])).item(),
            axis=1,
        )

    match_fac_ref.loc[:, "simil_arr"] = match_fac_ref.apply(
        lambda x: [
            [
                similx_flat.sel(r0=(i, x.loc[i]), r1=(j, x.loc[j])).item()
                for j in range(nrec)
            ]
            for i in range(nrec)
        ],
        axis=1,
    )
    match_fac_ref.loc[:, "simil"] = match_fac_ref.apply(
        lambda x: [
            similx_flat.sel(r0=(i, x.loc[i]), r1=(j, x.loc[j])).item()
            for i, j in itertools.combinations(range(nrec), 2)
        ],
        axis=1,
    )
    match_fac_ref.loc[:, "simil_score"] = match_fac_ref.loc[:, "simil"].apply(
        lambda x: min(x)
    )
    match_fac_ref = match_fac_ref.sort_values(
        "simil_score", ascending=False
    ).reset_index(drop=True)
    match_fac_ref = match_fac_ref.reset_index().rename(
        columns={"index": "matched_factor"}
    )
    return match_fac_ref


# ---------------------------------------------------------------------------
# Optimal reference selection
# ---------------------------------------------------------------------------


def match_metric(x, eps: float):
    return 1 - np.where((1 - x) < eps, (1 - x) ** 2, 1)


def optimal_refrec(match_fac_ref, eps: float = 0.3):
    """Return the recording index with highest mean match metric."""
    nrec = len(match_fac_ref)
    return np.argmax(
        [
            match_metric(match_fac_ref[iref].simil_score, eps).mean()
            for iref in range(nrec)
        ]
    )


def cutoff_lowsimil(match_df, threshold: float = 0.7):
    return match_df[match_df.simil_score > threshold]


def get_fac_match_df(
    ss_series, freq_threshold, nrec, nfac, simil_score_threshold,
    optimal_ref=True,
):
    simil_arr = get_similarity(ss_series, freq_threshold)
    optimal_remapping, _, _ = get_remapping(simil_arr, nrec)
    match_fac_ref = [
        get_match_fac_ref(
            simil_arr, optimal_remapping, refrec, nrec, nfac, simil_score_threshold
        )
        for refrec in range(nrec)
    ]
    if optimal_ref:
        opt_ref = optimal_refrec(match_fac_ref, eps=1 - simil_score_threshold)
    else:
        opt_ref = 0
    match_df = match_fac_ref[opt_ref].copy()
    match_df = cutoff_lowsimil(match_df, threshold=simil_score_threshold)
    return match_df, opt_ref


def compress_fac_match_df(match_df, nrec, nfac):
    match_df_compressed = match_df.iloc[:, : nrec + 1].copy()
    match_df_compressed.loc[:, "simil"] = match_df.simil_score
    return match_df_compressed.set_index("matched_factor")


# ---------------------------------------------------------------------------
# SpatSpec remapping / concatenation
# ---------------------------------------------------------------------------


def remap_fac_xr(arrx, remap):
    return arrx[remap].assign_coords(factor=range(len(remap)))


def remap_ss(ss, fac_remap):
    """Remap factors of a SpatSpecDecomposition."""
    ss_match = deepcopy(ss)
    ss_match.ldx = remap_fac_xr(ss_match.ldx, fac_remap)
    ss_match.ldx_fch = remap_fac_xr(ss_match.ldx_fch, fac_remap)
    ss_match.ldx_maxfreq = remap_fac_xr(ss_match.ldx_maxfreq, fac_remap)
    ss_match.ldx_slc_maxfreq = remap_fac_xr(ss_match.ldx_slc_maxfreq, fac_remap)
    ss_match.ldx_slc_maxch = remap_fac_xr(ss_match.ldx_slc_maxch, fac_remap)
    ss_match.ldx_df = ss_match.ldx_df.iloc[fac_remap].reset_index(drop=True)
    return ss_match


def concat_ss_series(ss_series):
    """Concatenate a series of SpatSpecDecompositions along a ``rec`` dim."""
    ss_list = [ss_series.iloc[i] for i in range(len(ss_series))]
    ss_concat = deepcopy(ss_list[0])
    ss_concat.ldx = xr.concat([ss.ldx for ss in ss_list], dim="rec")
    ss_concat.ldx_fch = xr.concat([ss.ldx_fch for ss in ss_list], dim="rec")
    ss_concat.ldx_maxfreq = xr.concat([ss.ldx_maxfreq for ss in ss_list], dim="rec")
    ss_concat.ldx_slc_maxfreq = xr.concat(
        [ss.ldx_slc_maxfreq for ss in ss_list], dim="rec"
    )
    ss_concat.ldx_slc_maxch = xr.concat([ss.ldx_slc_maxch for ss in ss_list], dim="rec")
    ss_concat.ldx_df = None
    return ss_concat
