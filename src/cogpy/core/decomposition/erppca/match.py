import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import itertools
from copy import deepcopy
from scipy.optimize import linear_sum_assignment
from .base import SpatSpecDecomposition
from .erpPCA import get_similarity


def match_factors(
    spatspec_series, nrec, nfac, freq_threshold=3, simil_score_threshold=0.7
):
    match_df, opt_ref = get_fac_match_df(
        spatspec_series, freq_threshold, nrec, nfac, simil_score_threshold
    )
    match_df_compressed = compress_fac_match_df(match_df, nrec, nfac)

    ss_match_series = spatspec_series.copy()
    for irec in range(nrec):
        fac_remap = match_df_compressed.iloc[:, irec].values
        ss_match_series.iloc[irec] = remap_ss(spatspec_series.iloc[irec], fac_remap)

    ss_match_cc = concat_ss_series(ss_match_series)

    # Normalize ldx and centroid
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


def get_remapping(simil_arr, nrec):
    """
    Parameters:
    - simil_arr: Array of similarity matrices
    - nrec: Number of recordings

    Returns:
    - optimal_remapping: Remapping of factors that maximizes similarity
    - optimal_similarity: Similarity of optimal remapping
    - mean_optimal_similarity: Mean similarity of optimal remapping
    """
    # remapping_split = np.zeros((nrec, nrec, nfac), dtype=int)
    optimal_remapping = np.zeros((nrec, nrec), dtype=object)
    optimal_similarity = np.zeros((nrec, nrec), dtype=object)
    mean_optimal_similarity = np.zeros((nrec, nrec), dtype=float)

    for i in np.ndindex((nrec, nrec)):
        # remapping_split[i] = remapping_(corr_split[i])
        # row_ind, col_ind = symmetric_assignment(corr_split[i])
        row_ind, col_ind = linear_sum_assignment(
            -simil_arr[i]
        )  # Using negative as it finds minimum
        mean_optimal_similarity[i] = simil_arr[i][row_ind, col_ind].mean()
        optimal_similarity[i] = simil_arr[i][row_ind, col_ind]
        optimal_remapping[i] = (row_ind, col_ind)

    return optimal_remapping, optimal_similarity, mean_optimal_similarity


def set_offdiag_elements(a, val):
    """
    Set the off-diagonal elements of a square matrix to a specified value.

    Parameters:
    a (numpy.ndarray): A square matrix (2D numpy array) whose off-diagonal elements are to be modified.
    val (numeric): The value to be assigned to the off-diagonal elements of the matrix.

    Returns:
    None: The function modifies the matrix in place and does not return a value.
    """
    a[np.where(~np.eye(a.shape[0], dtype=bool))] = val


def get_similx_flat(simil_arr, nrec):
    """
    Flatten a given similarity array into a stacked format and return as an xarray DataArray.

    This function iterates over a 2D similarity array, modifying its off-diagonal elements,
    then flattens and stacks the array into a new xarray DataArray format for further analysis.

    Parameters:
    simil_arr (numpy.ndarray): A 2D numpy array representing similarity values.
    nrec (int): The number of records. This is used to determine the size of the iterations and the final array dimensions.

    Returns:
    xarray.DataArray: A flattened and stacked version of the input similarity array.
                      The returned DataArray has dimensions ('rec0', 'rec1', 'fac0', 'fac1')
                      and stacked dimensions ('r0', 'r1').

    Note:
    The function modifies the off-diagonal elements of the `simil_arr[2,2]` sub-matrix to -2 for each iteration.
    This specific behavior may need to be adjusted based on the intended use.
    """
    for i in zip(range(nrec), range(nrec)):
        set_offdiag_elements(simil_arr[2, 2], -2)
    simil_arr_flat = np.stack([np.stack(simil_arr[i]) for i in range(nrec)])
    similx = xr.DataArray(simil_arr_flat, dims=["rec0", "rec1", "fac0", "fac1"])
    similx_flat = similx.stack(r0=("rec0", "fac0"), r1=("rec1", "fac1"))
    return similx_flat


def get_match_fac_ref(
    simil_arr, optimal_remapping, refrec, nrec, nfac, simil_score_threshold=0.7
):
    """
    Parameters:
    - simil_arr: Similarity array
    - optimal_remapping: Optimal remapping
    - refrec: Reference recording
    - nrec: Number of recordings
    - nfac: Number of factors

    Returns:
    - match_fac_ref: DataFrame containing the similarity score for each factor pair
    """
    similx_flat = get_similx_flat(simil_arr, nrec)
    match_fac_ref = pd.DataFrame(
        np.vstack(optimal_remapping[refrec])[1::2].T
    ).sort_values(refrec)
    for i, j in np.ndindex((nrec, nrec)):
        match_fac_ref.loc[:, f"s{i}_{j}"] = match_fac_ref.apply(
            lambda x: similx_flat.sel(r0=(i, x.loc[i]), r1=(j, x.loc[j])).item(), axis=1
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


# find optimal refrec
def match_metric(x, eps: float):
    # retunr squared distance to 1 if distance smaller than eps
    return 1 - np.where((1 - x) < eps, (1 - x) ** 2, 1)


def optimal_refrec(match_fac_ref, eps: float = 0.3):
    # return index of refrec with highest mean similarity score
    nrec = len(match_fac_ref)
    return np.argmax(
        [
            match_metric(match_fac_ref[iref].simil_score, eps).mean()
            for iref in range(nrec)
        ]
    )


def cutoff_lowsimil(match_df, threshold: float = 0.7):
    return match_df[match_df.simil_score > threshold]


def get_fac_match_df(ss_series, freq_threshold, nrec, nfac, simil_score_threshold):
    simil_arr = get_similarity(ss_series, freq_threshold)
    optimal_remapping, _, _ = get_remapping(simil_arr, nrec)
    match_fac_ref = [
        get_match_fac_ref(
            simil_arr, optimal_remapping, refrec, nrec, nfac, simil_score_threshold
        )
        for refrec in range(nrec)
    ]
    opt_ref = optimal_refrec(match_fac_ref, eps=1 - simil_score_threshold)
    match_df = match_fac_ref[1].copy()
    match_df = cutoff_lowsimil(match_df, threshold=simil_score_threshold)
    return match_df, opt_ref


def compress_fac_match_df(match_df, nrec, nfac):
    match_df_compressed = match_df.iloc[:, : nrec + 1].copy()
    match_df_compressed.loc[:, "simil"] = match_df.simil_score
    match_df_compressed = match_df_compressed.set_index("matched_factor")
    return match_df_compressed


def remap_fac_xr(arrx, remap):
    return arrx[remap].assign_coords(factor=range(len(remap)))


def remap_ss(ss, fac_remap):
    ss_match = deepcopy(ss)
    ss_match.ldx = remap_fac_xr(ss_match.ldx, fac_remap)
    ss_match.ldx_fch = remap_fac_xr(ss_match.ldx_fch, fac_remap)
    ss_match.ldx_maxfreq = remap_fac_xr(ss_match.ldx_maxfreq, fac_remap)
    ss_match.ldx_slc_maxfreq = remap_fac_xr(ss_match.ldx_slc_maxfreq, fac_remap)
    ss_match.ldx_slc_maxch = remap_fac_xr(ss_match.ldx_slc_maxch, fac_remap)
    ss_match.ldx_df = ss_match.ldx_df.iloc[fac_remap].reset_index(drop=True)
    return ss_match


def concat_ss_series(ss_series):
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


def plot_matched_facs(ss_match_cc, title, file):
    # plot matched factors ldx_slc_maxfreq
    nmatch_fac = len(ss_match_cc.factor)
    nrec = len(ss_match_cc.rec)

    fig, ax = plt.subplots(nmatch_fac, nrec, figsize=(25, 5 * nmatch_fac))
    for irec in range(nrec):
        for ifac in range(nmatch_fac):
            axis = ax[ifac, irec]
            ss_match_cc.ldx_slc_maxfreq.sel(factor=ifac)[irec].plot.imshow(
                cmap="jet", vmin=0, ax=axis
            )

    # For the title
    fig.suptitle(title, y=0.999, fontweight="bold")
    plt.tight_layout()

    # For the filename
    fig.savefig(file)


# loadings
def get_loadings_dict(ss: SpatSpecDecomposition):
    loadings_dict = {
        "loading": ss.ldx.data,
        "coo_AP": ss.ldx.h.values,
        "coo_ML": ss.ldx.w.values,
        "coo_Freq": ss.ldx.freq.values,
    }

    loadings_dict |= ss.ldx_df_mat()
    return loadings_dict
