import numpy as np
import pandas as pd
from .burst_phase import (
    get_burst_signal,
    get_burst_sig_at_ch,
    bandpass_filter,
    hilbert_transform,
)
from tqdm import tqdm


# %% burst-wave analysis
def get_burst_concurrent_wave_subsets(
    wave_rec_df_, burst_rec_df_, is_wave_burst_concurrent, compute_rel_coos=True
):
    """
    Get the burst-wave subset for each burst in the burst dataframe.
    The burst-wave subset is a dataframe that contains the wave detections that are concurrent with the burst detections.
    The function computes the relative coordinates of the wave detections with respect to the burst detections.
    The function returns a dataframe with the following columns:
        - AP: Anterior-Posterior coordinate
        - ML: Medio-Lateral coordinate
        - time: Time coordinate
        - iAP: Index of the Anterior-Posterior coordinate
        - iML: Index of the Medio-Lateral coordinate
        - iTime: Index of the time coordinate
        - ref_burst: Index of the reference burst
        - burst_idx: Index of the burst
        - AP_diff: Difference in the Anterior-Posterior coordinate with respect to the reference burst
        - ML_diff: Difference in the Medio-Lateral coordinate with respect to the reference burst
    Parameters
    ----------
    wave_rec_df_ : pd.DataFrame
        wave detection dataframe
    burst_rec_df_ : pd.DataFrame
        burst detection dataframe
    is_wave_burst_concurrent : np.ndarray
        A boolean array of shape (len(wave_rec_df_), len(burst_rec_df_)) indicating the colocalization of wave and burst detections
    compute_rel_coos : bool
        If True, compute the relative coordinates of the wave detections with respect to the burst detections.
        If False, do not compute the relative coordinates.
    Returns
    -------
    burst_concurrent_wave_subset : pd.DataFrame
        A dataframe with the wave detections that are concurrent with the burst detections.
        The dataframe has the following columns:
            - AP: Anterior-Posterior coordinate
            - ML: Medio-Lateral coordinate
            - time: Time coordinate
            - iAP: Index of the Anterior-Posterior coordinate
            - iML: Index of the Medio-Lateral coordinate
            - iTime: Index of the time coordinate
            - ref_burst: Index of the reference burst
            - burst_idx: Index of the burst
            - AP_diff: Difference in the Anterior-Posterior coordinate with respect to the reference burst
            - ML_diff: Difference in the Medio-Lateral coordinate with respect to the reference burst
    """

    # are concurrent waves more colocalized than non-concurrent waves?
    burst_concurrent_wave_subset_idx = [np.where(x) for x in is_wave_burst_concurrent.T]
    burst_concurrent_wave_subset = {}

    # for each wave subset subtract the reference burst's AP, ML
    # suppress pandas warning
    pd.options.mode.chained_assignment = None
    for iburst, wave_subset in enumerate(burst_concurrent_wave_subset_idx):
        reference_burst = burst_rec_df_.iloc[iburst]
        wave_subset = wave_rec_df_.iloc[wave_subset].copy()
        if compute_rel_coos:
            wave_subset["AP_diff"] = wave_subset.AP - reference_burst.AP
            wave_subset["ML_diff"] = wave_subset.ML - reference_burst.ML
        wave_subset["ref_burst"] = iburst
        burst_concurrent_wave_subset[iburst] = wave_subset.reset_index()

    # concat dict of dataframes adding burst key
    burst_concurrent_wave_subset = pd.concat(
        burst_concurrent_wave_subset, names=["burst"]
    )
    burst_concurrent_wave_subset.loc[:, "burst_idx"] = (
        burst_concurrent_wave_subset.index.get_level_values("burst")
    )
    burst_concurrent_wave_subset = burst_concurrent_wave_subset.reset_index(drop=True)
    return burst_concurrent_wave_subset


def get_is_wave_burst_colocalized_spatial_bins(wave_rec_df_, burst_rec_df_):
    """
    Get the colocalization of wave and burst detections in spatial bins

    Parameters
    ----------
    wave_rec_df_ : pd.DataFrame
        wave detection dataframe

    burst_rec_df_ : pd.DataFrame
        burst detection dataframe

    Returns
    -------
    is_wave_burst_colocalized : np.ndarray
        A boolean array of shape (len(wave_rec_df_), len(burst_rec_df_)) indicating the colocalization of wave and burst detections in spatial bins
    """
    wave_spatial_groups = wave_rec_df_.groupby(["AP_bin", "ML_bin"])
    burst_spatial_groups = burst_rec_df_.groupby(["AP_bin", "ML_bin"])
    is_wave_burst_colocalized = np.zeros(
        (len(wave_rec_df_), len(burst_rec_df_)), dtype=bool
    )
    for (ap_bin, ml_bin), wave_group in wave_spatial_groups:
        try:
            burst_group = burst_spatial_groups.get_group((ap_bin, ml_bin))
            wave_idxs = wave_group.index.values
            burst_idxs = burst_group.index.values
            is_wave_burst_colocalized[wave_idxs[:, None], burst_idxs] = True
        except:
            continue
    return is_wave_burst_colocalized


# spatial dist
def prep_wave_df(wave_df):
    wave_df.loc[:, "iAP"] = (wave_df.peak_x - 1).astype(int)
    wave_df.loc[:, "iML"] = (wave_df.peak_x - 1).astype(int)
    return wave_df


def get_burst_conco_wave_subset(burst_df, wave_df):
    wave_df = prep_wave_df(wave_df)

    ap_abs_dist = np.abs(wave_df.iAP.values[:, None] - burst_df.iAP.values)
    ml_abs_dist = np.abs(wave_df.iML.values[:, None] - burst_df.iML.values)
    spatial_dist = np.sqrt(ap_abs_dist**2 + ml_abs_dist**2)

    # is colocalized
    axial_dist_channels = 4
    spatial_dist_thresh = round(np.sqrt(2) * axial_dist_channels + 0.5)
    is_wave_burst_colocalized = spatial_dist < spatial_dist_thresh

    # temporal dist
    time_dist = np.abs(wave_df.peak_abs.values[:, None] - burst_df.time.values)

    # is concurrent
    time_dist_thresh = 1  # s
    is_wave_burst_concurrent = time_dist < time_dist_thresh

    # is concurrent and colocalized
    is_wave_burst_concurrent_and_colocalized = (
        is_wave_burst_colocalized & is_wave_burst_concurrent
    )

    # concurrent and colocalized wave subset
    burst_conco_wave_subset = get_burst_concurrent_wave_subsets(
        wave_df, burst_df, is_wave_burst_concurrent_and_colocalized
    )
    return burst_conco_wave_subset


def get_burst_wave_phase(burst_conco_wave_subset, burst_df, csig, time_halfwindow=1):
    burst_wave_phase = []
    burst_conco_wave_group = burst_conco_wave_subset.groupby("burst_idx")
    for burst_idx, conco_waves in tqdm(burst_conco_wave_group):
        burst = burst_df.loc[burst_idx]
        burst_sig = get_burst_signal(csig, burst, time_halfwindow=time_halfwindow)
        burst_sig = get_burst_sig_at_ch(burst_sig, burst)
        burst_sig_bp = bandpass_filter(
            burst_sig, burst, fs=csig.fs, freq_halfbandwidth=2.5
        )
        burst_analytic = hilbert_transform(burst_sig_bp)

        # sum burst_phase at each wave time
        wave_phase = conco_waves.peak_abs.values
        burst_wave_phase.append(burst_analytic.sel(time=wave_phase, method="nearest"))

    return burst_wave_phase
