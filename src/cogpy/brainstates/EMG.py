"""
Watson & Buzsaki
Network Homeostasis and State Dynamics of Neocortical Sleep 2016, J Neuron

from Methods:
((
Electromyogram (EMG) from the neck, jaw and face muscles was extracted from the
intracranially recorded signals by detecting the zero time-lag correlation coefficients (r)
between 300-600 Hz filtered signals (using a Butterworth filter at 300 – 600 Hz with
filter shoulders spanning to 275 – 625 Hz) recorded at different sites (Schomburg et al.,
2014). Pairwise Pearson correlations were calculated between pairs of channels that were
a minimum of two shanks away from each other (400 µm). One high-quality channel per
shank was randomly nominated for pairing. The mean of all pairwise correlations
measured in each 0.5-second bin was calculated and recorded as an EMG score. A high
correlation between intracranially derived and direct EMG recordings were reported
previously (Schomburg et al., 2014). In our dataset, intracranial EMG was found to co-
vary reliably highly with motion and accelerometer measures. In cases where EMG and
motion detection measures differed, video recordings were examined for movement. For
example, chewing movements resulted in strong EMG activity but no translocation of the
head.
))
"""

import numpy as np
import xarray as xr
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.spatial.distance import cdist
from functools import partial
from ..utils.sliding import running_measure
from ..preprocess.filtx import butterworth_bandpass_shoulder
from ..utils.time_series import seconds_to_samples


def running_corrcoef(
    xsig_flat: xr.DataArray,
    window_size: float,
    window_step: float,
    run_dim="time",
    corrcoef_dim="ch",
):
    try:
        fs = xsig_flat.fs
    except AttributeError:
        raise ValueError(
            "Input `xsig_flat` must have a `.fs` attribute indicating the sampling frequency."
        )

    assert (
        corrcoef_dim in xsig_flat.dims
    ), f"Dimension '{corrcoef_dim}' not found in input data."
    assert run_dim in xsig_flat.dims, f"Dimension '{run_dim}' not found in input data."

    num_channels = xsig_flat.sizes[corrcoef_dim]
    run_corrcoef = running_measure(
        np.corrcoef,
        xsig_flat,
        fs,
        slider_kwargs=dict(
            window_size=seconds_to_samples(window_size, fs),
            window_step=seconds_to_samples(window_step, fs),
        ),
        measure_input_core_dims=[[corrcoef_dim, "window"]],
        measure_output_core_dims=[[corrcoef_dim + "1", corrcoef_dim + "2"]],
        measure_output_sizes={"ch1": num_channels, "ch2": num_channels},
        run_dim=run_dim,
        window_dim="window",
        output_dtype=np.float64,
    )
    return run_corrcoef


def select_distant_channel_pairs(
    ch_stereotaxic_coords_df: pd.DataFrame, min_distance: float = 400.0
) -> np.ndarray:
    """
    Return a definitive (deterministic) boolean mask over ALL channel pairs
    whose Euclidean distance (in µm) is >= `min_distance`.

    Parameters
    ----------
    ch_stereotaxic_coords_df : pd.DataFrame
        Must contain columns ['ap', 'ml', 'dv'] in micrometers. Row order defines channel order.
    min_distance : float
        Minimum distance (µm) for a pair to be considered distant.

    Returns
    -------
    np.ndarray
        Boolean array of shape (N, N). mask[i, j] == True iff distance(i, j) >= min_distance.
        The mask is symmetric, with the diagonal set to False.
    """
    required = {"ap", "ml", "dv"}
    missing = required - set(ch_stereotaxic_coords_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Use row order as the definitive channel order
    coords = ch_stereotaxic_coords_df[["ml", "ap", "dv"]].to_numpy(
        dtype=float
    )  # (N, 3)

    # Pairwise Euclidean distances (µm)
    pairwise_euclid_dist = cdist(coords, coords, metric="euclidean")  # (N, N)

    # Boolean mask for all pairs (symmetric). Diagonal is not a pair -> False.
    mask = pairwise_euclid_dist >= float(min_distance)
    np.fill_diagonal(mask, False)
    return np.triu(mask)


# EMG-specific bandpass filter as a partial function
bandpass_emg = partial(
    butterworth_bandpass_shoulder,
    low=300.0,
    high=600.0,
    shoulder=25.0,
    rp=1.0,
    rs=40.0,
    time_dim="time",
)

running_corrcoef_emg = partial(
    running_corrcoef,
    window_size=0.5,  # seconds
    window_step=0.25,  # seconds
    run_dim="time",
    corrcoef_dim="ch",
)


def compute_emg_proxy(
    x_data,
    fs,
    coords_df,
    min_distance,
    window_size=0.5,
    window_step=0.25,
    return_corr_da=False,
):
    # bandpass the data
    x_data_bp = butterworth_bandpass_shoulder(
        x_data,
        fs=fs,
        low=300.0,
        high=600.0,
        shoulder=25.0,
        rp=1.0,
        rs=40.0,
        time_dim="time",
    )
    x_data_bp.attrs["fs"] = fs

    # rolling window 0.5 seconds
    corr_da = running_corrcoef(
        x_data_bp,
        window_size=window_size,
        window_step=window_step,
        run_dim="time",
        corrcoef_dim="ch",
    )
    # # Select distant channel pairs
    mask = select_distant_channel_pairs(coords_df, min_distance=min_distance)
    # Apply mask to correlation data
    ch1_idxs, ch2_idxs = np.where(mask)
    corr_da_masked = corr_da.data[:, ch1_idxs, ch2_idxs]
    # compute mean and std
    emg_proxy = np.mean(corr_da_masked, axis=1)
    emg_proxy_std = np.std(corr_da_masked, axis=1)
    # make dataframe
    emg_proxy_df = pd.DataFrame(
        {
            "time": corr_da.coords["time"].data,
            "emg_proxy": emg_proxy,
            "emg_proxy_std": emg_proxy_std,
        }
    )

    if return_corr_da:
        return emg_proxy_df, corr_da
    return emg_proxy_df


def synthetic_data():
    # Create example data
    fs = 2000  # 2 kHz sampling rate
    n_channels = 32
    duration = 10  # 10 seconds
    n_samples = int(fs * duration)

    # Create synthetic neural data with EMG-like activity
    time = np.arange(n_samples) / fs
    channels = [f"ch_{i:02d}" for i in range(n_channels)]

    # Generate synthetic data (neural signals + EMG contamination)
    data = np.random.randn(n_channels, n_samples) * 10  # baseline activity

    # Add some EMG-like high frequency correlations
    emg_signal = np.random.randn(n_samples) * 5
    emg_filt = butter(4, [300, 600], btype="bandpass", fs=fs)
    emg_signal = filtfilt(emg_filt[0], emg_filt[1], emg_signal)

    # Add EMG to some channels with varying strengths
    emg_channels = np.random.choice(n_channels, size=n_channels // 2, replace=False)
    for ch_idx in emg_channels:
        strength = np.random.uniform(0.2, 0.8)
        data[ch_idx, :] += strength * emg_signal

    # Create xarray
    x_data = xr.DataArray(
        data, coords={"ch": channels, "time": time}, dims=("ch", "time")
    )

    # Create synthetic coordinates (simulating electrode positions)
    coords_data = {
        "ch": channels,
        "ml": np.random.uniform(-2000, 2000, n_channels),  # medio-lateral (µm)
        "ap": np.random.uniform(-3000, 1000, n_channels),  # anterior-posterior (µm)
        "dv": np.random.uniform(-4000, -500, n_channels),  # dorso-ventral (µm)
    }
    coords_df = pd.DataFrame(coords_data)
    return x_data, coords_df, fs
