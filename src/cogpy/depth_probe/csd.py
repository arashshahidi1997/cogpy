import numpy as np
import scipy.ndimage as nd
from cogpy.utils.imports import import_optional

plt = import_optional("matplotlib.pyplot")
import_optional("skimage")
from skimage.morphology import extrema
import pandas as pd
import xarray as xr
from ..utils import xarr as xut


def simulate_depth_recordings(n_samples=1000, n_electrodes=16, smoothing_sigma=8):
    """Simulate depth recordings from multiple electrodes with smoothing."""
    # Create a random signal for each electrode
    data = np.random.randn(n_samples, n_electrodes)

    # Apply Gaussian smoothing along the time dimension for each electrode
    smoothed_data = nd.gaussian_filter1d(data, sigma=smoothing_sigma, axis=0)
    return smoothed_data


def compute_csd(dsigx):
    """
    Compute Current Source Density using the second spatial derivative.

    Parameters
    ----------
    dsigx : xarray.DataArray
        Depth recordings of shape (time, ch)

    Returns
    -------
    csd : xarray.DataArray
        Current Source Density of shape (time, ch)
    """
    csddata = np.diff(dsigx.transpose("ch", "time"), n=2, axis=0)
    csdx = xr.DataArray(
        csddata,
        dims=("ch", "time"),
        coords=dict(ch=dsigx.ch.values[1:-1], time=dsigx.time),
    )
    return csdx.transpose(*dsigx.dims)


def test_compute_csd():
    dsigx = xr.DataArray(
        np.random.randn(100, 16),
        dims=("time", "ch"),
        coords=dict(time=np.arange(100), ch=np.arange(16)),
    )
    csdx = compute_csd(dsigx)
    assert csdx.shape == (100, 14)


def identify_active_regions(csd_data, threshold_factor=3):
    """Identify active sinks and sources based on thresholding."""
    mean_csd = np.mean(csd_data)
    std_csd = np.std(csd_data)
    threshold = mean_csd + threshold_factor * std_csd

    sinks = csd_data > threshold
    sources = csd_data < -threshold
    return sinks, sources


def visualize_data(data, title=""):
    """Visualize depth recordings or CSD data."""
    plt.imshow(data.T, aspect="auto", cmap="bwr")
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Electrode")
    plt.show()


def main_pipeline():
    # 1. Simulate depth recordings
    recordings = simulate_depth_recordings()
    visualize_data(recordings, title="Simulated Depth Recordings")

    # 2. Compute CSD
    csd_data = compute_csd(recordings)
    visualize_data(csd_data, title="Current Source Density")

    # 3. Identify active regions
    sinks, sources = identify_active_regions(csd_data)
    visualize_data(sinks, title="Detected Sinks")
    visualize_data(sources, title="Detected Sources")


# Execute the pipeline
# main_pipeline()


def preprocess_depth_sig(dsigx, median_kernel):
    dsigx_sm = xut.xarr_wrap(lambda x: nd.median_filter(x, size=median_kernel))(dsigx)
    dsigx_sm = xut.xarr_wrap(lambda x: nd.gaussian_filter(x, sigma=median_kernel))(
        dsigx_sm
    )
    return dsigx_sm


# remove boundary extrema
def drop_boundary_extrema(pch, pt, boundary_ch):
    keep = ~np.isin(pch, boundary_ch)
    return pch[keep], pt[keep]


def find_max(image, boundary_ch, q=0.95):
    hmax = np.quantile(image, 0.95)
    h_maxima = extrema.h_maxima(image, hmax)
    pch, ptime = np.where(h_maxima)
    pch, ptime = drop_boundary_extrema(pch, ptime, boundary_ch)
    return pch, ptime


def get_max_df(x, time, ch, **kwargs):
    """
    x: np.array of shape (ch, time)
    """
    boundary_ch = ch[[0, -1]]
    pch, pit = find_max(x, boundary_ch, **kwargs)
    pa = x[pch, pit]
    pt = time[pit]
    max_df = pd.DataFrame(
        np.array([pt, pit, pch + 1, pa]).T, columns=["tpeak", "ipeak", "ch", "amp"]
    )
    return max_df


def detect_sinks_and_sources(csd_sm, time, ch_values):
    source_df = get_max_df(csd_sm, time, ch_values - 1)
    sink_df = get_max_df(-csd_sm, time, ch_values - 1)
    return sink_df, source_df
