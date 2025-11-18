import numpy as np
import xarray as xr
from scipy import signal
from cogpy.preprocess.filtx import bandpassx, bandpass_filt_params


def test_bandpassx():
    time = np.linspace(0, 10, 1000)
    fs = 1 / (time[1] - time[0])
    sig = np.sin(2 * np.pi * 10 * time) + np.sin(2 * np.pi * 20 * time)
    # repeat signal for 10 channels
    sig = np.tile(sig, (10, 1))
    sigx = xr.DataArray(sig, dims=["ch", "time"], coords={"time": time})
    sigx.attrs["fs"] = fs
    sigx_bp = bandpassx(sigx, 5, 15, 3, "time")

    # apply bandpass with regular numpy array
    b, a = bandpass_filt_params(3, 5, 15, fs)
    sig_bp = signal.filtfilt(b, a, sig, axis=1)

    assert np.allclose(sigx_bp.data, sig_bp)
