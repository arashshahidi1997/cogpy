import numpy as np
import xarray as xr
from scipy import signal
from cogpy.preprocess.filtx import bandpassx, bandpass_filt_params, notchx, notchesx


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


def test_notchesx_matches_sequential_notchx():
    fs = 1000.0
    t = np.arange(2000) / fs
    rng = np.random.default_rng(42)
    sig = rng.normal(size=(4, t.size))
    sigx = xr.DataArray(sig, dims=["ch", "time"], coords={"time": t})
    sigx.attrs["fs"] = fs

    freqs = [60.0, 120.0, 180.0]
    Q = 30.0

    out_seq = sigx
    for f in freqs:
        out_seq = notchx(out_seq, w0=f, Q=Q, time_dim="time")

    out_multi = notchesx(sigx, freqs=freqs, Q=Q, time_dim="time")

    # SOS cascade vs sequential filtfilt can differ at tiny numerical levels.
    assert np.allclose(out_multi.data, out_seq.data, rtol=1e-6, atol=1e-8)
    assert out_multi.attrs["filter_type"] == "notch"
    assert out_multi.attrs["w0_hz"] == freqs
