# %% compute scores

# %% process scores
import scipy.signal as scisig
from ...utils import xarr as xut
import xarray as xr


def scx_bandpass(scx: xr.DataArray, wl, wh, order, axis=0):
    fs = scx.fs
    b, a = scisig.butter(order, [wl, min(wh, fs / 2 - 0.1)], btype="bandpass", fs=fs)
    bp_filt = lambda x: scisig.filtfilt(b, a, x, axis=axis)
    scx_smooth = xut.xarr_wrap(bp_filt)(scx)
    return scx_smooth


def get_score_dict(scx: xr.DataArray):
    scx_dict = {
        "scores": scx.data,
        "time": scx.time.values,
        "factor": scx.factor.values,
        "fs": scx.fs,
    }
    return scx_dict
