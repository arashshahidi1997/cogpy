import numpy as np
import xarray as xr
from scipy import signal
from tqdm import tqdm
import pandas as pd

# Spindle Phase
DEFAULT_TIME_HALFWINDOW = 0.4
def slc_around(time, time_halfwindow=DEFAULT_TIME_HALFWINDOW):
    burst_time_slc = slice(time-time_halfwindow, time+time_halfwindow)
    return burst_time_slc

def get_burst_tslc(burst, time_halfwindow=DEFAULT_TIME_HALFWINDOW):
    burst_time_slc = slc_around(burst.time, time_halfwindow)
    return burst_time_slc

def get_burst_signal(csig, burst, time_halfwindow=DEFAULT_TIME_HALFWINDOW):
    burst_time_slc = get_burst_tslc(burst, time_halfwindow)
    burst_sig = csig.sel(time=burst_time_slc)
    return burst_sig

def get_burst_sig_at_ch(burst_sig, burst):
    iAP = burst.iAP
    iML = burst.iML
    # convert to int if it is a float
    if isinstance(iAP, float):
        iAP = int(iAP)
    if isinstance(iML, float):
        iML = int(iML)
    burst_sig = burst_sig.sel(AP=iAP, ML=iML)
    return burst_sig
    
def get_burst_sig_tslice_at_ch(burst, csig, time_halfwindow=DEFAULT_TIME_HALFWINDOW):
    burst_sig = get_burst_signal(csig, burst, time_halfwindow=time_halfwindow)
    burst_sig = get_burst_sig_at_ch(burst_sig, burst)
    return burst_sig

def bandpass_filter(burst_sig, burst, fs, freq_halfbandwidth=5):
    freq_center = burst.freq
    freq_band = [freq_center-freq_halfbandwidth, freq_center+freq_halfbandwidth]

    burst_sig_bp = signal.filtfilt(*signal.butter(2, freq_band, 'bandpass', fs=fs), burst_sig.data)
    burst_sig_bp = xr.DataArray(burst_sig_bp, coords=burst_sig.coords, dims=burst_sig.dims)
    return burst_sig_bp

def hilbert_transform(burst_sig_bp):
    burst_sig_hilb = signal.hilbert(burst_sig_bp.data)
    burst_analytic = xr.DataArray(burst_sig_hilb, coords=burst_sig_bp.coords, dims=burst_sig_bp.dims)
    return burst_analytic

def get_burst_sig_phase_and_amp(csig, burst, time_halfwindow=1, freq_halfbandwidth=5):
    burst_sig = get_burst_signal(csig, burst, time_halfwindow=time_halfwindow)
    burst_sig = get_burst_sig_at_ch(burst_sig, burst)
    burst_sig_bp = bandpass_filter(burst_sig, burst, fs=csig.fs, freq_halfbandwidth=freq_halfbandwidth)
    burst_analytic = hilbert_transform(burst_sig_bp)
    return burst_analytic

def get_burst_analytic(burst_df, csig, freq_halfbandwidth=2.5, time_halfwindow=1):
    burst_analytic_sig = []
    for burst in tqdm(burst_df.itertuples(), total=len(burst_df)):
        burst_sig = get_burst_signal(csig, burst, time_halfwindow=time_halfwindow)
        burst_sig = get_burst_sig_at_ch(burst_sig, burst)
        burst_sig_bp = bandpass_filter(burst_sig, burst, fs=csig.fs, freq_halfbandwidth=freq_halfbandwidth)
        burst_analytic = hilbert_transform(burst_sig_bp)
        burst_analytic_sig.append(burst_analytic)
    return burst_analytic_sig

def get_rel_time(burst, burst_analytic_sig):
    return burst_analytic_sig.assign_coords({'time': burst_analytic_sig.time - burst.time})

def get_burst_analytic_rel_time(burst_df, burst_analytic_sig):
    burst_analytic_sig_rel_time = []
    for burst, burst_analytic in zip(burst_df.itertuples(), burst_analytic_sig):
        burst_analytic = get_rel_time(burst, burst_analytic)
        burst_analytic_sig_rel_time.append(burst_analytic)
    return burst_analytic_sig_rel_time
