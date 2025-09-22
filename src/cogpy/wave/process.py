import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from .features import positive_boundaries

def extract_wave_df(x_, height=0, prominence_ratio=0.25, wlen_seconds=6):
    wlen = int(wlen_seconds * x_.fs.item())
    ipeak, peak_props = find_peaks(x_.data, height=height, distance=1, prominence=np.quantile(x_.data , 0.95) * prominence_ratio, wlen=wlen, width=0, rel_height=0.5)

    i2t = lambda i: x_.time[0].item() + i / x_.fs.item()

    tpeak = x_.time[ipeak]
    amp = x_.data[ipeak]
    ion = peak_props['left_bases']
    ioff = peak_props['right_bases']
    ton = i2t(peak_props['left_bases'])
    toff = i2t(peak_props['right_bases'])
    dur = toff - ton
    ndur = ioff - ion

    ileft = peak_props['left_bases']
    iright = peak_props['right_bases']
    tleft = i2t(peak_props['left_ips'])
    tright = i2t(peak_props['right_ips'])
    width = i2t(peak_props['widths'])
    nwidth = iright - ileft

    prominence = peak_props['prominences']
    contour_heights = x_[ipeak] - peak_props['prominences']
    width_heights = peak_props['width_heights']

    # make dataframe
    wave_df = pd.DataFrame({
        'ipeak':ipeak,
        'tpeak':tpeak,
        'ion':ion,
        'ioff':ioff,
        'ton':ton,
        'toff':toff,
        'dur':dur,
        'ndur':ndur,
        'ileft':ileft,
        'iright':iright,
        'tleft':tleft,
        'tright':tright,
        'width':width,
        'nwidth':nwidth,
        'amp':amp,
        'prominence':prominence,
        'contour_heights':contour_heights,
        'width_heights':width_heights
    })
    return wave_df

def positive_waves(x):
    # get boundaries of waves
    boundaries = positive_boundaries(x)
    # get wave df
    wave_df = []
    for ion, ioff in boundaries:
        wave = x[ion:ioff]
        df_ = process_wave(wave)
        df_['ion'] = ion
        df_['ioff'] = ioff
        df_['ipeak'] = ion + df_['ipeak_rel'].values[0]
        wave_df.append(df_)
    wave_df = pd.concat(wave_df, ignore_index=True)
    # reorganize columns
    wave_df = wave_df[['peak_time', 'ipeak', 'ipeak_rel', 't_on', 'ion', 't_off', 'ioff', 'dur', 'ndur', 'peak_amp']]
    return wave_df

def process_wave(wave):
    """
    wave: xr.DataArray | list of xr.DataArray
    coords:
        time
    attrs:
        fs: sampling frequency
    """
    # if wave is a list or generator apply process wave to each wave
    if isinstance(wave, list):
        return pd.concat([process_wave(w) for w in wave], ignore_index=True)

    peak_amplitude = np.max(wave).item()
    ipeak_rel = wave.argmax().item()
    peak_time = wave.time[ipeak_rel].item()
    half_amplitude = peak_amplitude / 2
    nsamples = len(np.where(wave.data > half_amplitude)[0])
    half_amplitude_duration = nsamples / wave.fs.item()
    t_on = wave.time.isel(time=0).item()
    t_off = wave.time.isel(time=-1).item()

    wave_df = pd.DataFrame([{
        'peak_time': peak_time,
        'ipeak_rel': ipeak_rel,
        'peak_amp': peak_amplitude,
        'dur': half_amplitude_duration,
        'ndur': nsamples,
        't_on': t_on,
        't_off': t_off,
    }])
    return wave_df

# with envelope removal
# is waves with constant zero breaks in between
# so to detect the waves and their properties, we need to remove the zero breaks
# then we can apply the same process_wave function to each wave
# split array with removing zero breaks
