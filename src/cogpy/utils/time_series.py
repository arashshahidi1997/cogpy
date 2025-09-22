import numpy as np
import pandas as pd
import quantities as pq
from scipy.signal import hilbert
from .xarr import xarr_wrap

def rolling_zscore(x: np.ndarray, axis=0, window_size=100, return_stats=False):
    """
    x: numpy array
    axis: axis to compute rolling zscore
    window_size: window size for rolling zscore
    """
    multidim = x.ndim > 2
    # reshape to 2d array
    if x.ndim == 1:
        x = x.reshape(-1,1)
    elif multidim:
        # move axis to 0
        x = np.moveaxis(x, axis, 0)
        # reshape the rest of the axes but save original shape
        x_shape = x.shape
        x = x.reshape(x.shape[0], -1)

    x_ = pd.DataFrame(x)
    
    # mirror half window size
    x_mirror = pd.concat([x_.iloc[window_size:0:-1], x_, x_.iloc[-2:-window_size-2:-1]])
    x_roll = x_mirror.rolling(window_size*2+1, center=True, axis=0)
    
    # compute rolling mean and std
    x_mean = x_roll.mean()
    x_std = x_roll.std()

    # drop half window size
    x_mean = x_mean.iloc[window_size:-window_size].to_numpy()
    x_std = x_std.iloc[window_size:-window_size].to_numpy()
    
    # compute zscore
    x_zscore = (x - x_mean) / x_std

    # reshape back to original shape
    if multidim:
        x_zscore = x_zscore.reshape(*x_shape)
        x_mean = x_mean.reshape(*x_shape)
        x_std = x_std.reshape(*x_shape)

    # move axis back
    x_zscore = np.moveaxis(x_zscore, 0, axis)
    x_mean = np.moveaxis(x_mean, 0, axis)
    x_std = np.moveaxis(x_std, 0, axis)

    if return_stats:
        return x_zscore, x_mean, x_std
    else:
        return x_zscore

@xarr_wrap
def lower_envelope(x, axis=-1):
    lower_envelope = -np.abs(hilbert(-x, axis=axis))
    return lower_envelope

@xarr_wrap
def threshold(x, quantile=0.25, axis=-1):
    quantile_value = np.expand_dims(np.quantile(x, quantile, axis=axis), axis=axis)
    return x - quantile_value

def seconds_to_samples(x, fs):
    if isinstance(x, pq.quantity.Quantity):
        xi = int(x.rescale('s').magnitude * fs)
    elif isinstance(x, int):
        xi = x
    elif isinstance(x, float):
        xi = int(x * fs)
    return xi

