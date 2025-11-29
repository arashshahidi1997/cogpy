import numpy as np
import xarray as xr
import dask.array as da
import ghostipy as gsp
from scipy import signal
from typing import Callable, Dict, List, Any
from cogpy.utils.convert import closest_power_of_two
from copgy.utils import sliding as sl


def nperseg_from_ncycle(fm, fs=1, ncycle=7, power_of_two=True):
    """
    rel_nperseg: number of cycles per segment

    Parameters
    ----------
    fm: center frequency
    fs: sampling frequency

    Returns
    -------
    nperseg: number of samples per segment
    """
    nperseg = int(fs * ncycle / fm)
    if power_of_two:
        nperseg = closest_power_of_two(nperseg)
    return nperseg
