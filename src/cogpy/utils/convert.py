import numpy as np


def to_int16_arr(float_arr):
    """
    Convert a numpy array to int16 format with safe clipping and normalization.

    Parameters:
    data (numpy.ndarray): The input array to convert.

    Returns:
    numpy.ndarray: The converted array.
    """
    # Determine the minimum and maximum values in the float64 array
    min_val = np.min(float_arr)
    max_val = np.max(float_arr)
    scaler = lambda x: 2 * (x - (min_val + max_val) / 2) / (max_val - min_val)
    scaled_arr = scaler(float_arr)
    int_arr = (scaled_arr * np.iinfo(np.int16).max).astype(
        np.int16
    )  # Scale to int16 range

    # Verify that no values exceed the range of int16
    assert np.all(int_arr >= np.iinfo(np.int16).min)
    assert np.all(int_arr <= np.iinfo(np.int16).max)
    return int_arr


def closest_power_of_two(num):
    """
    Find the closest power of two to the given number (can be up or down).

    Parameters:
    num (int or array-like): Input number or array of numbers.

    Returns:
    int or ndarray: Closest power(s) of two.
    """
    num_ = np.asarray(num)
    log2 = np.zeros_like(num_, dtype=float)
    mask = num_ > 0
    log2[mask] = np.log2(num_[mask])
    power = np.round(log2).astype(int)
    p2 = 2**power

    # Return scalar if input was scalar
    return p2.item() if np.isscalar(num) else p2
