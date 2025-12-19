from cogpy.utils.convert import closest_power_of_two


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
