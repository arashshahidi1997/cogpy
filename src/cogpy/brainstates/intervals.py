import numpy as np
import pandas as pd
import xarray as xr

from cogpy.datasets.schemas import Events, Intervals

__all__ = [
    "check_intervals_disjoint",
    "map_numbers_to_intervals",
    "map_numbers_to_disjoint_intervals",
    "subtract_intervals",
    "restrict",
    "perievent_epochs",
]


# %% Interval and State Mapping Functions:
def check_intervals_disjoint(intervals: list) -> bool:
    """
    Checks if a list of intervals is disjoint.

    Parameters
    ----------
    intervals : list
        List of intervals, each defined as a list or tuple with a start and an end.

    Returns
    -------
    bool
        True if the intervals are disjoint, False otherwise.

    Example
    -------
    intervals = [[1, 3], [4, 6], [7, 9]]
    check_intervals_disjoint(intervals)

    Output: True

    Note
    ----
    The function first sorts the intervals based on start and then checks if the start of any interval is less than the end of the next interval.
    """
    intervals = pd.DataFrame(intervals, columns=["start", "end"])
    intervals = intervals.sort_values("start")
    is_disjoint_df = intervals.start.shift(-1) < intervals.end
    is_disjoint = ~is_disjoint_df.any()
    return is_disjoint


def map_numbers_to_intervals(numbers: list, intervals: list) -> dict:
    """
    Maps indices of a sorted list of numbers to the indices of intervals they belong to.

    Parameters
    ----------
    numbers : list
        List of numbers to be mapped to intervals.

    intervals : list
        List of intervals, each defined as a list or tuple with a start and an end.

    Returns
    -------
    dict
        Dictionary with keys as indices of the sorted list of numbers and values as
          lists of indices of intervals (in the original order) that each number belongs to.

    Example
    -------
    numbers = [1, 2, 3, 4, 5]
    intervals = [[1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

    Output: {0: [0], 1: [0, 1], 2: [0, 1, 2], 3: [1, 2, 3], 4: [2, 3, 4]}

    Note
    ----
    The function first sorts both the numbers and the intervals before mapping.
    The indices in the output dictionary correspond to the sorted order of the numbers,
      and the indices in the value lists correspond to the original order of the intervals.

    end boundary is not included in the interval
    """
    # Sort the intervals based on start (and then by end if starts are equal)
    sorted_intervals = sorted(enumerate(intervals), key=lambda x: (x[1][0], x[1][1]))
    # Sort numbers and keep their original indices
    sorted_numbers = sorted(enumerate(numbers), key=lambda x: x[1])

    # Resultant dictionary
    result = {i: [] for i in range(len(numbers))}

    interval_index = 0
    for num_index, num in sorted_numbers:
        # While there are more intervals and the current interval's end is less than the current number
        while (
            interval_index < len(intervals)
            and sorted_intervals[interval_index][1][1] < num
        ):
            interval_index += 1

        # For each interval, if the number lies inside, add to result
        temp_interval_index = interval_index
        while (
            temp_interval_index < len(intervals)
            and sorted_intervals[temp_interval_index][1][0] <= num
        ):
            if sorted_intervals[temp_interval_index][1][1] > num:
                original_interval_index = sorted_intervals[temp_interval_index][0]
                result[num_index].append(original_interval_index)
            temp_interval_index += 1
    return result


def map_numbers_to_disjoint_intervals(numbers: list, intervals: list) -> np.ndarray:
    """
    Maps indices of a sorted list of numbers to the indices of disjoint intervals they belong to.

    Parameters
    ----------
    numbers : list
        List of numbers to be mapped to intervals.

    intervals : list
        List of intervals, each defined as a list or tuple with a start and an end.

    Returns
    -------
    np.ndarray
        Array with indices of intervals that each number belongs to. If a number does not belong to any interval, it is marked with -1.

    Note
    ----
    The intervals are considred as open ended, i.e., the end boundary is not included in the interval.

    Example
    -------
    numbers = [1, 2, 3, 4, 5]
    intervals = [[1, 3], [4, 6]]

    Output: array([ 0,  0,  -1,  1,  1])
    """
    # if intervals is empty, return -1 for all numbers
    tmap_arr = -np.ones(len(numbers), dtype=int)
    if len(intervals) == 0:
        return tmap_arr
    assert check_intervals_disjoint(intervals), "Intervals are not disjoint"
    tmap = map_numbers_to_intervals(numbers, intervals)
    # convert tmap dict to array
    for i, intervals in tmap.items():
        if len(intervals) == 0:
            tmap_arr[i] = -1
        elif len(intervals) == 1:
            tmap_arr[i] = intervals[0]
        else:
            raise AssertionError(
                f"Number belongs to more than one interval even though the intervals are disjoint, this should not happen. {i}, {intervals}"
            )
    return tmap_arr


def subtract_intervals(a, b):
    """
    Subtract intervals in a from intervals in b.

    Parameters
    ----------
    a : list
        List of intervals to be subtracted from b, each defined as a list or tuple with a start and an end.

    b : list
        List of intervals, each defined as a list or tuple with a start and an end.

    Returns
    -------
    list
        List of intervals after subtracting a from b.

    Example
    -------
    a = [[5, 8], [18, 22]]
    b = [[1, 10], [15, 20], [25, 30]]
    subtract_intervals(a, b)

    Output: [[1, 5], [8, 10], [15, 18], [22, 20], [25, 30]]

    Note
    ----
    The function subtracts each interval in a from each interval in b. The intervals in b are assumed to be disjoint.
    """
    assert check_intervals_disjoint(b), print(f"Intervals in b are not disjoint: {b}")

    result = []
    for interval_b in b:
        start_b, end_b = interval_b
        temp = [[start_b, end_b]]  # Start with the current interval from b

        for interval_a in a:
            start_a, end_a = interval_a
            new_temp = []

            for curr_start, curr_end in temp:
                if curr_end <= start_a or curr_start >= end_a:  # No overlap
                    new_temp.append([curr_start, curr_end])
                elif (
                    curr_start < start_a and curr_end > end_a
                ):  # A splits B into two parts
                    new_temp.append([curr_start, start_a])
                    new_temp.append([end_a, curr_end])
                elif curr_start < start_a and curr_end <= end_a:  # Overlap on the right
                    new_temp.append([curr_start, start_a])
                elif curr_start >= start_a and curr_end > end_a:  # Overlap on the left
                    new_temp.append([end_a, curr_end])

            temp = (
                new_temp  # Update the current working set with the adjusted intervals
            )

        result.extend(
            temp
        )  # Add the adjusted intervals for the current b interval to the result

    return result


def restrict(
    xsig: xr.DataArray,
    intervals,
    *,
    time_dim: str = "time",
) -> xr.DataArray:
    """
    Return signal samples whose time coordinate falls within any interval.

    Equivalent to pynapple's .restrict() but operates on xr.DataArray
    with a named time coordinate. Preserves all dims, coords, and attrs.

    Parameters
    ----------
    xsig      : xr.DataArray — signal with a time coordinate
    intervals : Intervals | np.ndarray (n,2) | list of [t0,t1]
                | dict {state: [[t0,t1],...]}
        If dict (cogpy brainstates format), all intervals across
        all states are used (union of all intervals).
    time_dim  : str — name of time dimension (default "time")

    Returns
    -------
    xr.DataArray — same dims/coords/attrs as input,
                   time axis restricted to valid samples.
                   Empty array if no samples fall in any interval.

    Notes
    -----
    Boundary convention: [t0, t1) — consistent with
    cogpy.brainstates.intervals.map_numbers_to_disjoint_intervals.
    """
    if time_dim not in xsig.dims:
        raise ValueError(f"xsig must have a {time_dim!r} dimension, got dims={tuple(xsig.dims)}")
    if time_dim not in xsig.coords:
        raise ValueError(f"xsig must have a {time_dim!r} coordinate")

    if isinstance(intervals, Intervals):
        iv = intervals.to_array()
    elif isinstance(intervals, dict):
        parts = []
        for v in intervals.values():
            if v is None or len(v) == 0:
                continue
            arr = np.asarray(v, dtype=float)
            if arr.ndim != 2 or arr.shape[1] != 2:
                raise ValueError(f"State intervals must be (n, 2), got shape {arr.shape}")
            parts.append(arr)
        iv = np.concatenate(parts, axis=0) if parts else np.zeros((0, 2), dtype=float)
    else:
        iv = np.asarray(intervals, dtype=float)
        if iv.ndim == 1 and len(iv) == 2:
            iv = iv[np.newaxis, :]
        if iv.ndim != 2 or iv.shape[1] != 2:
            raise ValueError(f"intervals must be (n, 2) array or Intervals object, got shape {iv.shape}")

    t = np.asarray(xsig.coords[time_dim].values, dtype=float)
    mask = np.zeros(len(t), dtype=bool)
    for t0, t1 in iv:
        mask |= (t >= t0) & (t < t1)

    return xsig.isel({time_dim: mask})


def perievent_epochs(
    xsig: xr.DataArray,
    events,
    fs: float,
    *,
    pre: float,
    post: float,
    time_dim: str = "time",
    fill_value: float = np.nan,
) -> xr.DataArray:
    """
    Extract signal epochs time-locked to events.

    Equivalent to pynapple's compute_perievent but operates on
    xr.DataArray and returns an xr.DataArray with an "event" dimension.

    Parameters
    ----------
    xsig       : xr.DataArray — continuous signal with time coordinate
                 Supported schemas: MultichannelTimeSeries (channel, time)
                                    IEEGGridTimeSeries (time, ML, AP)
                                    any DataArray with a time dim
    events     : Events | np.ndarray (n,) | list of float
                 Event times in seconds (same timebase as xsig).
    fs         : float — sampling rate in Hz
    pre        : float — seconds before event (positive = before)
    post       : float — seconds after event
    time_dim   : str — name of time dimension (default "time")
    fill_value : float — value for out-of-bounds samples (default NaN)

    Returns
    -------
    xr.DataArray with dims ("event", <other dims>, "lag")
        - "event" coordinate: event times in seconds
        - "lag" coordinate: time relative to event in seconds,
          from -pre to +post, length = round((pre + post) * fs) + 1
        - all other dims/coords from xsig preserved (excluding time coords)
        - attrs from xsig preserved, plus:
            pre, post, fs added to attrs

    Notes
    -----
    Events near the start or end of the recording where the full
    window is not available are padded with fill_value (NaN by default).
    Events are not dropped — the caller can filter NaN epochs if needed.
    """
    if time_dim not in xsig.dims:
        raise ValueError(f"xsig must have a {time_dim!r} dimension, got dims={tuple(xsig.dims)}")
    if time_dim not in xsig.coords:
        raise ValueError(f"xsig must have a {time_dim!r} coordinate")
    if fs <= 0:
        raise ValueError("fs must be positive")
    if pre < 0 or post < 0:
        raise ValueError("pre and post must be non-negative")

    if isinstance(events, Events):
        event_times = np.asarray(events.times, dtype=float)
    else:
        event_times = np.asarray(events, dtype=float)
        if event_times.ndim != 1:
            raise ValueError("events must be 1D array of times")

    t = np.asarray(xsig.coords[time_dim].values, dtype=float)
    n_time = len(t)

    n_pre = int(round(pre * fs))
    n_post = int(round(post * fs))
    n_samples = n_pre + n_post + 1
    lags = np.linspace(-pre, post, n_samples)

    other_dims = tuple(d for d in xsig.dims if d != time_dim)
    out_dims = ("event", *other_dims, "lag")

    fill_arr = np.asarray(fill_value)
    if fill_arr.dtype.kind == "f" and np.isnan(fill_arr):
        out_dtype = np.result_type(xsig.dtype, np.float64)
    else:
        out_dtype = np.result_type(xsig.dtype, fill_arr.dtype)

    out_shape = (len(event_times), *[xsig.sizes[d] for d in other_dims], n_samples)
    out = np.full(out_shape, fill_value, dtype=out_dtype)

    xsig_other_last = xsig.transpose(*other_dims, time_dim) if other_dims else xsig
    for ie, t_ev in enumerate(event_times):
        i_center = _nearest_sample_index(t, t_ev)
        i_start = i_center - n_pre
        i_end = i_center + n_post + 1

        src_start = max(0, i_start)
        src_end = min(n_time, i_end)
        if src_end <= src_start:
            continue

        dest_start = max(0, -i_start)
        dest_end = dest_start + (src_end - src_start)

        epoch = xsig_other_last.isel({time_dim: slice(src_start, src_end)})
        epoch_data = np.asarray(epoch.data)
        out[(ie, *([slice(None)] * len(other_dims)), slice(dest_start, dest_end))] = epoch_data

    coords = {}
    for name, coord in xsig.coords.items():
        if time_dim in coord.dims:
            continue
        coords[name] = coord
    coords["event"] = event_times
    coords["lag"] = lags

    epochs_da = xr.DataArray(out, dims=out_dims, coords=coords, attrs=dict(xsig.attrs))
    epochs_da.attrs.update({"pre": float(pre), "post": float(post), "fs": float(fs)})
    return epochs_da


def _nearest_sample_index(t: np.ndarray, t_ev: float) -> int:
    """
    Return index of sample time nearest to t_ev.

    Assumes t is 1D, sorted increasing.
    """
    n = len(t)
    if n == 0:
        raise ValueError("time coordinate is empty")
    i = int(np.searchsorted(t, t_ev, side="left"))
    if i <= 0:
        return 0
    if i >= n:
        return n - 1
    return i if abs(t[i] - t_ev) <= abs(t[i - 1] - t_ev) else i - 1
