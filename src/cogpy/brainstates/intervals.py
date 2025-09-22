import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm
from typing import List

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
    intervals = pd.DataFrame(intervals, columns=['start', 'end'])
    intervals = intervals.sort_values('start')
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
        while interval_index < len(intervals) and sorted_intervals[interval_index][1][1] < num:
            interval_index += 1

        # For each interval, if the number lies inside, add to result
        temp_interval_index = interval_index
        while temp_interval_index < len(intervals) and sorted_intervals[temp_interval_index][1][0] <= num:
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
    assert check_intervals_disjoint(intervals), 'Intervals are not disjoint'
    tmap = map_numbers_to_intervals(numbers, intervals)
    # convert tmap dict to array
    for i, intervals in tmap.items():
        if len(intervals) == 0:
            tmap_arr[i] = -1
        elif len(intervals) == 1:
            tmap_arr[i] = intervals[0]
        else:
            raise AssertionError(f'Number belongs to more than one interval even though the intervals are disjoint, this should not happen. {i}, {intervals}')
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
    assert check_intervals_disjoint(b), print(f'Intervals in b are not disjoint: {b}')

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
                elif curr_start < start_a and curr_end > end_a:  # A splits B into two parts
                    new_temp.append([curr_start, start_a])
                    new_temp.append([end_a, curr_end])
                elif curr_start < start_a and curr_end <= end_a:  # Overlap on the right
                    new_temp.append([curr_start, start_a])
                elif curr_start >= start_a and curr_end > end_a:  # Overlap on the left
                    new_temp.append([end_a, curr_end])

            temp = new_temp  # Update the current working set with the adjusted intervals

        result.extend(temp)  # Add the adjusted intervals for the current b interval to the result

    return result
