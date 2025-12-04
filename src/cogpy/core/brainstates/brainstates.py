"""Short summary of the module.
----------------------------
This module provides utilities for mapping numerical values to interval-based states, organizing DataFrame columns into state labels, validating interval structures, and analyzing state transitions and durations.

Status
------
WIP

Metadata
--------
Author : Arash Shahidi <A.Shahidi@campus.lmu.de>
Last Updated : 2025-08-26

Extended description of the module.
----------------------------------------------
The module focuses on interval-based state classification and transition analysis.  
It offers core tools for verifying disjoint intervals, mapping time series or numerical values to state definitions, computing state durations, handling micro- and macro-state relationships, and converting state dictionaries into structured DataFrames.  
Testing utilities are included to ensure correct mapping and labeling behavior.

Notes
-----
- State intervals are defined as `[start, end]` pairs in lists, tuples, or arrays.  
- Functions assume state labels are strings.  
- Many utilities operate on DataFrames containing time or numeric columns to be classified into state intervals.  
- Mapping functions return DataFrames that encode state membership using period indices or `-1` for non-membership.  
- The module includes tools for cleaning corrupt intervals and subtracting microstates from macrostates to produce purified state sets.  
- Transition analysis functions generate DataFrames describing when and how states change over time.

See Also
--------
cogpy.brainstates.intervals : Tools for signal segmentation, event detection, or additional state-processing utilities.

Examples
--------
"""


import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm
from typing import List
from .intervals import (
    check_intervals_disjoint,
    map_numbers_to_intervals,
    map_numbers_to_disjoint_intervals,
    subtract_intervals,
)


# %% DataFrame Operations for State Sorting and Labeling:
def get_states_df(states: dict) -> pd.DataFrame:
    """
    Converts a dictionary of state intervals to a DataFrame with state edges.

    Parameters
    ----------
    states : dict
        Dictionary of state intervals with state labels as keys and array of [start, end] intervals as values.
        E.g., {'PerSWS': array([[start1, end1], [start2, end2], ...]), 'PerREM': array([[start1, end1], [start2, end2], ...]), ...}

    Returns
    -------
    pd.DataFrame
        A DataFrame with rows corresponding to state intervals and columns for state, interval number, start time, and end time.

    Note
    ----
    The function first sorts the intervals based on start time and then assigns an interval number to each interval.

    Example
    -------
    states = {'PerSWS': [[10, 20], [30, 40]], 'PerREM': [[20, 30], [40, 50]]}
    get_states_df(states)

    Output:
    state	iseg	t0	t1
    0	PerSWS	0	10	20
    1	PerREM	0	20	30
    2	PerSWS	1	30	40
    3	PerREM	1	40	50
    """
    states_df = []
    for state, state_timesegs in states.items():
        for iseg, (t0, t1) in enumerate(state_timesegs):
            states_df.append({"state": state, "iseg": iseg, "t0": t0, "t1": t1})
    states_df = pd.DataFrame(states_df).sort_values("t0").reset_index(drop=True)
    return states_df


def label_numbers_by_state_intervals(
    tarr: np.ndarray, states: dict, progress_bar=True
) -> pd.DataFrame:
    """
    Parameters
    ----------
    tarr : np.ndarray
        Array of time values to be labeled by state intervals.

    states : dict
        Dictionary of state intervals with state labels as keys and array of [start, end] intervals as values.
        E.g., {'PerSWS': array([[start1, end1], [start2, end2], ...]), 'PerREM': array([[start1, end1], [start2, end2], ...]), ...}

    Returns
    -------
    pd.DataFrame
        A DataFrame with rows corresponding to `tarr` and columns for each state, filled with period index if the time value falls into the state interval
        and with -1 if it does not.

    Example
    -------
    tarr = np.arange(0, 100, 5)
    states = {'PerSWS': [[10, 20], [30, 40]], 'PerREM': [[20, 30], [40, 50]]}
    label_numbers_by_state_intervals(tarr, states)

    Output:
    PerSWS	PerREM
    0	-1	-1
    1	0	-1
    2	0	-1
    3	0	-1
    4	0	0
    5	-1	0
    6	1	0
    7	1	-1
    8	1	1
    9	-1	1
    10	-1	1
    11	-1	-1
    12	-1	-1
    13	-1	-1
    14	-1	-1
    15	-1	-1
    16	-1	-1
    17	-1	-1
    18	-1	-1
    19	-1	-1
    """
    tmap_df = {}
    iterator = (
        tqdm(states.items(), desc="Processing States")
        if progress_bar
        else states.items()
    )
    for state, state_intervals in iterator:
        tmap = map_numbers_to_disjoint_intervals(tarr, state_intervals)
        tmap_df[state] = tmap
    return pd.DataFrame(tmap_df)


def sort_col_into_states(
    df: pd.DataFrame, col: str, states: dict, progress_bar=True
) -> pd.DataFrame:
    """
    Sorts a column of a DataFrame into predefined state intervals and returns a DataFrame

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the column to be sorted into states.

    col : str
        Column name to be sorted into states. e.g. time

    states : dict
        Dictionary of state intervals with state labels as keys and array of [start, end] intervals as values.
        E.g., {'PerSWS': array([[start1, end1], [start2, end2], ...]), 'PerREM': array([[start1, end1], [start2, end2], ...]), ...}

    Returns
    -------
    pd.DataFrame
        A DataFrame with rows corresponding to `df` and columns for each state, filled with intervals
        indicating the state each time value falls into. If a time value does not fall into any interval of
        a particular state, it is indicated with a 0 in the respective state's column.
    """
    # raise warning if keys of states are already columns in the dataframe
    if any(state in df.columns for state in states.keys()):
        warnings.warn(
            "One or more state labels are already columns in the dataframe. The result will have duplicate column names.\
                        Consider changing the column names or the keys of the states by adding a prefix."
        )
    t_df = label_numbers_by_state_intervals(
        df[col].values, states, progress_bar=progress_bar
    )
    t_df.index = df.index
    df = pd.concat([df, t_df], axis=1)
    return df


def filter_by_states(
    df_labeled_states, include_states, exclude_states, return_complement=False
):
    """
    Filters a DataFrame based on the presence of certain states and the absence of others.

    Parameters
    ----------
    df_labeled_states : pd.DataFrame
        DataFrame containing columns for each state, filled with period index if the time value falls into the state interval
          and with -1 if it does not.

    include_states : list
        List of state column names to be included in the filtered DataFrame.

    exclude_states : list
        List of state column names to be excluded from the filtered DataFrame.

    return_complement : bool, optional
        If True, returns the complement of the filtered DataFrame as well. Default is False.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing only the rows that satisfy the inclusion and exclusion criteria.

    pd.DataFrame
        If `return_complement` is True, returns a tuple of two DataFrames: the first containing the rows that satisfy the inclusion and exclusion criteria,
          and the second containing the rows that do not satisfy the criteria.
    """
    cnd_map = lambda row: all(row[state] == -1 for state in exclude_states) & any(
        row[state] != -1 for state in include_states
    )
    condition = df_labeled_states.apply(cnd_map, axis=1)
    df_state = df_labeled_states[condition]
    if return_complement:
        df_complement = df_labeled_states[~condition]
        return df_state, df_complement
    return df_state


def get_exclusive_state_df(rec_df, col, state, states):
    """
    Parameters
    ----------
    rec_df : pd.DataFrame
        event dataframe with col to sort into states

    col : str
        usually 'time' or 'peak_abs' to sort into states

    state : str
        state to include

    states : dict
        dict of states intervals

    Returns
    -------
    rec_df_ : pd.DataFrame
        Dataframe with a multiindex of state periods and idx of event
        rec_df filtered by state
    """
    # Filter by states
    rec_df_ = sort_col_into_states(rec_df, col, states)
    exclude_states = [state_ for state_ in states.keys() if state_ != state]
    rec_df_ = filter_by_states(
        rec_df_, include_states=[state], exclude_states=exclude_states
    )
    # drop excluded states
    rec_df_ = rec_df_.drop(columns=exclude_states)
    rec_df_ = (
        rec_df_.sort_values(col)
        .reset_index(drop=True)
        .reset_index()
        .rename(columns={"index": "idx"})
        .set_index([state, "idx"])
    )
    return rec_df_


# %% State Transition and Event Analysis Functions:
def check_disjoint_states(states: dict) -> bool:
    """
    Checks if the state intervals in a dictionary are non-overlapping.

    Parameters
    ----------
    states : dict
        Dictionary of state intervals with state labels as keys and array of [start, end] intervals as values.
        E.g., {'PerSWS': array([[start1, end1], [start2, end2], ...]), 'PerREM': array([[start1, end1], [start2, end2], ...]), ...}

    Returns
    -------
    overlap_df: 2d bool array
        True if the state intervals overlap, False otherwise. rows and columns correspond to the states in the input dictionary.

    Example
    -------
    states = {'PerSWS': [[10, 20], [25, 30]], 'PerREM': [[20, 25], [30, 35]]}
    check_disjoint_states(states)

    Output:
        array([[True,  False],
            [ False, True]])
    """
    overlap_arr = np.zeros((len(states), len(states)), dtype=bool)
    for i, (_, intervals1) in enumerate(states.items()):
        for j, (_, intervals2) in enumerate(states.items()):
            # merge intervals into one list
            if i != j:
                intervals2 = np.concatenate([intervals1, intervals2])
            overlap_arr[i, j] = check_intervals_disjoint(intervals2)

    # convert to DataFrame
    overlap_df = pd.DataFrame(overlap_arr, index=states.keys(), columns=states.keys())
    return overlap_df


def state_transitions(states: dict, next_state_: bool = True) -> pd.DataFrame:
    """
    Converts a dictionary of state intervals to a DataFrame with state transitions.

    Parameters
    ----------
    states : dict
        Dictionary of state intervals with state labels as keys and array of [start, end] intervals as values.
        E.g., {'PerSWS': array([[start1, end1], [start2, end2], ...]), 'PerREM': array([[start1, end1], [start2, end2], ...]), ...}

    next_state_ : bool, optional
        If True, returns transitions to next state. If False, returns transitions from previous state. Default is True.

    Returns
    -------
    pd.DataFrame
        A DataFrame with rows corresponding to state transitions and columns for transition time, previous state, and next state.
    """
    # Extract and sort edge values
    state_times_df = get_states_df(states)

    edges = []
    n = state_times_df.shape[0]
    state_series = state_times_df["state"]
    t0_series = state_times_df["t0"]
    t1_series = state_times_df["t1"]
    for irow in range(n):
        # Get the current state and its time boundaries
        current_state = state_series[irow]
        start_time = t0_series[irow]
        end_time = t1_series[irow]
        # Define previous and next states
        if irow == 0:
            prev_state = current_state  # First row, no previous state change
        else:
            prev_state = state_series[irow - 1]

        if irow == n - 1:
            next_state = current_state  # Last row, no next state change
        else:
            next_state = state_series[irow + 1]

        # Append the transitions
        if next_state_:
            edges.append(
                {
                    "transition_time": end_time,
                    "prev_state": current_state,
                    "next_state": next_state,
                }
            )
        else:
            edges.append(
                {
                    "transition_time": start_time,
                    "prev_state": prev_state,
                    "next_state": current_state,
                }
            )

    # Sort by transition time
    edges_df = pd.DataFrame(edges).sort_values("transition_time")

    # Drop rows with the same prev_state and next_state
    edges_df = edges_df[edges_df["prev_state"] != edges_df["next_state"]]

    return edges_df


def state_transition_interval(
    states, start_state, end_state, window_before, window_after
):
    """
    Detects transitions between specified start and end states and creates intervals around the transition times.

    Parameters
    ----------
    states : dict
        Dictionary of state intervals with state labels as keys and lists of [start, end] intervals as values.
    start_state : str
        The state from which transitions are being detected.
    end_state : str
        The state to which transitions are being detected.
    window_before : int
        Duration of the window before the transition time.
    window_after : int
        Duration of the window after the transition time.

    Returns
    -------
    np.array
        Array of [start, end] intervals around the transition times.
    """
    df = state_transitions(states)

    mask = (df["prev_state"].astype(str) == start_state) & (
        df["next_state"].astype(str) == end_state
    )

    t = df.loc[mask, "transition_time"].astype(int)
    return [[ti - window_before, ti + window_after] for ti in t]


def append_transition_intervals(
    states, start_state, end_state, window_before=30, window_after=0
):
    """
    Appends intervals around transitions between specified states to the dictionary of states.

    Parameters
    ----------
    states : dict
        Dictionary of state intervals with state labels as keys and array of [start, end] intervals as values.
    start_state : str
        The state from which transitions are being detected.
    end_state : str
        The state to which transitions are being detected.
    window_before : int, optional
        Duration of the window before the transition time. Default is 30.
    window_after : int, optional
        Duration of the window after the transition time. Default is 0.

    Returns
    -------
    dict
        Updated dictionary of state intervals with new transition intervals appended.
    """
    # Detect specified state transitions and create intervals around them
    transition_intervals = state_transition_interval(
        states, start_state, end_state, window_before, window_after
    )
    transition_label = f"{start_state}_To_{end_state}"
    states_with_transitions = {**states, transition_label: transition_intervals}
    return states_with_transitions


def append_intermediate_sleep_intervals(states, IS_dur=30):
    """
    IS_dur = 30 # transition window size

    Parameters
    ----------
    states : dict
        Dictionary of state intervals with state labels as keys and array of [start, end] intervals as values.
        E.g., {'PerSWS': array([[start1, end1], [start2, end2], ...]), 'PerREM': array([[start1, end1], [start2, end2], ...]), ...}

    IS_dur : int, optional
        Duration of intermediate sleep (IS) state. Default is 30.

    Returns
    -------
    dict
        Dictionary of state intervals with intermediate sleep (PerIS) intervals appended.
    """
    # separate microstates from macrostates
    # states_micro = {k: v for k, v in states.items() if k in ['PerMicroA', 'PerHVS']}
    states_macro = {k: v for k, v in states.items() if k not in ["PerMicroA", "PerHVS"]}

    # detect SWS-REM (intermediate sleep) transitions
    state_df = state_transitions(states_macro, next_state_=False)
    IS_transitions = state_df[
        (state_df["prev_state"] == "PerSWS") & (state_df["next_state"] == "PerREM")
    ]
    IS_intervals = [
        [t_end - IS_dur, t_end] for t_end in IS_transitions.transition_time.values
    ]
    states_plus_IS = {**states, "PerIS": IS_intervals}
    return states_plus_IS


# %% state properties
def get_state_durations(states_df: pd.DataFrame) -> pd.Series:
    """
    Calculates the total duration of each state from the state intervals.

    Parameters
    ----------
    states_df : pd.DataFrame
        DataFrame with rows corresponding to state intervals and columns for state, interval number, start time, and end time.

    Returns
    -------
    pd.Series
        Series with state labels as index and total duration of each state as values.
    """
    # Explicitly selecting 't0' and 't1' columns after grouping, returning a Series
    durations = states_df.groupby("state", sort=False)[["t0", "t1"]].apply(
        lambda x: (x["t1"] - x["t0"]).sum()
    )

    return durations


# %% clean
def drop_corrupt_intervals(states: dict) -> dict:
    """
    Drops intervals with end time less than or equal to start time.

    Parameters
    ----------
    states : dict
        Dictionary of state intervals with state labels as keys and array of [start, end] intervals as values.

    Returns
    -------
    dict
        Dictionary of state intervals with corrupt intervals removed.
    """
    states_clean = {}
    for state, intervals in states.items():
        intervals_clean = [
            interval for interval in intervals if interval[1] > interval[0]
        ]
        states_clean[state] = intervals_clean
    return states_clean


def drop_micro_states(
    macro_state_intervals: np.ndarray, micro_state_intervals: np.ndarray
) -> np.ndarray:
    """
    Drops microstate intervals from macrostate intervals.

    Parameters
    ----------
    macro_state_intervals : np.ndarray
        List of [start, end] intervals for macrostates.

    micro_state_intervals : np.ndarray
        List of [start, end] intervals for microstates.

    Returns
    -------
    List
        List of [start, end] intervals for macrostates with microstate intervals removed.

    Example
    -------
    macro_state_intervals = [[10, 20], [30, 40], [50, 60]]
    micro_state_intervals = [[15, 23], [35, 37], [55, 57]]
    drop_micro_states(macro_state_intervals, micro_state_intervals)

    Output: [[10, 15], [30, 35], [37, 40], [50, 55], [57, 60]]
    """
    # sort macro into micro
    df_macro = pd.DataFrame(macro_state_intervals, columns=["t_start", "t_end"])
    df_macro = sort_col_into_states(
        df_macro, "t_start", {"micro_start": micro_state_intervals}, progress_bar=False
    )
    df_macro = sort_col_into_states(
        df_macro, "t_end", {"micro_end": micro_state_intervals}, progress_bar=False
    )
    # find cases where both micro_start and micro_end are the same but not -1
    macro_within_micro = df_macro[
        (df_macro["micro_start"] != -1)
        & (df_macro["micro_end"] != -1)
        & (df_macro["micro_start"] == df_macro["micro_end"])
    ]
    df_macro = df_macro.drop(macro_within_micro.index)
    # convert to numpy
    macro_state_intervals = df_macro[["t_start", "t_end"]].to_numpy()

    df = pd.DataFrame(micro_state_intervals, columns=["t_start", "t_end"])
    df_ = sort_col_into_states(
        df, "t_start", {"macro_start": macro_state_intervals}, progress_bar=False
    )
    df_micro = sort_col_into_states(
        df_, "t_end", {"macro_end": macro_state_intervals}, progress_bar=False
    )

    def _check_start_end_state(row):
        if row["macro_start"] == row["macro_end"]:
            return row["macro_start"]
        elif row["macro_start"] != -1:
            return row["macro_start"]
        elif row["macro_end"] != -1:
            return row["macro_end"]
        else:
            return -1

    df_micro["macro"] = df_micro.apply(_check_start_end_state, axis=1)
    df_micro_groupby_macro = df_micro.groupby("macro")
    split_macro_state_intervals = []
    for macro_state_idx, macro_interval in enumerate(macro_state_intervals):
        try:
            df_micro = df_micro_groupby_macro.get_group(macro_state_idx)
        except KeyError:
            split_macro_state_intervals.append(macro_interval)
            continue

        # get the row with macro_start == -1 and macro_end != -1, if there are any
        start_micro = df_micro[
            (df_micro["macro_start"] == -1) & (df_micro["macro_end"] != -1)
        ]
        if not start_micro.empty:
            macro_interval[0] = start_micro["t_end"].values[0]
        # get the row with macro_start != -1 and macro_end == -1, if there are any
        end_micro = df_micro[
            (df_micro["macro_start"] != -1) & (df_micro["macro_end"] == -1)
        ]
        if not end_micro.empty:
            macro_interval[1] = end_micro["t_start"].values[0]
        # get the rows with macro_start != -1 and macro_end != -1
        df_micro_within = df_micro[
            (df_micro["macro_start"] != -1) & (df_micro["macro_end"] != -1)
        ]
        micro_edges = df_micro_within[["t_start", "t_end"]].to_numpy().reshape(-1)

        macro_split_edges = np.zeros(1 + len(micro_edges) + 1)
        macro_split_edges[0] = macro_interval[0]
        macro_split_edges[-1] = macro_interval[1]
        if micro_edges.size > 0:
            macro_split_edges[1:-1] = micro_edges
        macro_split_intervals = list(macro_split_edges.reshape(-1, 2))
        split_macro_state_intervals = (
            split_macro_state_intervals + macro_split_intervals
        )
    return [list(interval_) for interval_ in split_macro_state_intervals]


def purify_macro_states(
    states: dict, macro_states: List[str], micro_states: List[str]
) -> dict:
    """
    Purifies the state intervals by removing microstate intervals from macrostate intervals.

    Parameters
    ----------
    trange : np.ndarray
        Array of time values to be labeled by state intervals.

    states : dict
        Dictionary of state intervals with state labels as keys and array of [start, end] intervals as values.
        E.g., {'PerSWS': array([[start1, end1], [start2, end2], ...]), 'PerREM': array([[start1, end1], [start2, end2], ...]), ...}

    macro_states : list
        List of macro state labels

    micro_states : list, optional
        List of micro state labels. Default is None.

    Returns
    -------
    dict
        Dictionary with state labels as keys and array of [start, end] intervals as values.
    """
    assert all(
        macro_state in states for macro_state in macro_states
    ), "All macro states should be present in the states dictionary"
    assert all(
        micro_state in states for micro_state in micro_states
    ), "All micro states should be present in the states dictionary"
    assert (
        len(set(macro_states).intersection(set(micro_states))) == 0
    ), "Macro states and micro states should be disjoint"

    states_purified = {}
    for macro_state in macro_states:
        macro_intervals = states[macro_state]
        for micro_state in micro_states:
            macro_intervals = subtract_intervals(states[micro_state], macro_intervals)
        states_purified[macro_state] = macro_intervals
    # append micro states
    for state in micro_states:
        states_purified[state] = states[state]
    return states_purified
