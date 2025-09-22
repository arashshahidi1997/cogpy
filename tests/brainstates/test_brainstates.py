import pytest
import numpy as np
import pandas as pd
import pandas.testing as pdt
from cogpy.brainstates.brainstates import *
import warnings

# %% Utility and Testing Functions
def test_check_intervals_disjoint():
    assert check_intervals_disjoint([[1, 3], [4, 6], [7, 9]]) == True
    assert check_intervals_disjoint([[1, 3], [2, 4], [3, 5]]) == False

def test_map_numbers_to_intervals():
    numbers = [1, 2, 3, 4, 5]
    intervals = [[1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
    expectesd_map = {0: [0], 1: [0, 1], 2: [1, 2], 3: [2, 3], 4: [3, 4]}
    output_map = map_numbers_to_intervals(numbers, intervals)
    assert output_map == expectesd_map, f"Mapping of numbers to intervals is incorrect. Expected: {expectesd_map}, Got: {output_map}"

def test_map_numbers_to_disjoint_intervals():
    numbers = [1, 2, 3, 4, 5]
    intervals = [[1, 3], [4, 6]]
    output_map = map_numbers_to_disjoint_intervals(numbers, intervals)
    expected_map = [0, 0, -1, 1, 1]
    assert np.array_equal(output_map, expected_map), f"Mapping of numbers to disjoint intervals is incorrect. Expected: {expected_map}, Got: {output_map}"

def test_subtract_intervals():
    a = [[5, 8], [18, 22]]
    b = [[1, 10], [15, 20], [25, 30]]
    output = subtract_intervals(a, b)
    expected_output = [[1, 5], [8, 10], [15, 18], [25, 30]]
    assert output == expected_output, f"Subtraction of intervals is incorrect. Expected: {expected_output}, Got: {output}"

def test_get_state_durations():
    states = {'PerSWS': [[10, 20], [30, 40]], 'PerREM': [[20, 30], [40, 45]]}
    states_df = get_states_df(states)
    output = get_state_durations(states_df)
    expected_output = pd.Series({'PerSWS': 20, 'PerREM': 15})
    expected_output.index.name = 'state'
    try:
        pdt.assert_series_equal(output, expected_output)
    except:
        pytest.fail(f"Getting state durations is incorrect. Expected: {expected_output}, Got: {output}")

def test_get_states_df():
    states = {'PerSWS': [[10, 20], [30, 40]], 'PerREM': [[20, 30], [40, 50]]}
    states_df = get_states_df(states)
    assert states_df.shape == (4, 4)
    assert states_df.columns.tolist() == ['state', 'iseg', 't0', 't1']
    assert states_df['state'].tolist() == ['PerSWS', 'PerREM', 'PerSWS', 'PerREM']
    assert states_df['iseg'].tolist() == [0, 0, 1, 1]
    assert states_df['t0'].tolist() == [10, 20, 30, 40]
    assert states_df['t1'].tolist() == [20, 30, 40, 50]

def test_label_numbers_by_state_intervals():
    tarr = np.arange(0, 100, 5)
    states = {'PerSWS': [[1, 20], [30, 40]], 'PerREM': [[20, 30], [40, 50]]}
    sws_ = [-1,  0,  0,  0,  -1, -1,  1,  1,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    rem_ = [-1, -1, -1, -1,  0,  0,  -1, -1,  1,  1,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    df_expected_out = pd.DataFrame({'PerSWS': sws_, 'PerREM': rem_})
    df_out = label_numbers_by_state_intervals(tarr, states)
    assert df_out.shape == (20, 2)
    pdt.assert_frame_equal(df_out, df_expected_out)

def test_sort_col_into_states():
    df = pd.DataFrame({'time': np.arange(5, 55, 10)})
    # 5, 15, 25, 35, 45
    states = {'PerSWS': [[10, 20], [30, 40]], 'PerREM': [[20, 30], [40, 50]]}
    df_labeled_states = sort_col_into_states(df, 'time', states)
    expected_df = pd.DataFrame({'time': np.arange(5, 55, 10), 'PerSWS': [-1, 0, -1, 1, -1], 'PerREM': [-1, -1, 0, -1, 1]})
    try:
        pdt.assert_frame_equal(df_labeled_states, expected_df)
    except:
        pytest.fail(f"Sorting column into states is incorrect. Expected: {expected_df}, Got: {df_labeled_states}")

def test_filter_by_states():
    df = pd.DataFrame({'time': np.arange(5, 55, 10)})
    states = {'PerSWS': [[10, 20], [30, 40]], 'PerREM': [[20, 30], [40, 50]]}
    df = sort_col_into_states(df, 'time', states)
    df_state = filter_by_states(df, include_states=['PerSWS'], exclude_states=['PerREM'])
    expected_df = pd.DataFrame({'time': [15, 35], 'PerSWS': [0, 1], 'PerREM': [-1, -1]}, index=[1, 3])
    try:
        pdt.assert_frame_equal(df_state, expected_df)
    except:
        pytest.fail(f"Filtering by states is incorrect. Expected: {expected_df}, Got: {df_state}")

def test_get_exclusive_state_df():
    rec_df = pd.DataFrame({'time': np.arange(5, 55, 10)})
    states = {'PerSWS': [[10, 20], [30, 40]], 'PerREM': [[20, 30], [40, 50]]}
    rec_df_ = get_exclusive_state_df(rec_df, 'time', 'PerSWS', states)
    expected_df = pd.DataFrame({'time': [15, 35], 'PerSWS': [0, 1]})
    expected_df = expected_df.sort_values('time').reset_index(drop=True).reset_index().rename(columns={'index': 'idx'}).set_index(['PerSWS', 'idx'])
    try:
        pdt.assert_frame_equal(rec_df_, expected_df)
    except:
        pytest.fail(f"Filtering by exclusive state is incorrect. Expected: {expected_df}, Got: {rec_df_}")

def test_check_disjoint_states():
    states = {'PerSWS': [[10, 20], [30, 35]], 'PerREM': [[20, 30], [35, 40]]}
    output = check_disjoint_states(states)
    expected_output = pd.DataFrame([[True, True], [True, True]], index=['PerSWS', 'PerREM'], columns=['PerSWS', 'PerREM'])
    try:
        pdt.assert_frame_equal(output, expected_output)
    except:
        pytest.fail(f"Checking disjoint states is incorrect. Expected: {expected_output}, Got: {output}")

    # example with overlapping intervals
    states = {'PerSWS': [[10, 20], [25, 30]], 'PerREM': [[20, 25], [30, 35]], 'PerIS': [[18, 19]]}
    output = check_disjoint_states(states)
    expected_output = pd.DataFrame([[True, True, False], [True, True, True], [False, True, True]], index=['PerSWS', 'PerREM', 'PerIS'], columns=['PerSWS', 'PerREM', 'PerIS'])
    try:
        pdt.assert_frame_equal(output, expected_output)
    except:
        pytest.fail(f"Checking disjoint states is incorrect. Expected: {expected_output}, Got: {output}")

def test_state_transitions():
    states = {'PerSWS': [[10, 20], [30, 40]], 'PerREM': [[20, 30], [40, 50]]}
    output = state_transitions(states)

    expected_output = pd.DataFrame(
        {
            'transition_time': [20, 30, 40],
            'prev_state': ['PerSWS', 'PerREM', 'PerSWS'],
            'next_state': ['PerREM', 'PerSWS', 'PerREM'],
        }
    )

    try:
        pdt.assert_frame_equal(output, expected_output)
    except:
        pytest.fail(f"State transitions are incorrect. Expected: {expected_output}, Got: {output}")

def test_state_transitions_interval():
    states = {'PerSWS': [[10, 20], [30, 40]], 'PerREM': [[20, 30], [40, 50]]}
    start_state = 'PerSWS'
    end_state = 'PerREM'
    window_before = 5
    window_after = 5
    output = state_transition_interval(states, start_state, end_state, window_before, window_after)
    expected_output = [[15, 25], [35, 45]]
    assert output == expected_output, f"State transitions are incorrect. Expected: {expected_output}, Got: {output}"

def test_append_transition_intervals():
    states = {'PerSWS': [[10, 20], [30, 40]], 'PerREM': [[20, 30], [40, 50]]}
    output = append_transition_intervals(states, 'PerSWS', 'PerREM', window_before=5, window_after=5)
    expected_output = {'PerSWS': [[10, 20], [30, 40]], 'PerREM': [[20, 30], [40, 50]], 'PerSWS_To_PerREM': [[15, 25], [35, 45]]}
    assert output == expected_output, f"Appending transition intervals is incorrect. Expected: {expected_output}, Got: {output}"

def test_append_intermediate_sleep_intervals():
    states = {'PerSWS': [[10, 20], [30, 40]], 'PerREM': [[20, 30], [40, 50]]}
    output = append_intermediate_sleep_intervals(states, IS_dur=5)
    expected_output = {'PerSWS': [[10, 20], [30, 40]], 'PerREM': [[20, 30], [40, 50]], 'PerIS': [[15, 20], [35, 40]]}
    assert output == expected_output, f"Appending intermediate sleep intervals is incorrect. Expected: {expected_output}, Got: {output}"

def test_drop_corrupt_intervals():
    states = {'PerSWS': [[10, 20], [30, 40], [40, 30]], 'PerREM': [[20, 30], [40, 50]]}
    output = drop_corrupt_intervals(states)
    expected_output = {'PerSWS': [[10, 20], [30, 40]], 'PerREM': [[20, 30], [40, 50]]}
    assert output == expected_output, f"Dropping corrupt intervals is incorrect. Expected: {expected_output}, Got: {output}"

def test_drop_micro_states():
    macro_state_intervals = [[10, 20], [30, 40], [50, 60]]
    micro_state_intervals = [[15, 23], [35, 37], [55, 57]]
    output = drop_micro_states(macro_state_intervals, micro_state_intervals)
    expected_output = [[10, 15], [30, 35], [37, 40], [50, 55], [57, 60]]
    assert output == expected_output, f"Dropping micro states is incorrect. Expected: {expected_output}, Got: {output}"

def test_purify_macro_states():
    states = {'PerSWS': [[10, 20], [30, 40], [50, 60]], 'PerREM': [[20, 30], [40, 50]], 'PerMicroA': [[15, 17], [35, 37]], 'PerHVS': [[55, 57]]}
    macro_states = ['PerSWS', 'PerREM']
    micro_states = ['PerMicroA', 'PerHVS']
    output = purify_macro_states(states, macro_states, micro_states)
    expected_output = {'PerSWS': [[10, 15], [17, 20], [30, 35], [37, 40], [50, 55], [57, 60]], 'PerREM': [[20, 30], [40, 50]], 'PerMicroA': [[15, 17], [35, 37]], 'PerHVS': [[55, 57]]}
    assert output == expected_output, f"Purifying macro states is incorrect. Expected: {expected_output}, Got: {output}"
    