import pytest
import numpy as np
from cogpy.utils.curve import *


def test_find_elbow():
    ddx = [1, 2, 3, 6, 9, 8, 7, 4, 1]
    dx = np.cumsum(ddx)  # [0, 1, 3, 6, 12, 21, 29, 36, 40, 41]
    x = np.cumsum(dx)  # [0, 0, 1, 4, 10, 22, 43, 72, 108, 148, 189]
    output = find_elbow(x)
    expected_output = x[4]
    assert output == expected_output, f"Expected {expected_output}, but got {output}"

    knee = find_elbow(x, return_knee=True)
    expected_output_x = 4 + 1
    expected_output_y = x[4]
    assert (
        knee.knee_y == expected_output_y
    ), f"Expected {expected_output_y}, but got {knee.knee_y}"
    assert (
        knee.knee == expected_output_x
    ), f"Expected {expected_output_x}, but got {knee.knee}"
