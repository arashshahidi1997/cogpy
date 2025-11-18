from cogpy.utils.convert import closest_power_of_two
import matplotlib.pyplot as plt
import numpy as np


def test_closest_power_of_two():
    x_ = [5, 14, 22, 256 * np.sqrt(2) + 1, 256 * np.sqrt(2) - 1]
    x_p2 = [closest_power_of_two(x_i) for x_i in x_]
    expected_output = [4, 16, 16, 512, 256]
    assert x_p2 == expected_output
