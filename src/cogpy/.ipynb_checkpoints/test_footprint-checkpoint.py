import pytest
from src import footprint as fp
# from src.datasets import load_sample

# sig = load_sample()

def test_rolling_window():
    #1D
    roll_win = fp._rolling_window(5, 15)
    assert roll_win == [[0,1,2,3,4], [3,4,5,6,7], [6,7,8,9,10], [9,10,11,12,13]]    