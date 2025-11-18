import numpy as np
from cogpy.preprocess.interpolate import interpolate_bads


# %% test
def test_interp():
    data = np.ones((3, 3, 1)) * np.array([1, 10, 3, 96]).reshape(1, 1, -1)
    nan_mask_ = np.zeros((3, 3), dtype=bool)
    nan_mask_[1, 1] = True
    nan_mask_[0, 0] = True
    idata = interpolate_bads(data, skip=nan_mask_, method="linear", gridshape=(3, 3))
    assert np.all(idata == data), print(idata)
    print("interpolation test passed")
