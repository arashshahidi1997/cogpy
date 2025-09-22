import numpy as np
import xarray as xr
import dask.array as da
import pytest
from cogpy.utils.xarr import xdim_subsample_around, dim_dur_slice, axis_dim_from_xarr

def test_dim_dur_slice():
    # 0..3600s with high sampling rate
    time_vec = np.linspace(0.0, 3600.0, 10001)

    # Happy path: center at 50%, duration 600s -> [1500, 2100]
    s = dim_dur_slice(time_vec, time_fraction=0.5, duration=600.0)
    assert isinstance(s, slice)
    assert s.start == pytest.approx(1500.0)
    assert s.stop == pytest.approx(2100.0)

    # Edge clamp at start: center at 0%, duration 600s -> [0, 600]
    s0 = dim_dur_slice(time_vec, time_fraction=0.0, duration=600.0)
    assert s0.start == pytest.approx(0.0)
    assert s0.stop == pytest.approx(600.0)

    # Edge clamp at end: center at 100%, duration 600s -> [3000, 3600]
    s1 = dim_dur_slice(time_vec, time_fraction=1.0, duration=600.0)
    assert s1.start == pytest.approx(3000.0)
    assert s1.stop == pytest.approx(3600.0)

def test_dim_dur_slice_invalid_inputs():
    # Unsorted time vector should raise
    unsorted_time = np.array([0.0, 2.0, 1.0, 3.0])
    with pytest.raises(ValueError, match="must be sorted"):
        dim_dur_slice(unsorted_time, time_fraction=0.5, duration=1.0)

    # Duration exceeding total span should raise
    time_vec = np.array([0.0, 10.0])
    with pytest.raises(ValueError, match="Duration exceeds total time span"):
        dim_dur_slice(time_vec, time_fraction=0.5, duration=20.0)

def test_sampling():
	test_x = xr.DataArray(np.arange(20), dims='t', coords={'t': np.arange(20)})
	it = test_x.get_index('t').get_indexer([11], method="nearest")[0]
	sample_win = np.arange(-2, 2) * 2
	tsamples = it + sample_win
	assert np.all(tsamples == np.array([7,9,11,13]))

	sample_win = np.arange(-3, 3) * 2
	tsamples = it + sample_win
	# clip to valid range
	tsamples = tsamples[(tsamples >= 0) & (tsamples < test_x.sizes['t'])]
	assert np.all(tsamples == np.array([5,7,9,11,13,15]))

	# write using xdim_subsample_around
	tsamples = xdim_subsample_around(test_x, dim='t', center=11, nsample=4, step=2, clip=True)
	assert np.array_equal(tsamples, np.array([7,9,11,13]))

	tsamples = xdim_subsample_around(test_x, dim='t', center=11, nsample=7, step=2, clip=True)
	
	assert np.array_equal(tsamples, np.array([5, 7, 9, 11, 13, 15, 17])), print(tsamples)

def test_axis_dim_from_xarr():
    x = xr.DataArray(np.random.randn(10, 10), dims=['time', 'space'])
    axis, dim = axis_dim_from_xarr(x, axis=0)
    assert axis == 0
    assert dim == 'time'
    axis, dim = axis_dim_from_xarr(x, dim='space')
    assert axis == 1
    assert dim == 'space'
    axis, dim = axis_dim_from_xarr(x, axis=0, dim='space')
    assert axis == 1
    assert dim == 'space'
    print('All tests passed.')

