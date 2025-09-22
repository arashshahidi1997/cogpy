import pytest
import numpy as np
import xarray as xr
import cogpy.utils.sliding as sl
from cogpy.utils.time_series import seconds_to_samples
import quantities as pq
import xarray as xr
from cogpy.datasets import load as ld
# import cartesian product for parametrize
from itertools import product

def _rolling_win_eager(
	xsig: xr.DataArray,
	window_size: int,
	window_step: int,
	dim: str = "time",
	window_dim: str = "window",
	min_periods: int = None,
):
	assert dim in xsig.dims, f"Dimension '{dim}' not found in input xarray."
	assert dim in xsig.coords, f"Dimension '{dim}' does not have coordinates in input xarray."
	assert window_dim not in xsig.dims, f"Dimension '{window_dim}' already exists in input xarray."

	if min_periods is None:
		min_periods = window_size

	# 1) rolling + construct with stride
	xroll = xsig.rolling({dim: window_size}, center=True, min_periods=min_periods)
	xwin = xroll.construct(window_dim, stride=window_step)  # <-- strided centers

	# 2) counts at every center, then align to the strided centers kept by construct
	counts_full = xroll.count(keep_attrs=False)
	# Align counts to the positions actually present in xwin[dim]
	counts = counts_full.sel({dim: xwin[dim]})

	# 3) build a 1-D validity mask along `dim` (AND over other dims if present)
	if counts.dims != (dim,):
		reduce_dims = tuple(d for d in counts.dims if d != dim)
		valid_1d = (counts >= min_periods).all(dim=reduce_dims)
	else:
		valid_1d = (counts >= min_periods)

	# 4) materialize mask (tiny) to avoid dask boolean indexing, then use integer isel
	if getattr(valid_1d, "chunks", None):
		valid_1d = valid_1d.compute()
	valid_idx = np.nonzero(valid_1d.values)[0]

	return xwin.isel({dim: xr.DataArray(valid_idx, dims=(dim,))})

def test_rolling_win_simple():
	# test 1
	xsig = xr.DataArray(np.arange(5), dims=['time'], coords={'time': np.arange(5)})
	xwin = sl.rolling_win(xsig, window_size=3, window_step=2, dim='time', window_dim='window')
	assert xwin.dims == ('time', 'window'), print(xwin.dims)
	assert np.array_equal(xwin['window'].values, np.array([0, 1, 2]))
	assert np.array_equal(xwin['time'].values, np.array([2]))
	assert np.array_equal(xwin.values, np.array([[1., 2., 3.]]))

def test_rolling_win_eager():
	# test 2
	N = 11
	xsig = xr.DataArray(np.arange(N), dims=['time'], coords={'time': np.linspace(0, 1, N)})
	xwin = _rolling_win_eager(xsig, window_size=3, window_step=1, dim='time', window_dim='window')
	expected = np.array([
		[0., 1., 2.],
		[1., 2., 3.],
		[2., 3., 4.],
		[3., 4., 5.],
		[4., 5., 6.],
		[5., 6., 7.],
		[6., 7., 8.],
		[7., 8., 9.],
		[8., 9., 10.]
	])
	assert np.allclose(xwin, expected, equal_nan=True)

	# test 3
	N = 11
	xsig = xr.DataArray(np.arange(N), dims=['time'], coords={'time': np.linspace(0, 1, N)}).chunk({'time': -1})
	xwin = _rolling_win_eager(xsig, window_size=3, window_step=1, dim='time', window_dim='window')
	expected = np.array([
		[0., 1., 2.],
		[1., 2., 3.],
		[2., 3., 4.],
		[3., 4., 5.],
		[4., 5., 6.],
		[5., 6., 7.],
		[6., 7., 8.],
		[7., 8., 9.],
		[8., 9., 10.]
	])
	assert np.allclose(xwin, expected, equal_nan=True)

@pytest.mark.parametrize("N, window_size, window_step", product(
	[5, 10, 11, 50],  # N
	[3, 4, 5, 7],     # window_size
	[1, 2, 3, 4]      # window_step
))
def test_rolling_win(N, window_size, window_step):
	# test 2
	N = 11
	xsig = xr.DataArray(np.arange(N), dims=['time'], coords={'time': np.linspace(0, 1, N)})
	xwin = sl.rolling_win(xsig, window_size=window_size, window_step=window_step, dim='time', window_dim='window')
	expected = _rolling_win_eager(xsig, window_size=window_size, window_step=window_step, dim='time', window_dim='window')
	assert np.allclose(xwin, expected, equal_nan=True)

@pytest.mark.parametrize("N, window_size, window_step", product(
	[5, 10, 11, 50],  # N
	[3, 4, 5, 7],     # window_size
	[1, 2, 3, 4]      # window_step
))
def test_rolling_win_chunked(N, window_size, window_step):
	xsig = xr.DataArray(np.arange(N), dims=['time'], coords={'time': np.linspace(0, 1, N)}).chunk({'time': -1})
	xwin = sl.rolling_win(xsig, window_size=window_size, window_step=window_step, dim='time', window_dim='window')
	expected = _rolling_win_eager(xsig, window_size=window_size, window_step=window_step, dim='time', window_dim='window')
	assert np.allclose(xwin, expected, equal_nan=True)

def test_running_measure():
	def running_corrcoef(xsig: xr.DataArray, fs: float, window_size: float, window_step:float, run_dim='time', corrcoef_dim='ch'):
		window_size_samples = seconds_to_samples(window_size, fs)
		window_step_samples = seconds_to_samples(window_step, fs)
		x_roll = sl.rolling_win(xsig, window_size=window_size_samples, window_step=window_step_samples, dim=run_dim)
		corr_da = xr.apply_ufunc(
			np.corrcoef, 
			x_roll, 
			input_core_dims=[[corrcoef_dim, 'window']],
			output_core_dims=[[corrcoef_dim+'1', corrcoef_dim+'2']],
			vectorize=True
		).rename(xsig.name + '_corrcoef')
		return corr_da

	xsig = ld.load_sample().compute().stack(ch=('AP', 'ML')).reset_index('ch')
	# xwin = xut.sl.rolling_win(xsig, window_size=128, window_step=64, dim='time').chunk({'AP':-1, 'ML':-1, 'window': -1})

	run_corrcoef1 = running_corrcoef(xsig, fs=xsig.fs, window_size=1*pq.s, window_step=0.1*pq.s, run_dim='time', corrcoef_dim='ch')
	xsig_flat = ld.load_sample().stack(ch=('AP', 'ML')).reset_index('ch')
	fs = xsig_flat.fs
	run_corrcoef = sl.running_measure(
		np.corrcoef, xsig_flat.compute(), fs=fs, 
		slider_kwargs=dict(window_size=seconds_to_samples(1*pq.s, fs), window_step=seconds_to_samples(0.1*pq.s, fs)), 
		measure_input_core_dims=[['ch', 'window']],
		measure_output_core_dims=[['ch1', 'ch2']],
		measure_output_sizes={'ch1': 256, 'ch2': 256},
		run_dim='time',
		window_dim='window',
		output_dtype=np.float64
	)
	assert np.allclose(run_corrcoef1, run_corrcoef), "sl.running_measure(np.corrcoef) mismatch."
