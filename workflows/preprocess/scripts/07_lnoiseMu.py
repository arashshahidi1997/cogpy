#!/usr/bin/env python
"""
Title: linenoiseMu.py
Status: WIP
Last Updated: 2025-09-15

Summary:
	Applies line noise removal using Multitaper Sinusoid regression.
"""

"""

%load_ext autoreload
%autoreload 2
import xarray as xr
from cogpy.preprocess.linenoise import LineNoiseEstimatorMultitaper
from cogpy.io import ecog_io

input_lnoise_ica = "interpolate/sample_raw_signal.zarr"
sigx_interp = ecog_io.from_zarr(input_lnoise_ica)['sigx_interp']

lnoiseMu_params = dict(
        linenoise_f0=50,
        halfbandwidth=4,
        nharmonics=2,
        ncomp=20
)

"""
import numpy as np
import xarray as xr
from cogpy.preprocess.linenoise import LineNoiseEstimatorICAArr
from cogpy.io import ecog_io

def _input(input_lnoise_ica):
	sigx_lnoise = ecog_io.from_zarr(input_lnoise_ica)['sigx']
	return sigx_lnoise

def estimate_mu_lnoise(sigx, lnoiseICA_params):
	sigx = sigx.compute()
	fs = sigx.fs
	# reshape to (time, ch)
	sigx = sigx.stack(ch=['AP', 'ML']).transpose('time', 'ch').reset_index('ch')

	# segment data into 5-min segments
	segment_size = int(5 * 60 * fs)
	sigx_coarsen = sigx.coarsen({'time': segment_size}, boundary='pad').construct(time=('segment', 'time'))

	# fit ICA on the first segment to initialize
	lnoise_init = LineNoiseEstimatorICAArr(fs, **lnoiseICA_params)
	X_init = sigx_coarsen.isel(segment=0).data
	lnoise_init.fit(X_init)

	# ICA estimator set up with initialization
	def _mu_lnoise(segment_data):
		# segment_data (time, ch)
		# detect NaN times
		nan_times = np.isnan(segment_data).any(axis=1) # (time,)
		# take the non-NaN times
		segment_data_nonan = segment_data[~nan_times]
		segment_lnoise_nonan = LineNoiseEstimatorICAArr(
				fs, 
				lnoise_estimator_init=lnoise_init,
				**lnoiseICA_params
			).fit_transform(segment_data_nonan)
	
		# append NaN times back
		segment_lnoise = np.full(segment_data.shape, np.nan)
		segment_lnoise[~nan_times] = segment_lnoise_nonan
		return segment_lnoise
	
	# apply ICA to each segment
	lnoise_estimate = xr.apply_ufunc(
		_ica_lnoise,
		sigx_coarsen,
		input_core_dims=[['time', 'ch']],
		output_core_dims=[['time', 'ch']],
		vectorize=True,
		dask='parallelized',
		keep_attrs=True,
		output_dtypes=[sigx.dtype]
	)

	# repatch segments
	lnoise_estimate = xr.concat([lnoise_estimate.isel(segment=i) for i in range(lnoise_estimate.sizes['segment'])], dim='time')
	
	# drop NaN times
	valid_time_idx_slice = slice(0, sigx.sizes['time'])
	lnoise_estimate = lnoise_estimate.isel(time=valid_time_idx_slice)
	assert np.allclose(np.isnan(lnoise_estimate.data), np.isnan(sigx.data)), "NaN times of linenoise estimate and original signal do not match!"
	return lnoise_estimate

def _output(lnoise_est, output_linenoise):
	lnoise_est.name = "sigx"
	ecog_io.to_zarr(output_linenoise, lnoise_est)

def main(input_lnoise_ica, output_linenoise, lnoiseICA_params):
	sigx_lnoise = _input(input_lnoise_ica)
	lnoise_est = estimate_mu_lnoise(sigx_lnoise, lnoiseICA_params)
	_output(lnoise_est, output_linenoise)

if __name__ == "__main__":
	# snakemake
	if 'snakemake' in globals():
		snakemake = globals()['snakemake']
		input_lnoise_ica = snakemake.input.noisy
		output_linenoise = snakemake.output.linenoise
		main(input_lnoise_ica, output_linenoise)
	else:
		raise RuntimeError("This script is intended to be run via Snakemake.")

