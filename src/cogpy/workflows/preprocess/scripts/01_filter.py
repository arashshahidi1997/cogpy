#!/usr/bin/env python
"""
Title: 01_lowpass.py
Status: REVIEW
Last Updated: 2025-09-09

Summary:
	Applies a lowpass filter to the ECoG data.
"""
import xarray as xr
from scipy import signal
from cogpy.io import ecog_io

def _input(input_raw):
	return ecog_io.from_zarr(input_raw)['sigx']

def _output(sigx_filtered, output_filtered):
	sigx_filtered.name = "sigx"
	ecog_io.to_zarr(output_filtered, sigx_filtered)
	print(f"Lowpass filtered data saved to\n\t{output_filtered}")
	
def butter_filter(sigx, order, f_cutoff, btype):	
	ecog_io.assert_ecog(sigx)
	fs = sigx.fs

	def _filter(arr):
		b, a = signal.butter(order, f_cutoff, btype=btype, output='ba', fs=fs)
		return signal.filtfilt(b, a, arr, axis=-1)

	sigx = sigx.chunk({'time': -1, 'AP': 1, 'ML': 1})

	sigx_filtered = xr.apply_ufunc(
		_filter,
		sigx,
		input_core_dims=[['time']],
		output_core_dims=[['time']],
		dask='parallelized',
		output_dtypes=[sigx.dtype],
		keep_attrs=True,
	)

	return sigx_filtered

def main(input_raw, output_filtered, filt_params):
	sigx = _input(input_raw)
	sigx_filtered = butter_filter(sigx, **filt_params)
	_output(sigx_filtered, output_filtered)

if __name__ == "__main__":
	if 'snakemake' in globals():
		snakemake = globals()['snakemake']
		
		# io
		input_raw = snakemake.input.raw
		filt_params = dict(
			order = snakemake.params.order,
			f_cutoff = snakemake.params.cutoff,
			btype = snakemake.params.btype
		)
		output_filtered = snakemake.output.filtered

		# main
		main(input_raw, output_filtered, filt_params)

	else:
		raise RuntimeError("This script is intended to be run via Snakemake.")
