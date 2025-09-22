#!/usr/bin/env python
"""
Title: 02_feature.py
Status: REVIEW
Last Updated: 2025-09-13

Summary:
	Extracts features from ECoG channels.

Usage:
	main(input_lowpass, output_feature, slider_kwargs=dict(window_size=512, window_step=64), zscore=True)
"""

from cogpy.preprocess.channel_feature import ChannelFeatures
import xarray as xr
from cogpy.io import ecog_io

def _input(input_zarr):
	sigx_lp_load = ecog_io.from_zarr(input_zarr)['sigx']
	return sigx_lp_load

def _output(feat_dataset, output_feature):
	ecog_io.to_zarr(output_feature, feat_dataset)

	assert isinstance(feat_dataset, xr.Dataset)
	print(f"Feature data saved to\n\t{output_feature}")
	return feat_dataset

def feature(sigx, slider_kwargs, zscore):
	AP = sigx.sizes['AP']
	ML = sigx.sizes['ML']
	fs = sigx.fs
	sigx.attrs['fs'] = fs
	chfeat = ChannelFeatures(nrows=AP, ncols=ML)

	sigx.chunk({'time': -1, 'AP': AP, 'ML': ML})
	feat_ds = chfeat.transform_dask(sigx, slider_kwargs=slider_kwargs, zscore=zscore)
	return feat_ds

def main(input_lowpass, output_feature, slider_kwargs, zscore):
	sigx = _input(input_lowpass)
	# test
	# sigx_slc = sigx.isel(time=slice(0, 1500))
	feat_dataset = feature(sigx, slider_kwargs, zscore)
	_output(feat_dataset, output_feature)

	# actual run
	# sigx_lp = sigx.chunk({'time': -1, 'AP': 16, 'ML': 16})
	# chfeat.transform_dask(sigx_lp)

if __name__ == "__main__":
	# snakemake
	if 'snakemake' in globals():
		snakemake = globals()['snakemake']
		input_lowpass = snakemake.input.lowpass
		output_feature = snakemake.output.feature
		slider_kwargs = snakemake.params.slider_kwargs
		zscore = snakemake.params.zscore
		main(input_lowpass, output_feature, slider_kwargs, zscore)
	else:
		raise RuntimeError("This script is intended to be run via Snakemake.")
