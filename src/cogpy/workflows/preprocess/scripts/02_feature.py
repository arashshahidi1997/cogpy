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

from cogpy.core.preprocess.channel_feature import ChannelFeatures, save_features
import xarray as xr
from cogpy.io import ecog_io

def _input(input_zarr):
	sigx_lp_load = ecog_io.from_zarr(input_zarr)['sigx']
	return sigx_lp_load

def _output(feat_dataset, output_feature):
	# print(f"preparing to save feature data to\n\t{output_feature}")
	# # encoding = encoding_from_ds_chunks(feat_dataset)
	
	print(f"writing feature data to\n\t{output_feature}")
	print("Computing and writing features to Zarr ...")
	save_features(feat_dataset, output_feature)

	# with ProgressBar():
	# 	feat_ds = feat_ds.persist()

	# ecog_io.to_zarr(output_feature, feat_dataset, encoding=encoding, compute=True)
	# assert isinstance(feat_dataset, xr.Dataset)
	print(f"Feature data saved to\n\t{output_feature}")

def encoding_from_ds_chunks(ds: xr.Dataset) -> dict:
	enc = {}
	for name, da in ds.data_vars.items():
		# da.dims gives the dim order used on disk
		# da.chunks is a tuple-of-tuples, one per dim
		# Use the *first* chunk size on each dim as the Zarr chunk size
		c = []
		for dim in da.dims:
			axis = da.get_axis_num(dim)
			first = da.chunks[axis][0]  # assumes uniform chunking after unify_chunks()
			c.append(first)
		enc[name] = {"chunks": tuple(c)}  # Zarr expects a tuple-of-ints in dim order
	return enc

def feature(sigx: xr.DataArray, window_size: int, window_step: int, zscore: bool) -> xr.Dataset:
	AP = sigx.sizes['AP']
	ML = sigx.sizes['ML']
	fs = sigx.fs
	sigx.attrs['fs'] = fs
	
	print("Setting up ChannelFeatures Extractor ...")
	chfeat = ChannelFeatures(nrows=AP, ncols=ML)

	print("Chunking input data ...")
	chunks = {"time": 16 * 4096, "AP": -1, "ML": -1}
	sigx = sigx.chunk(chunks)
	# take a small slice for testing
	sigx = sigx.isel(time=slice(0, 32 * 4096))

	print("Extracting features ...")
	return chfeat.compute_features(sigx, window_size=window_size, window_step=window_step, zscore=zscore)

def main(input_lowpass, output_feature, window_size, window_step, zscore):
	sigx = _input(input_lowpass)
	# test
	# sigx_slc = sigx.isel(time=slice(0, 1500))
	feat_dataset = feature(sigx, window_size=window_size, window_step=window_step, zscore=zscore)
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
		window_size = snakemake.params.window_size
		window_step = snakemake.params.window_step
		zscore = snakemake.params.zscore
		
		# from dask.distributed import Client, LocalCluster
		# from dask.diagnostics import ProgressBar
		# import atexit

		# Configure cluster
		# cluster = LocalCluster(
		# 	n_workers=2,
		# 	threads_per_worker=2,
		# 	memory_limit='2GB',
		# 	dashboard_address=':8787'
		# )
		# client = Client(cluster)
		
		# # Register cleanup
		# def cleanup():
		# 	print("Cleaning up Dask client and cluster...")
		# 	try:
		# 		client.close()
		# 		cluster.close()
		# 	except:
		# 		pass
		
		# atexit.register(cleanup)
		
		try:
			# print(f"Dashboard available at: {client.dashboard_link}")
			main(input_lowpass, output_feature, window_size, window_step, zscore)
		except Exception as e:
			print(f"Error during processing: {e}")
			# cleanup()
			raise e
	else:
		raise RuntimeError("This script is intended to be run via Snakemake.")
