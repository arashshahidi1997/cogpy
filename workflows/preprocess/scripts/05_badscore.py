#!/usr/bin/env python
"""
Title: Script Name
Status: WIP
Last Updated: 2025-09-14

Summary:
	Detects bad periods in ECoG data based on deviations of channel features from smoothness.
	Periods that show highly non-smooth behavior across multiple channels are marked as bad.
Usage:
	python 05_badperiod.py [--args]
"""
import numpy as np
import xarray as xr
from scipy.spatial.distance import cdist
from cogpy.io import ecog_io
from cogpy.utils.stats import mad_based_outlier_threshold

def _input(input_feature, input_labels):
	feature_ds = xr.open_zarr(input_feature, chunks='auto')
	labels = np.load(input_labels)
	return feature_ds, labels

def _output(badscore_ds, output_badscore):
	ecog_io.to_zarr(output_badscore, badscore_ds)
	print(f"Badscore data saved to\n\t{output_badscore}")

def mean_pairwise_distance(feature_arr: np.ndarray):
	"""
	Computes the mean pairwise distance for each channel based on pairwise distances.

	Parameters
	----------
	feature_arr: (ch, feature)
		Array of features for each channel.

	Returns
	-------
		(ch,) deviation z-score for each channel
	"""
	pairwise_dist = cdist(feature_arr, feature_arr) # pairwise distance (ch, ch)
	mean_pairwise_dist = np.nanmean(pairwise_dist) # mean over all pairs	
	return mean_pairwise_dist

def compute_badscore(feature_ds, labels):
	"""
	Note:
		For squared distances:  ||Xi - Xj||^2 ~ 2*Chi^2_p, mean=2p, var=8p.
		For distances:          ||Xi - Xj||   ~ sqrt(2)*Chi_p, mean ~ sqrt(2p).
		In both cases, the mean over pairs is a 2nd-order U-statistic,
		which is asymptotically Normal (by U-statistic CLT).
		Thus we can safely use a Normal cutoff for either squared or raw distances.
	"""
	featurex = feature_ds.to_array('feature').stack(ch=['AP', 'ML']).chunk({'feature':-1}).reset_index('ch')
	# select good channels
	featurex = featurex.sel(ch=labels)
	# compute badscore
	badscorex = xr.apply_ufunc(mean_pairwise_distance, featurex, input_core_dims=[['ch', 'feature']], vectorize=True, dask='parallelized', output_dtypes=[featurex.dtype])
	_, threshold = mad_based_outlier_threshold(badscorex.data, alpha=0.01)
	isbad = (badscorex > threshold).rename('isbad_time').astype(bool)
	
	# combine into a dataset
	badscore_ds = xr.Dataset({'badscore': badscorex, 'isbad_time': isbad})
	badscore_ds.attrs['threshold_upper'] = float(threshold)  # store as attr if you like
	return badscore_ds

def main(input_feature, input_labels, output_badscore):
	feature_ds, labels = _input(input_feature, input_labels)
	badscore_ds = compute_badscore(feature_ds, labels)
	# output
	_output(badscore_ds, output_badscore)

if __name__ == "__main__":
	# snakemake
	if 'snakemake' in globals():
		snakemake = globals()['snakemake']
		input_feature = snakemake.input.feature
		input_labels = snakemake.input.badlabel
		output_badscore = snakemake.output.badscore

		# main
		main(input_feature, input_labels, output_badscore)

	else:
		raise RuntimeError("This script is intended to be run via Snakemake.")