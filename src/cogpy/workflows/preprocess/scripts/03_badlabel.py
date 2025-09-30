#!/usr/bin/env python
"""
Title: 03_badlabel.py
Status: REVIEW
Last Updated: 2025-09-09

Summary:
	Identifies bad channels based on statistical features using an unsupervised outlier detection method.
"""

import numpy as np
import xarray as xr
from scipy.spatial.distance import cdist
from cogpy.preprocess.detect_bads import OutlierDetector
from cogpy.utils.stats import robust_zscore

def compute_quantile_features(features, qmin=0.75, qmax=0.95, nq=5):
	quantiles = np.linspace(qmin, qmax, nq)
	qfeat = features.quantile(quantiles, dim=['time']).stack(ch=['AP', 'ML'])
	return qfeat.transpose('ch', 'quantile')

def deviation_zscore(qfeat: np.ndarray):
	"""
	qfeat: (ch, qfeat)
	"""
	qfeat_dist = cdist(qfeat, qfeat) # (ch, ch)
	ch_deviation_score = np.nanmedian(qfeat_dist, axis=1)
	ch_deviation_zscore = robust_zscore(ch_deviation_score, scale='normal', nan_policy='omit')
	return ch_deviation_zscore

def find_outliers(feature_dataset, knn=10):
	qfeat = feature_dataset.apply(compute_quantile_features, qmin=0.8, qmax=0.95, nq=5)
	qfeat = qfeat.to_array('feature')
	# devz = qfeat.apply(deviation_zscore)
	qfeat = qfeat.stack(qfeat=['feature', 'quantile']).transpose('ch', 'qfeat')
	outlier = OutlierDetector(min_samples=knn, eps_optimize_k=knn)
	# qfeat_X = qfeat.data.compute()
	outlier_map = outlier.fit_predict(qfeat)
	# devz = qfeat.apply(deviation_zscore)
	return outlier_map

def _input(input_feature):
	feature_ds = xr.open_zarr(input_feature)
	return feature_ds

def _output_outlier(labels, output_labels):
	np.save(output_labels, labels)

def main(input_feature, output_labels, knn):
	feature_ds = _input(input_feature)
	outlier_map = find_outliers(feature_ds, knn=knn)
	_output_outlier(outlier_map, output_labels)

if __name__ == "__main__":
	if 'snakemake' in globals():
		snakemake = globals()['snakemake']

		# io
		input_feature = snakemake.input.feature
		output_labels = snakemake.output.badlabel
		knn = snakemake.params.knn

		# main
		main(input_feature, output_labels, knn)

	else:
		raise RuntimeError("This script is intended to be run via Snakemake.")

