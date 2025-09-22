import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from scipy import stats as sts

def mad_based_outlier_threshold(data, alpha=0.05):
    """
    Compute two-sided threshold for outliers using MAD with significance alpha.
    
    Parameters
    ----------
    data : array-like
        Input 1D data.
    alpha : float
        Significance level (e.g. 0.05 → 95% central region kept).
        
    Returns
    -------
    lower, upper : float
        Thresholds for outlier detection.
    """
    data = np.asarray(data)
    
    # Median and MAD
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    sigma_mad = 1.4826 * mad   # robust std estimator
    
    # Normal quantile for chosen alpha
    z = sts.norm.ppf(1 - alpha/2)  # two-sided
    
    lower = median - z * sigma_mad
    upper = median + z * sigma_mad
    return lower, upper

def robust_zscore(arr, scale='normal', nan_policy='omit', return_moments=False):
	arr_med = np.nanmedian(arr)
	arr_mad = sts.median_abs_deviation(arr, scale=scale, nan_policy=nan_policy)
	arr_zscore = (arr - arr_med) / arr_mad
	if return_moments:
		return arr_zscore, arr_med, arr_mad
	return arr_zscore

def hit_miss_table(x, y):
	hit = np.sum(x*y)
	miss = np.sum(x*(~y))
	false_alarm = np.sum((~x)*y)
	correct_rejection = np.sum((~x)*(~y))
	hit_miss_table = pd.DataFrame({'old bad': [hit, miss], 'old good': [false_alarm, correct_rejection]}, index=['new bad', 'new good'])
	return hit_miss_table

def bin_data_and_get_centers_edges(data, bin_size):
	"""
	Bin data into groups of 'bin_size', compute bin centers, and calculate bin edges.

	Parameters:
	data (np.array): 1D array of data points.
	bin_size (int): Number of data points in each bin.

	Returns:
	tuple: (bin_centers, bin_edges) where both are np.array
	"""
	# Reshape data into bins
	bins = data.reshape(-1, bin_size)
	
	# Calculate bin edges
	bin_edges = np.empty(bins.shape[0] + 1)
	half_step = (bins[0, 1] - bins[0, 0]) / 2
	bin_edges[0] = bins[0, 0] - half_step
	bin_edges[1:-1] = (bins[:-1, -1] + bins[1:, 0]) / 2
	bin_edges[-1] = bins[-1, -1] + half_step

	# Calculate bin centers
	bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
	return bin_centers, bin_edges

def summarize_performance(true_labels, predicted_labels):
	accuracy = metrics.accuracy_score(true_labels, predicted_labels)
	precision = metrics.precision_score(true_labels, predicted_labels)
	recall = metrics.recall_score(true_labels, predicted_labels)
	f1_score = metrics.f1_score(true_labels, predicted_labels)
	auc_score = metrics.roc_auc_score(true_labels, predicted_labels)
	confusion = metrics.confusion_matrix(true_labels, predicted_labels)
	tn, fn = confusion[:,0]
	fp, tp = confusion[:,1]

	# Create a dictionary with the performance metrics
	metrics_dict = {"TP":tp,
					"FP":fp,
					"TN":tn,
					"FN":fn,
					"Accuracy": accuracy,
					"Precision": precision,
					"Recall": recall,
					"F1 Score": f1_score,
					"AUC Score": auc_score}

	# Create a DataFrame from the dictionary
	df = pd.DataFrame.from_dict(metrics_dict, orient="index", columns=["Value"])

	return df
