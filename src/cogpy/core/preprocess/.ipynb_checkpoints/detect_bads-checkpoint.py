import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.ndimage as nd
import scipy.stats as sts
from scipy import signal
from sklearn.cluster import KMeans
import sklearn.metrics as metrics
from functools import partial

from src.spectral.spectral import freq_index, LOW_FREQ, HIGH_FREQ

# footprints
from ..utils.footprint import fp, fp_exclude, loc_exclude
EPSILON = 0.000001

def anticorrelation(a, gridshape):
    """
    Parameters:
    a: array (time, channel)

    Returns
    -------
    1-mean_corr: array (ch, ch)
    """
    # Use errstate to ignore RuntimeWarning from corrcoef  
    with np.errstate(divide='ignore', invalid='ignore'):
        corr = np.corrcoef(a.T)
    corr = corr.reshape(-1, *gridshape)
    corr = np.nan_to_num(corr)
    med_corr = np.array(
        [np.nanmedian(corr[ch][np.where(loc_exclude[ch])]) for ch in range(np.product(gridshape))]
    ).reshape(*gridshape)
    return 1 - med_corr

from src.spectral.spectral import LOW_FREQ, HIGH_FREQ

def diff_neighbor_median(arr):
    """
    a: array (time, channel)
    """
    med_arr = nd.filters.median_filter(arr, footprint=fp_exclude)
    return np.abs(arr - nd.filters.median_filter(arr, footprint=fp_exclude))

def ratio_neighbor_median(arr, log=True):
    """
    a: array (time, channel)
    """
    med_arr = nd.filters.median_filter(arr, footprint=fp_exclude)

    if log:
        res = np.log(arr + EPSILON) - np.log(med_arr + EPSILON)
    else:
        res /= nd.filters.median_filter(arr, footprint=fp_exclude)
    return res

def noise_to_signal_spectral_power(a, gridshape, fs=1):
    """
    a: array (time, channel)
    """
    nperseg = 512
    freq, psd = signal.welch(a.T, nperseg=nperseg, fs=fs, axis=-1)
    f_low, f_high = freq_index([LOW_FREQ, HIGH_FREQ], fs=fs, N=nperseg)
    power_low = np.nanmean(psd[:, slice(*f_low)], axis=-1).reshape(gridshape)
    power_high = np.nanmean(psd[:, slice(*f_high)], axis=-1).reshape(gridshape)
    nsr = power_high/(power_low+EPSILON)
    # power_low = rel_median(power_low)
    # power_high = rel_median(power_high)
    nsr = np.log10(nsr + EPSILON)
    nsr = np.abs(nsr-np.nanmedian(nsr))
    return nsr

def median_gradient(a, gridshape):
    """
    a: array (time, channel)
    """
    med_grad = np.zeros_like(a)
    # t = 0
    for ch in tqdm(range(np.product(gridshape)), position=0):
        for t in range(a.shape[0]):
            med_grad[t, ch] = np.abs(np.nanmedian(
                a.reshape(-1,*gridshape)[t]\
                [np.where(loc_exclude[ch])] - a[t, ch]
            ))
    mean_grad = np.nanmean(med_grad, axis=0).reshape(*gridshape)    
    return np.log(mean_grad + EPSILON)

def relative_variance(a, gridshape):
    """
    a: array (time, channel)
    """
    rel_var = np.var(a, axis=0).reshape(*gridshape)
    return ratio_neighbor_median(rel_var, log=True)

def deviation(a, gridshape):
    """
    a: array (time, channel)
    """
    rel_mean = np.nanmean(a, axis=0).reshape(*gridshape)
    rel_mean -= nd.filters.generic_filter(rel_mean, np.nanmedian, footprint=fp_exclude)
    rel_mean = np.abs(rel_mean)
    return np.log(rel_mean + EPSILON)

def amplitude(a, gridshape):
    """
    a: array (time, channel)
    """
    rel_amp = (np.max(a, axis=0) - np.min(a, axis=0)).reshape(*gridshape)
    assert np.all(rel_amp>=0), print('neg range')
    return ratio_neighbor_median(rel_amp, log=True)

def spatial_gradient(a, gridshape):
    """
    a: array (time, channel)
    """
    rel_grad = np.nanmean(
        np.linalg.norm(
            np.array(np.gradient(a.reshape(-1,*gridshape), axis=(1,2)))
            , axis=0),
        axis=0)
    rel_grad = rel_grad / nd.filters.median_filter(rel_grad, footprint=fp_exclude)
    return np.log(rel_grad + EPSILON)

def time_derivate(a, gridshape):
    """
    a: array (time, channel)
    """
    tder = np.nanmean(
            np.abs(np.array(np.gradient(a, axis=0))),
        axis=0).reshape(*gridshape)
    return ratio_neighbor_median(tder, log=True)

def hurst_exponent(a, gridshape):
    """
    a: array (time, channel)
    """
    mean = np.nanmean(a, axis=0)
    y = a - mean
    z = np.cumsum(y, axis=0)
    r = np.max(z, axis=0) - np.min(z, axis=0)
    std = np.std(a)
    hurst = (np.log10(r/std + EPSILON)/2).reshape(*gridshape)
    return diff_neighbor_median(hurst)

def kurtosis(a, gridshape):
    """
    a: array (time, channel)
    """
    kurt = sts.kurtosis(a, axis=0).reshape(*gridshape)
    return kurt

def is_disconnected(a):
    """
    a: array (time, channel)
    Returns
     (channel, ) bool True: disconnected (constant)
    """
    return (a[0, np.newaxis] == a).all(axis=0)

def get_channel_features(a, gridshape=(16,16), fs=1):
    """
    Parameters
    ----------
    a: array (time, channel)
        a smaller array would make computations faster,
        so only pass a segment of the full data

    Returns
    -------
    chan_feature: DataFrame
        rows: channels, columns: features
    """
    # features
    feature_dict = {'antiCor':anticorrelation, # high bad
                    'Var':relative_variance, # high bad
                    'Dev':deviation, # high bad
                    'Amp':amplitude, # high bad
                    'tDer':time_derivate, # high bad
                    'SpatGrad':spatial_gradient, # high bad
                    'Hurst':hurst_exponent, # 
                    'NSR':partial(noise_to_signal_spectral_power, fs=fs), # high bad
    }
                    # 'Kurt':kurtosis}
    
    chan_feature = {feature: func(a, gridshape=gridshape).reshape(-1) for feature, func in feature_dict.items()}
    chan_feature = pd.DataFrame.from_dict(chan_feature)
    return chan_feature

def leave_one_out_kmeans(channel_feature_arr, ch):
    """
    hold feature data point corresponding to channel `ch` out,
    perform kmeans clustering (ncluster=2),
    predict whether the channel `ch` belongs to the smaller (bad) or larger (good) cluster
    returns True if `ch` belongs to the smaller cluster, otherwise False

    Parameters
    ----------
    channel_feature_arr: array (channels, features)
        feature values for each channel

    Returns
    -------
    prediction: bool
        True if channel with index `ch` is predicted by kmeans to be bad.
        kmeans is trained on all data points in channel_feature_arr except for the `ch` row.
    
        True if ch is classified as part of the smallest cluster (which I assume to be the bad channel cluster)
    """
    km = KMeans(n_clusters=2)
    km.fit_transform(np.delete(channel_feature_arr, ch, 0))
    scores=[np.sum(km.labels_ == i) for i in range(km.n_clusters)]
    prediction = km.predict(channel_feature_arr[ch].reshape(1,-1))[0] == np.argmin(scores)
    return prediction

def detect_twomeans(channel_feature_arr):
    pred = [] # prediction of kmeans
    nchan = len(channel_feature_arr)
    for ch in range(nchan):
        pred.append(leave_one_out_kmeans(channel_feature_arr, ch))
    
    return np.array(pred, bool)

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

# def refine_bads(channel_feature_arr):
#     """
#     Note: this algorithm assumes that the majority of channels are good.

#     Parameters
#     ----------
#     channel_feature_arr: array (channels, features)
#         feature values for each channel
#     target: array (channels,)
#         0 for good channles 1 for bad channels

#     Returns
#     -------
#     refined_target:

#     refined_channels:
#     """
#     nchan = len(channel_feature_arr)
#     pred = kmeans_prediction(channel_feature_arr)
#     return 

# def compress_distribution():
#     pass
    

def detect_bads(arr, method=detect_twomeans, chan_feature=None):
    """
    Parameters
    ----------
    arr: 3D array (height, width, time), chunk of signal

    Retruns
    -------
    bad_channels: 1D bool array (channel,)

    Leave-one-out kmeans clustering prediction
    """
    gridshape = arr.shape[:-1]

    # reshape to (time, height, width)
    arr = np.moveaxis(arr, -1, 0)
    # reshape to (time, channel)
    nchan = np.prod(gridshape)
    arr = arr.reshape(-1, nchan)

    # compute channel features
    if chan_feature is None:
        chan_feature = get_channel_features(arr, gridshape=gridshape)
    # convert features dataframe to array; find disconnected (constant) channels
    disconnected = is_disconnected(arr)
    connected_chan_feature = chan_feature[np.invert(disconnected)]
    connected_chan_feature_arr = connected_chan_feature.to_numpy()
    # normalize features
    connected_chan_feature_arr = sts.zscore(connected_chan_feature_arr, axis=0)
    # detect bad channels
    bad_connected_channels = method(connected_chan_feature_arr)
    bad_connected_channels = connected_chan_feature.iloc[bad_connected_channels].index.values
    bad_channels = np.zeros(nchan, dtype=bool)
    bad_channels[bad_connected_channels] = True
    bad_channels[np.where(disconnected)] = True
    return bad_channels

