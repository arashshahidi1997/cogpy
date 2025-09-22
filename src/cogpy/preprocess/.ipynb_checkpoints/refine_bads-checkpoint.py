import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.ndimage as nd
import scipy.stats as sts
from scipy import signal
from sklearn.cluster import KMeans

# footprints
fp = nd.iterate_structure(nd.generate_binary_structure(2,1), 2)
fp_exclude = np.copy(fp)
fp_exclude[fp.shape[0]//2, fp.shape[1]//2] = False

loc_exclude = []
for ch in range(256):
    loc = np.zeros((16,16))
    loc[np.unravel_index(ch, (16,16))] = 1
    loc = signal.convolve2d(loc, fp, mode='same')
    loc[np.unravel_index(ch, (16,16))] = 0
    loc_exclude.append(loc)

loc_exclude = np.array(loc_exclude)


def get_channel_features(a, anat_map):
    """
    Parameters
    ----------
    a: array (time, channel)
        a smaller array would make computations faster,
        so only pass a segment of the full data

    anat_map: DataFrame
        id
        skip

    Returns
    -------
    chan_feature: DataFrame
        rows: channels, columns: id, skip, features
    """
    # features
    med_grad = np.zeros_like(a)
    # t = 0
    for ch in tqdm(range(256), position=0):
        for t in range(1000):
            med_grad[t, ch] = np.abs(np.median(
                a.reshape(-1,16,16)[t]\
                [np.where(loc_exclude[ch])] - a[t, ch]
            ))
    mean_grad = np.mean(med_grad, axis=0).reshape(16,16)    

    ### correlation
    corr = np.corrcoef(a.T)
    corr = corr.reshape(-1, 16, 16)
    mean_corr = np.array(
        [np.mean(corr[ch][np.where(loc_exclude[ch])]) for ch in range(256)]
    ).reshape(16,16)

    ### variance
    rel_var = np.var(a, axis=0).reshape(16,16)
    rel_var /= nd.filters.median_filter(rel_var, footprint=fp_exclude)

    ### mean
    rel_mean = np.mean(a, axis=0).reshape(16,16)
    rel_mean -= nd.filters.generic_filter(rel_mean, np.mean, footprint=fp_exclude)
    rel_mean = np.abs(rel_mean)

    ### amplitude
    rel_amp = (np.max(a, axis=0) - np.min(a, axis=0)).reshape(16,16)
    assert np.all(rel_amp>=0), print('neg range')
    rel_amp = rel_amp / nd.filters.median_filter(rel_amp, footprint=fp_exclude)

    ### spatial gradient
    rel_grad = np.mean(
        np.linalg.norm(
            np.array(np.gradient(a.reshape(-1,16,16), axis=(1,2)))
            , axis=0),
        axis=0)
    rel_grad = rel_grad / nd.filters.median_filter(rel_grad, footprint=fp_exclude)


    ### temporal gradient
    rel_tgrad = np.mean(
            np.abs(np.array(np.gradient(a, axis=0))),
        axis=0).reshape(16,16)
    rel_tgrad = rel_tgrad / nd.filters.median_filter(rel_tgrad, footprint=fp_exclude)

    ### Hurst exponent
    mean = np.mean(a, axis=0)
    y = a - mean
    z = np.cumsum(y, axis=0)
    r = np.max(z, axis=0) - np.min(z, axis=0)
    std = np.std(a)
    hurst = (np.log10(r/std)/2).reshape(16,16)

    ### kurtosis
    kurt = sts.kurtosis(a, axis=0).reshape(16,16)


    ### Features
    chan_feature = np.stack(
        [1-mean_corr, np.log(rel_var), np.log(rel_mean), 
        np.log(rel_amp), np.log(rel_tgrad), np.log(mean_grad),
        hurst, kurt]
    ).reshape(-1,256).T

    feature_list = ['Cor', 'Var', 'Dev', 'Amp', 'tGrad', 'Grad', 'Hurst', 'Kurt']

    chan_feature = pd.DataFrame(chan_feature, columns=feature_list)
    chan_feature = pd.concat([anat_map, chan_feature], axis=1)

    return chan_feature

def leave_one_out_kmeans(channel_feature_arr, target, ch):
    """
    returns boolean
    True if ch is classified as part of the smallest cluster (which I assume to be the bad channel cluster)
    """
    km = KMeans(n_clusters=2)
    km.fit_transform(np.delete(channel_feature_arr, ch, 0), np.delete(target, ch))
    scores=[np.sum(km.labels_ == i) for i in range(km.n_clusters)]
    return km.predict(channel_feature_arr[ch].reshape(1,-1))[0] == np.argmin(scores)

def refine_bads(channel_feature_arr, target):
    """
    channel_feature_arr: array (channels, features)
        feature values for each channel
    target: array (channels,)
        0 for good channles 1 for bad channels
    """
    pred = []
    for ch in tqdm(range(256), position=0):
        pred.append(leave_one_out_kmeans(channel_feature_arr, target, ch))
        
    skip = set(map(int, set(np.where(target==1)[0])))
    sel = set(map(int, set(np.where(pred)[0])))
    refined_channels = np.array(list(sel.difference(skip)))

    refined_target = np.zeros(256)
    refined_target[refined_channels] = 1
    refined_target[list(skip)] = 1

    return refined_target, refined_channels