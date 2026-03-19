import numpy as np
from .burst_phase import get_burst_analytic, DEFAULT_TIME_HALFWINDOW
import pandas as pd
import networkx as nx

# Merge bursts


def compute_correlation_matrix(burst_analytic_sig):
    burst_sig_data = np.concatenate(
        [burst_sig_.data[None, :] for burst_sig_ in burst_analytic_sig], axis=0
    )
    return np.corrcoef(np.real(burst_sig_data))


def compute_distances(burst_df):
    # Temporal distance
    burst_time_diff = np.abs(
        burst_df.time.values[:, None] - burst_df.time.values[None, :]
    )

    # Spatial distance
    burst_ML_diff = np.abs(burst_df.ML.values[:, None] - burst_df.ML.values[None, :])
    burst_AP_diff = np.abs(burst_df.AP.values[:, None] - burst_df.AP.values[None, :])
    burst_spatial_dist = np.sqrt(burst_ML_diff**2 + burst_AP_diff**2)

    return burst_time_diff, burst_spatial_dist


def assign_quadrants(burst_df):
    quadrant_names = {(-1, -1): "DL", (-1, 1): "UL", (1, -1): "DR", (1, 1): "UR"}
    burst_df.loc[:, "quadrant"] = burst_df.apply(
        lambda x: quadrant_names[(np.sign(x.AP), np.sign(x.ML))], axis=1
    )
    return burst_df


def create_corr_df(
    burst_corr,
    burst_time_diff,
    burst_spatial_dist,
    burst_df,
    time_thresh=DEFAULT_TIME_HALFWINDOW,
):
    # get boolean matrix with upper triangular True values
    mask = np.triu(np.ones_like(burst_corr, dtype=bool), k=1)
    corr_series = burst_corr[mask].reshape(-1)
    time_series = burst_time_diff[mask].reshape(-1)
    concurrent_series = time_series < time_thresh
    spat_series = burst_spatial_dist[mask].reshape(-1)
    is_colocalized = (
        burst_df.quadrant.values[:, None] == burst_df.quadrant.values[None, :]
    )
    colocalized_series = is_colocalized[mask].reshape(-1)
    burst_corr_df = pd.DataFrame(
        np.array(
            [
                spat_series,
                time_series,
                corr_series,
                colocalized_series,
                concurrent_series,
            ]
        ).T,
        columns=["spat_dist", "time_dist", "corr_val", "colocalized", "concurrent"],
    )
    # make colocalized and concurrent boolean
    burst_corr_df.loc[:, "colocalized"] = burst_corr_df.colocalized.astype(bool)
    burst_corr_df.loc[:, "concurrent"] = burst_corr_df.concurrent.astype(bool)
    # add column of burst_pair_idx to burst_corr
    burst_corr_df.loc[:, "burst_1"] = mask.nonzero()[0]
    burst_corr_df.loc[:, "burst_2"] = mask.nonzero()[1]
    return burst_corr_df


def filter_pairs_for_merging(burst_corr_df, corr_thresh=0.9):
    # get the burst pairs with high correlation values
    burst_corr_merge_df = burst_corr_df[
        (burst_corr_df.corr_val >= corr_thresh)
        & (burst_corr_df.colocalized)
        & (burst_corr_df.concurrent)
    ]
    return burst_corr_merge_df


def find_merge_clusters(burst_corr_merge_df):
    G_merge = nx.from_pandas_edgelist(burst_corr_merge_df, "burst_1", "burst_2")
    burst_merge_clusters_list = [
        list(cluster) for cluster in list(nx.connected_components(G_merge))
    ]
    return burst_merge_clusters_list


def merge_bursts(burst_df, burst_merge_clusters_list):
    # merge the bursts in each cluster using the highest amplitude burst as the representative
    burst_merge_clusters = []
    burst_drop_indices = []
    for cluster in burst_merge_clusters_list:
        cluster_df = burst_df.loc[cluster]
        max_amp_idx = cluster_df.amp.idxmax()
        burst_merge_clusters.append(max_amp_idx)
        # drop the max_amp_idx from the cluster
        cluster.remove(max_amp_idx)
        burst_drop_indices.extend(cluster)

    burst_df_merged = burst_df.drop(burst_drop_indices)
    return burst_df_merged


def merge_duplicate_bursts(
    burst_df,
    csig,
    time_thresh=0.5,
    freq_halfbandwidth=2.5,
    time_halfwindow=DEFAULT_TIME_HALFWINDOW,
    corr_thresh=0.9,
    verbose=False,
):
    burst_analytic_sig = get_burst_analytic(
        burst_df,
        csig,
        freq_halfbandwidth=freq_halfbandwidth,
        time_halfwindow=time_halfwindow,
    )
    burst_corr = compute_correlation_matrix(burst_analytic_sig)
    burst_time_diff, burst_spatial_dist = compute_distances(burst_df)
    burst_df = assign_quadrants(burst_df)
    burst_corr_df = create_corr_df(
        burst_corr,
        burst_time_diff,
        burst_spatial_dist,
        burst_df,
        time_thresh=time_thresh,
    )
    burst_corr_merge_df = filter_pairs_for_merging(
        burst_corr_df, corr_thresh=corr_thresh
    )
    burst_merge_clusters_list = find_merge_clusters(burst_corr_merge_df)
    burst_df_merged = merge_bursts(burst_df, burst_merge_clusters_list)
    if verbose:
        print(f"Merged {len(burst_df) - len(burst_df_merged)} bursts")
    return burst_df_merged
