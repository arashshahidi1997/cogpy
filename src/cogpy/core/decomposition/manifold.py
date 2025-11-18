from sklearn.manifold import TSNE
import numpy as np
import pandas as pd


def simplify_point_cloud(X, eps, random_state=None):
    """
    point cloud simplification using radial distance (euclidean metric).
    Start with the first point in in X and mark it as a key point. All consecutive points that have a distance less than a predetermined distance eps to the key point are removed. The first point that have a distance greater than eps to the key point is marked as the new key point. The process repeates itself from this new key point, and continues until it reaches the end of the point cloud.

    Parameters
    ----------
    X: pandas DataFrame (n_datapoints, n_features):

    eps: max radial distance - cutoff distance

    random_state: seed of random generator used for choosing the inital point

    Returns
    -------
    X_reduced: chosen data points

    indices: indices of the chosen data points
    """
    if random_state is not None:
        np.random.seed(random_state)

    ix0 = np.random.choice(X.shape[0])
    x0 = X.iloc[ix0]
    xt = x0
    ixt = ix0
    X_temp = X
    ind_reduced = [ix0]

    while True:
        dist = np.linalg.norm(X_temp.to_numpy() - xt.to_numpy(), axis=1)
        cond = dist < eps

        X_temp = X_temp.drop(X_temp.index[np.where(cond)])
        if len(X_temp) == 0:
            break

        where_not_cond = np.where(np.logical_not(cond))
        w = np.argmin(dist[where_not_cond])
        ixt = X_temp.index[w]
        xt = X.iloc[ixt]
        ind_reduced.append(ixt)

    return X.iloc[ind_reduced]
