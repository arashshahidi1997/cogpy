"""Point cloud simplification utilities."""

import numpy as np


def simplify_point_cloud(X, eps, random_state=None):
    """
    Point cloud simplification using radial distance (euclidean metric).

    Start with a random point in X and mark it as a key point. All consecutive
    points within distance *eps* are removed. The first point beyond *eps*
    becomes the new key point. Repeat until exhausted.

    Parameters
    ----------
    X : pandas.DataFrame
        Shape ``(n_datapoints, n_features)``.
    eps : float
        Cutoff radial distance.
    random_state : int or None
        Seed for random initial-point selection.

    Returns
    -------
    X_reduced : pandas.DataFrame
        Selected data points.
    """
    if random_state is not None:
        np.random.seed(random_state)

    ix0 = np.random.choice(X.shape[0])
    x0 = X.iloc[ix0]
    xt = x0
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
