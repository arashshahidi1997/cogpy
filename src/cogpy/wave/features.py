"""Scalar feature extraction for detected travelling waves.

Provides functions to compute summary features (duration, trajectory,
boundary detection) from wave objects or labelled extrema DataFrames.
"""

import numpy as np


def duration(wave):
    """Compute the duration of a wave in samples.

    Parameters
    ----------
    wave : object
        Wave object with a ``.t`` attribute containing time indices.

    Returns
    -------
    int
        Duration in samples (inclusive of endpoints).
    """
    return wave.t.max() - wave.t.min() + 1


def trajectory(lext_df):
    """Extract per-cluster spatial trajectories from labelled extrema.

    Groups coordinate columns by cluster label and returns an array of
    coordinate sequences for each cluster.

    Parameters
    ----------
    lext_df : pandas.DataFrame
        Labelled extrema table with a ``"Clu"`` column and one or more
        coordinate columns.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by ``"Clu"`` with a ``"coo"`` column containing
        coordinate arrays for each cluster.
    """
    return (
        lext_df.set_index("Clu")
        .apply(list, axis=1)
        .to_frame()
        .reset_index()
        .rename(columns={0: "coo"})
        .groupby("Clu")
        .agg({"coo": np.array})
    )


def positive_boundaries(x: np.ndarray):
    """Positive boundaries of a wave signal.

    Parameters
    ----------
    x : np.ndarray
        Input wave signal.

    Returns
    -------
    boundaries : np.ndarray
        Array of shape (n, 2) with the start and end indices of each
        positive segment.
    """
    # Find the boundaries of non-zero sequences
    xclip = np.clip(x, 0, None)
    y_ = np.concatenate(([0], xclip, [0])) > 0
    boundaries = np.where(np.diff(y_))[0].reshape(-1, 2)
    return boundaries
