import numpy as np
import scipy.ndimage as nd


def duration(wave):
    return wave.t.max() - wave.t.min() + 1


def trajectory(lext_df):
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
    """
    Positive boundaries of a wave signal.

    Parameters
    ----------
    x : np.ndarray
        Input wave signal.

    Returns
    -------
    boundaries : np.ndarray
        Array of shape (n, 2) with the start and end indices of each positive wave.
    """
    # Find the boundaries of non-zero sequences
    xclip = np.clip(x, 0, None)
    y_ = np.concatenate(([0], xclip, [0])) > 0
    boundaries = np.where(np.diff(y_))[0].reshape(-1, 2)
    return boundaries


def local_max_half_amp():
    pass


def peak(wave):
    return


def relext(wave):
    return


def remove_close(wave):
    pass


def laplace(wave):
    nd.filters.laplace()
    pass


def convexity():
    pass


def phase_coherence():
    pass


def contour(wave):
    pass


def split_waves(wave):
    waves = []
    return waves


def drop_short_waves(wave):
    pass


def eccentricity(df):
    pass


def wavelet(df):
    pass
