import numpy as np
from kneed import KneeLocator


def find_elbow(x, return_knee=False):
    kn = KneeLocator(
        np.arange(1, len(x) + 1), x, curve="convex", direction="increasing"
    )
    if return_knee:
        return kn
    return kn.knee_y
