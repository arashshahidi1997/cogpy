import numpy as np
import scipy.ndimage as nd

def gaussian_smoothed_random_signal(sigma):
    """
    Parameters
    ----------
    sigma: tuple of int
        The standard deviation of the Gaussian filter along each axis.

    Returns
    -------
    a: ndarray
        The random signal after Gaussian smoothing.
    """
    a = nd.gaussian_filter(np.random.rand(sigma.shape), sigma=sigma)
    return a

class Monopole:
    def __init__(self, pos, amp):
        self.pos = pos
        self.amplitude = amp
    
    def volume_conduction(self, pos):
        dist2 = np.linalg.norm(self.pos - pos) ** 2
        self.amplitude / dist2

class Dipole:
    def __init__(self, pos, amp):
        self.pos = pos
        self.amplitude = amp

    def volume_conduction(self, pos):
        dist2 = np.linalg.norm(self.pos - pos) ** 2
        self.amplitude / dist2

class DirectCurrent:
    """
    a positive pulse
    """
    pass

class AlternatingCurrent:
    """
    a positive followed by a negative pulse
    """
    pass
    
    