import numpy as np
import scipy.ndimage as nd
from scipy import signal

a = nd.gaussian_filter(np.random.rand(10,5,5), sigma=(1,1,1))
a_min = nd.filters.minimum_filter(a, size=(0,3,3)) == a
a_max = nd.filters.maximum_filter(a, size=(0,3,3)) == a


class Monopole:
    def __init__(self, pos, amp):
        self.pos = pos
        self.amplitude = amp
    
    def volume_conduction(self, pos):
        dist2 = np.linalg.norm(self.pos - pos) ** 2
        self.amplitude/dist2
        

class Dipole:
    
class DirectCurrent:
    
class AlternatingCurrent:
    
    
    

    
    
    