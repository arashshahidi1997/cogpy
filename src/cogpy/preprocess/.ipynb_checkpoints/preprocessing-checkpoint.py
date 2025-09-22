import numpy as np
import scipy.ndimage as nd
# from src import filtering as filt
from scipy.interpolate import griddata
# from numba import vectorize
from scipy import signal
import networkx as nx
import pandas as pd
from copy import copy

from src.footprint import loc_exclude

def filtering_pipeline(sig, order=4, size=(3,3,80), f_cutoff=50, fs=625):
    # lowpass filter
    b, a = signal.butter(order, f_cutoff, btype='lowpass', output='ba', fs=fs)
    sig_low = signal.filtfilt(b, a, sig, axis=axis)
    # spatio-temporal median filter (instead of high pass temporal filter and spatial gaussian filtering)
    sig_filt = nd.median_filter(sig_low, size)
    return sig_filt

def interpolate_bads(arr, skip, method='linear', extrapolate=True, remove_bads=False, **kwargs):
    """
    Parameters
    ##########
    
    arr: 3d array containing values (x, y, t)
    skip: 2d bool array True at bad channels
    
    Return
    ######
    iarr: 2d array with interpolated values at bad channels
    """
    
    def _griddata(_arr, nans, method):
        not_nans = np.invert(nans)
        x, y = np.where(not_nans) # reference coordinates
        z = _arr[x, y] # reference values
        ix, iy = np.where(nans) # bad channel coordinates
        iz = griddata((x,y), z, (ix, iy), method=method, **kwargs)
        _iarr = _arr.copy()
        _iarr[ix, iy] = iz
        return _iarr

    # interpolate
    iarr = _griddata(arr, skip, method=method)
    
    # extrapolate
    if extrapolate:
        iarr = extrapolate_bads(iarr)

    return iarr

def nan_mask(arr):
    isnan = np.array(np.isnan(arr), dtype=bool)
    abs_bad_mask = np.all(isnan, axis=-1)
    occasional_bad_mask = np.any(isnan, axis=-1)
    assert np.all(abs_bad_mask == occasional_bad_mask), print(occasional_bad_mask)
    return abs_bad_mask


def extrapolate_bads(x):
    """
    x: (grid, grid, [time])
    depth=2
    """
    x_ = np.copy(x)
    # find nans
    isnan = np.where(np.isnan(x_[:,:,0])) 
    # find bad channel indices where nans occur
    bad_chan = np.ravel_multi_index(isnan, (16,16)) 
    # replace nan at each bad channel with the mean of spatial neighborhood of the channel
    x_[isnan] = np.array([np.nanmean(x_[np.where(loc_exclude[ich])], axis=0) for ich in bad_chan]) 
    return x_

class SignalProcess:
    """
    Graph of signal processing steps with desired order
    nodes: signal at different states
    links: filters/processing step
    """
    def __init__(self, init_state:str, init_sig, init_desc=''):
        self.G = nx.DiGraph()
        self.init_state = init_state
        self.G.add_node(init_state, sig=init_sig, desc=init_desc)

    def run_processor(self, processor, istate):
        fstate = processor.acr
        self.G.add_node(fstate, lineage= istate +'->'+ fstate, desc=processor.abr)
        self.G.add_edge(istate, fstate, process=processor.info())
        
        _temp_sig = copy(self.G.nodes[istate]['sig'])
        self.G.nodes[fstate]['sig'] = processor._filt(_temp_sig)

        return fstate

    def run_pipeline(self, pipeline, istate):
        state = istate
        for processor in pipeline:
            state = self.run_processor(processor, state)
        
        return state

    @property
    def edge_labels(self):
        return {edge:pd.Series(params['process']['params']).to_string() for edge,params in self.G.edges.items()}

    @property
    def node_descriptions(self):
        return pd.Series({node:params['desc'] for node, params in self.G.nodes.items()}).to_string()

########
# other filters
# def pipeline(input, w=25, f1=0.1, f2=50, fs=625, sigma=1):
#     # median filter (temporal)
#     output = filt.median_filt(input, w=w)
#     # high pass - make it local on the grid
#     # h:3 x w:3 x t:50 median filter size

#     # temporal filtering
#     output = filt.butterworth(output, low=f1, high=f2, fs=fs)
#     # change to low pass

#     # spatial filtering
#     ## gaussian
#     output = filt.gaussian_spatial(output, sigma=sigma)

