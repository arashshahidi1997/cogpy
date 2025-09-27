"""
module for identifying local minima and maxima
"""
import numpy as np
from scipy.ndimage import filters
import scipy.ndimage as nd
import pandas as pd
from scipy.spatial import distance_matrix
from ..utils.batch_maker import batch_maker

# def df_wrapper(method):
#     def wrap_method(ext, *args, **kwargs):
        
#         [ for df in ext.df]

#         return ext.__dict__[method]
#     return wrap_method

# class ExtDataFrame:
#     def __init__(self):

def detect_extrema(x, footprint, minima=True):
    _filt = [filters.maximum_filter, filters.minimum_filter][minima]
    return _filt(x, footprint=footprint)==x
    
class Extrema:
    def __init__(self, grid_signal, minima=True, der_step=1, axes=-1):
        """
        Takes an array and detects the troughs using the local maximum filter.
        Returns a boolean mask of the troughs (i.e. 1 when
        the pixel's value is the neighborhood's maximum, 0 otherwise)
        """

        # read array
        arr = grid_signal.asarray().transpose(2,0,1) # axis 0 = time

        # construct footprint
        struct = nd.generate_binary_structure(2, 1)
        footprint = nd.iterate_structure(struct, der_step).astype(int)
        self.der_kernel = np.expand_dims(footprint, 0) # expand dims in the time direction
        
        # extrema
        self.ext_arr = detect_extrema(arr, footprint=self.der_kernel, minima=minima) # boolean True at minima if `minima`=True 
        df = pd.DataFrame(np.argwhere(self.ext_arr), columns=['t', 'h', 'w']) # DataFrame: coordinates of extrema
        df['val'] = pd.Series(arr[self.ext_arr.nonzero()]) # Return the values of the signal at non-zero (True) indices where extrema are located.
        
        # sort
        self.df = df.sort_values(['t', 'h', 'w'])

        # time index
        # self.df.reset_index(inplace=True)

        # wave id
        # self.cluster_extrema(ext_arr)

        # empty dataframe
        self.empty = pd.DataFrame(columns=self.df.columns)

        # owner grid_signal
        self.gs = grid_signal
    
    def __getitem__(self, t):
        try:
            return self.df[self.df.t == t]
        except:
            return self.empty

    def get_wave(self, clu):
        return self.df[self.df.Clu == clu]

    def channel_column(self):
        self.df['ch'] = pd.Series(np.ravel_multi_index((self.df.h, self.df.w), dims=self.gs.shape))

    def detect_waves(self, propagate_radius=1):
        propagate_size = 2 * propagate_radius + 1
        self.propagator_kernel = np.ones((3, propagate_size, propagate_size))
        not_ext_arr = np.invert(self.ext_arr)
        labels = self.ext_arr.astype(float)
        labels[self.ext_arr.nonzero()] = np.arange(1,1+len(self.df))
        labels[labels==0] = np.inf

        # frame by frame wave propagation
        clusters = np.copy(labels)

        for i in range(clusters.shape[0]-1):
            clusters[i:i+2] = nd.filters.minimum_filter(clusters[i:i+2], footprint=self.propagator_kernel)
            clusters[i:i+2][(not_ext_arr[i:i+2]).nonzero()] = np.inf
        
        # self.df['Clu1'] = clusters[ext_arr.nonzero()].astype(int)
        clu = clusters[tuple(self.df[['t','h','w']].to_numpy().T)].astype(int)
        
        # relabel clusters (this step is just for elegance and bears no computational significance!)
        x = np.zeros(np.max(clu)+1, dtype=int)
        uclu = np.unique(clu)
        x[uclu] = np.arange(len(uclu))
        clu = x[clu]

        # add cluster id to dataframe
        self.df['Clu'] = clu

    def get_waves(self, interval):
        pass
    
    def center_of_mass(self):
        df = self.df[self.df == 0]
        clusters = np.unique(df['Clu'])
        df_com_list = []
        for i, c in enumerate(clusters):
            h_mean, w_mean, val_mean = df[df['Clu']==c][['h','w','val']].mean()
            df_com_list.append([h_mean, w_mean, val_mean, c])

        df_com = pd.DataFrame(df_com_list, columns=['h','w','val','Clu'])
        return df_com

    # @get_wrapper('df')
    # def thresholding(self, amp_thr, sign):
    #     if sign == 'min':
    #         cnd = ext.df.val < amp_thr        
    #     if sign == 'max':
    #         cnd = ext.df.val > amp_thr        
            
    #     return [ext.df[cnd], ]

# Batch

ExtremaBatch = batch_maker(Extrema)

class Waves:
    def __init__(self, df:pd.DataFrame):
        self.df = df
        self.w = df.groupy('Clu')
        
    def duration(self):
        return self.w.apply(duration)
    




