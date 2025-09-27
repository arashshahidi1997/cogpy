# spatiotemporal filtering
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import scipy.ndimage as nd

def butteer_worth_widget(fs):
    @widgets.interact(order=widgets.IntSlider(4,0,10), f0=widgets.FloatSlider(value=1, min=0.1, max=10), f1=widgets.FloatSlider(value=50, min=10, max=100))

    def plot_filt(order, f0, f1):
        b, a = signal.butter(order, [f0, f1], btype='bandpass', output='ba', fs=fs)

        fig, ax = plt.subplots(1,1)
        w, h = signal.freqz(b, a)
        ax.plot(fs*w/np.pi/2, (abs(h)))
        ax.set(title ='filter', xlabel='Frequency [Hz]',
        ylabel='Amplitude', xlim=[0.1,100])
        ax.grid(which='both', axis='both')
        ax.set(xscale='log')
        # ax.axvline(100, color='green') # cutoff frequency

        fig.suptitle('Butterworth filter frequency response', weight='bold')
        plt.tight_layout()
        plt.show()

def butterworth(sig, order=4, low=1, high=50, fs=625, axis=-1):
    b, a = signal.butter(order, [low, high], btype='bandpass', output='ba', fs=fs)
    sig_filt = signal.filtfilt(b, a, sig, axis=axis)
    return sig_filt

def gaussian_spatial(arr, sigma=1):
    if arr.ndim==3:
        sigma = (sigma, sigma, 0)
    elif arr.ndim==2:
        sigma = (sigma, sigma)
    else:
        raise ValueError
        
    return nd.gaussian_filter(arr, sigma=sigma, mode='reflect')

def median_subtract(sig):
    return sig - np.median(sig, axis=(0,1), keepdims=True)

def nanpad(input, w):
    return np.pad(input, pad_width=w//2, mode='constant', constant_values=(np.nan,))[1:17, 1:17, :]

def median_filt(input, w=3):
    input_pad = nanpad(input, w)
    idx = sliding_window(input_pad.shape,(*input_pad.shape[:2], w))
    output = input - np.nanmedian(input_pad[:,:,idx], axis=(0,1,3))
    return output
    
def median_spatial(arr, size=1):
    if arr.ndim==3:
        size = (size, size, 0)
    elif arr.ndim==2:
        size = (size, size)
    else:
        raise ValueError
        
    return nd.median_filter(arr, size=size)

# nd.median_filter(arr, size=(h_size,w_size,t_size))

# duplicate of footprint._rolling_window
def sliding_window(signal_dim, kernel_dim):
    """
    vectorized sliding window
    """
    # assert kernel_dim.shape == signal_dim
    for d, w in zip(signal_dim, kernel_dim):
        idx = np.arange(d - w + 1)[:,None] + np.arange(w)
        # out = arr_1*arr_2[idx]
    return idx



# ----FILTERS----
class Filter:
    @property    
    def params(self):
        return self.__dict__
        
    @property
    def name(self):
        return self._dim+' '+self._type+' '+self._tech

    @property
    def acr(self):
        return (self._dim[0] + self._type[0] + self._tech[0])

    @property
    def abr(self):
        return self._dim[:4].capitalize()+self._type[0].upper()+'P '+self._tech.capitalize()

    def info(self):
        return {'name':self.abr, 'params':self.params}
    
class SpatialLowpassMedian(Filter):
    _dim = 'spatial'
    _type = 'lowpass'
    _tech = 'median'

    def __init__(self, size=(3,3)):
        self.size = size
    
    def _filt(self, arr):
        return nd.median_filter(arr, size=(*self.size,1))        
    
class SpatialLowpassGaussian(Filter):
    _dim = 'spatial'
    _type = 'lowpass'
    _tech = 'gaussian'

    def __init__(self, sigma=(1,1)):
        self.sigma = sigma
    
    def _filt(self, arr):
        return nd.gaussian_filter(arr, sigma=(*self.sigma,0))        

class TemporalHighpassMedian(Filter):
    _dim = 'temporal'
    _type = 'highpass'
    _tech = 'median'

    def __init__(self, size=(7,7,100)):
        self.size = size

    def _filt(self, arr):
        return arr - nd.median_filter(arr, size=self.size)

class TemporalLowpassButter(Filter):
    _dim = 'temporal'
    _type = 'lowpass'
    _tech = 'butter'
    
    def __init__(self, order=4, f_cutoff=50, fs=625):
        self.order = order
        self.f_cutoff = f_cutoff
        self.fs = fs
        
    def _filt(self, A):
        b, a = signal.butter(self.order, self.f_cutoff, btype='lowpass', output='ba', fs=self.fs)
        # descriptor = {'name':'temporal lowpass butter', 'params':f'f_cutoff={f_cutoff}, order={order}'}
        return signal.filtfilt(b, a, A, axis=-1)
    

class TemporalBandpassButter(Filter):
    _dim = 'temporal'
    _type = 'bandpass'
    _tech = 'butter'
    
    def __init__(self, order=4, f_band=[1, 15], fs=625):
        self.order = order
        self.f_band = f_band
        self.fs = fs
        
    def _filt(self, A):
        b, a = signal.butter(self.order, self.f_band, btype='bandpass', output='ba', fs=self.fs)
        # descriptor = {'name':'temporal lowpass butter', 'params':f'f_cutoff={f_cutoff}, order={order}'}
        return signal.filtfilt(b, a, A, axis=-1)


class Downsample(Filter):
    _dim = 'temporal'
    _type = 'downsample'
    _tech = 'decimate'

    def __init__(self, factor=2):
        self.factor = factor

    def _filt(self, A):
        return signal.decimate(A, self.factor, axis=-1)


# bands of interest
delta_band = [1,4]
theta_band = [4, 15]
spindle_band = [9, 18]
low_gamma_band = [30, 60]
band_dict = {'delta': delta_band, 'theta': theta_band,
         'spindle_band': spindle_band, 'low_gamma_band':low_gamma_band}

class Hilbert(Filter):
    _dim = 'temporal'
    _type = 'analytic'
    _tech = 'hilbert'
    
    def __init__(self, order=4, f_band=[1, 15], fs=625):
        self.order = order
        self.f_band = f_band
        self.fs = fs
        
    def _filt(self, A):
        b, a = signal.butter(self.order, self.f_band, btype='bandpass', output='ba', fs=self.fs)
        # descriptor = {'name':'temporal lowpass butter', 'params':f'f_cutoff={f_cutoff}, order={order}'}
        return signal.filtfilt(b, a, A, axis=-1)


