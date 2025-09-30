import numpy as np
from statsmodels.tsa.ar_model import AutoReg
from scipy.signal import lfilter
from ..preprocess.filt import Filter
from ..utils import xarr as xut

class AR_whiten(Filter):
    """
    Autoregressive whitening
    params:
    order: int
    """
    _dim = 'temporal'
    _type = 'whitening'
    _tech = 'autoregressive'

    def __init__(self, lags=[1, 2]):
        self.lags = lags
        self.coefs = None
        self.kernel = None

    @property
    def ar_filter(self):
        return np.array([1, *-np.array(self.kernel)])

    @property
    def ar_fir(self):
        return self.ar_filter, 1

    def fit(self, x):
        """
        Fit the autoregressive model to the input data.

        Parameters
        ----------
        x: array-like
            Input data array. The first dimension should be the number of channels.

        Returns
        -------
        None

        """
        params = []
        models = []
        for ich in range(x.shape[0]):
            model = AutoReg(x[ich], lags=self.lags).fit()
            params.append(model.params[1:])
            models.append(model)
        coefs = np.array(params)
        self.all_coefs = coefs.tolist()
        argmed = list(coefs[:,0]).index(np.percentile(coefs[:,0], 50, interpolation='nearest'))
        self.coefs = list(coefs[argmed])
        self.set_kernel()
        self.coefs_std = list(np.std(coefs, axis=0))

    @xut.xarr_wrap
    def _filt(self, x, axis=-1):
        """
        Apply AR filter to the input array along the specified axis.
        
        Parameters
        ----------
        x : xarray.DataArray
            Input data array.
        axis : int, optional
            Axis along which to apply the filter (default is -1).
        dim: str, optional
            Axis name to apply the filter along. If provided, the filter is applied along this axis.

        Returns
        -------
        xarray.DataArray
            Filtered data array.
        """
        b, a = self.ar_fir
        x_white = lfilter(b, a, x, axis=axis)
        return x_white

    def set_kernel(self):
        if self.lags[-1] != len(self.coefs):
            kernel = np.zeros(self.lags[-1])
            for ilag, lag_ in enumerate(self.lags):
                kernel[lag_-1] = self.coefs[ilag]
        else:
            kernel = self.coefs
        self.kernel = kernel
        
    def fit_transform(self, x):
        self.fit(x)
        return self._filt(x)
    
def AR_whitening(y, ar_params):
    """
    y: array (time,)
    ar_params: autoregressive kernel

    Returns
    -------
    whitened_y
    """
    ar_filt = [1, *-np.array(ar_params)]
    return np.convolve(ar_filt, y, mode='full')[:-len(ar_params)]

def autocovariance(X, p):
    '''
    calculate autocovariance for some data X
    p: time lag
    '''
    scale = len(X) - p
    autoCov = 0
    for i in np.arange(0, len(X)-p):
        autoCov += ((X[i+p]))*(X[i])
    return autoCov/scale - np.mean(X)**2

def AR_yule(p, X):
    '''
    estimate AR(p) coefficients a for data X by solving the Yule-
    Walker equation in matrix form
    
    p: order of AR model
    '''
    # x = zscore(X)
    # rxx = np.convolve(x, x[::-1], mode='full')[:len(x)-1][::-1][:p+1]/(np.arange(len(x)-p-1, len(x))[::-1])
    rxx = [autocovariance(X, i) for i in range(p+1)]

    Rxx_mat = np.zeros((p, p))
    for i in range(p):
        for j in range(i, p):
            Rxx_mat[i, j] = rxx[j-i]
            Rxx_mat[j, i] = Rxx_mat[i, j]

    a = np.linalg.inv(Rxx_mat) @ rxx[1:]
    return a

