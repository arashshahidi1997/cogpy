import numpy as np
import scipy.ndimage as nd
from scipy import signal

from numpy.lib.arraysetops import _unpack_tuple
# footprints

# depth 2
fp = nd.iterate_structure(nd.generate_binary_structure(2,1), 2)
fp_exclude = np.copy(fp)
fp_exclude[fp.shape[0]//2, fp.shape[1]//2] = False

channels_arr = np.arange(256).reshape(16,16)

def neighborhood_arr_generator(fp):
    """
    Parameters
    ----------
    ch: channel index
    fp: footprint 2d boolean array defining the neighborhood relative to the central element

    Returns
    -------
    loc_include_: (ch, h, w) including the picked channel
    loc_exclude_: (ch, h, w) excluding the picked channel
    """
    loc_exclude_ = []
    loc_include_ = []
    for ch in range(256):
        loc_exc = np.zeros((16,16))
        loc_exc[np.unravel_index(ch, (16,16))] = 1
        loc_exc = signal.convolve2d(loc_exc, fp, mode='same')
        loc_exc[np.unravel_index(ch, (16,16))] = 0
        loc_exclude_.append(loc_exc)

        loc_inc = np.zeros((16,16))
        loc_inc[np.unravel_index(ch, (16,16))] = 1
        loc_inc = signal.convolve2d(loc_inc, fp, mode='same')
        loc_include_.append(loc_inc)

    loc_exclude_ = np.array(loc_exclude_, dtype=bool)
    loc_include_ = np.array(loc_include_, dtype=bool)

    return loc_include_, loc_exclude_


loc_include, loc_exclude = neighborhood_arr_generator(fp)
fp_own = np.ones((1,1))
fp_collection = [fp, fp_own]
loc_collection = {np.sum(footp):neighborhood_arr_generator(footp) for footp in fp_collection}

def neighborhood_arr(ch, fp=fp, exclude=False):
    loc_ = loc_collection[np.sum(fp)]
    return loc_[exclude][ch]

def neighborhood(ch, **kwargs):
    loc_include_ch = neighborhood_arr(ch, **kwargs)
    neigh = channels_arr[np.where(loc_include_ch)]
    not_neigh = channels_arr[np.where(~loc_include_ch)]
    return neigh, not_neigh

def _rolling_window(win, ref, step=1, return_centers=False):
    """
    returns sliding windows indices of windowsize=win over a reference length=ref. 
    The output is useful for vectorized operations using sliding windows.

    Parameters
    ----------
    win: int
    ref: int
    
    Returns
    -------
    (roll_win, [centers])
    """
    roll_win = np.arange(win) + np.arange(0,ref-win+1,step)[:, None]
    ret = (roll_win.astype(int), )
    if return_centers:
        centers = roll_win[:,int(win//2)]
        ret += (centers, )
    return _unpack_tuple(ret)

def roll_dim(x, nroll):
    """
    reshapes the array x, for example if x.shape = (3,6,9,0)
    roll_dim(x,-1).shape = (6,9,0,3)
    Parameters
    ----------
    x: array (arbitrary dimensions)
    nroll: int

    Returns
    -------
    reshaped version of x with dimesnions rolled by `nroll` times.
    """
    return x.transpose(np.roll(np.arange(x.ndim), nroll))

def pad_left(kernel_shape, ndim, val=1):
    padded_kernel_shape = val * np.ones(ndim)
    padded_kernel_shape[-np.size(kernel_shape):] = kernel_shape
    return padded_kernel_shape

def generic_kernel(x, kernel_shape):
    """
    extracts windows of shape=kernel_shape around each element of x
    that is not close to boundaries of x such that kernel around the element
    lies inside x.
    
    Parameters
    ----------
    x: array (arbitrary dimensions)
        
    kernel_shape: tuple
        if len(kernel_shape) is smaller than the dimensionality of x \
        kernel_shape is paded by ones on the left side to match the \
        dimensionality of x
        
    Returns
    -------
    x_win: array (*x.shape, *(1,...,1), *kernel_shape)
        the shape of kernel to be extracted at each element of array x

    """
    padded_kernel_shape = pad_left(kernel_shape, x.ndim)

    # rolling window for each dimension
    roll = [_rolling_window(win, ref) for win, ref in zip(padded_kernel_shape, x.shape)]
    
    x_win = x
    for sliding_ind in roll:
        x_win = x_win[sliding_ind] # fancy indexing by sliding window
        x_win = x_win.transpose(np.roll(np.arange(x_win.ndim), -2)) 
        # roll dimensions by 2 (since we have the original dimension + the dimension
        # that contains the windows) to do fancy indexing over the next dimension
    
    even_dim_index = np.arange(0,2*x.ndim,2) # original dims and sliding window dims alternate
    odd_dim_index = even_dim_index + 1
    x_win = x_win.transpose(*even_dim_index, *odd_dim_index) # organizing dims such that shape=(original dims, sliding dims)
    return x_win
