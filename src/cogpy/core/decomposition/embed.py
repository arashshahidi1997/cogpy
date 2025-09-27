"""
Module for embedding techniques
"""

import numpy as np
import scipy.ndimage as nd

def _rolling_window(win, ref):
    """
    returns sliding windows indices of windowsize=win over a reference length=ref. 
    The output is useful for vectorized operations using sliding windows.

    Parameters
    ----------
    win: int
    ref: int
    """
    return np.arange(win, dtype=int).reshape(1,-1) + np.arange(ref-win+1, dtype=int).reshape(-1,1)

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

def embed_chan(x_embed, kernel_shape, bad_dims):
    """
    Parameters
    ----------
    x: array (grid_height, grid_width, ...)
    kernel_shape : (H,W,T)
    bad_dims : (grid_height, grid_width)

    Returns
    -------
    x_embed : array
    """
    kernel_grid = kernel_shape[0]
    kernel_halfwidth = kernel_grid//2
    valid_slice = slice(None)

    outside_win = bad_dims
    if kernel_halfwidth:
        valid_slice = slice(kernel_halfwidth, -kernel_halfwidth)
        outside_win = nd.maximum_filter(outside_win,
                                        footprint=np.ones((kernel_grid, kernel_grid))
                                       )
    outside_win = outside_win[valid_slice, valid_slice]
    x_embed = x_embed[np.where(~outside_win)]
    assert not np.isnan(x_embed).any(), 'NaN'
    return x_embed

def chan2grid(chan_embed, bad_dims, val=0):
    """
    Parameters
    ----------
    chan_embed: array (chan, ...)
    kernel_shape : (H,W,T)
    bad_dims : (grid_height, grid_width)

    Returns
    -------
    x_embed : array
    """
    grid_embed = np.ndarray((*bad_dims.shape, *chan_embed.shape[1:]))
    grid_embed[np.where(bad_dims)] = np.nan
    grid_embed[np.where(~bad_dims)] = roll_dim(chan_embed, -1)
    return grid_embed

