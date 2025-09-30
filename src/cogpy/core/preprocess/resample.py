import numpy as np
import xarray as xr
from scipy.signal import resample_poly

def resample_sigx(sigx: xr.DataArray, f_d=64, axis='time') -> xr.DataArray:
    """
    Resample an xarray.DataArray along a specified axis to a target frequency.
    
    Parameters:
        sigx (xr.DataArray): Input signal.
        f_d (float): Target sampling frequency (Hz). Default is 64 Hz.
        axis (str): Name of the axis to resample along. Default is 'time'.
        
    Returns:
        xr.DataArray: Resampled signal with updated coordinates and attributes.
    """
    f_s = sigx.attrs.get('fs', None)
    if f_s is None:
        raise ValueError("Input DataArray must have a 'fs' attribute specifying the sampling frequency.")
    
    # Get the axis number and coordinates
    axis_num = sigx.get_axis_num(axis)
    original_coords = sigx.coords[axis]
    
    # Compute upsampling and downsampling factors
    p = int(f_d * 100)  # Upsampling factor
    q = int(f_s * 100)  # Downsampling factor
    
    # Apply resampling along the specified axis
    resampled_array = resample_poly(sigx.values, p, q, axis=axis_num)
    
    # Generate the new time coordinates
    resampled_coords = np.linspace(original_coords[0], original_coords[-1], resampled_array.shape[axis_num])
    
    # Update the coordinates and dimensions
    new_coords = {k: v for k, v in sigx.coords.items() if k != axis}
    new_coords[axis] = resampled_coords
    new_dims = sigx.dims
    
    # Create a new DataArray
    resampled_sigx = xr.DataArray(resampled_array, coords=new_coords, dims=new_dims)
    
    # Update attributes
    resampled_sigx.attrs = sigx.attrs
    resampled_sigx.attrs['fs'] = f_d
    
    return resampled_sigx

def resample_poly_sigx(sigx: xr.DataArray, up, down, axis='time') -> xr.DataArray:
    """
    Resample an xarray.DataArray along a specified axis to a target frequency.
    
    Parameters:
        sigx (xr.DataArray): Input signal.
        f_d (float): Target sampling frequency (Hz). Default is 64 Hz.
        axis (str): Name of the axis to resample along. Default is 'time'.
        
    Returns:
        xr.DataArray: Resampled signal with updated coordinates and attributes.
    """
    f_s = sigx.attrs.get('fs', None)
    if f_s is None:
        raise ValueError("Input DataArray must have a 'fs' attribute specifying the sampling frequency.")
    
    # Get the axis number and coordinates
    axis_num = sigx.get_axis_num(axis)
    original_coords = sigx.coords[axis]
    

    # Apply resampling along the specified axis
    resampled_array = resample_poly(sigx.values, up, down, axis=axis_num)
    
    # Generate the new time coordinates
    resampled_coords = np.linspace(original_coords[0], original_coords[-1], resampled_array.shape[axis_num])
    
    # Update the coordinates and dimensions
    new_coords = {k: v for k, v in sigx.coords.items() if k != axis}
    new_coords[axis] = resampled_coords
    new_dims = sigx.dims
    
    # Create a new DataArray
    resampled_sigx = xr.DataArray(resampled_array, coords=new_coords, dims=new_dims)
    
    # Update attributes
    resampled_sigx.attrs = sigx.attrs
    f_d = f_s * up / down
    resampled_sigx.attrs['fs'] = f_d
    
    return resampled_sigx
