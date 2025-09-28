import numpy as np
import xarray as xr
import zarr
from functools import partial
from pathlib import Path
from dask.diagnostics import ProgressBar
from contextlib import nullcontext
from .channel_feature_functions import (
	anticorrelation,
	relative_variance,
	deviation,
	amplitude,
	time_derivative,
	hurst_exponent,
	temporal_mean_laplacian,
	local_robust_zscore
)
from ..utils.sliding import running_measure, running_measure_sane, rolling_win, xroll_apply
from ..utils.grid_neighborhood import make_footprint, adjacency_matrix

FEATURE_NAMES = [
	"anticorrelation",
	"relative_variance",
	"deviation",
	"amplitude",
	"time_derivative",
	"hurst_exponent",
	"temporal_mean_laplacian",
]

class ChannelFeatures:
	"""
	Class to compute running channel features over time using a sliding window approach.
	Parameters
	----------
	nrows : int
		Number of rows corresponding to Anterior-Postierior axis.
	ncols : int
		Number of columns corresponding to Medial-Lateral axis.
	connectivity_size : int, optional
		Size of the connectivity for the neighborhood footprint. If provided, it overrides `footprint
		`. Default is None.
	footprint : np.ndarray, optional
		Custom boolean footprint array for neighborhood operations. If provided, it overrides `connectivity_size
		`. Default is None.
	row_dim : str, optional
		Name of the row dimension in the input data. Default is 'AP'.
	col_dim : str, optional
		Name of the column dimension in the input data. Default is 'ML'.
	time_dim : str, optional
		Name of the time dimension in the input data. Default is 'time'.
	zscore : bool, optional
		Whether to apply local robust z-scoring to the features. Default is True.
	Attributes
	----------
	nrows : int
		Number of rows corresponding to Anterior-Postierior axis.
	ncols : int
		Number of columns corresponding to Medial-Lateral axis.
	grid_shape : tuple
		Tuple representing the grid shape (nrows, ncols).
	feature_dict : dict
		Dictionary mapping feature names to their corresponding functions.
	feature_names : list
		List of feature names.
	nfeatures : int
		Total number of features computed.
	slider_kwargs_default : dict
		Default keyword arguments for the sliding window function.
	Methods
	-------
	measure(arr, zscore)
		Compute all features for a given window of data.
	transform(xsig, slider_kwargs=None, zscore)
		Compute running channel features over time using a sliding window approach.
	__repr__()
		String representation of the ChannelFeatures object.
	Example
	-------
	>>> from cogpy.datasets import load as ld
	>>> from cogpy.preprocess.channel_feature import ChannelFeatures
	>>> xsig = ld.load_sample()
	>>> nrows = xsig.sizes['AP']
	>>> ncols = xsig.sizes['ML']
	>>> feats = ChannelFeatures(nrows=nrows, ncols=ncols)
	>>> features = feats.transform(xsig, slider_kwargs=dict(window_size=512, window_step=64), zscore=True)
	>>> isinstance(features, xr.Dataset)
	True
	"""
	def __init__(self, nrows=16, ncols=16, connectivity_size=None, footprint=None, row_dim='AP', col_dim='ML', time_dim='time'):
		self.nrows = nrows
		self.ncols = ncols
		self.row_dim = row_dim
		self.col_dim = col_dim
		self.time_dim = time_dim
		self.grid_shape = (nrows, ncols)
		if (footprint is not None) and (connectivity_size is not None):
			raise ValueError("footprint and connectivity_size cannot both be provided. Choose one or neither.")
		if connectivity_size is not None:
			if self.nrows == 1 or self.ncols == 1:
				footprint = make_footprint(rank=1, connectivity=1, niter=connectivity_size)
			if self.nrows > 1 and self.ncols > 1:
				footprint = make_footprint(rank=2, connectivity=1, niter=connectivity_size)
		elif footprint is not None:
			assert footprint.ndim == 2, "Footprint must be 2D."
			assert footprint.shape[0] % 2 == 1 and footprint.shape[1] % 2 == 1, "Footprint dimensions must be odd."
			# assert bool array
			assert np.issubdtype(footprint.dtype, np.bool_), "Footprint must be a boolean array."
		else:
			if self.nrows == 1 or self.ncols == 1:
				footprint = make_footprint(rank=1, connectivity=1, niter=2)
			if self.nrows > 1 and self.ncols > 1:
				footprint = make_footprint(rank=2, connectivity=1, niter=2)

		self.footprint = footprint
		self.adj = adjacency_matrix(self.grid_shape, footprint=self.footprint, exclude=True)

		self.feature_dict = {
			"anticorrelation": partial(anticorrelation, adj=self.adj),
			"relative_variance": relative_variance,
			"deviation": deviation,
			"amplitude": amplitude,
			"time_derivative": time_derivative,
			"hurst_exponent": hurst_exponent,
			"temporal_mean_laplacian": temporal_mean_laplacian,
		}
		self.zscore_func = partial(local_robust_zscore, footprint=self.footprint)
		self.feature_names = list(self.feature_dict.keys())
		self.nfeatures = len(self.feature_dict)
		self.slider_kwargs_default = dict(window_size=512, window_step=64)

	def compute_features(self, xsig, window_size=512, window_step=64, run_dim='time', zscore=True, run_chunk=1000):
		xsig = xsig.astype(np.float32)
		x_roll = rolling_win(xsig, window_size=window_size, window_step=window_step, dim=run_dim)
		x_roll = x_roll.chunk({self.row_dim: -1, self.col_dim: -1, run_dim: run_chunk, 'window': -1})
		
		feat_ds = {}
		for feature_name in self.feature_dict.keys():
			feat_ds[feature_name] = xroll_apply(
				self._feature_measure_func,
				x_roll,
				**self._roll_apply_kwargs(feature_name, zscore),
			)
		feat_ds = xr.Dataset(feat_ds)
		return feat_ds

	def _roll_apply_kwargs(self, feature_name, zscore):
		return dict(
			measure_kwargs={'feature_name': feature_name, 'zscore': zscore},
			measure_input_core_dims=[[self.row_dim, self.col_dim, 'window']],
			measure_output_core_dims=[[self.row_dim, self.col_dim]],
			measure_output_sizes={self.row_dim: self.nrows, self.col_dim: self.ncols},
			output_dtype=np.float32,
			name=feature_name,
			dask='parallelized',
		)

	def _running_apply_kwargs_single_feature(self, feature_name):
		"""
		Keyword arguments for the running_measure function.
		These define how the measure function is applied over sliding windows.

		Returns
		-------
		dict
			Dictionary of keyword arguments for running_measure.		
		"""
		return dict(
			measure_input_core_dims=[[self.row_dim, self.col_dim, 'window']],
			measure_output_core_dims=[[self.row_dim, self.col_dim]],
			measure_output_sizes={self.row_dim: self.nrows, self.col_dim: self.ncols},
			run_dim=self.time_dim,
			window_dim='window',
			output_dtype=np.float32,
			name=feature_name
		)

	def _feature_measure_func(self, arr, feature_name, zscore):
		if zscore:
			arr = self.zscore_func(self.feature_dict[feature_name](arr))
		else:
			arr = self.feature_dict[feature_name](arr)
		return arr

	def transform_feature(self, xsig, feature_name, slider_kwargs, zscore) -> xr.DataArray:
		"""
		Compute a single running channel feature over time using a sliding window approach.

		Parameters
		----------
		xsig : xarray.DataArray
			Input signal with dimensions ('AP', 'ML', 'time').
		feature_name : str
			Name of the feature to compute. Must be one of self.feature_names.
		slider_kwargs : dict, optional
			dict(
				window_size=,
				window_step=	
			)
			Dictionary of keyword arguments for the sliding window function.
			Defaults to None, which uses self.slider_kwargs_default.
			dict(window_size=512, window_step=64)
		zscore : bool, optional
			Whether to apply local robust z-scoring to the feature.

		Returns
		-------
		feature : xarray.DataArray
			DataArray containing the computed feature with dimensions ('AP', 'ML', 'time').

		Example
		--------
		>>> from cogpy.datasets import load as ld
		>>> from cogpy.preprocess.channel_feature import ChannelFeatures
		>>> xsig = ld.load_sample()
		>>> nrows = xsig.sizes['AP']
		>>> ncols = xsig.sizes['ML']
		>>> feats = ChannelFeatures(nrows=nrows, ncols=ncols)
		>>> feature = feats.transform_feature(xsig, feature_name='deviation', slider_kwargs=dict(window_size=512, window_step=64), zscore=True)
		>>> isinstance(feature, xr.DataArray)
		True
		"""
		if feature_name not in self.feature_names:
			raise ValueError(f"feature_name must be one of {self.feature_names}, got '{feature_name}'")
		
		feature = running_measure(
			self._feature_measure_func,
			xsig,
			measure_kwargs={'feature_name': feature_name, 'zscore': zscore},
			slider_kwargs=slider_kwargs or self.slider_kwargs_default,
			**self._running_apply_kwargs_single_feature(feature_name)
		)
		return feature.transpose(self.row_dim, self.col_dim, self.time_dim)

	def transform_dask(self, xsig_da: xr.DataArray, slider_kwargs, zscore, verbose=False) -> xr.Dataset:
		"""
		Make a dask dataset version of the features

		Parameters
		----------
		xsig_da : xr.DataArray
			Input signal with dimensions ('AP', 'ML', 'time').

		Returns
		-------
		features : xr.Dataset
			Dataset containing all computed features with dimensions ('feature', 'AP', 'ML', 'time').
		"""
		features = {}
		for feature_name in self.feature_names:
			if verbose:
				print(f"\t- {feature_name}")
			features[feature_name] = self.transform_feature(
				xsig_da, feature_name=feature_name,  slider_kwargs=slider_kwargs, zscore=zscore
				)
		features = xr.Dataset(features)
		return xr.Dataset(features)

	@property
	def _running_apply_kwargs(self):
		"""
		Keyword arguments for the running_measure function.
		These define how the measure function is applied over sliding windows.

		Returns
		-------
		dict
			Dictionary of keyword arguments for running_measure.		
		"""
		return dict(
			measure_input_core_dims=[[self.row_dim, self.col_dim, 'window']],
			measure_output_core_dims=[['feature', self.row_dim, self.col_dim]],
			measure_output_sizes=dict(feature=self.nfeatures, **{self.row_dim: self.nrows, self.col_dim: self.ncols}),
			run_dim=self.time_dim,
			window_dim='window',
			output_dtype=np.float32,
			name='feature_value'
		)

	def measure(self, arr, zscore):
		"""
		Joint computation of all features for a given window `arr` of data.

		Parameters
		----------
		arr : xarray.DataArray
			Input data array with dimensions ('AP', 'ML', 'time').
			Shape should be (self.AP, self.ML, window_size).

		Returns
		-------
		feature_arr : np.ndarray
			Array of computed features with shape (nfeatures, AP, ML).
		"""
		feature_arr = np.empty((self.nfeatures, self.nrows, self.ncols), dtype=np.float32)
		for i, (feat, feat_func) in enumerate(self.feature_dict.items()):
			feature_arr[i] = feat_func(arr)
			if zscore:
				feature_arr[i] = self.zscore_func(feature_arr[i])
		return feature_arr
	
	def transform(self, xsig, slider_kwargs, zscore) -> xr.Dataset:
		"""
		Compute running channel features over time using a sliding window approach.

		Parameters
		----------
		xsig : xarray.DataArray
			Input signal with dimensions ('AP', 'ML', 'time').
		slider_kwargs : dict, optional
			dict(
				window_size=,
				window_step=	
			)
			Dictionary of keyword arguments for the sliding window function.
			Defaults to None, which uses self.slider_kwargs_default.
			dict(window_size=512, window_step=64)

		Returns
		-------
		feature : xarray.DataArray
			DataArray containing the computed features with dimensions ('feature', 'AP', 'ML', 'time').

		Example
		--------
		>>> from cogpy.datasets import load as ld
		>>> from cogpy.preprocess.channel_feature import ChannelFeatures
		>>> xsig = ld.load_sample()
		>>> nrows = xsig.sizes['AP']
		>>> ncols = xsig.sizes['ML']
		>>> feats = ChannelFeatures(nrows=nrows, ncols=ncols)
		>>> features = feats.transform_dask(xsig, slider_kwargs=dict(window_size=512, window_step=64), zscore=True)
		>>> isinstance(features, xr.Dataset)
		True
		"""
		feature = running_measure(
			self.measure,
			xsig,
			measure_kwargs={'zscore': zscore},
			slider_kwargs=slider_kwargs or self.slider_kwargs_default,
			**self._running_apply_kwargs
		)
		# add coordinates
		feature = feature.assign_coords(feature=self.feature_names).transpose('feature', self.row_dim, self.col_dim, self.time_dim)
		return xr.Dataset({name: feature.sel(feature=name, drop=True) for name in self.feature_names})\
			.assign({'feature_stacked': feature})
	
	def __repr__(self) -> str:
		feature_descriptions = {
			"anticorrelation": "Spatial anti-correlation with neighbors",
			"relative_variance": "Relative variance",
			"deviation": "Deviation from mean",
			"amplitude": "Signal amplitude",
			"time_derivative": "Temporal derivative",
			"hurst_exponent": "Hurst exponent (long-term memory)",
			"temporal_mean_laplacian": "Mean Laplacian over time",
		}
		repr_mssg = "To compute features of xsig: xr.DataArray with dimensions corresponding to (rows, cols, time), use:\n\t.transform(xsig)\n\n"
		repr_mssg += "ChannelFeatures Object\n=======================\n"
		repr_mssg += f"nrows: {self.nrows}\nncols: {self.ncols}\nGrid Shape: {self.grid_shape}\nNumber of Features: {self.nfeatures}\n"
		repr_mssg += "Features:\n"
		repr_mssg += "+--------------------------+-----------------------------------------------+\n"
		repr_mssg += "| Feature                  | Description                                   |\n"
		repr_mssg += "+==========================+===============================================+\n"
		for feat in self.feature_dict.keys():
			desc = feature_descriptions.get(feat, "")
			repr_mssg += f"| {feat:<24} | {desc:<45} |\n"
		repr_mssg += "+------------------------+-----------------------------------------------+\n"
		return repr_mssg

def save_features(feat_ds, zarr_path, encoding_per_var=None, consolidate=True, show_progress=True):
	"""
	Compute features (using compute_features) and write them to a Zarr store
	feature by feature (one variable at a time).

	Parameters
	----------
	feat_ds : xr.Dataset
		Dataset containing all computed features as `data_vars` with dimensions ('AP', 'ML', 'time').
	zarr_path : str or Path
		Destination path for the Zarr store.
	encoding_per_var : dict, optional
		Per-variable encoding options (e.g., compression, chunking).
	consolidate : bool, default True
		Whether to consolidate Zarr metadata at the end.
    show_progress : bool, default True
        Show a Dask progress bar during computation (supports local and distributed schedulers).
	"""
	zarr_path = Path(zarr_path)

	# 1) Write each variable sequentially
	first = True
	for var in feat_ds.data_vars:
		ds_one = feat_ds[[var]]  # single-variable Dataset

		enc = None
		if encoding_per_var and var in encoding_per_var:
			enc = {var: encoding_per_var[var]}

		if first:
			# write coords + first variable
			print(f"{var}: ... \n\t")
			with ProgressBar() if show_progress else nullcontext():
				ds_one.to_zarr(zarr_path, mode="w", encoding=enc, zarr_format=2)
			first = False
		else:
			# append additional variables
			print(f"{var}: ... \n\t")
			with ProgressBar() if show_progress else nullcontext():
				ds_one.to_zarr(zarr_path, mode="a", encoding=enc, zarr_format=2)

	# 2) Consolidate metadata for efficient reopening
	if consolidate:
		zarr.consolidate_metadata(str(zarr_path))

def ensure_footprint(nrows, ncols, connectivity_size=None, footprint=None):
	if (footprint is not None) and (connectivity_size is not None):
		raise ValueError("footprint and connectivity_size cannot both be provided. Choose one or neither.")
	if connectivity_size is not None:
		if nrows == 1 or ncols == 1:
			footprint = make_footprint(rank=1, connectivity=1, niter=connectivity_size)
		if nrows > 1 and ncols > 1:
			footprint = make_footprint(rank=2, connectivity=1, niter=connectivity_size)
	elif footprint is not None:
		assert footprint.ndim == 2, "Footprint must be 2D."
		assert footprint.shape[0] % 2 == 1 and footprint.shape[1] % 2 == 1, "Footprint dimensions must be odd."
		# assert bool array
		assert np.issubdtype(footprint.dtype, np.bool_), "Footprint must be a boolean array."
	else:
		if nrows == 1 or ncols == 1:
			footprint = make_footprint(rank=1, connectivity=1, niter=2)
		if nrows > 1 and ncols > 1:
			footprint = make_footprint(rank=2, connectivity=1, niter=2)
	return footprint

def transform_dask(xsig_da: xr.DataArray, slider_kwargs, zscore, feature_names=FEATURE_NAMES, sane=False, verbose=False) -> xr.Dataset:
	"""
	Make a dask dataset version of the features

	Parameters
	----------
	xsig_da : xr.DataArray
		Input signal with dimensions ('AP', 'ML', 'time').

	Returns
	-------
	features : xr.Dataset
		Dataset containing all computed features with dimensions ('feature', 'AP', 'ML', 'time').
	"""
	features = {}
	for feature_name in feature_names:
		if verbose:
			print(f"\t- {feature_name}")
		features[feature_name] = transform_feature(
			xsig_da, feature_name=feature_name,  slider_kwargs=slider_kwargs, zscore=zscore, sane=sane
			)
	return xr.Dataset(features)

def _feature_measure_func(arr, feature_func_, zscore, footprint):
	arr = feature_func_(arr)
	if zscore:
		zscore_func = partial(local_robust_zscore, footprint=footprint)
		arr = zscore_func(arr)
	return arr

def transform_feature(xsig, feature_name, slider_kwargs, zscore, footprint=None, row_dim='AP', col_dim='ML', time_dim='time', sane=False) -> xr.DataArray:
	"""
	Compute a single running channel feature over time using a sliding window approach.

	Parameters
	----------
	xsig : xarray.DataArray
		Input signal with dimensions ('AP', 'ML', 'time').
	feature_name : str
		Name of the feature to compute. Must be one of self.feature_names.
	slider_kwargs : dict, optional
		dict(
			window_size=,
			window_step=	
		)
		Dictionary of keyword arguments for the sliding window function.
		Defaults to None, which uses self.slider_kwargs_default.
		dict(window_size=512, window_step=64)
	zscore : bool, optional
		Whether to apply local robust z-scoring to the feature.

	Returns
	-------
	feature : xarray.DataArray
		DataArray containing the computed feature with dimensions ('AP', 'ML', 'time').

	Example
	--------
	>>> from cogpy.datasets import load as ld
	>>> from cogpy.preprocess.channel_feature import ChannelFeatures
	>>> xsig = ld.load_sample()
	>>> nrows = xsig.sizes['AP']
	>>> ncols = xsig.sizes['ML']
	>>> feats = ChannelFeatures(nrows=nrows, ncols=ncols)
	>>> feature = feats.transform_feature(xsig, feature_name='deviation', slider_kwargs=dict(window_size=512, window_step=64), zscore=True)
	>>> isinstance(feature, xr.DataArray)
	True
	"""
	nrows = xsig.sizes[row_dim]
	ncols = xsig.sizes[col_dim]

	# check feature_name
	if feature_name not in FEATURE_NAMES:
		raise ValueError(f"feature_name must be one of {FEATURE_NAMES}, got '{feature_name}'")
	
	# get feature function
	feature_func_ = {
		"anticorrelation": partial(anticorrelation, adj=adjacency_matrix((nrows, ncols))),
		"relative_variance": relative_variance,
		"deviation": deviation,
		"amplitude": amplitude,
		"time_derivative": time_derivative,
		"hurst_exponent": hurst_exponent,
		"temporal_mean_laplacian": temporal_mean_laplacian,
	}[feature_name]

	# ensure footprint
	footprint = ensure_footprint(nrows, ncols, connectivity_size=None, footprint=footprint)

	# ensure slider_kwargs
	slider_kwargs = slider_kwargs or dict(window_size=512, window_step=64)	

	if sane:
		running_measure_callable = running_measure_sane
	else:
		running_measure_callable = running_measure
	feature = running_measure_callable(
		_feature_measure_func,
		xsig,
		measure_kwargs={'feature_func_': feature_func_, 'zscore': zscore, 'footprint': footprint},
		slider_kwargs=slider_kwargs,
		**_running_apply_kwargs_single_feature(feature_name, nrows, ncols, row_dim=row_dim, col_dim=col_dim, time_dim=time_dim)
	)
	return feature.transpose(row_dim, col_dim, time_dim)

def _running_apply_kwargs_single_feature(feature_name, nrows, ncols, row_dim='AP', col_dim='ML', time_dim='time'):
	"""
	Keyword arguments for the running_measure function.
	These define how the measure function is applied over sliding windows.

	Parameters
	----------
	feature_name : str
		Name of the feature to compute.
	row_dim : str
		Name of the row dimension in the input data.
	col_dim : str
		Name of the column dimension in the input data.
	time_dim : str
		Name of the time dimension in the input data.
	nrows : int
		Number of rows corresponding to Anterior-Postierior axis.
	ncols : int
		Number of columns corresponding to Medial-Lateral axis.

	Returns
	-------
	dict
		Dictionary of keyword arguments for running_measure.		
	"""
	return dict(
		measure_input_core_dims=[[row_dim, col_dim, 'window']],
		measure_output_core_dims=[[row_dim, col_dim]],
		measure_output_sizes={row_dim: nrows, col_dim: ncols},
		run_dim=time_dim,
		window_dim='window',
		output_dtype=np.float32,
		name=feature_name
	)
