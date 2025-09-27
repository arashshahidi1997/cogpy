"""
module for line noise estimation and removal
LineNoiseEstimator: class for line noise estimation and removal
	fit
	transform
	performance
	performance_sliding_window
	comparision_plot
	multitaper_func
	detect_linenoise_components
	sum_linenoise_harmonic_power
	set_attributes
	sort_linenoise_components

find_elbow: find elbow in a curve
get_linenoise_freqs: get line noise frequencies from mtx
drop_linenoise_freqs: drop line noise frequencies from mtx
drop_linenoise_harmonics: drop line noise harmonics from mtx
"""
from sklearn.base import TransformerMixin
from sklearn.decomposition import FastICA
import numpy as np
import pandas as pd
import xarray as xr
import scipy.stats as sts
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d
from ..spectral import multitaper as sp
from ..utils import sliding as sl
from ..utils.wrappers import ax_plot
from ..utils.curve import find_elbow
from ..utils.convert import closest_power_of_two

class LineNoiseEstimatorICA(TransformerMixin):
	def __init__(self, fs, linenoise_f0=50, halfbandwidth=4, nharmonics=2, ncomp=20, init_ica=None):
		self.fs = fs
		self.f0 = linenoise_f0
		self.w = halfbandwidth
		self.nh = nharmonics
		self.ncomp = ncomp
		hslices = self.f0 * np.arange(1,self.nh +1).reshape(-1, 1) + [-self.w, self.w]
		self.hslices = [slice(*band) for band in hslices]
		# mtx_scores: xr.DataArray (SpecGram) (comp, freq, time)
		window_size = closest_power_of_two(int(self.fs*1))
		window_step = window_size//8
		slider_kwargs = dict(window_size=window_size, window_step=window_step)
		zscore_threshold = 1.5
		NW = 7
		self.NW = NW
		self.slider_kwargs = slider_kwargs
		self.zscore_threshold = zscore_threshold

		# if init_ica is given, use its ICA components to initialize
		if init_ica is not None:
			self._set_init_from(init_ica)

	def fit(self, X):
		"""
		fit ICA to design matrix X
				
		Parameters
		----------
		X : array-like, shape (time, ch)
			Design matrix to fit the ICA model.

		Sets
		----
		self.ica
		self.ic_scores
		self.ic_loadings
		self.linenoise_components
		"""
		# drop NaN
		print("dropping potential NaNs at boundaries...")
		X = X[~np.isnan(X).any(axis=1)]
		w_init = getattr(self, 'w_init', None)
		self.ica = FastICA(self.ncomp, whiten='unit-variance', max_iter=1000, tol=0.001, w_init=w_init)
		print("Fitting ICA...")
		self.ic_scores = self.ica.fit_transform(X).T # (comp, ch)
		self.ic_loadings = xr.DataArray(self.ica.components_, dims=['comp', 'ch'])

		# detect line noise components
		self._detect_and_set_linenoise_components(self.NW, self.slider_kwargs, self.zscore_threshold)

		# pick and sort linenoise components
		self.lnoise_ic_scores = self.ic_scores[self.linenoise_components]
		self.lnoise_ic_loadings = self.ic_loadings[self.linenoise_components]
		self.lnoise_ica_mixing = self.ica.mixing_[:, self.linenoise_components]

	def transform(self, X):
		"""
		X: array-like (time, ch) signal
		Returns:
		linenoise_sig: array (time, ch)
			reconstruction line noise signal

		"""
		# get IC scores for full signal
		ic_scores = self.ica.transform(X).T # shape (comp, time)
		lnoise_ic_scores = ic_scores[self.linenoise_components]
		return self._construct_linenoise_signal(lnoise_ic_scores)

	def fit_transform(self, X):
		"""
		X: array-like (time, ch) signal
		"""
		self.fit(X)
		return self._construct_linenoise_signal(self.lnoise_ic_scores)

	def _set_init_from(self, init_ica):
		# The relation is: components_ ≈ W @ K   ⇒   W ≈ components_ @ pinv(K)
		K = getattr(init_ica, "whitening_", None)
		if K is None:
			raise RuntimeError("Your scikit-learn version may not expose `whitening_`. "
							"Upgrade scikit-learn (>=1.2-ish) or use Method 2 below.")
		self.w_init = init_ica.components_ @ np.linalg.pinv(K)

	def _construct_linenoise_signal(self, lnoise_ic_scores):
		# reconstruct line noise signal
		linenoise_sig = np.dot(self.lnoise_ica_mixing, lnoise_ic_scores).T + self.ica.mean_
		return linenoise_sig

	def _detect_and_set_linenoise_components(self, NW, slider_kwargs, zscore_threshold=1.5):
		mtm_kwargs = dict(NW=NW, fs=self.fs, **slider_kwargs)
		mtm_gsp_kwargs = sp.mtm_kwarg_to_gsp(**mtm_kwargs)
		ic_scx = xr.DataArray(self.ic_scores, dims=['comp', 'time'], coords={'time': np.arange(self.ic_scores.shape[1])/self.fs})
		# spectrogram of IC scores
		mtx_scores = sp.mtm_spectrogramx(ic_scx, **mtm_gsp_kwargs)
		# sum linenoise harmonics
		linenoise_power = xr.concat([mtx_scores.sel(freq=slc) for slc in self.hslices], dim='freq').sum(dim='freq')
		linenoise_power = linenoise_power.sum(dim='time') / mtx_scores.sum(dim=('freq', 'time'))
		zlinenoise = sts.zscore(linenoise_power)
		zlinenoise_df = pd.Series(zlinenoise, name='noise_zpower').reset_index().sort_values('noise_zpower', ascending=False)
		zlinenoise_df['is_linenoise'] = zlinenoise_df['noise_zpower'] > zscore_threshold

		self.linenoise_components = zlinenoise_df['index'][zlinenoise_df['is_linenoise']].values

def sliding_ICA(sigx: xr.DataArray, fs, segment_size, lnoiseICA_params):
	assert sigx.dims == ('time', 'ch'), "sigx must have dimensions ('time', 'ch')"
	sigx_coarsen = sigx.coarsen({'time': segment_size}, boundary='pad').construct(time=('segment', 'time'))

	# fit ICA on the first segment to initialize
	lnoise_init = LineNoiseEstimatorICA(fs, **lnoiseICA_params)
	X_init = sigx_coarsen.isel(segment=0).data
	lnoise_init.fit(X_init)
	lnoise_estimator = LineNoiseEstimatorICA(
				fs, 
				init_ica=lnoise_init.ica,
				**lnoiseICA_params
			)
	# ICA estimator set up with initialization
	def _ica_lnoise(segment_data):
		# segment_data (time, ch)
		# detect NaN times
		nan_times = np.isnan(segment_data).any(axis=1) # (time,)
		# take the non-NaN times
		segment_data_nonan = segment_data[~nan_times]
		segment_lnoise_nonan = lnoise_estimator.fit_transform(segment_data_nonan)
	
		# append NaN times back
		segment_lnoise = np.full(segment_data.shape, np.nan)
		segment_lnoise[~nan_times] = segment_lnoise_nonan
		return segment_lnoise
	
	# apply ICA to each segment
	lnoise_estimate = xr.apply_ufunc(
		_ica_lnoise,
		sigx_coarsen,
		input_core_dims=[['time', 'ch']],
		output_core_dims=[['time', 'ch']],
		vectorize=True,
		dask='parallelized',
		keep_attrs=True,
		output_dtypes=[sigx.dtype]
	)

	# repatch segments
	lnoise_estimate = xr.concat([lnoise_estimate.isel(segment=i) for i in range(lnoise_estimate.sizes['segment'])], dim='time')
	
	# drop NaN times
	valid_time_idx_slice = slice(0, sigx.sizes['time'])
	lnoise_estimate = lnoise_estimate.isel(time=valid_time_idx_slice)
	assert np.allclose(np.isnan(lnoise_estimate.data), np.isnan(sigx.data)), "NaN times of linenoise estimate and original signal do not match!"
	return lnoise_estimate

# def performance_sliding_window(sigx: xr.DataArray, clean_sigx: xr.DataArray):
# TODO: implement sliding window performance
#     measure = lambda x: LineNoiseEstimator.performance(x[0], x[1])
#     sliding_measure = SlidingMeasure(measure, window_size=1000, window_step=1000)
#     X = np.array([sigx.transpose('ch','time').data.T, clean_sigx.transpose('ch','time').data.T])
#     Xmeasure = sliding_measure.measure_view(X)
#     return Xmeasure

@ax_plot
def comparision_plot(self, sigx, clean_sigx, ax=None):
	"""
	sigx: xr.DataArray (ch, time)
	clean_sigx: xr.DataArray (ch, time)
	"""
	raw_mt_sig = self.multitaper_func(sigx).sum(dim='time')
	clean_mt_sig = self.multitaper_func(clean_sigx).sum(dim='time')
	raw_mt_sig.plot(ax=ax, label='raw')
	clean_mt_sig.plot(ax=ax, label='clean')
	ax.legend()
	return ax

def performance(self, sigx, clean_sigx):
	"""
	Parameters
	----------
	sigx: xr.DataArray (ch, time)
	clean_sigx: xr.DataArray (ch, time)

	Returns
	-------
	score: float
	"""
	raw_psd = self.multitaper_func(sigx).sum(dim=('time', 'ch'))
	clean_psd = self.multitaper_func(clean_sigx).sum(dim=('time', 'ch'))

	# baseline powe at 50Hz
	# baseline is measured from a line connecting the left and right edges of the linenoise band
	# left elbow
	lf_psd = raw_psd.sel(freq=slice(None, 50))
	df = lf_psd.freq.data[1] - lf_psd.freq.data[0]
	lf_psd_smooth = gaussian_filter1d(lf_psd.data, sigma=5/df)
	f1, p1 = find_elbow(lf_psd_smooth.data[::-1])
	
	# right elbow
	hf_psd = raw_psd.sel(freq=slice(50, None))
	hf_psd_smooth = gaussian_filter1d(hf_psd.data, sigma=5/df)
	f2, p2 = find_elbow(hf_psd_smooth.data)

	delta_f = f2 - f1
	# (45, left_edge), (55, right_edge) form line as baseline
	baseline = lambda x: (p2 - p1)/delta_f*(x-f1) + p1
	baseline_50 = baseline(50)

	# get power at 50Hz
	raw_psd_50 = raw_psd.sel(freq=50)
	clean_psd_50 = clean_psd.sel(freq=50)

	# percentage of linenoise power removed
	score = (clean_psd_50 - baseline_50)/(raw_psd_50 - baseline_50)
	return score

class LineNoiseEstimatorMultitaper:
	def __init__(self, N, NW):
		self.N = N
		self.tapers = sp.dpss_tapers(N, NW)
		self.K_max = self.tapers.shape[0]
		self.mt_func = partial(sp.multitaper_fft, tapers=self.tapers)

		# U(0) is the DC component of the tapers, calculated here by summing over time
		self.u0 = xr.DataArray(self.tapers.sum(axis=-1), dims=['taper'])
		self.signif_level = 1 - 1/self.N
		self.signif_threshold = self.compute_f_test_threshold()
		
	def mu_func(self, mt_fft: xr.DataArray, taper_dim='taper'):
		"""
		mt_fft: (..., taper, freq)
		"""
		return (mt_fft * self.u0).sum(taper_dim) / (self.u0 * self.u0).sum(taper_dim)

	def mu_f_stat_func(self, mt_fft: xr.DataArray, taper_dim='taper'):
		"""
		mu: (..., freq)
		mt_fft: (..., freq, taper)
		"""
		mu = self.mu_func(mt_fft)
		return np.abs(mu) ** 2 / (np.abs(mt_fft - mu * self.u0)).sum(dim=taper_dim)

	def compute_f_test_threshold(self):
		"""
		F-statistic significance threshold

		`Under the null hypothesis that there is no line present, F(f0) has an F distribution with (2, 2K 2) degrees of freedom.
		One obtains an independent F-statistic every Raleigh frequency, and since there are N Raleigh frequencies in the spectrum, 
		statistical significance level is chosen to be 1 1/N. This means that on an average, there will be at most one false detection 
		of a sinusoid across all frequencies.`
		"""
		f_dist_dof = (2, 2*self.K_max-2)
		signif_thresh = sts.f.ppf(self.signif_level, *f_dist_dof)
		return signif_thresh

	def f_test(self, mu_f_stat): 		
		return mu_f_stat > self.signif_threshold

	def reconstruct_lnoise_analytic(self, mt_fft, f0, t):
		mt_fft_at_f0 = mt_fft.sel(freq=f0, method='nearest')
		mu_at_f0 = self.mu_func(mt_fft_at_f0)
		return mu_at_f0 * np.exp(2*np.pi*f0*1j*t)

def interpolate_local_50Hz(tk, Ak, phik, t_out, fc=50.0):
	# unwrap and demodulate phase
	psi_k = np.unwrap(phik - 2*np.pi*fc*tk)
	# shape-preserving interpolation
	A_itp  = PchipInterpolator(tk, Ak)
	psi_itp = PchipInterpolator(tk, psi_k)
	Ahat  = A_itp(t_out)
	psih  = psi_itp(t_out)
	muhat = Ahat * np.exp(1j*psih)              # complex envelope
	xhat  = np.real( muhat * np.exp(1j*2*np.pi*fc*t_out) )  # real signal
	# finst = fc + np.gradient(psih, t_out) / (2*np.pi)       # optional
	return xhat


# %% Line Noise Harmonics
def get_linenoise_freqs(freqs, f0=50, w=5, nh=3):
	hslices = f0 * np.arange(1,nh +1).reshape(-1, 1) + [-w, w]
	ln_closefreqs = []
	for fl, fh in hslices:
		cnd = (freqs >= fl) & (freqs <= fh)
		if not cnd.any():
			continue
		else:
			ln_closefreqs.append(freqs[cnd])
	return np.concatenate(ln_closefreqs) if len(ln_closefreqs) else []

def drop_linenoise_freqs(freqs, f0=50, w=5, nh=3):
	ln_closefreqs = get_linenoise_freqs(freqs, f0, w, nh)
	return freqs[~np.isin(freqs, ln_closefreqs)]

def drop_linenoise_harmonics(mtx, f0=50, w=5, nh=3):
	"""
	mtx: xr.DataArray with freq coordinate
	f0: fundamental frequency of line noise
	w: width of the band to drop around the harmonics
	nh: number of harmonics to drop
	"""
	ln_closefreqs = get_linenoise_freqs(mtx.freq.values, f0, w, nh)
	return mtx.drop_sel(freq=ln_closefreqs) if len(ln_closefreqs) else mtx
