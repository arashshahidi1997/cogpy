import numpy as np
import xarray as xr
import pandas as pd
import scipy.ndimage as nd
import scipy.signal as scisig
from ...preprocess import linenoise as ln
from ...spectral.process_spectrogram import median_spec
from ...utils._functools import untuple
from ...utils import (xarr as xut, time_series as ts)
from .plot import plot_maxfreq_slice_loadings
from .scores import *

class DesignMatrixReshaper:
    def __init__(self):
        self.original_dims = None
        self.stacked_coords = None

    def stack_except(self, data_array, exclude_dim):
        if exclude_dim not in data_array.dims:
            raise ValueError(f"Dimension '{exclude_dim}' not found in data array dimensions: {data_array.dims}")

        self.original_dims = data_array.dims
        stack_dims = [dim for dim in data_array.dims if dim != exclude_dim]
        stacked_data = data_array.stack(all_dims=stack_dims).transpose(exclude_dim, 'all_dims')
        self.stacked_coords = stacked_data.indexes.get('all_dims')
        return stacked_data

    def unstack_to_original(self, stacked_data):
        if 'all_dims' not in stacked_data.dims:
            raise ValueError("The stacked data does not contain 'all_dims' dimension for unstacking.")

        if self.original_dims is None:
            raise ValueError("Original dimensions have not been set. Ensure 'stack_except' is called before 'unstack_to_original'.")

        return stacked_data.unstack('all_dims').transpose(*self.original_dims)

class SpatSpecDecomposition:
    def __init__(self, spatspec_coords, ld=None):
        expected_dims = ['h', 'w', 'freq']
        if isinstance(spatspec_coords, xr.DataArray):
            print('Warning: spatspec_coords is xr.DataArray, assuming spatspec_coords is spatspec_mtx')
            ld_coords = dict(spatspec_coords.coords)
            _ = ld_coords.pop('time')
            self.spatspec_coords = ld_coords
            self.spatspec_shape = spatspec_coords.transpose(*expected_dims,'time').shape[:-1]
        else:
            self.spatspec_coords = {k:spatspec_coords[k] for k in expected_dims}
            self.spatspec_shape = tuple(spatspec_coords[k].size for k in expected_dims)
        if ld is not None:
            self.ldx_set(ld)

    def ldx_set(self, ld):
        """
        ld: array (h*w*freq, nFac)
        """
        self.nFac = ld.shape[-1]
        self.ldx = xr.DataArray((ld.T).reshape(-1, *self.spatspec_shape),
                    coords={'factor':np.arange(self.nFac), **self.spatspec_coords},
                    dims=['factor', 'h', 'w', 'freq'])

    def ldx_set_direct(self, ldx):
        self.ldx = ldx
        self.nFac = ldx.shape[0]

    def scx_from_FSr(self, FSr, times):
        self.nFac = FSr.shape[-1]
        scx = xr.DataArray(FSr,
                    coords={'time':times, 'factor':np.arange(self.nFac)},
                    dims=['time', 'factor'])
        fs = 1/(times[1]-times[0])
        scx.attrs['fs'] = fs
        return scx

    def reconstruct(self, scx):
        mtx = self.ldx.dot(scx).transpose('h','w','freq','time')
        return mtx
    
    # Loading Processing
    def ldx_process(self):
        self.ldx_fch, self.ldx_maxfreq, self.ldx_slc_maxfreq, self.ldx_df = ldx_process(self.ldx)
        self.set_ldx_slc_maxch()

    def set_ldx_slc_maxch(self):
        self.ldx_slc_maxch = set_ldx_slc_maxch(self.ldx, self.ldx_df)
        
    def set_ldx_slc_maxch(self):
        ldx_slc_maxch = []
        for ifac in self.ldx.factor.values:
            slc_maxch = self.ldx.sel(factor=ifac, h=self.ldx_df.AP.loc[ifac], w=self.ldx_df.ML.loc[ifac])
            # drop h, w coords
            assert slc_maxch.ndim == 1, print(slc_maxch)
            slc_maxch = xr.DataArray(slc_maxch.data, dims=['freq'], coords={'freq':slc_maxch.freq})
            ldx_slc_maxch.append(slc_maxch)
        self.ldx_slc_maxch = xr.concat(ldx_slc_maxch, dim='factor')

    def ldx_df_mat(self):
        # convert to structured numpy array
        columns_rename_dict = {
            'hmax':'peak_iAP',
            'wmax':'peak_iML',
            'ifreqmax':'peak_ifreq',
            'AP':'peak_AP',
            'ML':'peak_ML',
            'freqmax':'peak_freq'
        }
        return self.ldx_df.reset_index().rename(columns=columns_rename_dict).to_dict('list')

    def scx_gaussian(self, scx, sigma=0.25):
        scx_smooth = xut.xarr_wrap(nd.gaussian_filter)(scx, sigma=(sigma * scx.fs, 0))
        return scx_smooth
    
    # Score Processing
    def scx_process(self, scx, return_all=False, sigma=0.25, quantile=0.25):
        # score processing
        # Lowpass: smooth factor scores with gaussian filter of size 0.5 seconds
        # Highpass (rolling zscore already takes care of it)
        # remove lower envelope
        print('scx processing ...')
        print('\t smoothing ...')
        scx_smooth = self.scx_gaussian(scx, sigma=sigma)
        # # scale by max of loading
        # scx_ = scx_ * self.ldx.max(dim=('h','w','freq'))
        scx_noenv_dict = self.scx_lowerenv(scx_smooth)
        print('\t thresholding ...')
        scx_thresh = ts.threshold(scx_noenv_dict['scx_noenv'], quantile=quantile, axis=0)
        print('scx processing done!')
        if return_all:
            # create dict
            scx_dict = {'scx': scx, 'scx_smooth': scx_smooth} | scx_noenv_dict | {'scx_thresh': scx_thresh}
            return scx_dict
        return scx_thresh
    
    def scx_lowerenv(self, scx):
        print('\t removing lower envelope ...')
        scx_lower = ts.lower_envelope(scx, axis=0)
        scx_noenv = xr.zeros_like(scx)
        scx_noenv.data[:] = scx - scx_lower
        return {'scx_lower': scx_lower, 'scx_noenv': scx_noenv}

    def scx_threshold(self, scx, quantile=0.25):
        print('\t thresholding ...')
        scx_thresh = ts.threshold(scx, quantile=quantile, axis=0)
        return scx_thresh

    def scx_spikes(self, scx, max_halfdur=5., return_trace=False):
        fs = scx.fs
        nfactor = scx.factor.size
        max_halfdur = max([ts.seconds_to_samples(max_halfdur, fs), 5])
        # find all local maxima (peaks) - use scipy.signal.argrelmax
        periwin = np.arange(-max_halfdur, max_halfdur+1)
        facpp = {}
        for ifactor in range(nfactor):
            scx_ = scx.data[:, ifactor]
            scx_peak_time = scisig.argrelmax(scx_)[0]
            scx_peak_amp = scx_[scx_peak_time]
            npeak = scx_peak_time.shape[0]
            # slice around peak time
            peripeak_wins = scx_peak_time.reshape(-1, 1) + periwin.reshape(1, -1) # (npeak, nwin)
            
            # check for out of bounds
            peripeak_wins = np.clip(peripeak_wins, 0, scx_.size-1)

            scx_peripeak_slc = scx_[peripeak_wins] # (npeak, nwin)

            # half-amplitude duration
            scx_halfamp_dur = np.sum(scx_peripeak_slc > np.expand_dims(scx_peak_amp / 2, axis=1), axis=1) # (npeak,)

            # factor point processes
            fppx = np.array([scx_peak_time, scx_peak_amp, scx_halfamp_dur]) # feature, peak

            # convert to xr.DataArray
            fppx = xr.DataArray(fppx,
                                dims=['feature', 'peak'],
                                    coords={'feature':['time','amp','dur'],
                                            'peak':np.arange(npeak)
                                            })
            
            # append to xr.Dataset
            res = fppx,
            if return_trace:
                scx_peripeak_slc = xr.DataArray(scx_peripeak_slc,
                                                dims=['time', 'win'], 
                                                coords={'time':scx_peak_time,
                                                        'win': periwin,
                                                        })                
                res += scx_peripeak_slc,
            facpp['factor'+str(ifactor)] = untuple(res)
        return facpp

    def designmat(self, mtx, log=True):
        mtx = mtx.transpose('h','w','freq','time')
        X = mtx.stack(hwf=('h', 'w', 'freq')).transpose('time', 'hwf')
        if log:
            X.data[np.where(X.data==0)] = 1e-10
            X = np.log10(X).reset_index('hwf')
        X.attrs = {'hwf_shape': self.spatspec_shape, 'fs': mtx.fs, 'step': mtx.window_step/mtx.fs}
        return X
    
    def designmat_preprocess(self, X, win=10):
        """
        rolling zscore
        win: seconds
        """
        # rolling zscore
        window_size = int(win / 2 / X.step) # ~10 seconds window
        Xz = xut.xarr_wrap(ts.rolling_zscore)(X, window_size=window_size)
        
        # replace nan with min
        Xz.data[np.isnan(Xz.data)] = np.nanmin(Xz.data)
        return Xz
    
    def mtx_from_designmat(self, X, mtx):
        """
        X.shape: cases, vars
        """
        mtx_ = mtx.copy()
        if isinstance(X, xr.DataArray):
            X = X.set_index(hwf=['h', 'w', 'freq'])
            mtx_ = X.unstack('hwf')
            mtx_ = mtx_.transpose('h','w','freq','time')
        else:
            mtx_.data[:] = X.reshape(X.shape[0], *self.spatspec_shape)
        return mtx_
      
    def plot_ldx_maxfreq(self):
        fig, axes = plot_maxfreq_slice_loadings(self.ldx_slc_maxfreq, self.ldx_df, nrow=self.nFac//4, ncol=4, figsize=(4*3, self.nFac//4 * 3))
        return fig, axes

    def mark_spindle_freq(self):
        # is spindle frequency
        spindle_facs = self.ldx_df[self.ldx_df.freqmax < 20].index.to_list()
        self.ldx_df.loc[:, 'is_spindle_freq'] = self.ldx_df.factor.isin(spindle_facs)

    def mark_lowgamma_freq(self):
        lowgamma_facs = self.ldx_df[30<self.ldx_df.freqmax < 80].index.to_list()
        self.ldx_df.loc[:, 'is_lowgamma_freq'] = self.ldx_df.factor.isin(lowgamma_facs)

    def mark_highgamma_freq(self):
        highgamma_facs = self.ldx_df[80<self.ldx_df.freqmax < 200].index.to_list()
        self.ldx_df.loc[:, 'is_highgamma_freq'] = self.ldx_df.factor.isin(highgamma_facs)

    def similarity_metric(self, other, fac1, fac2, freq_threshold=5):
        """Compute the distance metric between a factor of this instance and a factor of another instance."""
        freq_diff = abs(self.ldx_df.freqmax.iloc[fac1] - other.ldx_df.freqmax.iloc[fac2])
        
        # Check frequency threshold
        if freq_threshold is not None and freq_diff > freq_threshold:
            return 0

        correlation_val = np.corrcoef(self.ldx_slc_maxfreq[fac1].data.flatten(), other.ldx_slc_maxfreq[fac2].data.flatten())[0, 1]
        similarity = correlation_val
        return similarity

    def compute_similarity_matrix(self, other, freq_threshold=5):
        """Compute a distance matrix between the factors of this instance and another instance."""
        nfac = self.ldx.shape[0]
        similarity_matrix = np.zeros((nfac, nfac))
        
        for fac1 in range(nfac):
            for fac2 in range(nfac):
                similarity_matrix[fac1, fac2] = self.similarity_metric(other, fac1, fac2, freq_threshold)
        
        return similarity_matrix

    def get_loadings_dict(self):
        loadings_dict = {
            'loading':self.ldx.data,
            'coo_AP': self.ldx.h.values,
            'coo_ML': self.ldx.w.values,
            'coo_Freq': self.ldx.freq.values,
        }

        loadings_dict |= self.ldx_df_mat()
        return loadings_dict

def stack_and_reset_index(ldx):
    return ldx.stack(ch=('h', 'w')).reset_index('ch')

def compute_ifreqmax_and_maxfreq(ldx_fch):
    factor_ld_mean = ldx_fch.mean(dim=('ch'))
    ifreqmax = factor_ld_mean.argmax(dim='freq')
    ldx_maxfreq = ldx_fch.freq[ifreqmax]
    return ifreqmax, ldx_maxfreq

def compute_spatial_ldx(ldx, ifreqmax, ldx_maxfreq):
    spat_ldx_list = [ldx.sel(factor=ifactor, freq=ldx_maxfreq[ifactor], method='nearest') for ifactor in ldx.factor]
    spat_ldx_peak = np.array([np.unravel_index(spat_ldx.argmax(), spat_ldx.shape) for spat_ldx in spat_ldx_list])
    ldx_slc_maxfreq = xr.concat(spat_ldx_list, 'factor')
    return spat_ldx_peak, ldx_slc_maxfreq

def create_ldx_df(ldx, ifreqmax, ldx_maxfreq, spat_ldx_peak):
    ldx_df = pd.DataFrame(ldx_maxfreq.factor.data, columns=['factor'])
    ldx_df.loc[:, ['hmax', 'wmax']] = spat_ldx_peak
    ldx_df.loc[:, 'AP'] = ldx_df.apply(lambda x: ldx.h.values[int(x.hmax)], axis=1)
    ldx_df.loc[:, 'ML'] = ldx_df.apply(lambda x: ldx.w.values[int(x.wmax)], axis=1)
    ldx_df.loc[:, 'freqmax'] = ldx_maxfreq.data
    ldx_df.loc[:, 'ifreqmax'] = ifreqmax
    ldx_df.loc[:, 'norm'] = get_norm(ldx)  # Assuming get_norm is a separate function
    ldx_df = ldx_df.set_index('factor')
    return ldx_df[['AP', 'ML', 'freqmax', 'hmax', 'wmax', 'ifreqmax', 'norm']]

def ldx_process(ldx):
    ldx_fch = stack_and_reset_index(ldx)
    ifreqmax, ldx_maxfreq = compute_ifreqmax_and_maxfreq(ldx_fch)
    spat_ldx_peak, ldx_slc_maxfreq = compute_spatial_ldx(ldx, ifreqmax, ldx_maxfreq)
    ldx_df = create_ldx_df(ldx, ifreqmax, ldx_maxfreq, spat_ldx_peak)
    return ldx_fch, ldx_maxfreq, ldx_slc_maxfreq, ldx_df

def set_ldx_slc_maxch(ldx, ldx_df):
    ldx_slc_maxch = []
    for ifac in ldx.factor.values:
        slc_maxch = ldx.sel(factor=ifac, h=ldx_df.AP.loc[ifac], w=ldx_df.ML.loc[ifac])
        assert slc_maxch.ndim == 1
        slc_maxch = xr.DataArray(slc_maxch.data, dims=['freq'], coords={'freq': slc_maxch.freq})
        ldx_slc_maxch.append(slc_maxch)
    return xr.concat(ldx_slc_maxch, dim='factor')

def preprocessing_spec(mtx: xr.DataArray, freq_slice, median_kernel):
    # clip spectrogram to spindle frequency range
    mtx = mtx.sel(freq=freq_slice)
    # median filter spectrogram
    mtx_sm = median_spec(mtx, median_kernel)
    # remove frequency bins close to line noise and its harmonics
    mtx_sm = ln.drop_linenoise_harmonics(mtx_sm)
    return mtx_sm

def spatspec_designmat(mtx):
    mtx = mtx.transpose('h','w','freq','time')
    hwf_shape = mtx.shape[:-1]
    X = mtx.stack(hwf=('h', 'w', 'freq')).transpose('time', 'hwf')
    Xlog = np.log10(X).reset_index('hwf')
    Xlog.attrs = {'hwf_shape': hwf_shape}
    return Xlog

def get_norm(ldx):
    return np.linalg.norm(ldx.data.reshape(ldx.shape[0], -1), axis=1)