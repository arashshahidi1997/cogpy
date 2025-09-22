#!/usr/bin/env python
"""
Title: linenoiseICA.py
Status: WIP
Last Updated: 2025-09-15

Summary:
	Applies line noise removal using ICA.
"""

"""

%load_ext autoreload
%autoreload 2
import xarray as xr
from cogpy.preprocess.linenoise import LineNoiseEstimatorICAArr
from cogpy.io import ecog_io

input_interp = "interpolate/sample_raw_signal.zarr"
sigx_interp = ecog_io.from_zarr(input_interp)['sigx_interp']

lnoiseICA_params = dict(
        linenoise_f0=50,
        halfbandwidth=4,
        nharmonics=2,
        ncomp=20
)

"""
from cogpy.preprocess.linenoise import sliding_ICA
from cogpy.io import ecog_io

def _input(input_interp):
	sigx_hp = ecog_io.from_zarr(input_interp)['sigx']
	return sigx_hp

def estimate_lnoise(sigx, segment_duration, lnoiseICA_params):
	sigx = sigx.compute()
	fs = sigx.fs
	# reshape to (time, ch)
	sigx = sigx.stack(ch=['AP', 'ML']).transpose('time', 'ch').reset_index('ch')

	# segment data into 5-minute segments
	segment_size = int(segment_duration * fs)
	lnoise_estimate = sliding_ICA(sigx, fs, segment_size, lnoiseICA_params)
	return lnoise_estimate

def _output(lnoise_est, output_linenoise):
	lnoise_est.name = "sigx"
	ecog_io.to_zarr(output_linenoise, lnoise_est)

def main(input_interp, output_linenoise, segment_duration, lnoiseICA_params):
	sigx_hp = _input(input_interp)
	lnoise_est = estimate_lnoise(sigx_hp, segment_duration, lnoiseICA_params)
	_output(lnoise_est, output_linenoise)

if __name__ == "__main__":
	# snakemake
	if 'snakemake' in globals():
		snakemake = globals()['snakemake']
		input_interp = snakemake.input.noisy
		output_linenoise = snakemake.output.linenoise
		segment_duration = snakemake.params.segment_duration
		lnoiseICA_params = {
			"linenoise_f0": snakemake.params.linenoise_f0,
			"halfbandwidth": snakemake.params.halfbandwidth,
			"nharmonics": snakemake.params.nharmonics,
			"ncomp": snakemake.params.ncomp
		}
		main(input_interp, output_linenoise, segment_duration, lnoiseICA_params)
	else:
		raise RuntimeError("This script is intended to be run via Snakemake.")

