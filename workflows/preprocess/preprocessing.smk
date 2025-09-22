# rule downsample:
#     """Downsamples the input dat signal."""
#     params: 
#         factor=4  # Downsampling factor
#     input:
#         dat="raw/{session}.dat",
#         xml="raw/{session}.xml"
#     output:
#         downsample=directory("downsample/{session}.zarr")
#     script:
#         "scripts/00_downsample.py"

rule dat2zarr:
	"""Converts the input dat signal to zarr format."""
	input:
		dat="raw/{session}.dat",
		xml="raw/{session}.xml"
	output:
		zarr=directory("raw_zarr/{session}.zarr")
	run:
		from cogpy.io import ecog_io
		sigx = ecog_io.from_file(input.dat, input.xml)
		sigx.name = "sigx"
		ecog_io.to_zarr(output.zarr, sigx)

rule lowpass:
	"""Applies a lowpass filter to the input dat signal."""
	params: 
		cutoff=40,   # Cutoff frequency in Hz
		order=4,      # Order of the filter
		btype='lowpass'
	input:
		raw='raw_zarr/{session}.zarr'
	output:
		filtered=directory("lowpass/{session}.zarr")
	script:
		"scripts/01_filter.py"

rule feature:
	"""Extracts features from the lowpass filtered dat signal."""
	params:
		slider_kwargs=dict(window_size=512, window_step=64),
		zscore=True
	input:
		lowpass="lowpass/{session}.zarr"
	output:
		feature=directory("feature/{session}.zarr")
	script:
		"scripts/02_feature.py"

rule badlabel:
	"""Scores the quality of the features."""
	input:
		feature="feature/{session}.zarr",
	output:
		badlabel="label/{session}.npy",
	params:
		knn=10
	script:
		"scripts/03_badlabel.py"

rule interpolate:
	"""Interpolates the bad labels to create a smooth label signal."""
	input:
		dat="raw/{session}.dat",
		xml="raw/{session}.xml",
		badlabel="label/{session}.npy"
	output:
		interp=directory("interpolate/{session}.zarr")
	script:
		"scripts/04_interpolate.py"

rule badscore:
	"""Scores the smoothness quality of the features."""
	input:
		feature="feature/{session}.zarr",
		badlabel="label/{session}.npy"
	output:
		badscore=directory("badscore/{session}.zarr")
	script:
		"scripts/05_badscore.py"

rule highpass:
	"""Applies a lowpass filter to the input dat signal."""
	params: 
		order=4,      # Order of the filter
		cutoff=40,   # Cutoff frequency in Hz
		btype='highpass'
	input:
		raw="interpolate/{session}.zarr"
	output:
		filtered=temp(directory("highpass/{session}.zarr"))
	script:
		"scripts/01_filter.py"

rule linenoise_ica:
	"""Removes line noise from the dat signal."""
	params:
		segment_duration=300,  # in seconds
		linenoise_f0=50,
		halfbandwidth=4,
		nharmonics=2,
		ncomp=20
	input:
		noisy="highpass/{session}.zarr"
	output:
		linenoise=directory("linenoise/ICA_{session}.zarr")
	script:
		"scripts/06_linenoiseICA.py"

rule linenoise_mu:
	"""Removes line noise from the dat signal."""
	params:
	input:
		noisy="linenoise/ICA_{session}.zarr"
	output:
		linenoise=directory("linenoise/Mu_{session}.zarr")
	script:
		"scripts/06_linenoiseMu.py"