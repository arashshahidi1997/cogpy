# How-to Guides

Practical recipes for common tasks. These assume familiarity with cogpy's
data model and basic API (see {doc}`/tutorials/quickstart` if you are new).

{doc}`filtering`
: Bandpass, notch, spatial smoothing, CMR, and z-score normalization.

{doc}`event-detection`
: Use pre-built pipelines or configure custom detectors.

{doc}`batch-spatial-analysis`
: Run spatial measures (Moran's I, gradient anisotropy) over spectrograms.

{doc}`compose-artifact-analysis`
: Assemble cogpy primitives for artifact detection and template subtraction.

{doc}`custom-snakemake-pipeline`
: Build a Snakemake preprocessing pipeline from cogpy building blocks.

{doc}`load-ecog-data`
: Read binary LFP, Zarr, and BIDS-iEEG files into xarray.

{doc}`visualization`
: Interactive grid movies, multichannel traces, and topomaps with HoloViews.

{doc}`triggered-analysis`
: Lock signals to events, extract epochs, and estimate/subtract templates.

{doc}`brainstates-and-regression`
: Interval-based state classification and OLS regression for artifact removal.

```{toctree}
:hidden:
:maxdepth: 1

filtering
event-detection
batch-spatial-analysis
compose-artifact-analysis
custom-snakemake-pipeline
load-ecog-data
visualization
triggered-analysis
brainstates-and-regression
```
