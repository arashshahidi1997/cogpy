# How to build a Snakemake preprocessing pipeline

cogpy ships Snakemake workflows as package data. You can use the built-in
pipeline or compose your own from cogpy's building blocks.

## Using the built-in pipeline

```bash
# Run the full preprocessing pipeline
cogpy-preproc all --config /path/to/config.yml

# Run individual steps
cogpy-preproc lowpass --config /path/to/config.yml
cogpy-preproc feature --config /path/to/config.yml
cogpy-preproc badlabel --config /path/to/config.yml
```

The pipeline steps are:

1. **raw_zarr** — convert raw data to Zarr format
2. **lowpass** — lowpass filter
3. **downsample** — decimate to target sampling rate
4. **feature** — extract channel features (windowed)
5. **badlabel** — label bad channels (DBSCAN)
6. **plot_feature_maps** — generate QC visualizations
7. **interpolate** — interpolate bad channels

## Writing a custom Snakefile

A custom pipeline composes `cogpy.io` (load/save) with `cogpy.core`
(compute):

```python
# scripts/my_step.py
import cogpy.io.ecog_io as ecog_io
from cogpy.preprocess.filtering import bandpassx, notchesx

# Load
sig = ecog_io.from_file(snakemake.input[0])

# Compute (cogpy.core — no file I/O here)
sig = notchesx(sig, freqs=[60.0, 120.0, 180.0])
sig = bandpassx(sig, wl=0.5, wh=300.0, order=4, axis="time")

# Save
ecog_io.to_zarr(sig, snakemake.output[0])
```

```python
# Snakefile
rule filter_and_denoise:
    input: "{subject}/raw.zarr"
    output: "{subject}/filtered.zarr"
    script: "scripts/my_step.py"
```

## Design principles

- **Rules are thin orchestrators.** Heavy logic belongs in `cogpy.core`.
- **Use `cogpy.io` for all file operations.** Do not read/write files directly
  in core functions.
- **Sidecar management** (updating JSON metadata after resampling, etc.)
  happens in `cogpy.io`, not in Snakemake rules.

## Configuration

Pipelines use YAML configuration:

```yaml
# config.yml
subjects: ["sub-01", "sub-02"]
fs_target: 500.0
lowpass_freq: 200.0
line_freq: 60.0
badchannel:
  window_size: 2048
  window_step: 1024
  dbscan_eps: 1.5
  dbscan_min_samples: 5
```
