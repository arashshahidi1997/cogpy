# cogpy — ECoG processing toolkit

Minimal, modular layout for algorithms, I/O, CLI, and Snakemake workflows.

## Layout

```
src/cogpy/
├─ __init__.py            # public API + backward-compat shims
├─ io/                    # data I/O (stable import: cogpy.io)
├─ core/                  # algorithms & transforms
├─ cli/                   # thin CLI wrappers
└─ workflows/             # packaged Snakemake pipelines
   └─ preprocess/
      ├─ Snakefile
      ├─ config.yaml
      └─ scripts/
```

## Install

```bash
# dev install
pip install -e .

# with workflow dependencies (if defined)
pip install -e .[workflows]
```

## CLI

Root command (Typer/Click):

```bash
cogpy --help
```

### Run workflow (Snakemake)

```bash
# via subcommand
cogpy wf run lowpass devA/devA-S01/rec1 -c 8 --printshellcmds

# or dedicated entry (if enabled)
ecogpipe lowpass devA/devA-S01/rec1 -c 8 --printshellcmds
```

* **Input spec:** `device/device-session/filename` (no extension).
* **Targets:** `{output_dir}/{step}/{input_spec}{ext}`
  Steps → ext: `raw_zarr .zarr`, `lowpass .zarr`, `downsample .zarr`, `feature .zarr`, `badlabel .npy`, `interpolate .zarr`.

## Config

Defaults live in `workflows/preprocess/config.yaml` (packaged). Override via file edits or Snakemake flags as usual.

## Library usage

```python
from cogpy.io.ecog_io import from_file, to_zarr
from cogpy.core.preprocess.filters import lowpass

sig = from_file("rec.lfp", "rec.xml")
filt = lowpass(sig, cutoff=300, order=4)
to_zarr("out.zarr", filt)
```

## Dev tips

* Put new algorithms in `core/`, keep `cli/` thin.
* Workflows + small scripts go under `workflows/` and are included as package data.
* `__init__.py` re-exports selected symbols and provides shims so legacy imports continue to work.
