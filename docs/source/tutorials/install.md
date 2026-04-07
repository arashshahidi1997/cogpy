# Installation

## Requirements

- **Python 3.10+**
- pip (or any PEP 517 installer)

## Install from PyPI

```bash
pip install ecogpy
```

This pulls in the core dependencies (numpy, scipy, pandas, xarray,
scikit-learn) and is enough for filtering, spectral analysis, event
detection, and spatial measures.

## Extras

cogpy uses optional dependency groups so you only install what you need:

| Extra | What it adds | When you need it |
|-------|-------------|------------------|
| `viz` | matplotlib, seaborn, plotly, holoviews, panel, hvplot | Plotting and interactive visualization |
| `io` | h5py, tables, zarr, openpyxl, xmltodict | Reading/writing HDF5, Zarr, Excel, XML formats |
| `notebook` | ipykernel, ipywidgets | Running cogpy inside Jupyter notebooks |
| `signal` | ghostipy, scikit-image, emd | Advanced signal processing (e.g. EMD, morphological filtering) |
| `perf` | dask, pyfftw | Parallel computation and faster FFTs |
| `interop-mne` | mne | Interoperability with MNE-Python |
| `all` | All of the above (except `interop-mne`) | Full install for development or exploration |

Install one or more extras with bracket syntax:

```bash
pip install "ecogpy[viz]"            # just visualization
pip install "ecogpy[viz,io]"         # visualization + file I/O
pip install "ecogpy[all]"            # everything (except MNE)
pip install "ecogpy[all,interop-mne]"  # truly everything
```

## Development install

Clone the repo and install in editable mode:

```bash
git clone https://github.com/arashshahidi1997/cogpy.git
cd cogpy
pip install -e ".[all]"
```

This lets you edit source files and see changes immediately without
reinstalling.

## Verify

```bash
python -c "import cogpy; print(cogpy.__version__)"
```

You should see the installed version (e.g. `0.1.2`).

## Next steps

- {doc}`quickstart` — load data, compute a PSD, detect events
