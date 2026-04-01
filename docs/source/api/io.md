# `cogpy.io`

File I/O for electrophysiology formats: binary LFP, Zarr, HDF5, BIDS-iEEG,
and XML anatomy maps. I/O functions load data into xarray and save results
back to disk — they never do heavy compute.

**Guide:** {doc}`/howto/load-ecog-data` |
**Design:** {doc}`/explanation/architecture` (compute vs I/O separation)

## Submodules

```{eval-rst}
.. autosummary::
   :recursive:

   cogpy.io.converters
   cogpy.io.ecephys_io
   cogpy.io.ecog_io
   cogpy.io.ieeg_io
   cogpy.io.ieeg_sidecars
   cogpy.io.load_meta
   cogpy.io.save_utils
   cogpy.io.sidecars
   cogpy.io.xml_anat_map
   cogpy.io.xml_io
```

```{eval-rst}
.. automodule:: cogpy.io
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:
```
