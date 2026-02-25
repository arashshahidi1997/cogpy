# Datasets (Synthetic & Example)

This section specifies **example datasets** intended for:

- fast, deterministic debugging of `cogpy` plotting/GUI components (Panel/Bokeh/HoloViews),
- repeatable “responsiveness checks” on larger tensors without depending on external files,
- shared expectations about **schemas** (dims/coords/attrs) for core entities used across viewers.

The goal is to define **intentions + contracts first**, then implement the generators and bundles.

```{toctree}
:maxdepth: 1
:caption: Datasets

schemas
gui-bundles
modes
```

