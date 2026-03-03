# TensorScope Runtime Notes (MCP / Dev Environment)

TensorScope development depends on the **conda `cogpy` environment**. In this repo, the system `python3` is not guaranteed to have the required packages on `PATH`.

## Which Python?

Use:
```bash
/storage/share/python/environments/Anaconda3/condabin/conda run -n cogpy python
```

Avoid relying on:
```bash
python3
```

## Quick Checks

```bash
/storage/share/python/environments/Anaconda3/condabin/conda run -n cogpy python -c "import panel as pn; print('panel', pn.__version__)"
/storage/share/python/environments/Anaconda3/condabin/conda run -n cogpy python -c "import importlib.util; print('pytest', bool(importlib.util.find_spec('pytest')))"
```

## Common Commands

**Run the Phase 0 demo**
```bash
/storage/share/python/environments/Anaconda3/condabin/conda run -n cogpy panel serve code/lib/cogpy/examples/tensorscope/hello_tensorscope.py --show
```

**Run TensorScope stub tests**
```bash
/storage/share/python/environments/Anaconda3/condabin/conda run -n cogpy python -m pytest -q code/lib/cogpy/tests/core/plot/tensorscope
```

## Reference Template Pattern (FastGridTemplate + GridSpec)

The working reference layout is `code/lib/cogpy/notebooks/tensorscope/tensorscope_app.py`.

Key pattern for `panel==1.8.8`:
```python
tmpl = pn.template.FastGridTemplate(
    title="🧠 TensorScope",
    sidebar=[sidebar],  # pass at init
    theme="dark",
    row_height=60,
)

tmpl.main[0:6, 0:6] = spatial_card
tmpl.main[0:6, 6:12] = events_card
tmpl.main[6:12, 0:12] = ts_card
```

## Known Panel Quirk (1.8.8)

For `panel==1.8.8`, `FastGridTemplate.main` is a `GridSpec` and does not support `.append()`. See `tensorscope-issues.md`.
