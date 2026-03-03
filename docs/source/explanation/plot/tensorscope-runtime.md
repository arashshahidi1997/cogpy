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

## Known Panel Quirk (1.8.8)

For `panel==1.8.8`, `FastGridTemplate.main` is a `GridSpec` and does not support `.append()`. See `tensorscope-issues.md`.
