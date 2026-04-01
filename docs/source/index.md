# cogpy

**Python toolkit for ECoG / iEEG signal processing and analysis.**

cogpy provides composable, file-agnostic compute primitives for intracranial
electrophysiology: filtering, spectral analysis, event detection, bad-channel
identification, and spatial grid measures. It pairs a structured I/O layer with
reproducible Snakemake pipelines and serves as the backend for visualization
frontends.

---

::::{grid} 2
:gutter: 3

:::{grid-item-card} Tutorials
:link: tutorials/index
:link-type: doc

**Learning-oriented.** Step-by-step lessons that take you from installation
to working analysis pipelines. Start here if you are new to cogpy.

- {doc}`tutorials/install`
- {doc}`tutorials/quickstart`
- {doc}`tutorials/spectral-analysis`
- {doc}`tutorials/bad-channel-detection`
:::

:::{grid-item-card} How-to Guides
:link: howto/index
:link-type: doc

**Task-oriented.** Practical recipes for common tasks. Assumes you already
know the basics.

- {doc}`howto/filtering`
- {doc}`howto/event-detection`
- {doc}`howto/batch-spatial-analysis`
- {doc}`howto/custom-snakemake-pipeline`
:::

:::{grid-item-card} Explanation
:link: explanation/index
:link-type: doc

**Understanding-oriented.** Design decisions, data model, and architecture.
Read these to understand *why* cogpy works the way it does.

- {doc}`explanation/architecture`
- {doc}`explanation/data-model`
- {doc}`explanation/spectral-conventions`
- {doc}`explanation/detection-framework`
:::

:::{grid-item-card} API Reference
:link: api/index
:link-type: doc

**Information-oriented.** Complete function and class documentation
auto-generated from docstrings.

- {doc}`api/measures`
- {doc}`api/spectral`
- {doc}`api/preprocess`
- {doc}`api/detect`
:::

::::

---

## Installation

```bash
pip install -e ".[all]"        # full install (editable dev)
pip install ecogpy              # minimal core
pip install "ecogpy[viz]"      # with visualization deps
pip install "ecogpy[io]"       # with I/O format support
```

## Quick Example

```python
import cogpy

# Load sample grid ECoG data
sig = cogpy.datasets.load_sample()

# Compute multitaper PSD
from cogpy.spectral.psd import psd_multitaper
psd, freqs = psd_multitaper(sig.values, fs=sig.fs)

# Detect spectral bursts
from cogpy.detect import BURST_PIPELINE
events = BURST_PIPELINE.run(sig)
```

```{toctree}
:hidden:
:caption: Tutorials

tutorials/index
```

```{toctree}
:hidden:
:caption: How-to Guides

howto/index
```

```{toctree}
:hidden:
:caption: Explanation

explanation/index
```

```{toctree}
:hidden:
:caption: Reference

reference/index
```

```{toctree}
:hidden:
:caption: API Reference

api/index
```
