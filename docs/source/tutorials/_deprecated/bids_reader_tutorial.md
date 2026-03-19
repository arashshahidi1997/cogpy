---
title: BIDS Reader Tutorial
file_format: mystnb
kernelspec:
  name: cogpy
  display_name: cogpy
  language: python
mystnb:
  execution_mode: "off"
---

---

# Tutorial: `utils.bids_reader` path logic

This page demonstrates how output paths change based on config:

- `analysis_level`
- retained entities
- directory structure

Changing config → deterministic path identity.

---

## Setup (conceptual)

```{code-cell} python
from copy import deepcopy
from pathlib import Path

# Fake repo_abs for illustrative output; in repo use utils.repo_root.repo_abs
def repo_abs(p):
    PIXECOG_ROOT = Path("/storage2/arash/projects/pixecog")
    return PIXECOG_ROOT / p

# We'll assume FlowBidsContext exists as implemented in utils.bids_reader.
from utils.bids.bids_reader import FlowBidsContext
```
---

## Base config

```{code-cell} python
base_cfg = {
    "pipe": "preprocess",
    "bids_dir": "raw",
    "bids_spec": "v0_0_0",
    "analysis_level": "session",
    "pybids_inputs": {
        "ieeg": {
            "filters": {"datatype": "ieeg", "suffix": "ieeg", "extension": ".lfp"},
            "wildcards": ["subject", "session", "task"],
        }
    },
    "analysis_levels": {
        "ieeg": {
            "session": ["subject", "session", "task"],
            "subject": ["subject", "task"],
            "group": [],
        }
    },
}
```

---

## Session-level output

```{code-cell} python
cfg = deepcopy(base_cfg)
cfg["analysis_level"] = "session"

ctx = FlowBidsContext(cfg, repo_abs=repo_abs)
out = ctx.deriv(root="lowpass", datatype="ieeg")
print(out(suffix="ieeg", extension=".lfp",
          subject="01", session="01", task="track"))

print("Expected:")
print("derivatives/preprocess/lowpass/sub-01/ses-01/ieeg/sub-01_ses-01_task-track_ieeg.lfp")
```

---

## Subject-level output (drop session)

```{code-cell} python
cfg = deepcopy(base_cfg)
cfg["analysis_level"] = "subject"

ctx = FlowBidsContext(cfg, repo_abs=repo_abs)
out = ctx.deriv(root="qc", datatype="ieeg")
print(out(suffix='qc', extension='.tsv', subject='01', task='track'))

print("Expected:")
print("derivatives/preprocess/qc/sub-01/ieeg/sub-01_task-track_qc.tsv")
```

---

## Group-level output

```{code-cell} python
cfg = deepcopy(base_cfg)
cfg["analysis_level"] = "group"

ctx = FlowBidsContext(cfg, repo_abs=repo_abs)
out = ctx.deriv(root="report", datatype="ieeg")
print(out(suffix='summary', extension='.html'))

print("Expected:")
print("derivatives/preprocess/report/ieeg/summary.html")
```

---

## Subsets affect inputs only

```{code-cell} python
print("Example usage in Snakefile:")
print("ECE = ctx.inputs['ieeg']")
print("ECE_TRACK = ECE.filter(task='track')")
print("Outputs still built via ctx.deriv(...)")
```

---

## Rules of thumb

* Use `analysis_levels` → control output identity
* Use `analysis_level` → session/subject/group mode
* Use `pybids_inputs.filters` → global dataset restriction
* Use subsets → input selection only
* Aggregation when dropping entities must be explicit in rules

```
::contentReference[oaicite:0]{index=0}
```
