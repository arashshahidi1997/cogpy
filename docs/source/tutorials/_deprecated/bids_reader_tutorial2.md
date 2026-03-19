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

# Tutorial: `utils.bids_reader` path building across analysis scenarios

This tutorial demonstrates how `FlowBidsContext` builds BIDS-like derivative paths as a function of:

- `pipe`
- `analysis_level`
- `analysis_levels` (entity retention)
- `analysis_level_dirs` (subject/session directory toggles)
- `pybids_inputs` wildcard templates
- `subsets` (input filtering; shown conceptually)

The point: **you change config; path identity changes deterministically.**

---

## Setup: minimal scaffolding (conceptual)

In a real flow, Snakemake provides `config` and you pass `repo_abs`.
Here we emulate configs as dicts.

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

## Base config: `ieeg` component

We start with a config where `ieeg` has these input wildcards:
`subject`, `session`, `task`.

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
    "analysis_level_dirs": {
        "session": {"subject_dir": True, "session_dir": True},
        "subject": {"subject_dir": True, "session_dir": False},
        "group": {"subject_dir": False, "session_dir": False},
    },
    "subsets": {
        "ieeg": {
            "track": {"task": "track"},
            "sleep": {"task": ["pre", "post", "free"]},
        }
    },
}
```

---

## Scenario A: Session-level paths (default)

**Expectation**

* Output includes `sub-*/ses-*` directories.
* Filename identity includes `subject`, `session`, `task`.
* Path is rooted under `derivatives/<pipe>/...`.

```{code-cell} python
cfg = deepcopy(base_cfg)
cfg["analysis_level"] = "session"

ctx = FlowBidsContext(cfg, repo_abs=repo_abs)
lowpass = ctx.deriv(root="lowpass", datatype="ieeg")

# Example call with explicit entity values (like in scripts):
p = lowpass(suffix="ieeg", extension=".lfp", subject="01", session="04", task="track")
print(p)

print("Expected shape:")
print("derivatives/preprocess/lowpass/sub-01/ses-04/ieeg/sub-01_ses-04_task-track_ieeg.lfp")
```

---

## Scenario B: Subject-level paths (drop session identity)

**Change**

* `analysis_level: subject`

**Expectation**

* Output directory includes `sub-*` but **no `ses-*`**.
* Filename identity drops `session`.
* You must explicitly aggregate across sessions in rules; the builder only changes identity.

Subject-level: keep subject dir, drop session dir and entity

```{code-cell} python
cfg = deepcopy(base_cfg)
cfg["analysis_level"] = "subject"

ctx = FlowBidsContext(cfg, repo_abs=repo_abs)
summary = ctx.deriv(root="preprocess/qc", datatype="ieeg")
p = summary(suffix="qc", extension=".tsv", subject="01", task="track")
print(p)

print("Expected shape:")
print("derivatives/preprocess/qc/sub-01/ieeg/sub-01_task-track_qc.tsv")
```

---

## Scenario C: Group-level paths (drop subject + session identity)

**Change**

* `analysis_level: group`

**Expectation**

* No `sub-*` and no `ses-*` directories.
* Identity contains only retained entities (here: none), so path becomes a single global artifact per `root/suffix/extension`.

Group-level: drop subject and session dirs and entities

```{code-cell} python
cfg = deepcopy(base_cfg)
cfg["analysis_level"] = "group"

ctx = FlowBidsContext(cfg, repo_abs=repo_abs)
report = ctx.deriv(root="report", datatype="ieeg")
p = report(suffix="summary", extension=".html")
print(p)

print("Expected shape:")
print("derivatives/preprocess/report/ieeg/summary.html  (exact filename depends on suffix/prefix choices)")
```

**Collision warning**
At group level, it’s easy to accidentally create collisions. Use a meaningful suffix/prefix and/or encode stratification entities (task/acq) into the retained set if needed.

---

## Scenario D: Keep `task` at group level (stratified group outputs)

Sometimes you want “group-level per task”. That is still *group* level if you drop subject/session but retain task.

**Change**

* `analysis_levels.ieeg.group = ["task"]`

Group-level, stratified by task
```{code-cell} python
cfg = deepcopy(base_cfg)
cfg["analysis_level"] = "group"
cfg["analysis_levels"]["ieeg"]["group"] = ["task"]

ctx = FlowBidsContext(cfg, repo_abs=repo_abs)
grp = ctx.deriv(root="group_qc", datatype="ieeg")
p = grp(suffix="qc", extension=".tsv", task="track")
print(p)

print("Expected shape:")
print("derivatives/preprocess/group_qc/ieeg/task-track_qc.tsv  (exact tag formatting depends on spec)")
```

---

## Scenario E: Rich entity set (`ecephys` with acquisition/recording)

Here we extend the config with an `ecephys` component that has:
`subject`, `session`, `task`, `acquisition`, `recording`.

We also show how to *restrict* recording to `lf` using `pybids_inputs.filters` (global selection).

Add an ecephys component with more entities and a recording restriction

```{code-cell} python
cfg = deepcopy(base_cfg)
cfg["pybids_inputs"]["ecephys"] = {
    "filters": {
        "suffix": "ecephys",
        "extension": ".lfp",
        # Global restriction: only lf recording
        "recording": "lf",
    },
    "wildcards": ["subject", "session", "task", "acquisition", "recording"],
}

cfg["analysis_levels"]["ecephys"] = {
    "session": ["subject", "session", "task", "acq", "recording"],
    "subject": ["subject", "task", "acq", "recording"],
    "group": ["task", "acq", "recording"],
}

cfg["subsets"]["ecephys"] = {
    "track": {"task": "track"},
    "bshift": {"acq": "bshift"},
}

ctx = FlowBidsContext(cfg, repo_abs=repo_abs)
spwr = ctx.deriv(root="sharpwaveripple/events", datatype="ecephys")
p = spwr(suffix="sharpwaveripple", extension=".mat",
         subject="01", session="04", task="track", acquisition="bshift", recording="lf")
print(p)

print("Expected session-level shape (with acquisition + recording):")
print("derivatives/preprocess/sharpwaveripple/events/sub-01/ses-04/ecephys/sub-01_ses-04_task-track_acq-bshift_recording-lf_sharpwaveripple.mat")
```

---

## Scenario F: Named subsets affect *inputs*, not outputs

Subsets are used to filter the dataset *selection* for rules. They do not automatically change output identity.

```{code-cell} python
:caption: "Subsets: selection helper (conceptual usage in Snakefiles)"

print("In Snakefile:")
print("ECE = ctx.inputs['ecephys']")
print("ECE_TRACK = ECE.filter(**ctx.subset_filters('ecephys', 'track'))")
print("ECE_BSHIFT = ECE.filter(**ctx.subset_filters('ecephys', 'bshift'))")
print("")
print("Outputs still use ctx.deriv(...), governed by analysis_level + retained entities.")
```

If a subset should change output identity (e.g., separate artifacts for track vs sleep),
encode that explicitly by:

* using different `root`s, or
* retaining a discriminating entity, or
* using different `suffix`/`prefix`.

---

## Summary table (conceptual)

```{code-cell} python
:caption: "What changes with analysis level?"

print("session:  dirs=sub+ses, retain subject+session+... -> one output per session entity-set")
print("subject:  dirs=sub only, retain subject+...        -> one output per subject entity-set (aggregate sessions)")
print("group:    dirs=none, retain configured entities    -> one output per cohort/stratum (aggregate subjects)")
```

---

## Practical rule of thumb

* Use `pybids_inputs.filters` for **global** restrictions (e.g., recording=lf).
* Use `subsets` for **named selections** (track vs sleep, acq=bshift).
* Use `analysis_levels` for **output identity policy** (which entities define uniqueness).
* Use explicit Snakemake rules to define aggregation when entities are dropped.
