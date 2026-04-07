---
title: "QC: wave.optical_flow + vector_field vs NeuroPattToolbox"
date: 2026-04-01
status: done
actionable: true
project_primary: "cogpy"
tags: [task, qc, travelling-waves]
---

## Prompt

Quality-check `src/cogpy/wave/optical_flow.py` and `src/cogpy/wave/vector_field.py` against the MATLAB reference in `/storage/share/codelib/braindynamicsusyd--neuropatttoolbox`.

Also check `/storage/share/codelib/navvab-afrashteh--ofamm` for the optical flow implementation details.

### Steps

1. Read the NeuroPattToolbox MATLAB source — focus on:
   - How optical flow is applied to neural phase/amplitude maps
   - How velocity fields are computed (which solver, what input representation)
   - Divergence and curl computation
   - Critical point detection algorithm (how are sources/sinks/spirals/saddles classified?)
   - Pattern classification logic (what thresholds or criteria for planar vs rotational vs source etc.)
   - Surrogate/null testing approach for pattern significance

2. Read the OFAMM source for:
   - Horn-Schunck / CLG implementation details
   - How wave speed and direction are extracted from flow fields

3. Read `src/cogpy/wave/optical_flow.py` and `src/cogpy/wave/vector_field.py` — compare:
   - Does `compute_flow()` apply flow to the right representation (phase maps, amplitude, complex)?
   - Does `divergence()` / `curl()` use the same finite-difference scheme?
   - Does `critical_points()` match the NeuroPatt classification algorithm?
   - Does `classify_pattern()` use sensible criteria consistent with the literature?

4. Fix discrepancies. Ensure docstrings cite Townsend & Gong 2018 and Afrashteh et al. 2017.

5. Run `python -m pytest tests/wave/test_optical_flow.py tests/wave/test_vector_field.py -v`.

### References
- Townsend & Gong 2018, DOI: 10.1371/journal.pcbi.1006643
- Afrashteh et al. 2017, DOI: 10.1016/j.neuroimage.2017.03.034


## Batch Result

- status: done
- batch queue_id: `f9eff0043988`
- session: `d738b2f9-7755-4c25-a7fa-4e9b25695419`
- batch duration: 1349.7s
