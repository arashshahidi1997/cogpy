---
title: "QC: wave.generalized_phase vs mullerlab generalized-phase"
date: 2026-04-01
status: done
actionable: true
project_primary: "cogpy"
tags: [task, qc, travelling-waves]
---

## Prompt

Quality-check `src/cogpy/wave/generalized_phase.py` against the reference implementation in `/storage/share/codelib/mullerlab--generalized-phase`.

### Steps

1. Read the MATLAB/Python source in the mullerlab repo — look for:
   - The main generalized phase function
   - How it centers the analytic signal
   - How it corrects negative-frequency components
   - Any amplitude weighting or masking logic
   - README or documentation describing the algorithm

2. Read `src/cogpy/wave/generalized_phase.py` and compare:
   - Does `generalized_phase()` follow the same algorithm steps?
   - Is the negative-frequency correction implemented correctly?
   - Does it handle edge cases (low amplitude, DC offset)?

3. Check for:
   - Algorithmic fidelity to Davis et al. 2020 (Nature)
   - Numerical stability on edge cases
   - Proper docstring citing the source

4. Fix any discrepancies. Run `python -m pytest tests/wave/test_generalized_phase.py -v` if test exists, otherwise add a basic round-trip test.

### References
- Davis et al. 2020, DOI: 10.1038/s41586-020-2802-y


## Batch Result

- status: done
- batch queue_id: `f9eff0043988`
- session: `d738b2f9-7755-4c25-a7fa-4e9b25695419`
- batch duration: 1349.7s
