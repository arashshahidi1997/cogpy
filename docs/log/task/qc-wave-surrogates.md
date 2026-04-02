---
title: "QC: wave.surrogates vs travelling-waves-or-sequentially-activated"
date: 2026-04-01
status: done
actionable: true
project_primary: "cogpy"
tags: [task, qc, travelling-waves]
---

## Prompt

Quality-check `src/cogpy/wave/surrogates.py` against the MATLAB reference in `/storage/share/codelib/evolutionaryneuralcodinglab--travelling-waves-or-sequentially-activated-discrete-modules`.

### Steps

1. Read the MATLAB source — look for:
   - Phase randomization implementation (how is FFT phase shuffled while preserving amplitude?)
   - Spatial shuffle logic (channel permutation approach)
   - Any other surrogate/null-model methods
   - How surrogate testing is structured (how many surrogates, what statistic, how p-value is computed)

2. Read `src/cogpy/wave/surrogates.py` and compare:
   - Does `phase_randomize()` correctly preserve the amplitude spectrum and randomize only phase?
   - Does it handle multi-channel data (randomize per-channel or jointly)?
   - Does `spatial_shuffle()` permute correctly?
   - Does `surrogate_test()` compute p-values correctly (one-tailed vs two-tailed, rank-based)?

3. Check edge cases:
   - Even vs odd length signals in phase randomization
   - Nyquist handling
   - Reproducibility with seed parameter

4. Fix any issues. Run `python -m pytest tests/wave/test_surrogates.py -v`.

### References
- Bhattacharya et al. 2022, DOI: 10.1371/journal.pcbi.1009827


## Batch Result

- status: done
- batch queue_id: `f9eff0043988`
- session: `d738b2f9-7755-4c25-a7fa-4e9b25695419`
- batch duration: 1349.7s
