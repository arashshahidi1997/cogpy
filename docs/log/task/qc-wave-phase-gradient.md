---
title: "QC: wave.phase_gradient vs wm_travelingwaves_code"
date: 2026-04-01
status: done
result_note: /storage2/arash/worklog/workflow/captures/20260401-235625-c590e3/note.md
completed: 2026-04-01T23:56:27+02:00
actionable: true
project_primary: "cogpy"
tags: [task, qc, travelling-waves]
---

## Prompt

Quality-check `src/cogpy/wave/phase_gradient.py` against the MATLAB reference implementation in `/storage/share/codelib/sayak66--wm_travelingwaves_code`.

### Steps

1. Read the MATLAB source files — especially:
   - `phase_gradient_complex_multiplication.m` (gradient via complex conjugate multiplication)
   - `analytic_signal.m` or `analytic_signal_FIR.m` (Hilbert transform)
   - Any PGD / phase-gradient directionality computation
   - `circ_corrcc.m`, `circ_corrcl.m`, `circ_mean.m` (circular stats used in plane fitting)
   - `phase_distance_map.m`, `angular_rotation.m`, `rectify_rotation.m`

2. Read `src/cogpy/wave/phase_gradient.py` and compare algorithm logic:
   - Does `hilbert_phase()` match the MATLAB analytic signal approach?
   - Does `phase_gradient()` use complex conjugate multiplication (Arg(V * conj(V_shifted))) as in the MATLAB, or a different finite-difference scheme? The MATLAB uses centered differences on interior, forward on edges — verify the Python matches or document why it differs.
   - Does `pgd()` match the PGD definition from Zhang 2018 / Bhattacharya 2022?
   - Does `plane_wave_fit()` use circular-linear regression consistent with the MATLAB?

3. Check for:
   - Sign conventions (direction of gradient, coordinate system)
   - Edge handling (boundary conditions on gradient)
   - Phase unwrapping approach
   - Numerical equivalence on simple test cases

4. Fix any discrepancies found. If the Python implementation is intentionally different (e.g. simpler), add a comment explaining why.

5. Run `python -m pytest tests/wave/test_phase_gradient.py -v` to verify tests still pass after any fixes.

### References
- Zhang et al. 2018, DOI: 10.1016/j.neuron.2018.05.019
- Bhattacharya et al. 2022, DOI: 10.1371/journal.pcbi.1009827


## Batch Result

- status: done
- batch queue_id: `f9eff0043988`
- session: `d738b2f9-7755-4c25-a7fa-4e9b25695419`
- batch duration: 1349.7s
