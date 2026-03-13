# TensorScope Context Snapshot Template

Copy this template into a dated file under `snapshots/` (e.g. `YYYY-MM-DD.md`).

---

## Goal

- What are we trying to understand or fix?

## Observed Behavior

- What did the user see?
- What is the minimal repro (commands + dataset shape assumptions)?

## Suspected Root Cause (with code pointers)

- Hypothesis:
- Where to inspect (prefer **search patterns** over hardcoded paths):

```bash
rg -n "<pattern>" code/lib/cogpy/src/cogpy/core/plot/tensorscope -S
```

## Confirmed Root Cause

- What we confirmed and how (reading code, minimal tests, etc.)

## Minimal Fix Plan + Acceptance Checks

- Fix plan:
- Acceptance checks:
  - what should change in the UI/behavior
  - what should *not* regress

## What Changed in Understanding

- What was previously assumed?
- What is now known?

## Notes / Follow-ups

- Deferred improvements
- Open questions

