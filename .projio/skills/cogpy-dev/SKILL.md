---
name: cogpy-dev
description: Unified cogpy development skill — feature implementation, schema design, and testing. Modes: feature (research + plan + implement), schema (DIMS/validate/coerce), test (pytest cases), spec (test conventions).
metadata:
  short-description: cogpy feature + schema + test workflow
  tooling:
    mcp:
      - server: projio
        tools:
          - runtime_conventions
          - module_context
          - codio_vocab
          - codio_list
          - codio_get
          - codio_registry
          - codio_validate
          - note_list
---

# cogpy-dev — Unified Development Skill

Multi-mode skill for cogpy library development. Consolidates feature
implementation, schema design, and testing into one workflow with approval gates.

## Modes

Invoke with one of:

| Mode | When to use | Old skill |
|------|-------------|-----------|
| `feature` | Research + plan + implement a new cogpy feature | `add-feature-cogpy` |
| `schema` | Propose DIMS_* + validate_* + coerce_* for a new output shape | `cogpy-schema` |
| `test` | Write pytest cases for a new/modified function | `cogpy-test` |
| `spec` | (Re)generate test conventions doc (one-time setup) | `cogpy-test-spec` |
| `full` | Orchestrate feature → schema → implement → test with gates | all four |

If no mode is specified, infer from context:
- User mentions "test" → `test`
- User mentions "schema" / "dims" / "validate" → `schema`
- User mentions "add" / "implement" / "feature" → `feature`
- Default: `feature`

---

## Mode: feature

Implement or integrate a new feature into the cogpy library.

### Inputs

| Parameter | Required | Description |
|-----------|----------|-------------|
| `FEATURE_GOAL` | yes | One sentence: what changes for the user |
| `SCOPE` | yes | `core_compute` / `io_format` / `plot_viewer` / `cli` / `workflow_glue` |
| `API_SURFACE` | yes | `function` / `class` / `module` / `cli_entry` / `combination` |
| `DATA_MODEL` | yes | Expected in-memory type (xarray, numpy, etc.) |
| `PERF_SCALING` | no | Constraints (channels, duration, streaming, memory) |
| `REFERENCES` | no | Desired methods/capabilities |
| `CODELIB_NOTE` | no | Path to existing codelib-discovery output |

### Workflow

**Phase 0 — Locate insertion point**
- Identify likely cogpy module(s), scan for equivalents
- Decide placement: `core` vs `io` vs `plot` vs `cli`

**Phase 1 — Prior-art research**
- Check for existing codelib-discovery note; if found and recent, extract findings
- If no note, run codelib-discovery inline (search codio registry, RAG)
- Deep-read candidate libraries if wrapping (`codio_get()`)
- Determine path: `cogpy_existing` / `cogpy_wrap` / `cogpy_new` / `codelib_direct`

**Phase 2 — Integration plan** (STOP for approval)
- Public API signature, module path, config defaults
- Schema requirements (triggers `schema` mode if new output shape)
- Test plan, doc plan
- Link to discovery note

**Phase 3 — Implement**
- Pure compute in `cogpy.core`, IO wrappers in `cogpy.io`
- Follow existing patterns in the target module
- If new output shape needed, run `schema` mode first

**Phase 4 — Validate**
- `py_compile` check
- Run smallest relevant test subset
- Trigger `test` mode for full test coverage

### Guardrails
- Do not import runtime code from `code/lib/` mirrors (read-only)
- Runtime deps must be pip/conda packages
- Respect cogpy layering: pure compute in `cogpy.core`, formats in `cogpy.io`
- Keep diffs tight — implement only the requested feature
- Stop after Phase 2 for approval before implementing

---

## Mode: schema

Propose and implement schema additions to `cogpy/datasets/schemas.py`.
Use as a gate before implementing any feature that produces a new output shape.

### Inputs

| Parameter | Required | Description |
|-----------|----------|-------------|
| `FEATURE_GOAL` | yes | What the new feature produces |
| `OUTPUT_SHAPE` | yes | e.g., `(event, channel, lag)` |
| `COORDS` | yes | Coord names and what they represent |
| `ATTRS` | yes | Required attrs (always include `fs` if time-series) |

### Workflow

**Phase 1 — Propose** (STOP for approval)
- Read current `schemas.py`; note existing patterns
- Check for existing schema match — propose reuse if found
- Propose: `DIMS_*` constant, `validate_*` function, `coerce_*` function
- Print summary

**Phase 2 — Implement**
- Add `DIMS_*` after last existing constant; update `__all__`
- Add `validate_*` using existing private helpers (`_check_type`, `_check_dims`, etc.)
- Add `coerce_*` (transpose, inject coords/attrs, call `validate_*` at end)
- Verify import works
- Run existing schema tests

### Guardrails
- Never modify existing `validate_*` or `coerce_*` functions
- Never remove from `__all__`
- Always use existing private helpers — never inline checks
- Use canonical coord names: `time`, `channel`, `freq`, `lag`, `event`, `ML`, `AP`, `time_win`, `channel_i`, `channel_j`

---

## Mode: test

Write pytest test cases for a new or modified cogpy function.

### Inputs

| Parameter | Required | Description |
|-----------|----------|-------------|
| `MODULE` | yes | Dotted path (e.g., `cogpy.core.measures.temporal`) |
| `FUNCTION` | yes | Function or class name |
| `SCHEMA` | no | `DIMS_*` constant for output (if xr.DataArray) |
| `IS_WRAPPER` | no | `true` if wraps external library |

### Prerequisite

Test conventions spec must exist at `docs/reference/infra/testing/cogpy-test-conventions.md`.
If missing, run `spec` mode first.

### Workflow

1. Read test conventions spec
2. Read function source: signature, return type, docstring, edge cases
3. Read nearest existing test file to match style and fixtures
4. Determine test file path: `tests/<domain>/test_<module_stem>.py`
5. Write tests by category:
   - **Shape** — output.shape == expected
   - **Known-signal** — white noise range, pure sinusoid peak, coupled signals
   - **Schema** (if xr.DataArray) — validate_*/coerce_* valid/invalid
   - **Edge cases** — single_channel, single_sample, preserves_attrs
   - **Regression anchor** (if IS_WRAPPER) — wrapper vs direct library call
6. Run new tests: `pytest <file> -v`
7. Run full suite: `pytest tests/ -q`

### Guardrails
- Never modify source function to make tests pass
- Never use `@pytest.mark.skip` without a comment
- Test observable behavior, not implementation details
- Use deterministic seeds: `rng = np.random.default_rng(42)`
- All tolerances must be justified in a comment
- Import via public shim path (`from cogpy.<domain> import <function>`)

---

## Mode: spec

(Re)generate test conventions spec. One-time setup; update when conventions change.

### Inputs
None — self-contained discovery task.

### Workflow

1. Read at least 5 existing test files across domains (datasets, measures, spectral, brainstates, + 2 newest)
2. Extract conventions: file naming, imports, fixtures, test naming, parametrize, tolerances
3. Write spec to `docs/reference/infra/testing/cogpy-test-conventions.md`
4. Include two copy-paste examples: measure function test + schema test
5. Run all existing tests: `pytest tests/ -q` — report pass count

### Guardrails
- Do not fix failing tests — only note them
- If conventions are inconsistent across files, note the ambiguity

---

## Mode: full

Orchestrate the complete workflow with approval gates:

1. `feature` Phase 0-2 → STOP for plan approval
2. `schema` Phase 1 (if new output shape) → STOP for schema approval
3. `schema` Phase 2 (implement schema)
4. `feature` Phase 3 (implement feature)
5. `test` (write + run tests)
6. Final validation

Each gate requires explicit user approval before proceeding.
