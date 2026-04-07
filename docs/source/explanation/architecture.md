# Architecture And Scope

This document defines the intended architectural boundary of `cogpy`: what the
package should own, what it should leave to external projects, and what
technical conventions make the library composable over time.

`cogpy` is primarily a reusable **compute + IO toolkit** for ECoG / iEEG
analysis. It should stay maintainable by keeping core algorithms file-agnostic,
keeping file-format concerns in `cogpy.io`, and keeping full workflow
orchestration outside the package whenever possible.

## Package Goals

`cogpy` should provide:

1. **Reusable in-memory compute primitives** for preprocessing, spectral
   analysis, measures, and event detection.
2. **Structured IO helpers** for translating between file formats and the
   package's internal xarray-centered representations.
3. **Stable enough internal conventions** that external pipelines and frontends
   can compose `cogpy` outputs without per-project glue everywhere.
4. **Backend-facing utilities** for notebooks and visualization frontends (e.g.
   [TensorScope](https://github.com/arashshahidi1997/tensorscope), a separate
   React + TypeScript application).

## What `cogpy` Is And Is Not

`cogpy` is:

- a library of reusable computational building blocks
- an IO layer for loading, coercing, validating, and saving structured data
- a place to define stable internal conventions for common signal and output
  shapes
- a backend that external pipelines and frontends can call

`cogpy` is not:

- the owner of full project orchestration
- a replacement for project-level Snakemake DAGs, derivative registries,
  scheduling, or publication workflows
- a GUI application or frontend framework
- a claim that every schema in the package is already fully mature

The intended split is:

- `cogpy.*` (top-level subpackages) = reusable in-memory compute primitives
- `cogpy.io` = load/save, sidecars, and file-format translation
- external projects such as PixECoG = orchestration, dataset-specific configs,
  large-scale execution, and derivative layout

The repository may still contain packaged workflows or CLI wrappers for
convenience, testing, and examples. Those should remain thin composition layers
around stable library APIs, not grow into a second workflow engine inside
`cogpy`.

## Design Principles

- Keep core processing code file-agnostic and testable.
- Put file-format and sidecar logic in `cogpy.io`.
- Prefer stable data conventions over ad hoc array shapes.
- Make outputs structured enough for downstream pipelines and frontends.
- Allow lightweight composition helpers, but keep orchestration minimal.
- Be honest about evolving areas instead of freezing premature abstractions.

## Layering And Composition Philosophy

The intended layering is:

- `cogpy.io`: read files, translate to internal representations, validate or
  coerce schemas, save results, update sidecars
- `cogpy.*` (compute subpackages): transform in-memory arrays, compute measures,
  detect events, build reusable compute abstractions
- external pipelines: choose inputs, parameter sets, derivative paths,
  scheduling, caching, and execution order
- external frontends: own UI state, interaction design, visualization layout,
  and application logic

This means external project pipelines should usually be **thin orchestrators**
around stable `cogpy` APIs. A Snakemake rule or project script should mostly:

1. load via `cogpy.io`
2. call one or more `cogpy` compute functions or lightweight composition helpers
3. validate/coerce outputs where needed
4. save via `cogpy.io`

If reusable composition is needed inside `cogpy`, it should stay small and
local: detector pipelines, convenience wrappers, or helper objects are fine.
Owning project-scale DAG semantics is not.

## Core Data Model And Schema Expectations

Core functions should operate on structured in-memory objects, typically
`xarray.DataArray` and sometimes `xarray.Dataset`.

The intended direction is an xarray-centered internal model with common dims
such as:

- `time`
- `channel`
- `freq`
- `time_win` or other window-like dims where the operation is windowed
- `AP` / `ML` for grid-aware spatial layouts
- `ap` / `ml` in some frontend-oriented or orthoslicer-oriented views

Common expectations:

- dims should be named, not implied by axis position alone
- key coordinates such as `time`, `freq`, `channel`, `AP`, and `ML` should be
  present when the abstraction requires them
- sampling rate should be recoverable in a stable way, currently most often via
  `attrs["fs"]`
- metadata should travel with the array when it materially affects downstream
  interpretation

`DataArray` is usually the right representation for a single typed tensor such
as a signal, spectrogram, or feature map. `Dataset` is appropriate when a
result is naturally a named collection of aligned arrays.

The important architectural point is not that every object already has a final
frozen schema. It is that **common schemas should converge**, and boundary code
should validate or coerce inputs toward those schemas so independent pieces of
the package remain composable.

Today, `src/cogpy/datasets/schemas.py` already defines several canonical dim
orders and `validate_*` / `coerce_*` helpers. That is the right direction, but
the full schema story is still evolving. In particular, dim naming and case are
not yet perfectly uniform across all modules, so new code should favor existing
canonical validators rather than inventing new one-off conventions.

## Conceptual Layers Inside `cogpy`

The package should remain conceptually clear even when the exact package layout
evolves. The main roles are:

### Preprocess / transforms

These operations reshape or clean signals without claiming to be a final
scientific measure. Examples include filtering, rereferencing, line-noise
handling, resampling, interpolation, and normalization.

### Spectral transforms

These convert signals into representations such as PSDs or spectrograms. They
are still transforms: their main job is to produce a new representation for
later interpretation or downstream computation.

### Measures / feature maps

These compute interpretable quantities from signals or transforms. Outputs may
be scalars, spectra, channel-wise feature vectors, grid feature maps, or
windowed maps. Examples include channel features, spatial measures, and
frequency-domain summary measures.

### Detectors / event extraction

Detectors consume a signal or transformed representation and produce
event-oriented outputs such as catalogs, intervals, or peak tables.
Architecturally, this is different from a measure: a detector returns discrete
events with provenance and event-specific metadata, not only another dense
tensor.

### Plotting and frontend-facing backend utilities

`cogpy` may provide backend utilities that make structured tensors and event
outputs easier to inspect or hand to a frontend. The library should not absorb
UI state management or frontend application logic.

This distinction matters because transforms, measures, and detectors should not
all collapse into one generic "analysis" bucket. They have different contracts,
different output types, and different downstream uses.

## IO Responsibilities

`cogpy.io` owns:

- file-format translation
- metadata and sidecar handling
- construction of valid internal xarray objects from raw files
- saving structured outputs back to external formats

This is also the right place for thin convenience wrappers that combine compute
with required file bookkeeping. For example, updating sidecars after resampling
belongs in IO, not in compute subpackages.

## Structured Outputs

Core operations should return structured outputs that downstream code can rely
on. Depending on the abstraction, those outputs may be:

- transformed signals
- PSDs and spectrograms
- feature maps
- summary measures
- event tables or catalogs
- interval-like outputs

Stable structured outputs matter for two reasons:

1. external pipelines need outputs they can validate, serialize, and route
   without bespoke per-step parsing
2. frontends need predictable tensors and event-like tables for visualization,
   overlays, and linked inspection

The package should therefore prefer explicit output contracts over
loosely-typed tuples or ad hoc dicts when an output becomes a reusable boundary.

## Grid-Aware Processing

Grid-aware ECoG processing is a first-class concern in `cogpy`, not an
afterthought layered onto flat channels.

Important supported patterns include:

- AP×ML-aware filtering and spatial transforms
- neighborhood-based normalization relative to local grid structure
- grid adjacency and footprint helpers
- spatial measures over feature maps or spectral maps
- conversions between grid views and channel-stacked views when needed by
  viewers or downstream algorithms

This is why stable spatial conventions matter. A function that understands
`AP`/`ML` layout can do more than a generic channel-only function: it can use
local neighborhoods, directional structure, and electrode geometry in a
reusable way.

## Snakemake And External Pipeline Guidance

Project pipelines should remain thin orchestration layers around `cogpy`, even
when they are authored in the same repository or shipped as examples.

Good uses of Snakemake or project-level orchestration:

- dataset selection and batching
- dependency ordering and caching
- resource management
- derivative path layout
- project-specific configuration and provenance

Good uses of `cogpy` inside those pipelines:

- loading and schema coercion
- reusable compute primitives
- detector and transform composition helpers
- saving outputs and sidecars

The architectural target is not "no workflows anywhere in the repo." The target
is that workflow logic stays lightweight and replaceable, while the durable
technical value lives in stable `cogpy` APIs.

If companion conceptual docs such as `docs/reference/feature-registry` or
`docs/reference/example-snakemake-pipeline` are maintained, they should sit
above this architecture and describe:

- what feature families exist
- which `cogpy` outputs are intended to be stable boundaries
- how external orchestration should call those APIs without reimplementing them

## Preprocessing Structure

The preprocessing stack is still one of the clearest examples of the intended
architecture:

- focused core modules for filtering, resampling, interpolation, and line-noise
  handling
- grid-aware bad-channel feature extraction and neighborhood normalization
- thin wrappers or scripts that compose those pieces for pipeline use

Legacy modules may remain as compatibility shims, but new code should prefer the
more explicit modules that preserve schema clarity and keep compute separate
from orchestration.

## Frontend Boundary

[TensorScope](https://github.com/arashshahidi1997/tensorscope) is the primary
visualization frontend. It is a **standalone React + TypeScript application**
in its own repository — it is not part of cogpy.

`cogpy` acts as the **compute and data backend** for TensorScope and similar
tools:

- provide stable tensor representations
- provide transforms, measures, and detector outputs that can be visualized
- provide event-like outputs that a frontend can overlay or inspect

TensorScope itself owns:

- UI state
- interaction logic
- layout and rendering
- frontend-specific persistence and application behavior

Archived TensorScope design docs remain in `explanation/plot/_archive/` for
historical reference on backend requirements.

## Open Questions

- How far should schema normalization go across current dim-name variants such
  as `AP`/`ML`, `ap`/`ml`, and `time_win`?
- Which outputs should be normalized around `DataArray`, which around
  `Dataset`, and which around table-like types such as `EventCatalog`?
- Which structured outputs are mature enough to be treated as public contracts
  for external projects today, versus internal conventions still settling?
- How much packaged workflow support should remain in-repo as examples or
  convenience wrappers without pulling `cogpy` toward full workflow ownership?
