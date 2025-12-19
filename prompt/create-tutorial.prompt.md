---
title: Create Tutorial Prompt
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.3
kernelspec:
  display_name: Python 3 (cogpy)
  language: python
  name: python3
---

Use the shared path variables in `prompt/prompt-paths.yaml` (`python`, `conda`,
`conda_env`).

# Create Tutorial Prompt

Codex, draft a new executable tutorial notebook under
`docs/source/tutorials/`. Keep the walkthrough runnable end-to-end so it can be
executed by Sphinx via `myst_nb`.

## Notebook skeleton

1. Pick a short kebab-case slug (for example, `erppca-overview`) and create
   `docs/source/tutorials/<slug>.md`.
2. Start the file with YAML front matter:

   ```yaml
   ---
   title: <Human-Readable Title>
   ---
   ```
3. Follow with a level-1 heading that repeats the title and states the tutorial
   goal in one sentence.
4. Add `##` sections for Overview, Setup, Data, Workflow, and Wrap-up (rename as
   needed, but keep the structure consistent with `sliding-core.md`).
5. Use MyST `code-cell` directives for runnable examples, and keep code blocks
   short so they can run within the docs build timeout.

## Required content

- **Imports & data:** synthesize or load a lightweight dataset that ships in the
  repo so the tutorial runs offline.
- **Core workflow:** demonstrate the API being documented with focused code
  cells and short explanations.
- **Validation:** include at least one assertion, log statement, or plot to show
  that the tutorial output is correct.

## Register the notebook

Add the new file to `docs/source/tutorials/index.md` so it appears in the
Tutorials toctree. Keep the list alphabetized.

## Build docs to verify

```{code-cell} ipython3
%%bash
set -euo pipefail
cd /storage2/arash/sirocampus/code/cogpy
make docs
```

Re-run the docs build after every major edit and fix warnings emitted by
Sphinx.
