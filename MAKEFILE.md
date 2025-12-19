# Makefile Guide

This guide explains the primary targets defined in the repository `Makefile`.
Run `make help` to see the same summary in the terminal.

## Core targets

- `make all` – runs formatting, linting, type checking, tests, and builds the
  Sphinx docs.
- `make check` – runs `format`, `lint`, `typecheck`, and `tests` without building docs.
- `make format` – `black .`
- `make lint` – `ruff check . --fix`
- `make typecheck` – `mypy src`
- `make tests` – `pytest`

## Docs targets

- `make docs` – builds HTML docs via `make -C docs html`, output in `docs/build/html`.
- `make docs-clean` – cleans Sphinx build artifacts.
- `make docs-serve` – builds docs and serves them via `python -m http.server` on port 8000 (configurable via `DOCS_PORT`).
- `make clean` – alias for `docs-clean`.

## DataLad targets

- `make save` – runs `datalad save -m "$(SAVE_MSG)"` (default message `chore: cogpy update`).
- `make push` – `datalad push --to origin`.
- `make deploy` – `datalad push --to gitlab`.
- `make website` – prints where docs are built and how to serve them locally.

## Customization

At the top of the `Makefile` you can override:

- `PYTHON` – interpreter to run tools (defaults to CogPy env).
- `CONDA_ENV` – environment used for `datalad` commands.
- `SAVE_MSG`, `ORIGIN_REMOTE`, `PAGES_REMOTE`, `DOCS_PORT` – self‑explanatory.

Override any variable inline, e.g. `make SAVE_MSG="docs update" save`.
