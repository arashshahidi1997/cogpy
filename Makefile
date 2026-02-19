# --- Config -------------------------------------------------------------------
SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

PYTHON     ?= /storage/share/python/environments/Anaconda3/envs/cogpy/bin/python
CONDA_ENV  ?= cogpy
DATALAD    := conda run -n $(CONDA_ENV) datalad
SPHINX_MAKE := $(MAKE) -C docs
DOCS_HTML  := docs/build/html
DOCS_PORT  ?= 8000
SAVE_MSG   ?= chore: cogpy update
ORIGIN_REMOTE ?= origin
PAGES_REMOTE  ?= gitlab

# --- Phony --------------------------------------------------------------------
.PHONY: help \
        all check format lint typecheck tests \
        docs docs-clean docs-serve clean \
        save push deploy website

# --- Help ---------------------------------------------------------------------
help:
	@echo "Usage:"
	@echo "  make all         Run format + lint + typecheck + tests + docs build"
	@echo "  make check       Run format + lint + typecheck + tests"
	@echo "  make format      Format Python sources with black"
	@echo "  make lint        Run Ruff with autofix"
	@echo "  make typecheck   Type-check src/ via mypy"
	@echo "  make tests       Run pytest"
	@echo "  make docs        Build Sphinx docs into $(DOCS_HTML)"
	@echo "  make docs-serve  Serve $(DOCS_HTML) via python -m http.server"
	@echo "  make docs-clean  Clean Sphinx build artifacts"
	@echo "  make update      datalad update --merge -s $(ORIGIN_REMOTE)"
	@echo "  make save        datalad save with SAVE_MSG='$(SAVE_MSG)'"
	@echo "  make push        datalad push to $(ORIGIN_REMOTE)"
	@echo "  make deploy      datalad push to $(PAGES_REMOTE)"

# --- Core workflows -----------------------------------------------------------
all: check docs

check: format lint typecheck tests

format:
	@echo ">> Running black"
	@$(PYTHON) -m black .

lint:
	@echo ">> Running Ruff (autofix)"
	@$(PYTHON) -m ruff check . --fix

typecheck:
	@echo ">> Running mypy on src/"
	@$(PYTHON) -m mypy src

tests:
	@echo ">> Running pytest"
	@$(PYTHON) -m pytest

# --- Docs ---------------------------------------------------------------------
docs:
	@echo ">> Building Sphinx docs into $(DOCS_HTML)"
	@$(SPHINX_MAKE) html

docs-clean:
	@echo ">> Cleaning Sphinx build directory"
	@$(SPHINX_MAKE) clean

serve:
	@echo ">> Serving docs from $(DOCS_HTML) at http://0.0.0.0:$(DOCS_PORT)"
	@cd "$(DOCS_HTML)" && $(PYTHON) -m http.server $(DOCS_PORT)

clean: docs-clean

# --- DataLad helpers ----------------------------------------------------------
update:
	@echo ">> datalad update --merge -s $(ORIGIN_REMOTE)"
	@$(DATALAD) update --merge -s $(ORIGIN_REMOTE)

save:
	@echo ">> datalad save -m \"$(SAVE_MSG)\""
	@$(DATALAD) save -m "$(SAVE_MSG)"

push:
	@echo ">> datalad push --to $(ORIGIN_REMOTE)"
	@$(DATALAD) push --to $(ORIGIN_REMOTE)

deploy:
	@echo ">> datalad push --to $(PAGES_REMOTE)"
	@$(DATALAD) push --to $(PAGES_REMOTE)

website:
	@echo ">> Docs build directory: $(DOCS_HTML)"
	@echo ">> Serve locally with: make docs-serve"
