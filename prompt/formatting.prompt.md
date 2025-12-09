Use the shared path variables in `prompt/prompt-paths.yaml` (`python`, `conda`, `conda_env`).

Codex, ensure the codebase is clean by running:

1. `black .`
2. `ruff check . --fix`
3. `mypy src`
