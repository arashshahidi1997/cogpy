Use the shared path variables in `prompt/prompt-paths.yaml` (`python`, `conda`, `conda_env`).

Codex, please do the following:

1. After completing the requested changes, write a markdown log entry at `logs/<timestamp>-<title>.md`. Include the same timestamp (with seconds) and title inside the file.
2. Run:
   ```
   conda run -n <conda_env> datalad save -F logs/<timestamp>-<title>.md
   ```
