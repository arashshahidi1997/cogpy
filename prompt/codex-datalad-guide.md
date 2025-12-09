# Datalad Command Template

Use this template whenever you need Codex to run `datalad` commands inside a specific conda environment.

## Prompt File

- Location: `prompt/datalad.prompt.md`
- Fill in `<message>` with a concise description of the change.
- Shared configuration lives in `prompt/prompt-paths.yaml` (`python`, `conda`, `conda_env`).

Copy the prompt file contents into the Codex chat after updating the placeholders. Codex will execute the command exactly as written.
