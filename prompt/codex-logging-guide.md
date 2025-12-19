# Logging Prompt Template

Use this prompt whenever you want Codex to record work logs and reuse them as
`datalad` commit messages.

## Prompt File

- Location: `prompt/logging.prompt.md`
- Update `<timestamp>` and `<title>` placeholders before using.
- Shared configuration lives in `prompt/prompt-paths.yaml` (`python`, `conda`, `conda_env`).

After editing the prompt file, paste it into Codex. Codex will follow the steps to create the log and reuse it for `datalad save -F`.
