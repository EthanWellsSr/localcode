# Working on localcode

localcode is a terminal coding agent for local Ollama models. The agent is in `localcode.py`; terminal UI helpers and the picker are in `localcode_ui.py`.

## Workflow

- Python 3.9+, standard library only. No dependencies.
- Read the relevant code before changing it.
- Do not add or run unit tests for now. The user explicitly wants to defer unit testing because it bogs down local models.
- For anything bigger than a one-line fix, or anything unclear, tell me what you understood and what you plan to change, then wait for my go-ahead.
- Keep changes small and focused on what was asked.

## Releasing

- Use this sequence for every release tag, including small-model-assisted releases:
  1. Read `VERSION` in `localcode.py`, `version` in `pyproject.toml`, the top of `CHANGELOG.md`, and the latest Git tags. If the user specifies a version, use it exactly; otherwise choose the next patch version for a fix or minor version for a feature.
  2. Update `VERSION`, `pyproject.toml`, and a new top entry in `CHANGELOG.md` to the same version. Do not change any existing tag.
  3. Check that the intended release files are the only changes, and confirm the version values match. Do not add or run unit tests for now.
  4. Commit the release changes with an imperative subject ending in the version, such as `Fix spinner leftovers (v0.8.1)`.
  5. Before tagging, check that the tag does not already exist locally or remotely. If it exists, stop; never move, delete, or reuse it.
  6. Create an annotated tag on the release commit and push the commit and tag:
     `git tag -a vX.Y.Z -m "localcode vX.Y.Z: <what changed>"`
     `git push origin main`
     `git push origin vX.Y.Z`
  7. Verify `localcode --version`, the package version, the commit, and the pushed tag. Report any check that could not be completed.

## Design lessons

- Don't guess intent from wording; give explicit commands/mode for it (e.g., `/plan`, `/auto`).
- Approval rules that skip asking must be conservative.
- Error messages must not contradict each other.

## Auto mode

Auto mode is enabled by default. It approves writes and edits inside the project, web fetches, read-only commands, git queries, and test runs, while asking about risky actions.
