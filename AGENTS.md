# Working on localcode

localcode is a terminal coding agent for local Ollama models. All of it is in `localcode.py`.

## Workflow

- Python 3.9+, standard library only. No dependencies.
- Read the relevant code before changing it.
- For anything bigger than a one-line fix, or anything unclear, tell me what you understood and what you plan to change, then wait for my go-ahead.
- Keep changes small and focused on what was asked.

## Releasing

- Commit message: imperative subject ending in the version (e.g., `Fix spinner leftovers (v0.6.6)`).
- Push `main`: `git push origin main`
- Tag: `git tag -a v0.7.3 -m "localcode v0.7.3: <what changed>"` + `git push origin v0.7.3`

## Design lessons

- Don't guess intent from wording; give explicit commands/mode for it (e.g., `/plan`, `/auto`).
- Approval rules that skip asking must be conservative.
- Error messages must not contradict each other.
