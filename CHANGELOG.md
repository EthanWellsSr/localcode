# Changelog

Each version is a git tag. `localcode --version`, the header and `/status` show the version you're running.

## v0.6.1 — 2026-11-02

- Approval prompts keep the mouse wheel and PgUp/PgDn inside localcode, so you can scroll back through a long diff before answering. Previously the terminal scrolled its own history instead and the prompt went out of view.

## v0.6.0 — 2026-11-02

- `web_fetch` tool: the model can read web pages (docs, API references, JSON endpoints) as readable text, in parts for long pages.
- Auto mode (`--auto`, `/auto`): approves edits inside the project, web fetches, read-only commands, git queries and test runs; still asks about anything risky.
- The version is shown in the header, the welcome screen and `/status`.

## v0.5.0 — 2026-11-02

- Scroll the conversation while the model is thinking or streaming; output is kept while you're scrolled back.
- Approval prompts reliably take over keyboard input during a run.

## v0.4.0 — 2026-11-02

- Multiline composing, searchable command and model choosers, `/history`, `/copy` and `/export` (PR #1).

## v0.3.0 — 2026-10-31

- Terminal polish, session guidance and more reliable input.
- Terminal regression tests portable across macOS and Python 3.9.

## v0.2.0 — 2026-10-30

- Full-window interface: pinned banner and input box, alternate screen, scrollable conversation, `/` command menu.
- Markdown rendering while replies stream, `/model` picker, saved input history, collapsed thinking.
- Batch file reads with paging, `AGENTS.md` instructions, `pip install -e .`.

## v0.1.0 — 2026-10-30

- First version: a coding agent for local Ollama models with file, search and shell tools.
