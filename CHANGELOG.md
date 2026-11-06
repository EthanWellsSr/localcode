# Changelog

Each version is a git tag. `localcode --version`, the header and `/status` show the version you're running.

## v0.7.5 — 2026-11-06

- Repeated reads of unchanged files return the requested lines again, instead of telling the model to reuse an earlier response.

## v0.7.1 — 2026-11-05

- Auto mode is now enabled by default. Use `--no-auto` to start with approval required for all writes, commands, and web fetches.

## v0.7.0 — 2026-11-05

- Enter `!<command>` to run a shell command in the project directory. Its output and exit status appear in the chat and are sent to the model as context.

## v0.6.7 — 2026-11-02

- The cursor is hidden while the model works, instead of sitting at the end of the output. It shows at an approval prompt while you answer, and in the input box once the model is done.
- The hint under the input box no longer mentions PgUp (the mouse wheel and PgUp still scroll).

## v0.6.6 — 2026-11-02

- Fix a crash when the window is resized while localcode is writing to a slow terminal: the redraw now waits until the write finishes instead of failing with "reentrant call". This was also why CI failed now and then.
- The terminal tests keep reading localcode's output until it exits, and print every thread's stack if it hangs.

## v0.6.5 — 2026-11-02

- The welcome screen no longer shows suggested prompts.

## v0.6.4 — 2026-11-02

- Plan mode (`/plan`): the model can read, search, fetch and run read-only commands, and replies with its plan. Anything that changes files or state is refused until you turn plan mode off.
- The system prompt asks the model to share its understanding first when you ask for that, and on large or unclear tasks, and to follow instructions about how to work even at the end of a long message.
- Reasoning models get their own reasoning back between the tool calls of a task, which keeps gpt-oss from thinking out loud in its replies. Reasoning from finished tasks is still dropped.

## v0.6.3 — 2026-11-02

- Auto mode approves `git add` and `git commit -m …`, and a plain `git push` when your latest message asks to push. Force-pushes, deleting branches, pushing tags, amending and skipping hooks still ask.

## v0.6.2 — 2026-11-02

- After a failed `edit_file`, reading that file again returns its text instead of "you already read this", so the model can copy the exact text and retry.
- The repeat-read guard only refuses a read that asks for no more than an earlier one; asking for more lines (a bigger `limit`) is answered.
- A reply that is only a tool's arguments with no tool name runs that tool when exactly one tool fits. Otherwise localcode asks the model once to retry, instead of ending the turn silently.
- The spinner clears the rest of its line, so switching from "waiting for model" to "thinking" leaves no leftover text.

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
