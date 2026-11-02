# Working on localcode

localcode is a terminal coding agent for local Ollama models. All of it is in `localcode.py`; the tests are in `tests/test_localcode.py`.

## Rules

- Python 3.9 or newer, standard library only. Never add a dependency or run `pip install`.
- It must work on macOS and Linux. CI runs the tests on Ubuntu with Python 3.9 and 3.12, and on macOS with 3.12.
- Match the code around your change: its naming, its idioms, and its comment style. Comments are short and say why, not what.
- User-facing text (messages, help, README) is plain and specific. Say what happened and what to do next.
- Keep changes small and focused on what the user asked. Mention other problems you notice; don't fix them unasked.

## Workflow

1. Read the relevant code before changing it. `localcode.py` is long, so use `grep` to find the right place, then read that part.
2. For anything bigger than a one-line fix, or anything unclear, tell the user what you understood and what you plan to change, then wait for their go-ahead.
3. Make the change.
4. Add or update tests in `tests/test_localcode.py`:
   - The tests never contact Ollama. Session tests run localcode with `FAKE_BACKEND`, which replaces `lc.ollama` with a fake.
   - Terminal behavior is tested in a real pseudo-terminal (see `test_full_screen_paste_scroll_resize_and_exit`).
   - `web_fetch` tests use a local HTTP server, never the internet.
   - For a bug fix, add a test that fails without the fix.
5. Run all the tests and make sure they pass:
   - `python3 -m unittest discover -s tests`
   - On macOS, also run them with Python 3.9: `/usr/bin/python3 -m unittest discover -s tests`
6. Update the docs when behavior changes: `README.md` for users, and a new entry at the top of `CHANGELOG.md`.
7. Bump the version in both places: `VERSION` in `localcode.py` and `version` in `pyproject.toml`. Use the patch number (0.6.5 → 0.6.6) for fixes and small changes, and the minor number (0.6.x → 0.7.0) for new features.
8. Tell the user what changed, what you tested, and what you couldn't test. Then ask whether to commit, push and tag.

## Releasing (only when the user asks)

- Commit message: a short imperative subject ending in the version, like `Fix spinner leftovers (v0.6.6)`. Add a body that explains what changed and why.
- Push `main`: `git push origin main`
- Tag the commit and push the tag:
  - `git tag -a v0.6.6 -m "localcode v0.6.6: <what changed>"`
  - `git push origin v0.6.6`
- After pushing, check that CI passed: `gh run list --limit 4`. If a run failed, read why with `gh run view <id> --log-failed` and tell the user. Tests that pass locally can still fail on Linux.
- Never force-push, rewrite history, or move an existing tag.

## Design lessons

- Smaller models miss instructions. Don't enforce a behavior by guessing intent from the user's wording; give the user an explicit command or mode for it instead (like `/plan` and `/auto`).
- Approval rules that skip asking must be conservative. When unsure whether a command is safe, ask.
- Error messages returned to the model must not contradict each other. For example, a failed edit says to read the file again, so a re-read must work.
