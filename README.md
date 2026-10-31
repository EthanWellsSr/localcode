# localcode

A terminal coding agent built around local [Ollama](https://ollama.com) models. One Python file, standard library only.

## Install

Needs Python 3.9+ and [Ollama](https://ollama.com).

```bash
git clone https://github.com/EthanWellsSr/localcode.git
cd localcode
pip3 install -e .
```

That adds a `localcode` command you can run from any folder. `-e` keeps it pointed at the clone, so `git pull` updates it with no reinstall. To remove it: `pip3 uninstall localcode`.

If your shell can't find `localcode` after installing, pip's script folder isn't on your PATH; `python3 -m pip show -f localcode` shows where it went.

## Run

```bash
ollama serve                     # if Ollama isn't already running
localcode                        # interactive, works in the current directory
localcode -C ~/code/myproject
localcode "fix the failing tests"   # one-shot
localcode -m qwen3-coder:30b        # pick a model
```

## Tools the model gets

`read_file`, `read_files` (several files in one call), `write_file`, `edit_file`, `list_dir`, `glob`, `grep`, `bash`.

Long files are returned in pages with the exact `offset` to read next. Re-reading an unchanged file gets a short "you already have this" reply instead of the full text again.

Writes, edits and shell commands show a diff or the command and ask before running: `[y]es / [n]o / [a]lways`. `--yolo` skips the prompts.

## Output

localcode runs full-window on the terminal's alternate screen, like `vim` or `htop`: your earlier terminal output is out of view while it runs and comes back unchanged when you quit. The LOCALCODE banner stays pinned at the top, the conversation scrolls in the middle, and the input box stays pinned to the bottom with a status line under it showing the model, the folder, how much of the context window is used, and the last reply's speed. Scroll back through the conversation with the mouse wheel, PgUp/PgDn or Shift+↑/↓. Because the mouse wheel is handled by localcode, selecting text with the mouse needs your terminal's override key held while dragging (Fn in Terminal.app, Option in iTerm2). Reasoning models' thinking is collapsed to a single "✻ Thought for Ns" line (set `LOCALCODE_SHOW_THINKING=1` to see it).

Replies are rendered as Markdown while they stream: headings and **bold** in bold, `code` in color, bullets as •, code blocks with a side bar, and tables drawn as aligned boxes that fit the terminal width. When output is piped instead of shown in a terminal, the raw Markdown is passed through.

## Commands and keys

Type `/` and a menu of commands appears under the input: ↑/↓ to choose, Tab to complete, Enter to run, Esc to close. After `/model ` it suggests your installed models.

| Command | |
| --- | --- |
| `/model` | Open the model picker, or `/model <name>` to switch straight to one |
| `/clear` | Start a fresh conversation (re-reads `AGENTS.md`) |
| `/yolo` | Toggle auto-approve for writes and commands |
| `/help` | Show commands and keys |
| `/exit` | Quit |

| Key | |
| --- | --- |
| Mouse wheel, PgUp/PgDn, Shift+↑/↓ | Scroll back through the conversation |
| ↑/↓ | Step through past entries (saved across sessions in `~/.config/localcode/history.jsonl`) |
| `\` then Enter | Start a new line; pasting multi-line text also works |
| Ctrl-C | Clear the line; on an empty line, quit |
| Ctrl-A / Ctrl-E, Ctrl-U / Ctrl-K, Ctrl-W | Start/end of line, delete to start/end, delete word |

The model you pick is remembered for next time (in `~/.config/localcode/state.json`); switching keeps the conversation.

## Settings

| Env var | Default |
| --- | --- |
| `LOCALCODE_MODEL` | the last model you picked, else `gpt-oss:20b` |
| `LOCALCODE_NUM_CTX` | `32768` |
| `OLLAMA_HOST` | `http://localhost:11434` |
| `LOCALCODE_SHOW_THINKING` | off; `1` prints reasoning models' thinking in full |

On startup, localcode looks for `AGENTS.md` (any capitalization) in the project and quietly adds it to the system prompt. A line under the banner says whether it was found and ingested.

## Model notes

Some models print tool calls as text instead of returning them as structured calls (qwen2.5-coder writes bare JSON; qwen3-coder writes `<function=...>` XML). localcode parses both formats.

The default is `gpt-oss:20b` (`ollama pull gpt-oss:20b`). It's a mixture-of-experts model that uses about 3.6B parameters per token, so on a 24 GB M-series MacBook Air it runs fully on the GPU at a 32k context (12 GB) and generates at about 18 tok/s, roughly twice as fast as the dense `qwen2.5-coder:14b`. `qwen3-coder:30b` doesn't fully fit on the GPU on that machine and runs slowly.
