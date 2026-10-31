# localcode

A small opencode-style coding agent for local models running in [Ollama](https://ollama.com). One Python file, standard library only.

## Run

```bash
ollama serve                     # if Ollama isn't already running
python3 localcode.py             # interactive, works in the current directory
python3 localcode.py -C ~/code/myproject
python3 localcode.py "fix the failing tests"   # one-shot
python3 localcode.py -m qwen3-coder:30b        # pick a model
```

## Tools the model gets

`read_file`, `read_files` (several files in one call), `write_file`, `edit_file`, `list_dir`, `glob`, `grep`, `bash`.

Long files are returned in pages with the exact `offset` to read next. Re-reading an unchanged file gets a short "you already have this" reply instead of the full text again.

Writes, edits and shell commands show a diff or the command and ask before running: `[y]es / [n]o / [a]lways`. `--yolo` skips the prompts.

## REPL commands

`/model [name]`, `/models`, `/clear`, `/yolo`, `/help`, `/exit`. End a line with `\` to keep typing on the next line.

## Settings

| Env var | Default |
| --- | --- |
| `LOCALCODE_MODEL` | `qwen2.5-coder:14b` |
| `LOCALCODE_NUM_CTX` | `32768` |
| `OLLAMA_HOST` | `http://localhost:11434` |

On startup, localcode looks for `AGENTS.md` (any capitalization) in the project and quietly adds it to the system prompt. A line under the banner says whether it was found and ingested.

## Model notes

Some models print tool calls as text instead of returning them as structured calls (qwen2.5-coder writes bare JSON; qwen3-coder writes `<function=...>` XML). localcode parses both formats.

On a 24 GB Mac, `qwen3-coder:30b` with a 32k context doesn't fully fit on the GPU and runs slowly; the 14b model fits comfortably.
