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

`read_file`, `write_file`, `edit_file`, `list_dir`, `glob`, `grep`, `bash`.

Writes, edits and shell commands show a diff or the command and ask before running: `[y]es / [n]o / [a]lways`. `--yolo` skips the prompts.

## REPL commands

`/model [name]`, `/models`, `/clear`, `/yolo`, `/help`, `/exit`. End a line with `\` to keep typing on the next line.

## Settings

| Env var | Default |
| --- | --- |
| `LOCALCODE_MODEL` | `qwen2.5-coder:14b` |
| `LOCALCODE_NUM_CTX` | `32768` |
| `OLLAMA_HOST` | `http://localhost:11434` |

If an `AGENTS.md` exists in the project, it's added to the system prompt.

## Model notes

Some models print tool calls as text instead of returning them as structured calls (qwen2.5-coder writes bare JSON; qwen3-coder writes `<function=...>` XML). localcode parses both formats.

On a 24 GB Mac, `qwen3-coder:30b` with a 32k context doesn't fully fit on the GPU and runs slowly; the 14b model fits comfortably.
