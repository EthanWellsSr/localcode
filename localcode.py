#!/usr/bin/env python3
"""localcode: a small opencode-style coding agent for local Ollama models.

Usage:
    python3 localcode.py                     # interactive REPL
    python3 localcode.py "fix the tests"     # one-shot
    python3 localcode.py -m qwen3-coder:30b --yolo

Stdlib only. Talks to Ollama's /api/chat with native tool calling, and falls
back to parsing JSON tool calls out of plain text for models that do that.
"""

import argparse
import difflib
import fnmatch
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
if not OLLAMA_HOST.startswith("http"):
    OLLAMA_HOST = "http://" + OLLAMA_HOST
DEFAULT_MODEL = os.environ.get("LOCALCODE_MODEL", "qwen2.5-coder:14b")
NUM_CTX = int(os.environ.get("LOCALCODE_NUM_CTX", "32768"))
MAX_STEPS = 40
MAX_TOOL_OUTPUT = 16000
IGNORED_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", ".next", ".cache"}

# ---------------------------------------------------------------- terminal UI

USE_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def c(code, text):
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text


def dim(t): return c("2", t)
def bold(t): return c("1", t)
def red(t): return c("31", t)
def green(t): return c("32", t)
def yellow(t): return c("33", t)
def cyan(t): return c("36", t)


class Spinner:
    """Shows 'waiting for model… Ns' until stop() is called, so a slow model load doesn't look like a hang."""

    FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, label="waiting for model"):
        self.label = label
        self.done = threading.Event()
        self.thread = None
        if sys.stdout.isatty():
            self.thread = threading.Thread(target=self._spin, daemon=True)
            self.thread.start()

    def _spin(self):
        start = time.monotonic()
        i = 0
        while not self.done.wait(0.1):
            elapsed = int(time.monotonic() - start)
            sys.stdout.write("\r" + dim(f"{self.FRAMES[i % len(self.FRAMES)]} {self.label}… {elapsed}s"))
            sys.stdout.flush()
            i += 1
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()

    def stop(self):
        if self.thread and not self.done.is_set():
            self.done.set()
            self.thread.join()


def print_diff(old, new, path):
    lines = difflib.unified_diff(old.splitlines(), new.splitlines(), f"a/{path}", f"b/{path}", lineterm="", n=2)
    for line in lines:
        if line.startswith("+") and not line.startswith("+++"):
            print(green(line))
        elif line.startswith("-") and not line.startswith("---"):
            print(red(line))
        else:
            print(dim(line))


# ---------------------------------------------------------------- tools


class Tools:
    def __init__(self, root, auto_approve=False):
        self.root = Path(root).resolve()
        self.auto_approve = auto_approve
        self.always = set()  # tool names approved for the rest of the session
        self.read_files = set()

    def resolve(self, path):
        p = Path(os.path.expanduser(path or "."))
        return (p if p.is_absolute() else self.root / p).resolve()

    def rel(self, p):
        try:
            return str(p.relative_to(self.root))
        except ValueError:
            return str(p)

    def confirm(self, tool, summary):
        if self.auto_approve or tool in self.always:
            return True
        print(yellow(f"  allow {tool}? ") + summary)
        try:
            ans = input(yellow("  [y]es / [n]o / [a]lways: ")).strip().lower()
        except EOFError:
            return False
        if ans in ("a", "always"):
            self.always.add(tool)
            return True
        return ans in ("y", "yes", "")

    # each tool returns a string that is fed back to the model

    def read_file(self, path, offset=1, limit=400):
        p = self.resolve(path)
        if not p.is_file():
            return f"Error: {path} is not a file"
        lines = p.read_text(errors="replace").splitlines()
        offset = max(1, int(offset or 1))
        limit = int(limit or 400)
        chunk = lines[offset - 1: offset - 1 + limit]
        self.read_files.add(p)
        out = "\n".join(f"{i:5}| {l}" for i, l in enumerate(chunk, offset))
        if offset - 1 + limit < len(lines):
            out += f"\n... ({len(lines)} lines total; use offset to read more)"
        return out or "(empty file)"

    def write_file(self, path, content):
        p = self.resolve(path)
        old = p.read_text(errors="replace") if p.is_file() else ""
        print_diff(old, content, self.rel(p))
        if not self.confirm("write_file", self.rel(p)):
            return "User denied the write."
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"Wrote {len(content.splitlines())} lines to {self.rel(p)}"

    def edit_file(self, path, old_string, new_string, replace_all=False):
        p = self.resolve(path)
        if not p.is_file():
            return f"Error: {path} does not exist. Use write_file to create it."
        text = p.read_text(errors="replace")
        count = text.count(old_string)
        if count == 0:
            return ("Error: old_string not found. Read the file again and copy the exact text, "
                    "including whitespace and indentation (without the line-number prefix).")
        if count > 1 and not replace_all:
            return f"Error: old_string appears {count} times. Include more surrounding lines to make it unique, or set replace_all."
        new = text.replace(old_string, new_string) if replace_all else text.replace(old_string, new_string, 1)
        print_diff(text, new, self.rel(p))
        if not self.confirm("edit_file", self.rel(p)):
            return "User denied the edit."
        p.write_text(new)
        return f"Edited {self.rel(p)}"

    def list_dir(self, path="."):
        p = self.resolve(path)
        if not p.is_dir():
            return f"Error: {path} is not a directory"
        entries = sorted(p.iterdir(), key=lambda e: (not e.is_dir(), e.name))
        return "\n".join(e.name + ("/" if e.is_dir() else "") for e in entries if e.name not in IGNORED_DIRS) or "(empty)"

    def _walk(self, base):
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS and not d.startswith(".")]
            for f in filenames:
                yield Path(dirpath) / f

    def glob(self, pattern, path="."):
        base = self.resolve(path)
        hits = [self.rel(f) for f in self._walk(base)
                if fnmatch.fnmatch(str(f.relative_to(base)), pattern) or fnmatch.fnmatch(f.name, pattern)]
        hits.sort()
        return "\n".join(hits[:200]) + (f"\n... {len(hits)} matches total" if len(hits) > 200 else "") if hits else "No matches"

    def grep(self, pattern, path=".", include=None):
        try:
            rx = re.compile(pattern)
        except re.error as e:
            return f"Error: bad regex: {e}"
        base = self.resolve(path)
        files = [base] if base.is_file() else self._walk(base)
        out = []
        for f in files:
            if include and not fnmatch.fnmatch(f.name, include):
                continue
            try:
                with open(f, errors="strict") as fh:
                    for n, line in enumerate(fh, 1):
                        if rx.search(line):
                            out.append(f"{self.rel(f)}:{n}: {line.rstrip()[:200]}")
                            if len(out) >= 150:
                                return "\n".join(out) + "\n... (truncated)"
            except (UnicodeDecodeError, OSError):
                continue
        return "\n".join(out) or "No matches"

    def bash(self, command, timeout=120):
        print(dim(f"  $ {command}"))
        if not self.confirm("bash", ""):
            return "User denied the command."
        try:
            r = subprocess.run(command, shell=True, cwd=self.root, capture_output=True,
                               text=True, timeout=int(timeout or 120))
            out = (r.stdout + ("\n" + r.stderr if r.stderr else "")).strip()
            return f"exit code {r.returncode}\n{out}" if out else f"exit code {r.returncode}"
        except subprocess.TimeoutExpired:
            return f"Error: command timed out after {timeout}s"

    def call(self, name, args):
        fn = getattr(self, name, None)
        if name not in TOOL_NAMES or fn is None:
            return f"Error: unknown tool {name}. Available: {', '.join(TOOL_NAMES)}"
        if not isinstance(args, dict):
            return "Error: arguments must be a JSON object"
        try:
            return fn(**args)
        except TypeError as e:
            return f"Error: bad arguments for {name}: {e}"
        except Exception as e:
            return f"Error: {type(e).__name__}: {e}"


def schema(name, desc, props, required):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": required}}}


S = {"type": "string"}
TOOL_SCHEMAS = [
    schema("read_file", "Read a text file. Returns lines prefixed with line numbers.",
           {"path": S, "offset": {"type": "integer", "description": "1-based start line"},
            "limit": {"type": "integer", "description": "max lines (default 400)"}}, ["path"]),
    schema("write_file", "Create or overwrite a file with the full content.",
           {"path": S, "content": S}, ["path", "content"]),
    schema("edit_file", "Replace an exact string in a file. old_string must match exactly once "
           "(unless replace_all). Prefer this over write_file for changes to existing files.",
           {"path": S, "old_string": S, "new_string": S, "replace_all": {"type": "boolean"}},
           ["path", "old_string", "new_string"]),
    schema("list_dir", "List a directory's entries.", {"path": S}, []),
    schema("glob", "Find files by name pattern, e.g. '*.py' or 'src/**/*.ts'.",
           {"pattern": S, "path": S}, ["pattern"]),
    schema("grep", "Search file contents with a regex. Returns path:line: text.",
           {"pattern": S, "path": S, "include": {"type": "string", "description": "filename glob filter, e.g. '*.py'"}},
           ["pattern"]),
    schema("bash", "Run a shell command in the project directory (zsh/bash). Use for tests, git, builds.",
           {"command": S, "timeout": {"type": "integer"}}, ["command"]),
]
TOOL_NAMES = [t["function"]["name"] for t in TOOL_SCHEMAS]


def system_prompt(root):
    git = ""
    if (root / ".git").exists():
        try:
            git = subprocess.run("git status --short --branch | head -20", shell=True, cwd=root,
                                 capture_output=True, text=True, timeout=5).stdout
        except Exception:
            pass
    try:
        listing = "\n".join(sorted(e.name + ("/" if e.is_dir() else "")
                                   for e in root.iterdir() if e.name not in IGNORED_DIRS)[:60])
    except OSError:
        listing = ""
    prompt = f"""You are localcode, a coding agent running in the user's terminal. You help with software tasks by using tools to read, search, edit files and run commands.

Working directory: {root}
Platform: {sys.platform}

Top-level files:
{listing}
{"Git status:" + chr(10) + git if git else ""}
Rules:
- Use tools to look at the code before changing it. Never guess file contents.
- To change an existing file, read it first, then use edit_file with an exact old_string copied from the file (without the line-number prefix).
- Make one tool call at a time when later steps depend on earlier results.
- After making changes, verify them (run tests, the build, or the script) when that is possible.
- Keep replies short. When the task is done, say what you changed in a few sentences and stop calling tools.
"""
    for name in ("AGENTS.md", "LOCALCODE.md", "CLAUDE.md"):
        f = root / name
        if f.is_file():
            prompt += f"\nProject instructions from {name}:\n{f.read_text(errors='replace')[:8000]}\n"
            break
    return prompt


# ---------------------------------------------------------------- model client


def ollama(path, payload=None, stream=False):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(OLLAMA_HOST + path, data=data, headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=600)
    if not stream:
        return json.loads(resp.read())
    return (json.loads(line) for line in resp if line.strip())


TEXT_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>|```(?:json)?\s*(\{.*?\})\s*```", re.S)


XML_CALL_RE = re.compile(r"<function=([\w-]+)>(.*?)</function>", re.S)
XML_PARAM_RE = re.compile(r"<parameter=([\w-]+)>\n?(.*?)\n?</parameter>", re.S)


def coerce(value):
    """XML-style params arrive as strings; turn numbers/bools/JSON back into values."""
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def parse_text_tool_calls(text):
    """Fallback for models that print tool calls as text instead of using native tool calls.

    Handles JSON ({"name":..., "arguments":...} bare, in ```json fences or <tool_call> tags)
    and the qwen3-coder XML style (<function=name><parameter=k>v</parameter></function>).
    """
    calls = []
    for name, body in XML_CALL_RE.findall(text):
        if name in TOOL_NAMES:
            args = {k: (v if k in ("content", "old_string", "new_string", "command", "pattern", "path") else coerce(v))
                    for k, v in XML_PARAM_RE.findall(body)}
            calls.append({"function": {"name": name, "arguments": args}})
    if calls:
        return calls
    candidates = [a or b for a, b in TEXT_CALL_RE.findall(text)]
    stripped = text.strip()
    if not candidates and stripped.startswith("{") and stripped.endswith("}"):
        candidates = [stripped]
    for raw in candidates:
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("name") in TOOL_NAMES:
            args = obj.get("arguments", obj.get("parameters", {}))
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    continue
            calls.append({"function": {"name": obj["name"], "arguments": args}})
    return calls


class Agent:
    def __init__(self, model, tools):
        self.model = model
        self.tools = tools
        self.reset()

    def reset(self):
        self.messages = [{"role": "system", "content": system_prompt(self.tools.root)}]

    def chat_once(self):
        """Stream one assistant turn. Returns (content, tool_calls)."""
        payload = {"model": self.model, "messages": self.messages, "tools": TOOL_SCHEMAS,
                   "stream": True, "options": {"num_ctx": NUM_CTX}}
        content, tool_calls, thinking_shown = "", [], False
        stats = {}
        spinner = Spinner()
        try:
            for chunk in ollama("/api/chat", payload, stream=True):
                spinner.stop()
                content, thinking_shown, done = self._handle_chunk(chunk, content, tool_calls, thinking_shown)
                if done:
                    stats = chunk
        finally:
            spinner.stop()
        if content or thinking_shown:
            print()
        if stats.get("eval_count"):
            secs = stats.get("eval_duration", 1) / 1e9
            ctx = stats.get("prompt_eval_count", 0) + stats["eval_count"]
            print(dim(f"  [{stats['eval_count']} tok, {stats['eval_count'] / secs:.1f} tok/s, ctx ~{ctx}/{NUM_CTX}]"))
        return content, tool_calls

    def _handle_chunk(self, chunk, content, tool_calls, thinking_shown):
        """Print one streamed chunk. Returns (content, thinking_shown, done)."""
        if "error" in chunk:
            raise RuntimeError(chunk["error"])
        msg = chunk.get("message", {})
        if msg.get("thinking"):
            if not thinking_shown:
                print(dim("thinking: "), end="")
                thinking_shown = True
            print(dim(msg["thinking"]), end="", flush=True)
        if msg.get("content"):
            if thinking_shown:
                print()
                thinking_shown = False
            content += msg["content"]
            print(msg["content"], end="", flush=True)
        tool_calls.extend(msg.get("tool_calls") or [])
        return content, thinking_shown, bool(chunk.get("done"))

    def run(self, user_input):
        self.messages.append({"role": "user", "content": user_input})
        for _ in range(MAX_STEPS):
            content, tool_calls = self.chat_once()
            native = bool(tool_calls)
            if not native:
                tool_calls = parse_text_tool_calls(content)
            self.messages.append({"role": "assistant", "content": content,
                                  **({"tool_calls": tool_calls} if native else {})})
            if not tool_calls:
                return
            for call in tool_calls:
                fn = call.get("function", {})
                name, args = fn.get("name", ""), fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                preview = ", ".join(f"{k}={json.dumps(v)[:60]}" for k, v in args.items()
                                    if k not in ("content", "old_string", "new_string"))
                print(cyan(f"● {name}") + dim(f"({preview})"))
                result = str(self.tools.call(name, args))
                if len(result) > MAX_TOOL_OUTPUT:
                    result = result[:MAX_TOOL_OUTPUT] + f"\n... (truncated {len(result) - MAX_TOOL_OUTPUT} chars)"
                first = result.splitlines()[0] if result else ""
                print(dim(f"  └ {first[:120]}" + (f"  (+{result.count(chr(10))} lines)" if "\n" in result else "")))
                if native:
                    self.messages.append({"role": "tool", "content": result, "tool_name": name})
                else:
                    self.messages.append({"role": "user", "content": f"Tool result for {name}:\n{result}"})
        print(yellow(f"Stopped after {MAX_STEPS} steps."))


# ---------------------------------------------------------------- REPL

HELP = """Commands:
  /model [name]   show or switch model
  /models         list installed Ollama models
  /clear          start a fresh conversation
  /yolo           toggle auto-approve for writes and commands
  /exit           quit
End a line with \\ to continue on the next line."""


def list_models():
    return [m["name"] for m in ollama("/api/tags").get("models", [])]


def read_input():
    lines = []
    while True:
        line = input(bold("> ") if not lines else "  ")
        if line.endswith("\\"):
            lines.append(line[:-1])
            continue
        lines.append(line)
        return "\n".join(lines).strip()


def main():
    ap = argparse.ArgumentParser(description="Coding agent for local Ollama models")
    ap.add_argument("prompt", nargs="*", help="run one task and exit")
    ap.add_argument("-m", "--model", default=DEFAULT_MODEL)
    ap.add_argument("-C", "--cwd", default=".", help="project directory")
    ap.add_argument("--yolo", action="store_true", help="auto-approve writes and shell commands")
    args = ap.parse_args()

    try:
        installed = list_models()
    except (urllib.error.URLError, OSError):
        sys.exit(red(f"Can't reach Ollama at {OLLAMA_HOST}. Start it with `ollama serve`."))
    if args.model not in installed:
        sys.exit(red(f"Model {args.model} isn't installed. Installed: {', '.join(installed)}"))

    try:
        import readline  # noqa: F401  (arrow keys and history in input())
    except ImportError:
        pass

    tools = Tools(args.cwd, auto_approve=args.yolo)
    agent = Agent(args.model, tools)

    if args.prompt:
        agent.run(" ".join(args.prompt))
        return

    print(bold("localcode") + dim(f"  {agent.model}  ·  {tools.root}  ·  /help for commands"))
    while True:
        try:
            text = read_input()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not text:
            continue
        if text.startswith("/"):
            cmd, _, rest = text.partition(" ")
            if cmd in ("/exit", "/quit"):
                return
            elif cmd == "/help":
                print(HELP)
            elif cmd == "/clear":
                agent.reset()
                print(dim("Conversation cleared."))
            elif cmd == "/models":
                for m in list_models():
                    print(("* " if m == agent.model else "  ") + m)
            elif cmd == "/model":
                if rest.strip():
                    if rest.strip() in list_models():
                        agent.model = rest.strip()
                    else:
                        print(red("Not installed. Try /models."))
                print(dim(f"Model: {agent.model}"))
            elif cmd == "/yolo":
                tools.auto_approve = not tools.auto_approve
                print(dim(f"Auto-approve {'on' if tools.auto_approve else 'off'}."))
            else:
                print(red("Unknown command.") + " " + HELP)
            continue
        try:
            agent.run(text)
        except KeyboardInterrupt:
            print(yellow("\nInterrupted."))
        except (urllib.error.URLError, RuntimeError) as e:
            print(red(f"Model error: {e}"))


if __name__ == "__main__":
    main()
