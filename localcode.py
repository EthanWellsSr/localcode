#!/usr/bin/env python3
"""localcode: a terminal coding agent built around local Ollama models.

Usage:
    python3 localcode.py                     # interactive REPL
    python3 localcode.py "fix the tests"     # one-shot
    python3 localcode.py -m qwen3-coder:30b --yolo

Stdlib only. Talks to Ollama's /api/chat with native tool calling, and falls
back to parsing JSON tool calls out of plain text for models that do that.
"""

import argparse
import atexit
import difflib
import fnmatch
import json
import os
import re
import shutil
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
DEFAULT_MODEL = "gpt-oss:20b"
NUM_CTX = int(os.environ.get("LOCALCODE_NUM_CTX", "32768"))
MAX_STEPS = 40
SHOW_THINKING = os.environ.get("LOCALCODE_SHOW_THINKING") == "1"  # else collapsed to "Thought for Ns"
MAX_TOOL_OUTPUT = 24000
READ_BUDGET = 12000  # chars per read_file call
READ_FILES_BUDGET = 24000  # chars per read_files call
IGNORED_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", ".next", ".cache"}

# ---------------------------------------------------------------- terminal UI

USE_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def c(code, text):
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text


DIM_ON = "\033[2m"  # re-enable dim after a nested color reset


def dim(t): return c("2", t)
def bold(t): return c("1", t)
def red(t): return c("31", t)
def green(t): return c("32", t)
def yellow(t): return c("33", t)
def cyan(t): return c("36", t)


def prompt(text):
    """Wrap color codes in \\001/\\002 so readline doesn't count them as visible width.

    Without this, readline thinks the prompt is wider than it is and backspace/redraw
    lands in the wrong column, eating into earlier output.
    """
    return re.sub(r"(\033\[[0-9;]*m)", "\001\\1\002", text)


def term_out():
    """The real terminal stream, bypassing the full-window conversation recorder when it's installed."""
    return getattr(sys.stdout, "raw", sys.stdout)


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
            out = term_out()
            out.write("\r" + dim(f"{self.FRAMES[i % len(self.FRAMES)]} {self.label}… {elapsed}s"))
            out.flush()
            i += 1
        out = term_out()
        out.write("\r\033[K")
        out.flush()

    def stop(self):
        if self.thread and not self.done.is_set():
            self.done.set()
            self.thread.join()


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
BOLD, NOT_BOLD, CODE, NOT_CODE = "\033[1m", "\033[22m", "\033[36m", "\033[39m"


def visible_len(s):
    return len(ANSI_RE.sub("", s))


def render_inline(s):
    """**bold** and `code` for text we have in full (headings, table cells)."""
    s = re.sub(r"\*\*(.+?)\*\*", BOLD + r"\1" + NOT_BOLD, s)
    return re.sub(r"`([^`]+)`", CODE + r"\1" + NOT_CODE, s)


def wrap_ansi(s, width):
    """Word-wrap text that contains color codes, counting only visible characters."""
    lines, line, vis, word, wvis = [], "", 0, "", 0
    for tok in re.findall(r"\x1b\[[0-9;]*m|.", s) + [" "]:
        if tok != " ":
            word += tok
            wvis += 0 if tok.startswith("\x1b") else 1
            continue
        if not word:
            continue
        if vis and vis + 1 + wvis > width:
            lines.append(line)
            line, vis = "", 0
        line += (" " if vis else "") + word
        vis += (1 if vis else 0) + wvis
        word, wvis = "", 0
    lines.append(line)
    return lines


class MarkdownStream:
    """Renders streamed Markdown for the terminal, line by line.

    Paragraphs, bullets and quotes stream as they arrive (with **bold** and `code` styled on the fly).
    Headings, horizontal rules and fence lines wait for their newline; tables are held until the
    table ends, then drawn as an aligned box that fits the terminal width.
    """

    def __init__(self):
        self.width = shutil.get_terminal_size((100, 24)).columns
        self.line = ""        # the current source line
        self.mode = None      # how the current line is rendered, once we know
        self.done = 0         # chars of self.line already rendered
        self.in_code = False  # inside a ``` block
        self.table = []
        self.bold = self.code = False
        self.star = False     # saw one '*', waiting to see if it's '**'
        self.mid_line = False

    def _w(self, s):
        if s:
            sys.stdout.write(s)
            self.mid_line = not s.endswith("\n")

    def feed(self, text):
        if not USE_COLOR:  # piped output: leave the Markdown as-is
            self._w(text)
            sys.stdout.flush()
            return
        for ch in text:
            if ch == "\n":
                self._end_line()
                continue
            self.line += ch
            if self.mode is None:
                self._classify()
            if self.mode in ("inline", "code"):
                self._stream()
        sys.stdout.flush()

    def _classify(self):
        """Decide how to render the current line, or return with mode None to wait for more text."""
        s = self.line.lstrip()
        indent = self.line[:len(self.line) - len(s)]
        if not s:
            return
        if self.table and not s.startswith("|"):
            self._flush_table()
        if self.in_code:
            if s.startswith("```"):
                self.mode = "fence"
            elif "```".startswith(s):
                return
            else:
                self._w(dim("│ "))
                self.mode = "code"
            return
        if s.startswith("|"):
            self.mode = "table"
        elif s.startswith("```"):
            self.mode = "fence"
        elif re.fullmatch(r"#{1,6}", s) or re.fullmatch(r"\d{1,3}[.)]?", s) or s in ("`", "``", ">") \
                or re.fullmatch(r"([-*_+])\1*", s):
            return  # could still become a heading, list item, fence, quote or rule
        elif re.match(r"#{1,6} ", s):
            self.mode = "heading"
        elif m := re.match(r"[-*+] ", s):
            self._w(indent + cyan("•") + " ")
            self.mode, self.done = "inline", len(indent) + m.end()
        elif m := re.match(r"\d{1,3}[.)] ", s):
            self._w(indent + cyan(m.group(0)))
            self.mode, self.done = "inline", len(indent) + m.end()
        elif s.startswith(">"):
            self._w(indent + dim("│ "))
            self.mode, self.done = "inline", len(indent) + (2 if s.startswith("> ") else 1)
        else:
            self.mode = "inline"

    def _stream(self):
        rest = self.line[self.done:]
        self.done = len(self.line)
        if self.mode == "code":
            self._w(rest)
            return
        for ch in rest:
            if self.star:
                self.star = False
                if ch == "*":
                    self.bold = not self.bold
                    self._w(BOLD if self.bold else NOT_BOLD)
                    continue
                self._w("*")
            if ch == "*":
                self.star = True
            elif ch == "`":
                self.code = not self.code
                self._w(CODE if self.code else NOT_CODE)
            else:
                self._w(ch)

    def _end_line(self):
        s = self.line.strip()
        if self.mode is None and s:
            self._classify()
        if self.mode is None:
            if not s:
                if self.table:
                    self._flush_table()
                self._w("\n")
            elif re.fullmatch(r"([-*_])\1{2,}", s):
                self._w(dim("─" * min(self.width, 60)) + "\n")
            else:  # a lone "#", "1." or similar: just show it
                self.mode = "code" if self.in_code else "inline"
                self._stream()
        if self.mode == "heading":
            level = len(s) - len(s.lstrip("#"))
            text = render_inline(s[level:].strip().rstrip("#").strip())
            self._w(("\033[1;4m" if level == 1 else BOLD) + text + "\033[0m\n")
        elif self.mode == "fence":
            self.in_code = not self.in_code
        elif self.mode == "table":
            self.table.append(s)
        elif self.mode in ("inline", "code"):
            if self.star:
                self._w("*")
            if self.bold or self.code:
                self._w("\033[0m")
            self._w("\n")
        self.line, self.mode, self.done = "", None, 0
        self.bold = self.code = self.star = False

    def _flush_table(self):
        rows, header = [], False
        for raw in self.table:
            cells = [c.strip() for c in raw.strip().strip("|").split("|")]
            if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c) and any(cells):
                header = len(rows) == 1
                continue
            rows.append([render_inline(c) for c in cells])
        self.table = []
        if not rows:
            return
        ncol = max(len(r) for r in rows)
        rows = [r + [""] * (ncol - len(r)) for r in rows]
        widths = [max(visible_len(r[i]) for r in rows) for i in range(ncol)]
        avail = self.width - (3 * ncol + 1)
        while sum(widths) > avail and max(widths) > 8:
            widths[widths.index(max(widths))] -= 1

        def border(left, mid, right):
            return dim(left + mid.join("─" * (w + 2) for w in widths) + right) + "\n"

        out = border("┌", "┬", "┐")
        for n, row in enumerate(rows):
            cells = [wrap_ansi(c, w) for c, w in zip(row, widths)]
            for k in range(max(len(c) for c in cells)):
                parts = []
                for c, w in zip(cells, widths):
                    text = c[k] if k < len(c) else ""
                    if n == 0 and header:
                        text = BOLD + text + NOT_BOLD
                    parts.append(" " + text + "\033[0m" + " " * max(0, w - visible_len(text)) + " ")
                out += dim("│") + dim("│").join(parts) + dim("│") + "\n"
            if n == 0 and header and len(rows) > 1:
                out += border("├", "┼", "┤")
        self._w(out + border("└", "┴", "┘"))

    def finish(self):
        if not USE_COLOR:
            return
        if self.line or self.mode:
            self._end_line()
        if self.table:
            self._flush_table()
        sys.stdout.flush()


class StreamPrinter:
    """Prints streamed model text, but holds back anything that might be a text-format tool call.

    Once a line starts like a tool call (``` fence, '{', <function=, <tool_call>), everything from
    there on is held. finish() then prints it only if it turned out not to be a tool call.
    """

    MARKERS = ("```", "{", "<function=", "<tool_call>")

    def __init__(self):
        self.text = ""
        self.printed = 0
        self.holding = False
        self.md = MarkdownStream()

    @property
    def needs_newline(self):
        return self.md.mid_line

    def _emit(self, s):
        if s:
            self.md.feed(s)

    def _could_be_marker(self, line):
        line = line.lstrip()
        return not line or any(line.startswith(m) or m.startswith(line) for m in self.MARKERS)

    def feed(self, s):
        self.text += s
        if self.holding:
            return
        # Look at each complete or partial line we haven't printed yet.
        start = self.text.rfind("\n", 0, self.printed) + 1
        while True:
            nl = self.text.find("\n", start)
            line = self.text[start:] if nl == -1 else self.text[start:nl]
            stripped = line.lstrip()
            if stripped and any(stripped.startswith(m) for m in self.MARKERS):
                self._emit(self.text[self.printed:start])
                self.printed = start
                self.holding = True
                return
            if nl == -1:
                # Partial line: print it unless it might still turn into a marker.
                if not self._could_be_marker(line):
                    self._emit(self.text[self.printed:])
                    self.printed = len(self.text)
                else:
                    self._emit(self.text[self.printed:start])
                    self.printed = max(self.printed, start)
                return
            start = nl + 1

    def finish(self, show_held):
        if show_held:
            self._emit(self.text[self.printed:])
        self.printed = len(self.text)
        self.md.finish()


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
        self.seen = {}  # (path, offset) -> (mtime, last line) for the repeat-read guard

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
            question = yellow("  [y]es / [n]o / [a]lways: ")
            ask = getattr(sys.stdout, "ask", None)  # full-window mode records the prompt and answer
            ans = ask(question) if ask else input(prompt(question))
        except EOFError:
            return False
        ans = ans.strip().lower()
        if ans:
            drop_last_history()  # keep y/n answers out of the up-arrow history
        if ans in ("a", "always"):
            self.always.add(tool)
            return True
        return ans in ("y", "yes", "")

    # each tool returns a string that is fed back to the model

    def forget(self):
        """Drop the record of what's been read (the model no longer has it after /clear)."""
        self.seen.clear()

    def read_file(self, path, offset=1, limit=None, budget=READ_BUDGET):
        p = self.resolve(path)
        if not p.is_file():
            return f"Error: {path} is not a file"
        offset = max(1, int(offset or 1))
        mtime = p.stat().st_mtime
        if self.seen.get((p, offset), (None,))[0] == mtime:
            end = self.seen[(p, offset)][1]
            return (f"You already read lines {offset}-{end} of {self.rel(p)} earlier in this conversation "
                    "and the file hasn't changed since. Use that output instead of reading it again.")
        lines = p.read_text(errors="replace").splitlines()
        if not lines:
            return "(empty file)"
        if offset > len(lines):
            return f"Error: offset {offset} is past the end ({len(lines)} lines)"
        # Fill up to the character budget on whole lines, so results are never cut mid-line.
        out, used, end = [], 0, offset - 1
        stop = len(lines) if not limit else min(len(lines), offset - 1 + int(limit))
        for i in range(offset - 1, stop):
            row = f"{i + 1:5}| {lines[i]}"
            if out and used + len(row) + 1 > budget:
                break
            out.append(row)
            used += len(row) + 1
            end = i + 1
        self.seen[(p, offset)] = (mtime, end)
        text = "\n".join(out)
        if end < len(lines):
            text += (f"\n... showing lines {offset}-{end} of {len(lines)}. "
                     f"Call read_file with offset={end + 1} to continue.")
        return text

    def read_files(self, paths):
        if isinstance(paths, str):
            paths = [paths]
        parts, used, skipped = [], 0, []
        for path in paths:
            remaining = READ_FILES_BUDGET - used
            if remaining < 2000:
                skipped.append(path)
                continue
            body = self.read_file(path, budget=min(READ_BUDGET, remaining))
            parts.append(f"==> {path} <==\n{body}")
            used += len(parts[-1]) + 1
        if skipped:
            parts.append(f"Not read (output limit reached): {', '.join(skipped)}. Read them in another call.")
        return "\n\n".join(parts)

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
    schema("read_file", "Read one text file. Returns lines prefixed with line numbers. Long files come back "
           "in pages; the result says which offset to pass to continue.",
           {"path": S, "offset": {"type": "integer", "description": "1-based start line"},
            "limit": {"type": "integer", "description": "max lines (optional)"}}, ["path"]),
    schema("read_files", "Read several text files in one call. Prefer this over repeated read_file calls "
           "when you need more than one file, e.g. to get familiar with a project.",
           {"paths": {"type": "array", "items": S}}, ["paths"]),
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
- When you need several files (for example to get familiar with a project), read them together with one read_files call.
- To change an existing file, read it first, then use edit_file with an exact old_string copied from the file (without the line-number prefix).
- Make one tool call at a time when later steps depend on earlier results.
- After making changes, verify them (run tests, the build, or the script) when that is possible.
- Keep replies short. When the task is done, say what you changed in a few sentences and stop calling tools.
"""
    instructions = find_instructions(root)
    if instructions:
        prompt += f"\nProject instructions from {instructions.name}:\n{instructions.read_text(errors='replace')[:8000]}\n"
    return prompt, instructions


def find_instructions(root):
    """Return the project's instructions file (AGENTS.md first, any capitalization), or None."""
    try:
        files = {e.name.lower(): e for e in root.iterdir() if e.is_file()}
    except OSError:
        return None
    for name in ("agents.md", "localcode.md", "claude.md"):
        if name in files:
            return files[name]
    return None


def instructions_status(path):
    if path:
        return green(f"✓ {path.name} found and ingested")
    return dim("✗ no AGENTS.md found")


# ---------------------------------------------------------------- model client


def ollama(path, payload=None, stream=False):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(OLLAMA_HOST + path, data=data, headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=600)
    if not stream:
        return json.loads(resp.read())
    return (json.loads(line) for line in resp if line.strip())


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

    Handles JSON ({"name":..., "arguments":...} anywhere in the text, including fences and <tool_call> tags)
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
    # Scan for JSON objects anywhere in the text, so calls after prose or inside fences are found.
    decoder = json.JSONDecoder()
    i = text.find("{")
    while i != -1:
        try:
            obj, end = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            i = text.find("{", i + 1)
            continue
        i = text.find("{", end)
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
        self.show_stats = True
        self.reset()

    def reset(self):
        prompt, self.instructions = system_prompt(self.tools.root)
        self.messages = [{"role": "system", "content": prompt}]
        self.tools.forget()
        self.ctx_used, self.last_speed = 0, 0.0

    def chat_once(self):
        """Stream one assistant turn. Returns (content, tool_calls, native)."""
        payload = {"model": self.model, "messages": self.messages, "tools": TOOL_SCHEMAS,
                   "stream": True, "options": {"num_ctx": NUM_CTX}}
        tool_calls = []
        printer = StreamPrinter()
        stats = {}
        spinner = Spinner()
        think_start = None
        try:
            for chunk in ollama("/api/chat", payload, stream=True):
                if "error" in chunk:
                    raise RuntimeError(chunk["error"])
                msg = chunk.get("message", {})
                if msg.get("thinking"):
                    # Reasoning models think before answering: keep the spinner going and collapse it to one line.
                    if think_start is None:
                        think_start = time.monotonic()
                        spinner.label = "thinking"
                    if SHOW_THINKING:
                        spinner.stop()
                        print(dim(msg["thinking"]), end="", flush=True)
                if not (msg.get("content") or msg.get("tool_calls") or chunk.get("done")):
                    continue
                spinner.stop()
                if think_start is not None:
                    print(("\n" if SHOW_THINKING else "") + dim(f"✻ Thought for {time.monotonic() - think_start:.0f}s"))
                    think_start = None
                if msg.get("content"):
                    printer.feed(msg["content"])
                tool_calls.extend(msg.get("tool_calls") or [])
                if chunk.get("done"):
                    stats = chunk
        finally:
            spinner.stop()
        content = printer.text
        native = bool(tool_calls)
        if not native:
            tool_calls = parse_text_tool_calls(content)
        # Text that turned out to be a tool call stays hidden; the ● line below shows it instead.
        printer.finish(show_held=native or not tool_calls)
        if printer.needs_newline:
            print()
        if stats.get("eval_count"):
            secs = stats.get("eval_duration", 1) / 1e9
            self.ctx_used = stats.get("prompt_eval_count", 0) + stats["eval_count"]
            self.last_speed = stats["eval_count"] / secs
            if self.show_stats:
                print(dim(f"  [{stats['eval_count']} tok, {self.last_speed:.1f} tok/s, ctx ~{self.ctx_used}/{NUM_CTX}]"))
        return content, tool_calls, native


    def run(self, user_input):
        self.messages.append({"role": "user", "content": user_input})
        for _ in range(MAX_STEPS):
            content, tool_calls, native = self.chat_once()
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
                    cut = result.rfind("\n", 0, MAX_TOOL_OUTPUT)
                    cut = cut if cut > 0 else MAX_TOOL_OUTPUT
                    result = result[:cut] + f"\n... (output cut off here; {len(result) - cut} more chars not shown)"
                first = result.splitlines()[0] if result else ""
                print(dim(f"  └ {first[:120]}" + (f"  (+{result.count(chr(10))} lines)" if "\n" in result else "")))
                if native:
                    self.messages.append({"role": "tool", "content": result, "tool_name": name})
                else:
                    self.messages.append({"role": "user", "content": f"Tool result for {name}:\n{result}"})
        print(yellow(f"Stopped after {MAX_STEPS} steps."))


# ---------------------------------------------------------------- REPL

COMMANDS = {
    "/model": "pick a model, or /model <name> to switch straight to one",
    "/clear": "start a fresh conversation",
    "/yolo": "toggle auto-approve for writes and commands",
    "/help": "show commands and keys",
    "/exit": "quit localcode",
}
HELP = "Commands:\n" + "\n".join(f"  {c:<8} {d}" for c, d in COMMANDS.items()) + """

Keys:
  /          suggest commands (↑/↓ to choose, Tab to complete, Enter to run, Esc to close)
  ↑/↓        step through past entries
  \\ + Enter  start a new line
  Ctrl-C     clear the line; on an empty line, quit"""
STATE_PATH = Path(os.environ.get("XDG_CONFIG_HOME") or "~/.config").expanduser() / "localcode" / "state.json"


def load_state():
    try:
        return json.loads(STATE_PATH.read_text())
    except (OSError, ValueError):
        return {}


def save_state(**updates):
    try:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps({**load_state(), **updates}, indent=2))
    except OSError:
        pass  # remembering the model is a convenience; never fail over it


def list_models():
    return sorted(m["name"] for m in ollama("/api/tags").get("models", []))


def model_details():
    """Installed models with size, parameter count, and whether they support tool calling."""
    details = []
    for m in sorted(ollama("/api/tags").get("models", []), key=lambda m: m["name"]):
        try:
            caps = ollama("/api/show", {"model": m["name"]}).get("capabilities", [])
        except (urllib.error.URLError, OSError, ValueError):
            caps = []
        details.append({"name": m["name"], "size": m.get("size", 0),
                        "params": m.get("details", {}).get("parameter_size", ""), "tools": "tools" in caps})
    return details


def pick(rows, start=0, title=""):
    """Arrow-key list picker. Returns the chosen index, or None if cancelled."""
    if not rows:
        return None
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        for i, r in enumerate(rows, 1):
            print(f"  {i}. {r}")
        try:
            n = input("Number (blank to cancel): ").strip()
        except EOFError:
            return None
        return int(n) - 1 if n.isdigit() and 1 <= int(n) <= len(rows) else None

    import select
    import termios
    import tty

    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    sel = start if 0 <= start < len(rows) else 0
    out = term_out()

    def draw(first=False):
        if not first:
            out.write(f"\033[{len(rows)}A")
        for i, r in enumerate(rows):
            out.write("\r\033[K" + (cyan(f"❯ {r}") if i == sel else f"  {r}") + "\n")
        out.flush()

    out.write(dim(title) + "\n")
    out.write("\033[?25l")
    result = None
    try:
        tty.setcbreak(fd)
        draw(first=True)
        while True:
            ch = os.read(fd, 1)
            if ch == b"\x1b":
                if not select.select([fd], [], [], 0.05)[0]:
                    break  # bare Esc
                seq = os.read(fd, 2)
                if seq == b"[A":
                    sel = (sel - 1) % len(rows)
                elif seq == b"[B":
                    sel = (sel + 1) % len(rows)
            elif ch in (b"\r", b"\n"):
                result = sel
                break
            elif ch == b"k":
                sel = (sel - 1) % len(rows)
            elif ch == b"j":
                sel = (sel + 1) % len(rows)
            elif ch.isdigit() and 1 <= int(ch) <= len(rows):
                sel = int(ch) - 1
            elif ch in (b"q", b"\x03"):
                break
            draw()
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)
        out.write(f"\033[{len(rows) + 1}A\r\033[J\033[?25h")  # erase the picker
        out.flush()
    return result


def pick_model(current):
    models = model_details()
    width = max(len(m["name"]) for m in models) if models else 0
    rows = [f"{m['name']:<{width}}  {m['params']:>6}  {m['size'] / 1e9:5.1f} GB"
            + ("  (current)" if m["name"] == current else "")
            + ("" if m["tools"] else "  · no tool support") for m in models]
    names = [m["name"] for m in models]
    i = pick(rows, names.index(current) if current in names else 0,
             "Select a model  (↑/↓ to move, Enter to choose, Esc to cancel)")
    return names[i] if i is not None else None


HISTORY_PATH = STATE_PATH.parent / "history"
HISTORY_LENGTH = 1000


def drop_last_history(only_if_duplicate=False):
    """Remove the newest history entry (approval answers), or just a repeat of the one before it."""
    readline = sys.modules.get("readline")  # only in the plain-prompt fallback; the full-window editor has its own
    if readline is None:
        return
    try:
        n = readline.get_current_history_length()
        if n and (not only_if_duplicate or (n > 1 and readline.get_history_item(n) == readline.get_history_item(n - 1))):
            readline.remove_history_item(n - 1)
    except ValueError:
        pass


def setup_readline():
    """Up/down history saved across sessions, plus Tab-completion of slash commands and model names."""
    try:
        import readline
    except ImportError:
        return
    try:
        readline.read_history_file(HISTORY_PATH)
    except OSError:
        pass
    readline.set_history_length(HISTORY_LENGTH)

    def save_history():
        try:
            HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
            readline.write_history_file(HISTORY_PATH)
        except OSError:
            pass

    atexit.register(save_history)
    cache = {}

    def complete(text, state):
        buf = readline.get_line_buffer()
        if buf.startswith("/model "):
            if "models" not in cache:
                try:
                    cache["models"] = list_models()
                except (urllib.error.URLError, OSError):
                    cache["models"] = []
            options = [m for m in cache["models"] if m.startswith(text)]
        elif buf.startswith("/") and " " not in buf:
            options = [c for c in COMMANDS if c.startswith(text)]
        else:
            options = []
        return options[state] if state < len(options) else None

    readline.set_completer(complete)
    readline.set_completer_delims(" \t\n")
    if "libedit" in (readline.__doc__ or ""):
        readline.parse_and_bind("bind ^I rl_complete")
    else:
        readline.parse_and_bind("tab: complete")


def read_input(screen=None):
    if screen and screen.active:
        return screen.read_line().strip()
    lines = []
    while True:
        line = input(prompt(bold("❯ " if not lines else "… ")))
        drop_last_history(only_if_duplicate=True)  # like a shell, don't store the same line twice in a row
        if line.endswith("\\"):
            lines.append(line[:-1])
            continue
        lines.append(line)
        return "\n".join(lines).strip()


def fmt_tokens(n):
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


def short_path(p):
    home = str(Path.home())
    p = str(p)
    return "~" + p[len(home):] if p == home or p.startswith(home + os.sep) else p


BIG = {  # 3-row block letters for the banner
    "L": ["█   ", "█   ", "▀▀▀▀"], "O": ["█▀▀█", "█  █", "▀▀▀▀"], "C": ["█▀▀▀", "█   ", "▀▀▀▀"],
    "A": ["█▀▀█", "█▀▀█", "▀  ▀"], "D": ["█▀▀▄", "█  █", "▀▀▀ "], "E": ["█▀▀▀", "█▀▀ ", "▀▀▀▀"],
}
PLACEHOLDER = "Ask anything, or type / for commands"
EDITOR_HISTORY_PATH = STATE_PATH.parent / "history.jsonl"


def header_lines(agent, width, height):
    """The pinned banner: big LOCALCODE lettering when there's room, else a single line."""
    info = dim(f"{agent.model} · Ollama · {short_path(agent.tools.root)}")
    if width >= 50 and height >= 24:
        big = [cyan(" ".join(BIG[ch][r] for ch in "LOCALCODE")) for r in range(3)]
        return ["", *big, "", info, instructions_status(agent.instructions), ""]
    return [bold("localcode") + "  " + info, ""]


def hard_wrap(line, width):
    """Split a line (with color codes) into terminal rows exactly as the terminal would wrap it."""
    rows, row, vis, active = [], "", 0, ""
    for tok in re.findall(r"\x1b\[[0-9;]*m|.", line):
        if tok.startswith("\x1b"):
            row += tok
            active = "" if tok in ("\x1b[0m", "\x1b[m") else active + tok
            continue
        if vis == width:
            rows.append(row + "\x1b[0m")
            row, vis = active, 0
        row += tok
        vis += 1
    rows.append(row)
    return rows


class Transcript:
    """Stands in for sys.stdout in full-window mode: passes output through and keeps a copy of the
    conversation as lines, so it can be redrawn after resizes and scrolled back through."""

    MAX_LINES = 5000
    TOKEN = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\r\n|\n|\r|\x1b|[^\x1b\r\n]+")

    def __init__(self, raw):
        self.raw = raw
        self.lines, self.cur = [], ""

    def write(self, s):
        self.raw.write(s)
        for tok in self.TOKEN.findall(s):
            if tok in ("\n", "\r\n"):
                self._newline()
            elif tok == "\r":
                self.cur = ""  # a carriage return means the line is about to be overwritten
            elif tok.startswith("\x1b["):
                if tok.endswith("m"):
                    self.cur += tok  # keep colors; other cursor/erase codes aren't part of the text
            elif tok != "\x1b":
                self.cur += tok
        return len(s)

    def _newline(self):
        self.lines.append(self.cur)
        self.cur = ""
        if len(self.lines) > self.MAX_LINES:
            del self.lines[:len(self.lines) - self.MAX_LINES]

    def ask(self, question):
        """A y/n style prompt inside the conversation: mouse reporting off so wheel scrolls don't type garbage."""
        self.raw.write("\033[?1000l")
        self.write(question)
        self.flush()
        try:
            answer = sys.stdin.readline()
            if not answer:
                raise EOFError
        finally:
            self.raw.write("\033[?1000h")
        answer = answer.rstrip("\n")
        self.cur += answer  # the terminal echoed it; keep the copy in step
        self._newline()
        return answer

    def clear(self):
        self.lines, self.cur = [], ""

    def flush(self):
        self.raw.flush()

    def isatty(self):
        return self.raw.isatty()

    @property
    def encoding(self):
        return self.raw.encoding


def load_editor_history():
    """One JSON string per line. On first run, import the old readline history file."""
    try:
        return [json.loads(l) for l in EDITOR_HISTORY_PATH.read_text().splitlines() if l.strip()][-HISTORY_LENGTH:]
    except (OSError, ValueError):
        pass
    try:
        lines = HISTORY_PATH.read_text().splitlines()
    except OSError:
        return []
    if lines and lines[0] == "_HiStOrY_V2_":  # libedit escapes spaces and backslashes
        lines = [l.replace("\\040", " ").replace("\\\\", "\\") for l in lines[1:]]
    return [l for l in lines if l][-HISTORY_LENGTH:]


class Screen:
    """Full-window layout on the terminal's alternate screen, like vim or htop.

    Rows 1..top hold the pinned banner. The conversation scrolls in a terminal scroll region
    (top+1..bottom). Below it: a rule, the prompt row, a rule, then two rows for the status and
    hint lines; while the / menu is open those rows show suggestions, and the box grows upward if
    it needs more. Everything printed to the conversation is also kept (see Transcript), so it
    can be redrawn on resize and scrolled back through with the mouse wheel or PgUp/PgDn.
    Input is read by a small raw-mode line editor (not readline) so the menu can update per key.
    Falls back to plain inline prompts when the output isn't a terminal or the window is too small.
    """

    FRAME = 5
    MIN_ROWS = 12
    MAX_MENU = 8

    def __init__(self, agent):
        self.agent = agent
        self.active = False
        self.reading = False  # True while read_line() owns the prompt row
        self.extra = 0        # rows the box has grown upward for the menu
        self.offset = 0       # how many rows the view is scrolled back from the newest output
        self.gap = False      # print a blank line before the next echoed message
        self.history = []
        self.models = None    # installed model names, fetched when the /model menu first needs them

    # ---- layout

    def start(self):
        if not (USE_COLOR and sys.stdin.isatty()) or shutil.get_terminal_size().lines < self.MIN_ROWS:
            return
        self.active = True
        self.history = load_editor_history()
        self.raw = sys.stdout
        self.rec = sys.stdout = Transcript(self.raw)
        atexit.register(self.stop)
        try:
            import signal
            signal.signal(signal.SIGWINCH, lambda *_: self.on_resize())
        except (ImportError, AttributeError, ValueError):
            pass
        # Alternate screen (the earlier terminal output stays out of view and comes back on exit),
        # plus mouse reporting so the wheel scrolls the conversation.
        self._w("\033[?1049h\033[?1000h\033[?1006h")
        self.redraw_all()

    def _layout(self):
        size = shutil.get_terminal_size()
        self.w, self.h = size.columns, size.lines
        self.header = header_lines(self.agent, self.w, self.h)
        self.top = len(self.header)
        self.bottom = self.h - self.FRAME - self.extra

    def _w(self, s):
        self.raw.write(s)
        self.raw.flush()

    def draw_header(self):
        self._layout()
        out = "\0337"
        for row, line in enumerate(self.header, 1):
            pad = " " * max(0, (self.w - visible_len(line)) // 2)
            out += f"\033[{row};1H\033[2K" + self.fit(pad + line)
        self._w(out + "\0338")

    def redraw_all(self):
        """Redraw everything: banner, the conversation (from the kept copy) and the input box."""
        self._layout()
        self._w("\033[2J" + f"\033[{self.top + 1};{self.bottom}r")
        self.draw_header()
        self.render_view()
        self.draw_frame()

    def clear_conversation(self):
        self.rec.clear()
        self.offset, self.gap = 0, False
        self.redraw_all()

    def on_resize(self):
        if not self.active:
            return
        self.redraw_all()
        if self.reading:
            self.render()

    def grow(self, n):
        """Make the box n rows taller than normal (for the menu); the conversation view shrinks to fit."""
        n = max(0, min(n, self.h - self.FRAME - self.top - 3))
        if n == self.extra:
            return
        self.extra = n
        self._layout()
        self._w(f"\033[{self.top + 1};{self.bottom}r")
        self.render_view()

    def render_view(self):
        """Draw the conversation rows from the kept copy, scrolled back self.offset rows.

        When showing the newest output (and not at the prompt), leaves the cursor where the
        next output goes, so streaming continues exactly where the copy ends.
        """
        height = self.bottom - self.top
        rows = []
        for line in reversed(self.rec.lines + [self.rec.cur]):
            rows[:0] = hard_wrap(line, self.w)
            if len(rows) >= height + self.offset:
                break
        self.offset = max(0, min(self.offset, len(rows) - height))
        end = len(rows) - self.offset
        shown = rows[max(0, end - height):end]
        out = ""
        for k in range(height):
            out += f"\033[{self.top + 1 + k};1H\033[2K" + (shown[k] + "\033[0m" if k < len(shown) else "")
        if not self.reading and shown:
            out += f"\033[{self.top + len(shown)};{min(self.w, visible_len(shown[-1]) + 1)}H"
        self._w(out)

    def scroll(self, rows):
        self.offset = max(0, self.offset + rows)
        self.render_view()

    def status_text(self):
        a = self.agent
        used = a.ctx_used
        pct = round(100 * used / NUM_CTX) if NUM_CTX else 0
        color = green if pct < 50 else yellow if pct < 80 else red
        parts = [a.model, short_path(a.tools.root),
                 "Context: " + color(f"{fmt_tokens(used)} / {fmt_tokens(NUM_CTX)} ({pct}%)") + DIM_ON]
        if a.last_speed:
            parts.append(f"{a.last_speed:.1f} tok/s")
        return dim("  " + "  |  ".join(parts))

    def hint_text(self):
        if self.offset:
            return "  " + cyan(f"↑ scrolled back {self.offset} lines") + dim(" · scroll down or PgDn to return")
        if self.agent.tools.auto_approve:
            return "  " + yellow("⏵⏵ auto-approve on") + dim(" (/yolo to turn off)")
        return dim("  ? /help for commands · /model to switch models · scroll or PgUp to look back")

    def fit(self, line):
        """Cut a line to the window width; a line that wraps would spill into the rows below it."""
        return hard_wrap(line, self.w)[0] + "\033[0m"

    def draw_frame(self, save=True, below=None):
        """Draw the rules and the rows under the prompt (status + hint, or the menu rows passed in)."""
        rule = dim("─" * self.w)
        row = self.bottom + 1
        below = below if below is not None else [self.status_text(), self.hint_text()]
        out = ("\0337" if save else "") + f"\033[{row};1H\033[2K" + rule + f"\033[{row + 2};1H\033[2K" + rule
        for i in range(2 + self.extra):
            out += f"\033[{row + 3 + i};1H\033[2K" + (self.fit(below[i]) if i < len(below) else "")
        self._w(out + ("\0338" if save else ""))

    # ---- the line editor

    def menu_items(self):
        """Suggestions for what's typed so far: slash commands, or model names after '/model '."""
        if self.menu_closed or "\n" in self.text():
            return []
        t = self.text()
        if t.startswith("/model "):
            if self.models is None:
                try:
                    self.models = list_models()
                except (urllib.error.URLError, OSError):
                    self.models = []
            q = t[len("/model "):]
            return [("/model " + m, m, "current" if m == self.agent.model else "")
                    for m in self.models if m.startswith(q)][:self.MAX_MENU]
        if t.startswith("/") and " " not in t:
            return [(c, c, d) for c, d in COMMANDS.items() if c.startswith(t)][:self.MAX_MENU]
        return []

    def text(self):
        return "".join(self.buf)

    def render(self):
        items = self.menu_items()
        self.sel = min(self.sel, max(0, len(items) - 1))
        self.grow(max(0, len(items) - 2))
        if items:
            width = max(len(label) for _, label, _ in items) + 3
            below = [(cyan("  ❯ " + label.ljust(width)) if i == self.sel else "    " + label.ljust(width)) + dim(desc)
                     for i, (_, label, desc) in enumerate(items)]
        else:
            below = None
        self.draw_frame(save=False, below=below)
        # The prompt row: scroll horizontally so the cursor stays visible; show newlines as ↵.
        shown = self.text().replace("\n", "↵")
        avail = max(10, self.w - 3)
        start = max(0, self.pos - avail)
        row = self.bottom + 2
        if shown:
            line = cyan("❯ ") + shown[start:start + avail]
        else:
            line = cyan("❯ ") + dim(PLACEHOLDER)
        self._w(f"\033[{row};1H\033[2K" + line + f"\033[{row};{3 + self.pos - start}H")
        return items

    def read_line(self):
        """Read one entry in the pinned prompt row, then echo it into the transcript."""
        import codecs
        import select
        import termios

        fd = sys.stdin.fileno()
        saved_tty = termios.tcgetattr(fd)
        raw = termios.tcgetattr(fd)
        raw[3] &= ~(termios.ICANON | termios.ECHO | termios.ISIG | termios.IEXTEN)
        raw[0] &= ~(termios.IXON | termios.ICRNL)
        raw[6][termios.VMIN], raw[6][termios.VTIME] = 1, 0
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

        self.buf, self.pos, self.sel, self.menu_closed, self.in_paste = [], 0, 0, False, False
        hist_i, draft = len(self.history), ""
        self._w("\033[?2004h")  # bracketed paste, so multi-line pastes arrive as one piece
        self.reading = True
        termios.tcsetattr(fd, termios.TCSADRAIN, raw)
        result = None
        pending = ""
        try:
            items = self.render()
            while result is None:
                data = pending + decoder.decode(os.read(fd, 4096))
                pending = ""
                if data == "\x1b" and select.select([fd], [], [], 0.03)[0]:
                    pending = data  # the rest of an escape sequence is on its way
                    continue
                i = 0
                while i < len(data) and result is None:
                    key, i = self._next_key(data, i)
                    if key in self.SCROLL_KEYS:
                        self.scroll(self.SCROLL_KEYS[key] * (max(1, self.bottom - self.top - 2) if key.startswith("page") else 1))
                        items = self.render()
                        continue
                    result = self._handle_key(key, items)
                    if result is None:
                        if key in ("up", "down") and not items:
                            hist_i, draft = self._history_step(key, hist_i, draft)
                        items = self.render()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, saved_tty)
            self.reading = False
            self.offset = 0
            self.grow(0)
            self.draw_frame(save=False)
            self._w(f"\033[{self.bottom + 2};1H\033[2K" + "\033[?2004l")
            self.render_view()  # back to the newest output, cursor where the next output goes
        if isinstance(result, BaseException):
            raise result
        if result and (not self.history or self.history[-1] != result):
            self.history.append(result)
        if result.strip():
            if self.gap:
                print()
            lines = result.strip().split("\n")
            print(cyan("❯ ") + bold(lines[0]))
            for extra in lines[1:]:
                print("  " + bold(extra))
            self.gap = True
        return result

    def _next_key(self, data, i):
        """Split raw input into keys: printable text, control keys, escape sequences and pastes."""
        if self.in_paste or data.startswith("\x1b[200~", i):
            # Bracketed paste: everything up to the end marker is text, even if it spans several reads.
            start = i if self.in_paste else i + 6
            end = data.find("\x1b[201~", start)
            self.in_paste = end == -1
            if self.in_paste:
                return ("paste", data[start:]), len(data)
            return ("paste", data[start:end]), end + 6
        ch = data[i]
        if ch != "\x1b":
            return ch, i + 1
        if i + 1 >= len(data):
            return "esc", i + 1
        nxt = data[i + 1]
        if nxt in "[O":
            j = i + 2
            while j < len(data) and not ("@" <= data[j] <= "~"):
                j += 1
            seq = data[i + 2:j + 1]
            names = {"A": "up", "B": "down", "C": "right", "D": "left", "H": "home", "F": "end",
                     "1~": "home", "7~": "home", "4~": "end", "8~": "end", "3~": "delete",
                     "1;3D": "wordleft", "1;5D": "wordleft", "1;3C": "wordright", "1;5C": "wordright",
                     "5~": "pageup", "6~": "pagedown", "1;2A": "lineup", "1;2B": "linedown"}
            if seq.startswith("<"):  # SGR mouse report: 64/65 are wheel up/down, the rest is ignored
                button = seq[1:].split(";")[0]
                return {"64": "wheelup", "65": "wheeldown"}.get(button, "unknown"), j + 1
            return names.get(seq, "unknown"), j + 1
        alt = {"b": "wordleft", "f": "wordright", "\x7f": "wordback"}
        return alt.get(nxt, "unknown"), i + 2

    def _handle_key(self, key, items):
        """Apply one key to the buffer. Returns the finished text, an exception to raise, or None."""
        buf, pos, before = self.buf, self.pos, self.text()
        if isinstance(key, tuple):  # pasted text goes in as-is, newlines included
            text = key[1].replace("\r\n", "\n").replace("\r", "\n")
            buf[pos:pos] = list(text)
            self.pos += len(text)
        elif key in ("\r", "\n"):
            if items:
                return items[self.sel][0]
            if pos and buf[pos - 1] == "\\":  # backslash + Enter: new line
                buf[pos - 1] = "\n"
                return None
            return self.text()
        elif key == "\t":
            if items:
                full = items[self.sel][0]
                full += " " if full == "/model" else ""
                self.buf, self.pos = list(full), len(full)
        elif key in ("up", "down") and items:
            self.sel = (self.sel + (1 if key == "down" else -1)) % len(items)
            return None
        elif key == "esc":
            self.menu_closed = True
            return None
        elif key == "\x03":  # Ctrl-C: clear the line, or quit if it's already empty
            if not buf:
                return KeyboardInterrupt()
            self.buf, self.pos = [], 0
        elif key == "\x04":  # Ctrl-D: quit on an empty line, else delete forward
            if not buf:
                return EOFError()
            if pos < len(buf):
                del buf[pos]
        elif key in ("\x7f", "\x08"):
            if pos:
                del buf[pos - 1]
                self.pos -= 1
        elif key == "delete":
            if pos < len(buf):
                del buf[pos]
        elif key in ("left", "\x02"):
            self.pos = max(0, pos - 1)
        elif key in ("right", "\x06"):
            self.pos = min(len(buf), pos + 1)
        elif key in ("home", "\x01"):
            self.pos = 0
        elif key in ("end", "\x05"):
            self.pos = len(buf)
        elif key == "\x15":  # Ctrl-U
            del buf[:pos]
            self.pos = 0
        elif key == "\x0b":  # Ctrl-K
            del buf[pos:]
        elif key in ("\x17", "wordback"):
            j = pos
            while j and buf[j - 1] == " ":
                j -= 1
            while j and buf[j - 1] != " ":
                j -= 1
            del buf[j:pos]
            self.pos = j
        elif key == "wordleft":
            j = pos
            while j and buf[j - 1] == " ":
                j -= 1
            while j and buf[j - 1] != " ":
                j -= 1
            self.pos = j
        elif key == "wordright":
            j = pos
            while j < len(buf) and buf[j] == " ":
                j += 1
            while j < len(buf) and buf[j] != " ":
                j += 1
            self.pos = j
        elif len(key) == 1 and key.isprintable():
            buf.insert(pos, key)
            self.pos += 1
        else:
            return None
        if self.text() != before:
            self.menu_closed = False  # typing again reopens a menu closed with Esc
        return None

    def _history_step(self, key, i, draft):
        """Up/Down through past entries, keeping what was being typed as the newest one."""
        if i == len(self.history):
            draft = self.text()
        i = max(0, i - 1) if key == "up" else min(len(self.history), i + 1)
        text = self.history[i] if i < len(self.history) else draft
        self.buf, self.pos = list(text), len(text)
        self.menu_closed = True  # a recalled "/command" shouldn't pop the menu open and take over ↑/↓
        return i, draft

    SCROLL_KEYS = {"wheelup": 3, "wheeldown": -3, "lineup": 1, "linedown": -1, "pageup": 1, "pagedown": -1}

    def stop(self):
        if not self.active:
            return
        self.active = False
        sys.stdout = self.raw
        try:
            EDITOR_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
            EDITOR_HISTORY_PATH.write_text("".join(json.dumps(h) + "\n" for h in self.history[-HISTORY_LENGTH:]))
        except OSError:
            pass
        # Mouse and paste modes off, scroll region released, and back to the screen from before.
        self._w("\033[?1006l\033[?1000l\033[?2004l\033[r\033[?1049l")


def main():
    ap = argparse.ArgumentParser(description="Coding agent for local Ollama models")
    ap.add_argument("prompt", nargs="*", help="run one task and exit")
    ap.add_argument("-m", "--model", help="model to use (default: last one picked, else " + DEFAULT_MODEL + ")")
    ap.add_argument("-C", "--cwd", default=".", help="project directory")
    ap.add_argument("--yolo", action="store_true", help="auto-approve writes and shell commands")
    args = ap.parse_args()

    try:
        installed = list_models()
    except (urllib.error.URLError, OSError):
        sys.exit(red(f"Can't reach Ollama at {OLLAMA_HOST}. Start it with `ollama serve`."))
    if not installed:
        sys.exit(red(f"No models installed. Try `ollama pull {DEFAULT_MODEL}`."))
    # -m flag, then LOCALCODE_MODEL, then the model picked last time, then the built-in default
    model = args.model or os.environ.get("LOCALCODE_MODEL") or load_state().get("model") or DEFAULT_MODEL
    if model not in installed:
        if args.prompt or not sys.stdin.isatty():
            sys.exit(red(f"Model {model} isn't installed. Installed: {', '.join(installed)}"))
        print(yellow(f"Model {model} isn't installed."))
        model = pick_model(None)
        if not model:
            return
        save_state(model=model)

    tools = Tools(args.cwd, auto_approve=args.yolo)
    agent = Agent(model, tools)

    if args.prompt:
        agent.run(" ".join(args.prompt))
        return

    screen = Screen(agent)
    screen.start()
    agent.show_stats = not screen.active  # the status line shows them instead
    if not screen.active:
        setup_readline()  # plain-prompt fallback: readline's editing and history
        print(bold("localcode") + dim(f"  {agent.model}  ·  {tools.root}  ·  /help for commands"))
        print(instructions_status(agent.instructions))
    while True:
        try:
            text = read_input(screen)
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not text:
            continue
        if screen.active:
            print()
        if text.startswith("/"):
            cmd, _, rest = text.partition(" ")
            if cmd in ("/exit", "/quit"):
                return
            elif cmd == "/help":
                print(HELP)
            elif cmd == "/clear":
                agent.reset()
                if screen.active:
                    screen.clear_conversation()
                else:
                    print(dim("Conversation cleared. ") + instructions_status(agent.instructions))
            elif cmd in ("/model", "/models"):
                name = rest.strip()
                if not name:
                    name = pick_model(agent.model)
                elif name not in list_models():
                    print(red(f"{name} isn't installed.") + dim(" Run /model to pick from the installed ones."))
                    continue
                if name and name != agent.model:
                    agent.model = name
                    save_state(model=name)
                    if screen.active:
                        screen.draw_header()
                    print(dim("Switched to ") + bold(name) + dim(". The conversation carries over; /clear to start fresh."))
                else:
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
