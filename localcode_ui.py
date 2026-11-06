"""Terminal rendering, output helpers, and interactive picker for localcode."""

import difflib
import os
import re
import shutil
import sys
import threading
import time
import unicodedata


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


class TerminalWriter:
    """The terminal stream, tracking when the main thread is in the middle of a write.

    A resize redraw runs in the SIGWINCH handler, which can interrupt a write that is waiting on a slow
    terminal; writing again from there raises "reentrant call". The redraw waits until the write finishes.
    """

    def __init__(self, stream):
        self.stream, self.depth, self.deferred = stream, 0, None

    def write(self, s):
        return self._call(self.stream.write, s)

    def flush(self):
        return self._call(self.stream.flush)

    def _call(self, fn, *args):
        if threading.current_thread() is not threading.main_thread():
            return fn(*args)  # signal handlers only run on the main thread
        self.depth += 1
        try:
            return fn(*args)
        finally:
            self.depth -= 1
            if not self.depth and self.deferred:
                run, self.deferred = self.deferred, None
                run()

    def __getattr__(self, name):
        return getattr(self.stream, name)


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
            self._write("\r" + dim(f"{self.FRAMES[i % len(self.FRAMES)]} {self.label}… {elapsed}s") + "\033[K")
            i += 1
        self._write("\r\033[K")

    def _write(self, text):
        screen = getattr(sys.stdout, "screen", None)
        if screen:
            screen.write_transient(text)
        else:
            out = term_out()
            out.write(text)
            out.flush()

    def stop(self):
        if self.thread and not self.done.is_set():
            self.done.set()
            self.thread.join()


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
BOLD, NOT_BOLD, CODE, NOT_CODE = "\033[1m", "\033[22m", "\033[36m", "\033[39m"


def visible_len(s):
    return sum(char_width(ch) for ch in ANSI_RE.sub("", s))


def char_width(ch):
    """Terminal cells for ordinary text, including CJK and combining accents."""
    if unicodedata.category(ch) in ("Mn", "Me", "Cf", "Cc"):
        return 0
    return 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1


def clip_ansi(text, width):
    """Clip without wrapping, preserving complete color escapes."""
    out, used = [], 0
    for tok in re.findall(r"\x1b\[[0-9;]*m|.", text):
        cells = 0 if tok.startswith("\x1b") else char_width(tok)
        if used + cells > max(0, width):
            break
        out.append(tok)
        used += cells
    return "".join(out)


def abbreviate(text, width):
    return text if visible_len(text) <= width else clip_ansi(text, max(0, width - 1)) + ("…" if width else "")


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
            wvis += 0 if tok.startswith("\x1b") else char_width(tok)
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


def picker_matches(rows, query):
    words = query.casefold().split()
    return [i for i, row in enumerate(rows)
            if all(word in ANSI_RE.sub("", row).casefold() for word in words)]


def pick(rows, start=0, title="", nested=False):
    """Searchable, scrollable chooser. Returns the original row index, never a filtered index."""
    if not rows:
        return None
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        for i, row in enumerate(rows, 1):
            print(f"  {i}. {row}")
        try:
            answer = input("Number (blank to cancel): ").strip()
        except EOFError:
            return None
        return int(answer) - 1 if answer.isdigit() and 1 <= int(answer) <= len(rows) else None

    import codecs
    import select
    import termios
    import tty

    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    out = term_out()
    query, selected, matches, pending = "", start if 0 <= start < len(rows) else 0, list(range(len(rows))), ""
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def draw():
        width, height = shutil.get_terminal_size()
        capacity = max(1, height - 6)
        first = max(0, min(selected - capacity // 2, len(matches) - capacity))
        lines = [bold(title), cyan("Search ❯ ") + query, dim("─" * width)]
        for position in range(first, min(len(matches), first + capacity)):
            text = rows[matches[position]].replace("\n", " ↵ ")
            lines.append((cyan("❯ ") + bold(text)) if position == selected else "  " + text)
        if not matches:
            lines.append(dim("No matches. Backspace to broaden your search."))
        footer = f"{selected + 1 if matches else 0}/{len(matches)} · ↑↓ choose · Enter select · Esc cancel"
        out.write("\033[H\033[2J" + "\r\n".join(clip_ansi(line, max(1, width - 1)) + "\033[0m"
                                                     for line in lines[:max(1, height - 1)]))
        out.write(f"\033[{max(1, height)};1H" + clip_ansi(dim(footer), max(1, width - 1)))
        out.flush()

    try:
        if not nested:
            out.write("\033[?1049h")
        out.write("\033[r\033[?25l\033[?1000l")
        tty.setraw(fd)
        draw()
        while True:
            if not select.select([fd], [], [], 0.2)[0]:
                draw()  # also adapt if the window changed size
                continue
            chunk = os.read(fd, 4096)
            if not chunk:
                return None
            data = pending + decoder.decode(chunk)
            pending = ""
            if data == "\x1b" and select.select([fd], [], [], 0.05)[0]:
                pending = data
                continue
            index = 0
            while index < len(data):
                ch = data[index]
                if ch == "\x1b" and index + 1 < len(data) and data[index + 1] == "[":
                    end = index + 2
                    while end < len(data) and not ("@" <= data[end] <= "~"):
                        end += 1
                    if end == len(data):
                        pending = data[index:]
                        break
                    key = data[index + 2:end + 1]
                    selected += {"A": -1, "B": 1, "5~": -max(1, shutil.get_terminal_size().lines - 6),
                                 "6~": max(1, shutil.get_terminal_size().lines - 6)}.get(key, 0)
                    index = end + 1
                else:
                    index += 1
                    if ch in ("\x1b", "\x03", "\x04"):
                        return None
                    if ch in ("\r", "\n"):
                        return matches[selected] if matches else None
                    if ch in ("\x7f", "\x08"):
                        query = query[:-1]
                    elif ch == "\x15":
                        query = ""
                    elif ch.isprintable():
                        query += ch
                    else:
                        continue
                    matches, selected = picker_matches(rows, query), 0
                selected = max(0, min(selected, len(matches) - 1))
            draw()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)
        out.write("\033[0m\033[?25h")
        if nested:
            out.write("\033[?1000h")
        else:
            out.write("\033[?1049l")
        out.flush()

