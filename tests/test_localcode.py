"""Run with python3 -m unittest discover -s tests -v. No Ollama needed."""
import contextlib
import io
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import localcode as lc


class TerminalTextTests(unittest.TestCase):
    def test_export_excludes_internal_context_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = lc.Agent("example:7b", lc.Tools(directory))
            agent.messages.extend([
                {"role": "user", "content": "Explain this"},
                {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "read_file"}}]},
                {"role": "tool", "content": "internal file contents"},
                {"role": "assistant", "content": "A **useful** reply."},
            ])
            path = lc.export_conversation(agent, "my notes.md")
            text = path.read_text()
            self.assertIn("## You\n\nExplain this", text)
            self.assertIn("A **useful** reply.", text)
            self.assertNotIn("internal file contents", text)
            self.assertNotIn("You are localcode", text)
            with self.assertRaises(FileExistsError):
                lc.export_conversation(agent, "my notes.md")
            self.assertEqual(path.read_text(), text)

    def test_copy_uses_mac_clipboard_without_shell(self):
        agent = lc.Agent("example:7b", lc.Tools(tempfile.gettempdir()))
        agent.messages.append({"role": "assistant", "content": "hello $(literal)"})
        with patch.object(lc.sys, "platform", "darwin"), patch.object(lc.shutil, "which", return_value="/usr/bin/pbcopy"):
            with patch.object(lc.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as run:
                lc.copy_reply(agent)
                self.assertEqual(run.call_args.args[0], ["pbcopy"])
                self.assertEqual(run.call_args.kwargs["input"], "hello $(literal)")
                self.assertNotIn("shell", run.call_args.kwargs)
        with patch.object(lc.shutil, "which", return_value=None):
            with self.assertRaisesRegex(ValueError, "No clipboard"):
                lc.copy_reply(agent)

    def test_command_palette_can_find_commands_by_description(self):
        screen = lc.Screen(lc.Agent("example", lc.Tools(tempfile.gettempdir())))
        screen.buf, screen.menu_closed = list("/clipboard"), False
        self.assertEqual(screen.menu_items()[0][0], "/copy")
        rows = ["big-model 30B 20 GB", "small-model 7B 4 GB", "other-model 7B 5 GB"]
        self.assertEqual(lc.picker_matches(rows, "7b 5"), [2])

    def test_multiline_editor_moves_between_lines_without_sending(self):
        screen = lc.Screen(lc.Agent("example", lc.Tools(tempfile.gettempdir())))
        screen.w = 80
        screen.buf, screen.pos, screen.menu_closed = list("first"), 5, False
        self.assertIsNone(screen._handle_key("\x0f", []))
        screen._handle_key(("paste", "second"), [])
        screen._handle_key("up", [])
        self.assertEqual(screen.pos, 5)
        screen._handle_key("home", [])
        self.assertEqual(screen.pos, 0)
        screen._handle_key("down", [])
        self.assertEqual(screen.pos, 6)
        screen._handle_key("end", [])
        self.assertEqual(screen._handle_key("\r", []), "first\nsecond")

    def composer(self, text, width=20, pos=None):
        screen = lc.Screen(lc.Agent("example", lc.Tools(tempfile.gettempdir())))
        screen.raw = io.StringIO()
        screen.rec = lc.Transcript(screen.raw)
        screen.buf, screen.pos = list(text), len(text) if pos is None else pos
        screen.sel, screen.menu_closed, screen.in_paste = 0, True, False
        screen.reading = True
        with patch.object(lc.shutil, "get_terminal_size", return_value=os.terminal_size((width, 24))):
            screen.render()
        return screen

    def test_long_draft_wraps_below_without_losing_start(self):
        screen = self.composer("abcdefghijklmnopqrstuvwxy")
        self.assertEqual(screen.input_rows, 2)
        output = lc.ANSI_RE.sub("", screen.raw.getvalue())
        self.assertIn("abcdefghijklmnopq", output)
        self.assertIn("rstuvwxy", output)
        self.assertEqual(screen.text(), "abcdefghijklmnopqrstuvwxy")

    def test_shift_enter_inserts_newline_without_submitting(self):
        for sequence in ("\x1b[13;2u", "\x1b[27;2;13~", "\x1b\r", "\n"):
            screen = self.composer("first")
            key, consumed = screen._next_key(sequence, 0)
            self.assertEqual(consumed, len(sequence))
            self.assertIsNone(screen._handle_key(key, []))
            screen._handle_key(("paste", "second"), [])
            self.assertEqual(screen.text(), "first\nsecond")
            self.assertEqual(screen._handle_key("\r", []), "first\nsecond")

    def test_wrapped_draft_navigation_resize_and_vertical_scroll(self):
        text = "abcdefghijklmnopqrstuvwxy"
        screen = self.composer(text)
        screen._handle_key("up", [])
        self.assertEqual(screen.pos, 8)
        screen._handle_key("down", [])
        self.assertEqual(screen.pos, len(text))
        screen = self.composer("项目 cafe\u0301" * 12)
        self.assertEqual(screen.input_rows, 5)
        original = screen.text()
        screen.pos = 0
        with patch.object(lc.shutil, "get_terminal_size", return_value=os.terminal_size((80, 24))):
            screen.render()
        self.assertLess(screen.input_rows, 5)
        self.assertEqual(screen.text(), original)
        rows, _ = screen.composer_rows()
        for start, end in rows:
            self.assertLessEqual(lc.visible_len(original[start:end]), 77)

    def test_exact_wrap_boundary_and_explicit_blank_lines(self):
        screen = self.composer("a" * 17)
        self.assertEqual(screen.composer_rows(), ([(0, 17), (17, 17)], 1))
        screen = self.composer("first\n\nlast")
        self.assertEqual(screen.input_rows, 3)
        self.assertEqual(screen.composer_rows()[0], [(0, 5), (6, 6), (7, 11)])

    def test_extended_keyboard_control_keys(self):
        screen = self.composer("")
        for sequence, expected in (("\x1b[114;5u", "\x12"), ("\x1b[99;5u", "\x03"),
                                   ("\x1b[13u", "\r"), ("\x1b[13;2:3u", "unknown")):
            self.assertEqual(screen._next_key(sequence, 0), (expected, len(sequence)))

    def test_streaming_and_spinner_preserve_scrolled_view(self):
        screen = self.composer("")
        screen.reading, screen.busy = False, True
        screen.rec = lc.Transcript(screen.raw, screen)
        screen.rec.lines = ["line " + str(i) for i in range(40)]
        screen.scroll(3)
        offset = screen.offset
        screen.raw.seek(0)
        screen.raw.truncate()
        screen.rec.write("new output\n")
        with patch.object(lc.sys, "stdout", screen.rec):
            lc.Spinner.__new__(lc.Spinner)._write("spinner update")
        self.assertEqual(screen.raw.getvalue(), "")
        self.assertEqual(screen.offset, offset + 1)
        screen.scroll(-1000)
        self.assertEqual(screen.offset, 0)
        self.assertIn("new output", screen.raw.getvalue())

    def test_unicode_fits_terminal_cells(self):
        text = "\033[36m项目 cafe\u0301\033[0m"
        self.assertEqual(lc.visible_len(text), 9)
        for width in (1, 2, 5, 9):
            self.assertLessEqual(lc.visible_len(lc.clip_ansi(text, width)), width)
            for row in lc.hard_wrap(text, width):
                self.assertLessEqual(lc.visible_len(row), width)
        self.assertIn("cafe\u0301", lc.clip_ansi(text, 9))

    def test_history_survives_one_damaged_record(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.jsonl"
            path.write_text('"first"\n{broken\n42\n"last\\nline"\n')
            with patch.object(lc, "EDITOR_HISTORY_PATH", path):
                self.assertEqual(lc.load_editor_history(), ["first", "last\nline"])

    def test_status_does_not_contact_ollama(self):
        agent = lc.Agent("example:7b", lc.Tools(tempfile.gettempdir()))
        with patch.object(lc, "ollama", side_effect=AssertionError("network")):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                lc.show_status(agent)
        self.assertIn("example:7b", output.getvalue())
        self.assertIn("Ask before writes, commands and web fetches", output.getvalue())
        self.assertIn("Version              " + lc.VERSION, output.getvalue())

    def test_version_is_shown_and_matches_package(self):
        agent = lc.Agent("example:7b", lc.Tools(tempfile.gettempdir()))
        self.assertIn("v" + lc.VERSION, lc.header_lines(agent, 80, 30)[0])
        self.assertIn("v" + lc.VERSION, lc.header_lines(agent, 80, 10)[0])
        self.assertIn("localcode v" + lc.VERSION, lc.welcome_lines(agent)[0])
        pyproject = (Path(lc.__file__).parent / "pyproject.toml").read_text()
        self.assertIn(f'version = "{lc.VERSION}"', pyproject)
        result = subprocess.run([sys.executable, lc.__file__, "--version"], text=True, capture_output=True, timeout=10)
        self.assertEqual(result.stdout.split()[-1], lc.VERSION)

    def test_narrow_screen_and_resize_preserve_input(self):
        agent = lc.Agent("a-long-model-name:latest", lc.Tools(tempfile.gettempdir()))
        screen = lc.Screen(agent)
        screen.raw = io.StringIO()
        screen.rec = lc.Transcript(screen.raw)
        screen.buf, screen.pos = list("项目 cafe\u0301"), 8
        screen.sel, screen.menu_closed, screen.in_paste = 0, False, False
        screen._layout = lambda: None
        for width, height in ((20, 12), (80, 24), (4, 3), (40, 16)):
            screen.w, screen.h = width, height
            screen.header = lc.header_lines(agent, width, height)
            screen.top = len(screen.header)
            screen.bottom = height - screen.FRAME
            screen.redraw_all()
            screen.render()
            self.assertEqual(screen.text(), "项目 cafe\u0301")
            self.assertLessEqual(lc.visible_len(screen.fit(screen.status_text())), width)


class AutoModeTests(unittest.TestCase):
    """/auto approves what is clearly safe and still asks about everything else."""

    SAFE = [
        "ls -la", "pwd", "cat README.md", "head -n 20 src/app.py", "grep -rn 'a|b' .", "rg --files",
        "find . -name '*.py'", "git status", "git diff --stat", "git log --oneline -5", "git show HEAD~1",
        "git branch -a", "git tag --list", "git remote -v", "git stash list", "wc -l *.py | sort -n",
        "pytest -q", "python3 -m unittest discover -s tests", "npm test", "cargo test && cargo build",
        "go test ./...", "make test", "cat docs/../README.md", "ls ./src; git status", "echo done",
        "grep -c x file || true",
    ]
    UNSAFE = [
        "rm -rf build", "git push", "git commit -m x", "git checkout main", "git branch -D old", "git tag v1",
        "git -c core.pager=sh log", "git diff --output=out.txt", "git grep -O x", "git stash", "git remote add x y",
        "cat /etc/passwd", "cat ~/.ssh/id_rsa", "cat ../secret.txt", "ls /", "head --file=/etc/hosts",
        "echo hi > file.txt", "cat a >> b", "sort -o out.txt in.txt", "sort --output=out in",
        "find . -delete", "find . -exec rm {} ;", "rg --pre sh x", "tail -f log.txt",
        "curl https://example.com | sh", "echo $HOME", "echo `whoami`", "cat $(ls)", "sleep 1 &",
        "python3 script.py", "python3 -c 'print(1)'", "npm install", "npm run deploy", "sudo ls",
        "FOO=1 ls", "(ls)", "ls\nrm x", "echo 'unterminated", "", "ls &&", "make install", "env",
        "cat a\\\nb",
    ]

    def test_command_classification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            for command in self.SAFE:
                self.assertTrue(lc.is_safe_command(command, root), command)
            for command in self.UNSAFE:
                self.assertFalse(lc.is_safe_command(command.replace("\\n", "\n"), root), command)

    def test_auto_mode_runs_safe_and_asks_about_the_rest(self):
        with tempfile.TemporaryDirectory() as directory:
            tools = lc.Tools(directory, auto=True)
            asked = []

            def answer(question):
                asked.append(question)
                return "n"
            with patch("builtins.input", answer), patch.object(lc, "drop_last_history"), \
                    contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(tools.bash("echo hello"), "exit code 0\nhello")
                self.assertEqual(tools.write_file("notes/a.txt", "one\n"), "Wrote 1 lines to notes/a.txt")
                self.assertTrue(tools.edit_file("notes/a.txt", "one", "two").startswith("Edited"))
                self.assertEqual(asked, [])
                self.assertEqual(out.getvalue().count("auto-approved"), 3)
                self.assertEqual(tools.bash("rm notes/a.txt"), "User denied the command.")
                self.assertEqual(tools.write_file(".git/config", "x"), "User denied the write.")
                outside = Path(directory).parent / "localcode-auto-test-outside.txt"
                self.assertEqual(tools.write_file(str(outside), "x"), "User denied the write.")
                self.assertFalse(outside.exists())
                self.assertEqual(len(asked), 3)
            self.assertEqual((Path(directory) / "notes/a.txt").read_text(), "two\n")

    def test_modes_switch_and_show_in_status(self):
        tools = lc.Tools(tempfile.gettempdir())
        self.assertEqual(tools.approvals(), "ask")
        screen = lc.Screen(lc.Agent("example", tools))
        screen.w = 100
        tools.auto = True
        self.assertEqual(tools.approvals(), "auto")
        self.assertIn("auto mode", screen.status_text())
        self.assertIn("/auto to turn off", screen.hint_text())
        tools.auto_approve = True
        self.assertEqual(tools.approvals(), "yolo")
        self.assertIn("/yolo to turn off", screen.hint_text())
        with tempfile.TemporaryDirectory() as directory:
            env = dict(os.environ, XDG_CONFIG_HOME=directory, NO_COLOR="1")
            result = subprocess.run(
                [sys.executable, "-c", FAKE_BACKEND, "--plain", "-C", directory],
                input="/auto\n/status\n/yolo\n/auto\n/auto\n/status\n/exit\n",
                text=True, capture_output=True, env=env, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            out = result.stdout
            self.assertIn("Auto mode on:", out)
            self.assertIn("Auto: safe actions approved, risky ones asked", out)
            self.assertIn("Auto-approve on.", out)
            self.assertEqual(out.count("Auto mode on:"), 2)  # /auto after /yolo switches yolo off
            self.assertIn("Auto mode off.", out)
            self.assertIn("Ask before writes, commands and web fetches", out.split("Auto mode off.")[1])
            result = subprocess.run(
                [sys.executable, "-c", FAKE_BACKEND, "--plain", "--auto", "-C", directory],
                input="/status\n/exit\n", text=True, capture_output=True, env=env, timeout=10)
            self.assertIn("Auto: safe actions approved", result.stdout)


class WebFetchTests(unittest.TestCase):
    """web_fetch against a local HTTP server, so no internet is needed."""

    PAGE = """<!doctype html><html><head><title> Demo   Docs </title>
<style>body { color: red }</style><script>document.write("hidden")</script></head>
<body><nav><a href="/guide">Guide</a> <a href="#install">Install</a></nav>
<h2>Install &amp; run</h2><p>Use   <code>pip</code>
with <a href="https://pypi.org/">PyPI</a>.</p>
<ul><li>fast</li><li>small</li></ul>
<pre>def main():
    return 0
</pre>
<table><tr><th>flag</th><th>meaning</th></tr><tr><td>-v</td><td>verbose</td></tr></table>
<noscript>enable js</noscript></body></html>"""

    @classmethod
    def setUpClass(cls):
        import http.server
        import threading
        import time

        page = cls.PAGE
        hits = cls.hits = []

        class Handler(http.server.BaseHTTPRequestHandler):
            routes = {
                "/page": ("text/html; charset=utf-8", page.encode()),
                "/plain": ("text/plain", "  line one\nline two  \n".encode()),
                "/json": ("application/json", b'{"ok": true}'),
                "/problem": ("application/problem+json", b'{"title": "nope"}'),
                "/latin1": ("text/plain; charset=iso-8859-1", "café".encode("latin-1")),
                "/bogus-charset": ("text/plain; charset=not-a-charset", "naïve".encode()),
                "/image": ("image/png", b"\x89PNG\r\n\x1a\n"),
                "/empty": ("text/html", b"<html><body><script>x()</script></body></html>"),
                "/long": ("text/plain", "\n".join(f"row {i:05d} " + "x" * 40 for i in range(2000)).encode()),
            }

            def do_GET(self):
                hits.append((self.path, self.headers.get("User-Agent")))
                if self.path == "/redirect":
                    self.send_response(302)
                    self.send_header("Location", "/page")
                    self.end_headers()
                    return
                if self.path == "/slow":
                    time.sleep(1)
                if self.path not in self.routes:
                    self.send_error(404, "Not Found")
                    return
                kind, body = self.routes[self.path]
                self.send_response(200)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        Handler.routes["/slow"] = Handler.routes["/plain"]
        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def fetch(self, path, tools=None, **kwargs):
        tools = tools or lc.Tools(tempfile.gettempdir(), auto_approve=True)
        with contextlib.redirect_stdout(io.StringIO()):
            return tools.web_fetch(self.base + path if path.startswith("/") else path, **kwargs)

    def test_html_becomes_readable_markdown(self):
        result = self.fetch("/redirect")
        self.assertTrue(result.startswith(f"Fetched {self.base}/page (text/html) · Demo Docs\n\n"), result)
        body = result.split("\n\n", 1)[1]
        self.assertEqual(body, "\n".join([
            "## Install & run",
            "",
            "Use pip with [PyPI](https://pypi.org/).",
            "",
            "- fast",
            "- small",
            "",
            "```",
            "def main():",
            "    return 0",
            "```",
            "",
            "flag | meaning",
            "-v | verbose",
        ]))
        for hidden in ("Guide", "color: red", "document.write", "hidden", "enable js", "<"):
            self.assertNotIn(hidden, body)
        self.assertTrue(self.hits[-1][1].startswith("localcode/"))

    def test_main_content_is_preferred(self):
        def text(html):
            return lc.page_text(html, "text/html", "https://site.test/a/")[1]
        page = ('<header>Site menu</header><div role="main"><div><h1>Article</h1></div><p>Body '
                '<a href="b">next</a></p><br></div><footer>Footer</footer>')
        self.assertEqual(text(page), "# Article\n\nBody [next](https://site.test/a/b)")
        self.assertEqual(text("<main><p>One</p></main><aside>ads</aside><main><p>Two</p></main>"), "One")
        self.assertEqual(text("<main><p>Unclosed"), "Unclosed")
        self.assertEqual(text("<p>No main</p><div>here</div>"), "No main\n\nhere")
        self.assertEqual(text("<ul><li></li><li>kept</li></ul><h2><img></h2><pre>-\n#</pre>"), "- kept\n\n```\n-\n#\n```")

    def test_text_json_and_charsets(self):
        self.assertTrue(self.fetch("/plain").endswith("\n\nline one\nline two"))
        self.assertTrue(self.fetch("/json").endswith('\n\n{"ok": true}'))
        self.assertIn('{"title": "nope"}', self.fetch("/problem"))
        self.assertTrue(self.fetch("/latin1").endswith("\n\ncafé"))
        self.assertTrue(self.fetch("/bogus-charset").endswith("\n\nnaïve"))
        self.assertTrue(self.fetch("/empty").endswith("\n\n(no readable text)"))

    def test_errors_come_back_as_text(self):
        self.assertEqual(self.fetch("/image"), f"Error: {self.base}/image is image/png, not a text page")
        self.assertEqual(self.fetch("/missing"), f"Error: HTTP 404 Not Found from {self.base}/missing")
        import socket
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            closed = f"http://127.0.0.1:{probe.getsockname()[1]}/"
        self.assertTrue(self.fetch(closed).startswith(f"Error: couldn't fetch {closed}: "))
        with patch.object(lc, "WEB_TIMEOUT", 0.2):
            self.assertIn("timed out", self.fetch("/slow"))
        self.assertTrue(self.fetch("http://no-such-host.invalid/").startswith("Error: couldn't fetch"))

    def test_bad_urls_and_denial_make_no_request(self):
        before = len(self.hits)
        for url in ("", "example.com", "file:///etc/passwd", "ftp://example.com/x", "https://"):
            self.assertEqual(self.fetch(url), "Error: url must be a full http:// or https:// address", url)
        tools = lc.Tools(tempfile.gettempdir())
        with patch("builtins.input", return_value="n"), patch.object(lc, "drop_last_history"):
            self.assertEqual(self.fetch("/page", tools), "User denied the request.")
        self.assertEqual(len(self.hits), before)
        with patch("builtins.input", return_value="a"), patch.object(lc, "drop_last_history"):
            self.assertIn("Demo Docs", self.fetch("/page", tools))
        self.assertIn("Demo Docs", self.fetch("/page", tools))  # "always" isn't asked again
        with patch("builtins.input", side_effect=AssertionError("asked")):
            self.assertIn("Demo Docs", self.fetch("/page", lc.Tools(tempfile.gettempdir(), auto=True)))
        self.assertEqual(len(self.hits), before + 3)

    def test_long_pages_come_in_parts_without_gaps(self):
        with patch.object(lc, "WEB_BUDGET", 10 ** 9):
            full = self.fetch("/long").split("\n\n", 1)[1]
        self.assertNotIn("more chars", full)
        with patch.object(lc, "WEB_BUDGET", 5000):
            pieces, start = [], 0
            while True:
                result = self.fetch("/long", start=start)
                body = result.split("\n\n", 1)[1]
                more = re.search(r"\n\n\.\.\. \((\d+) more chars; call web_fetch with start=(\d+) to continue\)$", body)
                pieces.append(body[:more.start()] if more else body)
                if not more:
                    break
                self.assertLessEqual(len(pieces[-1]), 5000)
                self.assertEqual(int(more.group(1)), len(full) - int(more.group(2)))
                start = int(more.group(2))
            self.assertGreater(len(pieces), 10)
            self.assertEqual("".join(pieces), full)
            self.assertTrue(all(p.startswith(("row", "\n")) for p in pieces))
            self.assertEqual(self.fetch("/long", start=len(full)),
                             f"Error: start={len(full)} is past the end of the page ({len(full)} chars)")

    def test_download_size_is_capped(self):
        with patch.object(lc, "WEB_MAX_BYTES", 2_000_000), patch.object(lc, "WEB_BUDGET", 10 ** 9):
            self.assertNotIn("download stopped", self.fetch("/long"))
        with patch.object(lc, "WEB_MAX_BYTES", 1000):
            result = self.fetch("/long")
        self.assertIn("· download stopped at 0 MB", result.splitlines()[0])
        self.assertLessEqual(len(result.split("\n\n", 1)[1]), 1000)

    def test_model_can_call_web_fetch(self):
        self.assertIn("web_fetch", lc.TOOL_NAMES)
        self.assertIn("web_fetch", lc.system_prompt(Path(tempfile.gettempdir()))[0])
        url = self.base + "/page"
        replies = iter([
            [{"message": {"tool_calls": [{"function": {"name": "web_fetch", "arguments": {"url": url}}}]}},
             {"done": True}],
            [{"message": {"content": "The docs say to use pip."}}, {"done": True}],
        ])
        agent = lc.Agent("example", lc.Tools(tempfile.gettempdir(), auto_approve=True))
        with patch.object(lc, "ollama", lambda path, payload=None, stream=False: iter(next(replies))), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            agent.run("what do the docs say?")
        tool_message = next(m for m in agent.messages if m["role"] == "tool")
        self.assertEqual(tool_message["tool_name"], "web_fetch")
        self.assertIn("## Install & run", tool_message["content"])
        self.assertIn("● web_fetch", out.getvalue())
        self.assertIn(f"└ Fetched {url}", out.getvalue())


FAKE_BACKEND = r'''
import localcode as lc
def fake(path, payload=None, stream=False):
    if path == "/api/tags":
        return {"models": [{"name": "gpt-oss:20b"}, {"name": "second:7b"}]}
    if path == "/api/show":
        return {"capabilities": ["tools"]}
    if path == "/api/chat":
        assert payload["options"]["num_ctx"] == lc.NUM_CTX
        assert payload["tools"] == lc.TOOL_SCHEMAS
        return iter([{"message": {"content": "Received: " + payload["messages"][-1]["content"]}},
                     {"done": True, "eval_count": 8, "eval_duration": 1000000000}])
    raise AssertionError(path)
lc.ollama = fake
lc.main()
'''


class SessionTests(unittest.TestCase):
    def test_plain_session_commands_and_model_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            env = dict(os.environ, XDG_CONFIG_HOME=directory, NO_COLOR="1")
            result = subprocess.run(
                [sys.executable, "-c", FAKE_BACKEND, "--plain", "-C", directory],
                input="/status\n/model second:7b\n/yolo\nhello\n/clear\n/exit\n",
                text=True, capture_output=True, env=env, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Your project. Your models.", result.stdout)
            self.assertIn("Switched to second:7b", result.stdout)
            self.assertIn("Received: hello", result.stdout)
            self.assertIn("Conversation cleared", result.stdout)
            self.assertNotIn("\x1b", result.stdout)

    def test_one_shot_has_no_welcome_or_terminal_codes(self):
        result = subprocess.run([sys.executable, "-c", FAKE_BACKEND, "-m", "gpt-oss:20b", "hello"],
                                text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Received: hello", result.stdout)
        self.assertNotIn("Your project", result.stdout)
        self.assertNotIn("\x1b", result.stdout)

    def test_model_menu_connection_error_keeps_session_open(self):
        backend = FAKE_BACKEND.replace('lc.main()', '''
real_fake = lc.ollama
calls = 0
def disconnected(path, payload=None, stream=False):
    global calls
    if path == "/api/tags":
        calls += 1
        if calls > 1:
            raise lc.urllib.error.URLError("server stopped")
    return real_fake(path, payload, stream)
lc.ollama = disconnected
lc.main()
''')
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, "-c", backend, "--plain", "-m", "gpt-oss:20b"],
                                    input="/model second:7b\n/status\n/exit\n", text=True,
                                    capture_output=True, timeout=10,
                                    env=dict(os.environ, XDG_CONFIG_HOME=directory, NO_COLOR="1"))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Command failed:", result.stdout)
        self.assertIn("Allowed this session", result.stdout)

    def test_invalid_directory_fails_before_model_connection(self):
        result = subprocess.run([sys.executable, "localcode.py", "-C", "/nonexistent/localcode-project"],
                                text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 2)
        self.assertIn("project directory", result.stderr)
        self.assertNotIn("Can't reach Ollama", result.stderr)

    @unittest.skipUnless(sys.platform != "win32", "requires a POSIX pseudo-terminal")
    def test_busy_approval_error_and_interrupt_restore_terminal(self):
        import fcntl
        import pty
        import select
        import signal
        import struct
        import termios
        import time

        backend = FAKE_BACKEND.replace('lc.main()', """
import time
def run(self, text):
    if text == "approve":
        accepted = self.tools.confirm("test", "permission test")
        print("approval accepted" if accepted else "approval rejected", flush=True)
    elif text == "fail":
        raise RuntimeError("simulated failure")
    else:
        print("interrupt ready", flush=True)
        time.sleep(30)
lc.Agent.run = run
lc.main()
""")
        for scenario, expected in (("approve", "approval accepted"), ("fail", "Model error:"),
                                   ("interrupt", "Interrupted.")):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as directory:
                master, slave = pty.openpty()
                original = termios.tcgetattr(slave)
                fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 80, 0, 0))
                env = dict(os.environ, XDG_CONFIG_HOME=directory, TERM="xterm-256color")
                env.pop("NO_COLOR", None)
                process = subprocess.Popen([sys.executable, "-c", backend, "-C", directory],
                                           stdin=slave, stdout=slave, stderr=slave, env=env)
                output, cursor = bytearray(), 0

                def wait_for(text):
                    nonlocal cursor
                    deadline = time.monotonic() + 5
                    needle = text.encode()
                    while time.monotonic() < deadline:
                        position = output.find(needle, cursor)
                        if position >= 0:
                            cursor = position + len(needle)
                            return
                        if select.select([master], [], [], 0.05)[0]:
                            output.extend(os.read(master, 65536))
                    self.fail("Missing " + repr(text) + "\n" + output.decode(errors="replace"))

                try:
                    wait_for("Ask about this project")
                    os.write(master, scenario.encode() + b"\r")
                    if scenario == "approve":
                        wait_for("[Y]es / [n]o / [a]lways for this tool:")
                        self.assertTrue(termios.tcgetattr(slave)[3] & termios.ICANON)
                        os.write(master, b"y\r")
                    elif scenario == "interrupt":
                        wait_for("interrupt ready")
                        self.assertFalse(termios.tcgetattr(slave)[3] & termios.ECHO)
                        process.send_signal(signal.SIGINT)
                    wait_for(expected)
                    wait_for("\x1b[?2004h")
                    os.write(master, b"\x04")
                    wait_for("\x1b[?1049l")
                    process.wait(timeout=5)
                    restored = termios.tcgetattr(slave)
                    pending = getattr(termios, "PENDIN", 0)
                    restored[3] &= ~pending
                    original[3] &= ~pending
                    self.assertEqual(restored, original)
                    self.assertEqual(process.returncode, 0)
                    self.assertNotIn(b"Traceback", output)
                finally:
                    if process.poll() is None:
                        process.kill()
                        process.wait()
                    os.close(master)
                    os.close(slave)

    @unittest.skipUnless(sys.platform != "win32", "requires a POSIX pseudo-terminal")
    def test_full_screen_paste_scroll_resize_and_exit(self):
        import fcntl
        import pty
        import select
        import signal
        import struct
        import termios
        import time

        master, slave = pty.openpty()
        original = termios.tcgetattr(slave)
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 80, 0, 0))
        output = bytearray()
        with tempfile.TemporaryDirectory() as directory:
            env = dict(os.environ, XDG_CONFIG_HOME=directory, TERM="xterm-256color")
            env.pop("NO_COLOR", None)
            backend = FAKE_BACKEND.replace('lc.main()', """
original_fake = lc.ollama
def slow_fake(path, payload=None, stream=False):
    if path != "/api/chat":
        return original_fake(path, payload, stream)
    def chunks():
        import time
        print("\\n".join("context line " + str(i) for i in range(30)), flush=True)
        yield {"message": {"thinking": "checking"}}
        time.sleep(1)
        yield from original_fake(path, payload, stream)
    return chunks()
lc.ollama = slow_fake
lc.main()
""")
            process = subprocess.Popen([sys.executable, "-c", backend, "-C", directory],
                                       stdin=slave, stdout=slave, stderr=slave, env=env)

            search_cursor = 0

            def wait_for(text):
                nonlocal search_cursor
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    position = output.find(text.encode(), search_cursor)
                    if position != -1:
                        search_cursor = position + len(text.encode())
                        return
                    if select.select([master], [], [], 0.1)[0]:
                        output.extend(os.read(master, 65536))
                self.fail("Terminal did not show " + repr(text) + "\n" + output.decode(errors="replace"))

            try:
                wait_for("Ask about this project")
                # Split both CSI opening and closing paste markers across reads.
                for chunk in (b"\x1b[20", b"0~", "项目".encode(), b"\x1b[20", b"1~"):
                    os.write(master, chunk)
                    time.sleep(0.05)
                for chunk in (b"\x1b[13;", b"2u", b"hello"):
                    os.write(master, chunk)
                    time.sleep(0.05)
                os.write(master, b"\r")
                wait_for("thinking…")
                self.assertFalse(termios.tcgetattr(slave)[3] & termios.ECHO,
                                 "busy terminal must not echo mouse reports")
                for chunk in (b"\x1b[<65;", b"56;35M", b"\x1b[5~"):
                    os.write(master, chunk)
                    time.sleep(0.02)
                wait_for("scrolled back")
                fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 70, 0, 0))
                process.send_signal(signal.SIGWINCH)
                os.write(master, b"\x1b[6~")
                wait_for("Received:")
                wait_for("\x1b[?2004h")
                os.write(master, b"/status\r")
                wait_for("Allowed this session")
                wait_for("\x1b[?2004h")
                os.write(master, b"\x12")
                wait_for("Prompt history")
                os.write(master, b"hello")
                time.sleep(0.1)
                os.write(master, b"\r")
                wait_for("/ commands")
                os.write(master, b"\x12")
                wait_for("Prompt history")
                os.write(master, b"\x1b")
                wait_for("/ commands")
                # Selected history is a draft, not an automatic model request.
                os.write(master, b"\x15/model\r")
                wait_for("Choose your Ollama model")
                os.write(master, b"second")
                time.sleep(0.1)
                os.write(master, b"\r")
                wait_for("Switched to ")
                wait_for("\x1b[?2004h")
                os.write(master, b"\x1b[5~")
                wait_for("scrolled back")
                fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 12, 30, 0, 0))
                process.send_signal(signal.SIGWINCH)
                time.sleep(0.1)
                os.write(master, b"draft\x03")  # first Ctrl-C clears; second quits
                time.sleep(0.1)
                os.write(master, b"\x03")
                wait_for("\x1b[?1049l")
                process.wait(timeout=5)
                self.assertEqual(process.returncode, 0)
                restored = termios.tcgetattr(slave)
                # Darwin may set PENDIN when canonical input is restored; this
                # kernel-managed flag schedules pending input for reprinting.
                pending_input = getattr(termios, "PENDIN", 0)
                restored[3] &= ~pending_input
                original[3] &= ~pending_input
                self.assertEqual(restored, original)
                self.assertNotIn(b"Traceback", output)
                self.assertNotIn(b"^[[<", output)
                self.assertNotIn(b"\x1b[<65;56;35M", output)
                self.assertEqual(output.count(b"\x1b[>1u"), output.count(b"\x1b[<u"))
                history = json.loads((Path(directory) / "localcode/history.jsonl").read_text().splitlines()[0])
                self.assertEqual(history, "项目\nhello")
                entries = [json.loads(line) for line in (Path(directory) / "localcode/history.jsonl").read_text().splitlines()]
                self.assertEqual(entries.count("项目\nhello"), 1)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
                os.close(master)
                os.close(slave)


if __name__ == "__main__":
    unittest.main()
