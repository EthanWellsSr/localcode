"""Run with python3 -m unittest discover -s tests -v. No Ollama needed."""
import contextlib
import io
import json
import os
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
        self.assertIn("Ask before writes and commands", output.getvalue())

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
