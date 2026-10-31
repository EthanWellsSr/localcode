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
            process = subprocess.Popen([sys.executable, "-c", FAKE_BACKEND, "-C", directory],
                                       stdin=slave, stdout=slave, stderr=slave, env=env)

            def wait_for(text):
                deadline = time.monotonic() + 5
                start = len(output)
                while time.monotonic() < deadline:
                    if text.encode() in output[start:]:
                        return
                    if select.select([master], [], [], 0.1)[0]:
                        output.extend(os.read(master, 65536))
                self.fail("Terminal did not show " + repr(text) + "\n" + output.decode(errors="replace"))

            try:
                wait_for("Ask about this project")
                # Split both CSI opening and closing paste markers across reads.
                for chunk in (b"\x1b[20", b"0~", "项目\nhello".encode(), b"\x1b[20", b"1~"):
                    os.write(master, chunk)
                    time.sleep(0.05)
                os.write(master, b"\r")
                wait_for("Received:")
                os.write(master, b"/status\r")
                wait_for("Allowed this session")
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
                history = json.loads((Path(directory) / "localcode/history.jsonl").read_text().splitlines()[0])
                self.assertEqual(history, "项目\nhello")
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
                os.close(master)
                os.close(slave)


if __name__ == "__main__":
    unittest.main()
