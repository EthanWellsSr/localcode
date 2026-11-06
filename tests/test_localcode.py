import tempfile
import unittest
from pathlib import Path

import localcode


class ReadFileRepeatTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.tools = localcode.Tools(self.tempdir.name)
        self.file = Path(self.tempdir.name) / "notes.txt"
        self.file.write_text("first line\nsecond line\nthird line\n")

    def test_repeated_read_returns_requested_lines_again(self):
        first = self.tools.read_file("notes.txt")
        repeated = self.tools.read_file("notes.txt", limit=1)

        self.assertIn("first line", first)
        self.assertIn("unchanged since your earlier read", repeated)
        self.assertIn("first line", repeated)
        self.assertNotIn("second line", repeated)

    def test_repeated_read_reflects_file_changes(self):
        self.tools.read_file("notes.txt")
        self.file.write_text("updated line\n")

        result = self.tools.read_file("notes.txt")

        self.assertNotIn("unchanged since your earlier read", result)
        self.assertIn("updated line", result)


if __name__ == "__main__":
    unittest.main()
