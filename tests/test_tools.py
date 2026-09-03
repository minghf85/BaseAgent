"""Tests for the six built-in tools against a temp workspace."""

from __future__ import annotations

import os
import tempfile
import unittest

from baseagent.config import Config
from baseagent.tools.base import ToolContext
from baseagent.tools.registry import ToolRegistry


class ToolTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ws = self._tmp.name
        cfg = Config.from_dict({"workspace": self.ws})
        self.reg = ToolRegistry(cfg)
        self.ctx = ToolContext(cwd=self.ws, max_result_chars=cfg.tools.max_result_chars, config=cfg)

    def tearDown(self):
        self._tmp.cleanup()

    def _write(self, rel, content):
        path = os.path.join(self.ws, rel)
        os.makedirs(os.path.dirname(path) or self.ws, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    async def _run(self, name, **kw):
        return await self.reg.run(self.ctx, name, kw)


class TestWriteReadEdit(ToolTestCase):
    def test_write_then_read(self):
        import asyncio

        r = asyncio.run(self._run("Write", file_path="a/b.txt", content="hello\nworld\n"))
        self.assertEqual(r.status, "success", r.output)
        r = asyncio.run(self._run("Read", file_path="a/b.txt"))
        self.assertEqual(r.status, "success")
        self.assertIn("hello", r.output)
        self.assertIn("world", r.output)

    def test_read_offset_limit(self):
        import asyncio

        content = "".join(f"line {i}\n" for i in range(1, 11))
        self._write("m.txt", content)
        r = asyncio.run(self._run("Read", file_path="m.txt", offset=5, limit=2))
        self.assertIn("line 5", r.output)
        self.assertIn("line 6", r.output)
        self.assertNotIn("line 7", r.output)

    def test_edit_unique_match(self):
        import asyncio

        self._write("e.txt", "one\ntwo\nthree\n")
        r = asyncio.run(self._run("Edit", file_path="e.txt", old_string="two", new_string="TWO"))
        self.assertEqual(r.status, "success")
        with open(os.path.join(self.ws, "e.txt")) as f:
            self.assertIn("TWO", f.read())

    def test_edit_ambiguous_fails(self):
        import asyncio

        self._write("e2.txt", "dup\ndup\n")
        r = asyncio.run(self._run("Edit", file_path="e2.txt", old_string="dup", new_string="x"))
        self.assertEqual(r.status, "error")
        self.assertEqual(r.error_type, "multiple_occurrences")


class TestGlobGrep(ToolTestCase):
    def test_glob(self):
        import asyncio

        self._write("src/main.py", "print(1)\n")
        self._write("src/util.py", "print(2)\n")
        self._write("README.md", "# title\n")
        r = asyncio.run(self._run("Glob", pattern="src/*.py"))
        self.assertEqual(r.status, "success")
        self.assertIn("main.py", r.output)
        self.assertIn("util.py", r.output)

    def test_grep(self):
        import asyncio

        self._write("g.txt", "apple\nbanana\napple pie\n")
        r = asyncio.run(self._run("Grep", pattern="apple"))
        self.assertEqual(r.status, "success")
        self.assertIn("g.txt:1", r.output)
        self.assertIn("g.txt:3", r.output)

    def test_grep_case_insensitive(self):
        import asyncio

        self._write("h.txt", "Hello\nworld\n")
        r = asyncio.run(self._run("Grep", pattern="hello", i=True))
        self.assertEqual(r.status, "success")
        self.assertIn("h.txt:1", r.output)


class TestBashTool(ToolTestCase):
    def test_bash_echo(self):
        import asyncio

        r = asyncio.run(self._run("Bash", command="echo hi_from_test"))
        self.assertEqual(r.status, "success", r.output)
        self.assertIn("hi_from_test", r.output)

    def test_bash_cwd_is_workspace(self):
        import asyncio

        self._write("cwd_marker.txt", "x")
        r = asyncio.run(self._run("Bash", command="ls"))
        self.assertEqual(r.status, "success")
        self.assertIn("cwd_marker.txt", r.output)


class TestSandbox(ToolTestCase):
    def test_path_escape_rejected(self):
        import asyncio

        outside = os.path.join(tempfile.gettempdir(), "outside_secret.txt")
        with open(outside, "w") as f:
            f.write("secret")
        try:
            r = asyncio.run(self._run("Read", file_path=outside))
            self.assertEqual(r.status, "error")
            self.assertEqual(r.error_type, "path_outside_sandbox")
        finally:
            os.remove(outside)


if __name__ == "__main__":
    unittest.main()
