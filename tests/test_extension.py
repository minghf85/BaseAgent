"""Extension-host tests: lifecycle hooks fire and can inject a redo message."""

from __future__ import annotations

import asyncio
import tempfile
import unittest

from baseagent.config import Config
from baseagent.engine.agent import Agent
from baseagent.extension import AgentExtension, ExtensionContext


class RecordingExtension(AgentExtension):
    name = "rec"

    def __init__(self):
        self.starts = 0
        self.ends = 0
        self.turns_seen = []

    def on_run_start(self, ctx):
        self.starts += 1

    def on_turn_end(self, ctx):
        self.turns_seen.append(ctx.turn_count)
        return None

    def on_run_end(self, ctx):
        self.ends += 1


class InjectExtension(AgentExtension):
    name = "inject"

    def __init__(self):
        self.injected = 0

    def on_turn_end(self, ctx):
        # Inject once, on the first turn boundary, then let the loop proceed.
        if self.injected == 0:
            self.injected += 1
            return "TURN_ERROR: your previous step failed validation; please redo it."
        return None


def _cfg(workspace, ext, script=None, limits=None):
    provider = {"type": "mock"}
    if script:
        provider["extra_body"] = {"script": script}
    d = {"workspace": workspace, "provider": provider}
    if limits:
        d["limits"] = limits
    cfg = Config.from_dict(d)
    cfg.extensions = [ext]
    return cfg


class TestExtensionHost(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ws = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def test_hooks_fire(self):
        ext = RecordingExtension()
        agent = Agent(_cfg(self.ws, ext))
        t = asyncio.run(agent.run("hi"))
        self.assertEqual(t.reason, "stop")
        self.assertEqual(ext.starts, 1)
        self.assertEqual(ext.ends, 1)
        # Default mock: one tool_use turn + one text turn -> one on_turn_end.
        self.assertEqual(len(ext.turns_seen), 1)

    def test_injected_message_adds_history(self):
        ext = InjectExtension()
        script = [
            {"tool": {"name": "Write", "input": {"file_path": "a.txt", "content": "1\n"}}},
            {"text": "done"},
        ]
        agent = Agent(_cfg(self.ws, ext, script=script))
        t = asyncio.run(agent.run("go"))
        self.assertEqual(t.reason, "stop")
        self.assertGreater(ext.injected, 0)
        # The injected user message is present in history.
        texts = [m.text for m in agent.history]
        self.assertTrue(any("failed validation" in s for s in texts))

    def test_bad_extension_does_not_break_run(self):
        class Bad(AgentExtension):
            def on_turn_end(self, ctx):
                raise RuntimeError("boom")

        agent = Agent(_cfg(self.ws, Bad()))
        t = asyncio.run(agent.run("hi"))
        self.assertEqual(t.reason, "stop")


if __name__ == "__main__":
    unittest.main()
