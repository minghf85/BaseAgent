"""Engine-loop tests driven by the deterministic MockProvider (no network)."""

from __future__ import annotations

import tempfile
import unittest

from baseagent.config import Config
from baseagent.engine.agent import Agent


def _cfg(workspace, extra_provider=None, limits=None):
    provider = {"type": "mock"}
    if extra_provider:
        provider.update(extra_provider)
    d = {"workspace": workspace, "provider": provider}
    if limits:
        d["limits"] = limits
    return Config.from_dict(d)


class MockLoopTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ws = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()


class TestLoop(MockLoopTest):
    def test_stop_after_1_tool_then_text(self):
        import asyncio

        agent = Agent(_cfg(self.ws))
        t = asyncio.run(agent.run("write a file"))
        self.assertEqual(t.reason, "stop")
        self.assertEqual(t.turn_count, 2)  # one tool_use turn + final text turn

    def test_max_turns_terminates(self):
        import asyncio

        script = [{"tool": {"name": "Bash", "input": {"command": "echo"}}}]
        agent = Agent(_cfg(self.ws, {"extra_body": {"script": script}}, {"max_iterations": 3}))
        t = asyncio.run(agent.run("run"))
        self.assertEqual(t.reason, "max_turns")
        self.assertEqual(t.turn_count, 3)

    def test_budget_tokens_terminates(self):
        import asyncio

        script = [{"tool": {"name": "Bash", "input": {"command": "echo"}}}]
        # Each mock request reports 20 output tokens.
        agent = Agent(_cfg(self.ws, {"extra_body": {"script": script}}, {"max_tokens_total": 25, "max_iterations": 50}))
        t = asyncio.run(agent.run("run"))
        self.assertEqual(t.reason, "budget")

    def test_usage_accumulates(self):
        import asyncio

        agent = Agent(_cfg(self.ws))
        asyncio.run(agent.run("hi"))
        info = agent.usage
        self.assertGreater(info["requests"], 0)
        self.assertGreater(info["usage"]["output_tokens"], 0)

    def test_reset_clears_history(self):
        import asyncio

        agent = Agent(_cfg(self.ws))
        asyncio.run(agent.run("first"))
        self.assertGreater(len(agent.history), 0)
        agent.reset()
        self.assertEqual(len(agent.history), 0)


class TestMockToolExecution(MockLoopTest):
    def test_tool_actually_runs(self):
        """The mock model invokes Write; verify the file really appears."""
        import asyncio
        import os

        script = [
            {"tool": {"name": "Write", "input": {"file_path": "made.txt", "content": "hello\n"}}},
            {"text": "done"},
        ]
        agent = Agent(_cfg(self.ws, {"extra_body": {"script": script}}))
        t = asyncio.run(agent.run("make a file"))
        self.assertEqual(t.reason, "stop")
        self.assertTrue(os.path.exists(os.path.join(self.ws, "made.txt")))


if __name__ == "__main__":
    unittest.main()
