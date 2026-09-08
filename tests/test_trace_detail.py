"""Trace-detail tests: the engine emits an assembled ``turn`` event and enriched
``tool_result`` events, so serialized traces carry complete tool-call info."""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest

from baseagent.config import Config
from baseagent.engine.agent import Agent


def _cfg(workspace, script):
    provider = {
        "type": "mock",
        "extra_body": {"script": script},
    }
    return Config.from_dict({"workspace": workspace, "provider": provider})


async def _events(agent, prompt):
    out = []
    async for item in agent.submit(prompt):
        out.append(item.to_dict())
    return out


class TestTraceDetail(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ws = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def test_tool_result_event_is_enriched(self):
        script = [
            {"tool": {"name": "Write", "input": {"file_path": "a.txt", "content": "hi\n"}}},
            {"text": "done"},
        ]
        agent = Agent(_cfg(self.ws, script))
        events = asyncio.run(_events(agent, "write a file"))
        results = [e for e in events if e.get("type") == "tool_result"]
        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual(r["tool_name"], "Write")
        self.assertEqual(r["status"], "success")
        self.assertFalse(r["is_error"])
        self.assertIn("error_type", r)
        self.assertIn("error_message", r)

    def test_turn_event_carries_assembled_content_and_stop_reason(self):
        script = [
            {"tool": {"name": "Write", "input": {"file_path": "a.txt", "content": "hi\n"}}},
            {"text": "done"},
        ]
        agent = Agent(_cfg(self.ws, script))
        events = asyncio.run(_events(agent, "write a file"))
        turns = [e for e in events if e.get("type") == "turn"]
        self.assertEqual(len(turns), 2)  # one tool_use turn + one text turn
        first, second = turns
        self.assertEqual(first["stop_reason"], "tool_use")
        self.assertEqual(first["content"][0]["type"], "tool_use")
        self.assertEqual(first["content"][0]["name"], "Write")
        self.assertEqual(first["content"][0]["input"]["file_path"], "a.txt")
        self.assertIn("usage", first)
        self.assertEqual(second["stop_reason"], "stop")
        self.assertEqual(second["text"], "done")
        # The final assistant text turn has no tool_use blocks.
        self.assertEqual(second["content"][0]["type"], "text")

    def test_turn_event_is_json_serializable(self):
        script = [
            {"tool": {"name": "Write", "input": {"file_path": "a.txt", "content": "hi\n"}}},
            {"text": "done"},
        ]
        agent = Agent(_cfg(self.ws, script))
        events = asyncio.run(_events(agent, "go"))
        json.dumps(events)  # must not raise


if __name__ == "__main__":
    unittest.main()
