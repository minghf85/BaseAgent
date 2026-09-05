"""Tests for the HTTP server: auth enforcement and per-request overrides."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from fastapi.testclient import TestClient

from baseagent.config import Config
from baseagent.server.app import create_app

# Verify the mock provider host sways the agent to Write into the given
# workspace, so we can observe the override in a deterministic way.
def _script_cfg(workspace, tools=None, **auth):
    provider = {
        "type": "mock",
        "extra_body": {
            "script": [{"tool": {"name": "Write", "input": {"file_path": "out.txt", "content": "hi\n"}}}, {"text": "ok"}],
        },
    }
    d = {
        "workspace": workspace,
        "provider": provider,
        "auth": auth,
        "limits": {"max_iterations": 6},
    }
    if tools:
        d["tools"] = {"enabled": tools}
    return Config.from_dict(d)


def _sse_events(client, url, body, token=None):
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    with client.stream("POST", url, json=body, headers=headers) as resp:
        status = resp.status_code
        events = []
        for line in resp.iter_lines():
            if line.startswith("data:"):
                events.append(json.loads(line[6:]))
        return status, events


class AuthTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ws = self._tmp.name
        self.token = "sekret"
        self.app = create_app(_script_cfg(self.ws, enabled=True, token=self.token))
        self.client = TestClient(self.app)

    def tearDown(self):
        self._tmp.cleanup()

    def test_run_requires_token(self):
        status, _ = _sse_events(self.client, "/run", {"prompt": "x"})
        self.assertEqual(status, 401)

    def test_run_accepts_token(self):
        status, events = _sse_events(self.client, "/run", {"prompt": "x"}, token=self.token)
        self.assertEqual(status, 200)
        self.assertEqual(events[-1]["type"], "terminal")

    def test_run_rejects_wrong_token(self):
        status, _ = _sse_events(self.client, "/run", {"prompt": "x"}, token="wrong")
        self.assertEqual(status, 401)

    def test_health_does_not_require_token(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)


class OverrideTest(unittest.TestCase):
    def test_workspace_override(self):
        # No auth here; the point is the override lands in a different dir.
        base_ws = self._make_ws()
        alt_ws = self._make_ws()
        app = create_app(_script_cfg(base_ws, enabled=False))
        client = TestClient(app)
        status, _ = _sse_events(
            client, "/run", {"prompt": "x", "workspace": alt_ws}
        )
        self.assertEqual(status, 200)
        self.assertTrue(os.path.exists(os.path.join(alt_ws, "out.txt")))
        self.assertFalse(os.path.exists(os.path.join(base_ws, "out.txt")))

    def test_tools_override(self):
        base_ws = self._make_ws()
        app = create_app(_script_cfg(base_ws, tools=["Read", "Glob"], enabled=False))
        client = TestClient(app)
        status, _ = _sse_events(
            client, "/run", {"prompt": "x", "tools": ["Bash", "Write"]}
        )
        self.assertEqual(status, 200)

    def _make_ws(self):
        return tempfile.mkdtemp()


if __name__ == "__main__":
    unittest.main()
