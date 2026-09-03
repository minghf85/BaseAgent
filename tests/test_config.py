"""Config parsing/validation and provider message-conversion tests."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from baseagent.config import Config
from baseagent.providers.anthropic import AnthropicProvider
from baseagent.providers.gemini import GeminiProvider
from baseagent.providers.openai import OpenAIProvider
from baseagent.config import ProviderConfig
from baseagent.types import Message, TextBlock, ToolResultBlock, ToolUseBlock


class ConfigTest(unittest.TestCase):
    def test_defaults(self):
        cfg = Config.from_dict({"workspace": "C:/ws"})
        self.assertEqual(cfg.provider.type, "anthropic")
        self.assertEqual(cfg.tools.enabled, ["Bash", "Edit", "Read", "Write", "Glob", "Grep"])
        self.assertEqual(cfg.limits.max_iterations, 20)
        self.assertEqual(cfg.workspace, os.path.abspath("C:/ws"))

    def test_load_from_yaml_file(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as f:
            f.write(
                "workspace: ./sub\n"
                "provider:\n"
                "  type: ollama\n"
                "  model: qwen2.5:7b\n"
                "tools:\n"
                "  enabled: [Bash, Read]\n"
                "limits:\n"
                "  max_iterations: 5\n"
            )
            path = f.name
        try:
            cfg = Config.load(path)
            self.assertEqual(cfg.provider.type, "ollama")
            self.assertEqual(cfg.provider.model, "qwen2.5:7b")
            self.assertEqual(cfg.tools.enabled, ["Bash", "Read"])
            self.assertEqual(cfg.limits.max_iterations, 5)
            # relative workspace resolves against the config's directory
            self.assertTrue(cfg.workspace.endswith(os.path.join("sub")))
        finally:
            os.remove(path)

    def test_empty_yaml_block_defaults(self):
        cfg = Config.from_dict({"workspace": "C:/ws", "tools": {"options": None}})
        self.assertEqual(cfg.tools.options, {})
        cfg2 = Config.from_dict({"workspace": "C:/ws", "tools": {"options": None, "enabled": None}})
        self.assertEqual(cfg2.tools.options, {})
        self.assertEqual(cfg2.tools.enabled, ["Bash", "Edit", "Read", "Write", "Glob", "Grep"])

    def test_bad_provider_rejected(self):
        with self.assertRaises(ValueError):
            Config.from_dict({"workspace": "C:/ws", "provider": {"type": "nope"}})

    def test_prompt_default_when_empty(self):
        cfg = Config.from_dict({"workspace": "C:/ws", "prompt": {"system": ""}})
        self.assertTrue(cfg.prompt.build().startswith("You are a software engineering agent"))


class ProviderConversionTest(unittest.TestCase):
    def _messages(self):
        assistant_calls = Message(
            role="assistant",
            content=[
                TextBlock(text="Let me write."),
                ToolUseBlock(id="call_1", name="Write", input={"file_path": "a.txt", "content": "hi"}),
            ],
        )
        tool_result = Message(
            role="user",
            content=[ToolResultBlock(tool_use_id="call_1", content="Wrote a.txt", is_error=False)],
        )
        assistant_final = Message(role="assistant", content=[TextBlock(text="Done.")])
        return [assistant_calls, tool_result, assistant_final]

    def test_openai_conversion_shape(self):
        p = OpenAIProvider(ProviderConfig(model="gpt-4o"))
        msgs = p._to_api_messages(self._messages())
        roles = [m["role"] for m in msgs]
        self.assertEqual(roles, ["assistant", "tool", "assistant"])
        self.assertEqual(msgs[0]["tool_calls"][0]["function"]["name"], "Write")
        self.assertEqual(msgs[1]["tool_call_id"], "call_1")
        parsed = json.loads(msgs[0]["tool_calls"][0]["function"]["arguments"])
        self.assertEqual(parsed["file_path"], "a.txt")

    def test_anthropic_conversion_shape(self):
        p = AnthropicProvider(ProviderConfig(model="claude-3-5-sonnet"))
        body = p._build_body("sys", self._messages(), [{"name": "Write", "description": "w", "input_schema": {}}])
        self.assertEqual(body["system"], "sys")
        roles = [m["role"] for m in body["messages"]]
        self.assertEqual(roles, ["assistant", "user", "assistant"])
        self.assertIn("tool_use", [c["type"] for c in body["messages"][0]["content"]])

    def test_gemini_conversion_shape(self):
        p = GeminiProvider(ProviderConfig(model="gemini-2.0-flash"))
        body = p._build_body("sys", self._messages(), [{"name": "Write", "description": "w", "parameters": {}}])
        self.assertEqual(body["systemInstruction"]["parts"][0]["text"], "sys")
        contents = body["contents"]
        # assistant(tool use), user(function response), assistant(text)
        self.assertIn("functionCall", contents[0]["parts"][1])
        self.assertIn("functionResponse", contents[1]["parts"][0])


if __name__ == "__main__":
    unittest.main()
