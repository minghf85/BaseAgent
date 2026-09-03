# Provider integration notes

The automated test suite runs fully offline using the built-in `MockProvider`,
so it never needs API keys or network. This page covers manually validating each
*real* provider by issuing a one-shot `baseagent run` against a config that
points at the vendor. The adapters send the documented wire formats.

## Prerequisites

- Set the API key env var before running (or inline `provider.api_key`).
- The workspace path in each config must exist (tools create dirs as needed).

## Anthropic

```yaml
provider:
  type: anthropic
  model: claude-sonnet-4-6
  api_key_env: ANTHROPIC_API_KEY
```

```bash
ANTHROPIC_API_KEY=sk-ant-... baseagent run -c my-anthropic.yaml -p "Say hello"
```

## OpenAI (and OpenAI-compatible)

```yaml
provider:
  type: openai            # or openai_compatible
  model: gpt-4o
  api_key_env: OPENAI_API_KEY
```

```bash
OPENAI_API_KEY=sk-... baseagent run -c my-openai.yaml -p "Say hello"
```

For a self-hosted server (vLLM, Together, local proxy), set `type:
openai_compatible` and `base_url`:
`http://localhost:8000/v1`.

## Google Gemini

```yaml
provider:
  type: gemini
  model: gemini-2.0-flash
  api_key_env: GOOGLE_API_KEY
```

```bash
GOOGLE_API_KEY=... baseagent run -c my-gemini.yaml -p "Say hello"
```

## Ollama / local

Start Ollama and pull a model, then:

```yaml
provider:
  type: ollama
  model: qwen2.5:7b        # any model you have pulled
  base_url: http://localhost:11434/v1
```

```bash
baseagent run -c my-ollama.yaml -p "Say hello"
```

## What "working" looks like

A successful real-provider run streams `request_start` events, one or more
`assistant_message` text deltas, possibly `tool_use` / `tool_result` events (if
the model chose to act), and finally a `terminal` event with nonzero
`usage.*tokens`. If a provider rejects the request, the run ends with a
`terminal {reason: "error"}` whose `error` field contains the normalized message
and HTTP status.
