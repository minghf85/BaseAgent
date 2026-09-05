# BaseAgent

A modular, configuration-driven **agent harness** built in the spirit of Claude Code — inspired by `ref/` (the reference `query.ts` and `QueryEngine.ts` loop). It exposes a persistent agent conversation over an **HTTP/SSE server**, with exactly six tools (`Bash`, `Edit`, `Read`, `Write`, `Glob`, `Grep`), a pluggable **provider layer** (Anthropic, OpenAI, Google Gemini, Ollama, or local OpenAI-compatible endpoints), and full **token/budget accounting**.

Everything — workspace, prompts, tool set, iteration limit, token & dollar budgets, model/provider — is configured from a single `config.yaml` and needs no code changes.

## Highlights

- **The harness loop** — build messages → call the model (streaming) → collect `tool_use` blocks → execute tools → append `tool_result`s → repeat until the model stops or `max_iterations` is reached (mirrors the reference `query.ts` / `QueryEngine.ts`).
- **Provider-neutral core** — the engine speaks a single content-block message model; each provider adapter translates to its native wire format over raw `httpx` (no vendor SDKs), so a new provider is one adapter class.
- **Six tools** — `Bash`, `Edit`, `Read`, `Write`, `Glob`, `Grep`, all sandboxed to the workspace directory (paths outside it are rejected).
- **Token & budget control** — per-request usage is accumulated from provider-reported values; optional `max_tokens` and `max_budget_usd` guards stop the loop cleanly at a turn boundary, plus an optional pre-call context-window estimator.
- **HTTP/SSE server** — `POST /run` streams agent events then a `terminal` event; `GET /health`, `GET /config`, `POST /reset`, `POST /abort`; a `/stream` endpoint for fresh-conversation runs. Per-request overrides (`workspace` / `tools` / `model` / `provider`) and optional bearer-token auth.
- **Fully testable offline** — ships with a deterministic `MockProvider` and a passing test suite.

## Documentation

- [`docs/project_info.md`](docs/project_info.md) — project intro written for **interviews & resume**: the key design principles, the problems each solves, architecture, and how to talk about the design decisions.
- [`docs/integration.md`](docs/integration.md) — manual per-provider validation steps (Anthropic / OpenAI / Gemini / Ollama).

## Install

Requires Python ≥ 3.12 and `uv`.

```bash
uv sync                 # install deps + the `baseagent` console script
```

## Quick start

The bundled `configs/base.yaml` currently points at a **local Ollama**
(`qwen3:14b`), so you can drive a real model with no cloud key:

```bash
# One-shot run (uses the Ollama provider / qwen3:14b)
baseagent run -c configs/base.yaml -p "Create a demo file"

# Launch the HTTP/SSE server
baseagent serve configs/base.yaml
```

```bash
# In another terminal:
curl -s http://127.0.0.1:8000/health
curl -N -X POST http://127.0.0.1:8000/run \
     -H 'Content-Type: application/json' \
     -d '{"prompt": "Create a demo file"}'
```

> **No model at all?** Switch the config to `provider.type: mock` for a
> scripted, offline demo (writes a file, then replies) — no keys, no network.
> See `configs/base.yaml` comments for the script format.

## Configuration (`configs/base.yaml`)

| Section | Knobs |
|--------|-------|
| (top) `workspace` | Absolute directory the agent's tools are sandboxed to (Read/Write/Edit/Glob/Grep cannot escape it). |
| `server` | `host`, `port`, `max_concurrency`. |
| `auth` | `enabled` (when true, `/run` `/stream` `/reset` `/abort` require a bearer token), `token` (literal) or `token_env` (env var holding the token). |
| `provider` | `type` (`anthropic` \| `openai` \| `openai_compatible` \| `gemini` \| `ollama` \| `mock`), `model`, `base_url`, `api_key_env`/`api_key`, `max_tokens`, `temperature`, `top_p`, `thinking`, `timeout_seconds`, plus `extra_headers`/`extra_body` escape hatches. |
| `prompt` | `system` (empty → built-in default) and `appends` (extra system paragraphs). |
| `tools` | `enabled` list of the six tools, `max_result_chars` (result cap), `bash_timeout_ms`, `bash_shell`, and per-tool `options`. |
| `limits` | `max_iterations`, `max_tokens_total` (0 = unlimited), `max_budget_usd` (0 = unlimited), `max_context_tokens` (pre-call guard, 0 = off), and a per-model `pricing` table in \$/token. |
| `logging` | `level`. |

Set an API key via the env var named in `api_key_env` (or inline `api_key`):

```bash
export ANTHROPIC_API_KEY=sk-ant-...     # for provider.type: anthropic
export OPENAI_API_KEY=sk-...            # for openai
export GOOGLE_API_KEY=...               # for gemini
```

### Provider notes

- **Anthropic**: default `https://api.anthropic.com`, `model` e.g. `claude-sonnet-4-6`.
- **OpenAI**: default `https://api.openai.com/v1`, `model` e.g. `gpt-4o`. Also use `type: openai_compatible` with a `base_url` for any OpenAI-compatible server.
- **Gemini**: default `https://generativelanguage.googleapis.com/v1beta`, `model` e.g. `gemini-2.0-flash`.
- **Ollama/local**: `type: ollama` (defaults to `http://localhost:11434/v1`), `model` e.g. `qwen3:14b`. Point `base_url` at any local OpenAI-compatible endpoint. (Validated end-to-end against a real local `qwen3:14b` — the model autonomously called the `Write` tool during a run.)
- **Mock**: no key needed; script the assistant with `provider.extra_body.script` (a list of `{tool: {...}}` or `{text: "..."}` turns). Used by the tests.

## Per-request overrides & auth

`POST /run` and `POST /stream` accept auth-protected overrides that change
behavior for just that request — without touching `config.yaml`:

| Field | Effect |
|-------|--------|
| `workspace` | Override the sandbox directory the tools operate in. |
| `tools` | Override the enabled tool list, e.g. `["Bash","Read"]`. |
| `model` | Override the model name. |
| `provider` | Override the provider type. |
| `session_id` | Reuse this to continue a conversation across calls. |

Each distinct `session_id` + override combination gets its own isolated
conversation (its own engine, history, and usage), so a `/run` override never
leaks into another conversation.

**Auth**: set `auth.enabled: true` and a token via `auth.token` or the
`auth.token_env` env var. Callers then authenticate with either
`Authorization: Bearer <token>` or `X-Api-Token: <token>`. `/health` and
`/config` stay open; `/run`, `/stream`, `/reset`, `/abort` are protected.

```bash
BASEAGENT_API_TOKEN=sekret baseagent serve configs/base.yaml

curl -N -X POST http://127.0.0.1:8006/run \
     -H 'Authorization: Bearer sekret' \
     -H 'Content-Type: application/json' \
     -d '{"prompt":"list files","workspace":"/tmp/other","tools":["Bash","Glob"],"model":"qwen2.5:7b"}'
```

## Architecture

```
src/baseagent/
├── main.py              # CLI: serve / run / config-dump
├── config.py            # Config dataclasses + YAML loader + validation
├── types.py             # provider-neutral messages, blocks, usage, events, Terminal
├── providers/           # LLMProvider ABC + Anthropic/OpenAI/Gemini/Ollama/Mock + factory/transport
├── tools/               # BaseTool ABC + Bash/Edit/Read/Write/Glob/Grep + registry + path sandbox
├── engine/              # QueryEngine (the loop), UsageTracker, context estimator, Agent facade
└── server/              # FastAPI app + SSE encoding
```

The **modular seams**:

- *Add a model backend*: subclass `LLMProvider` and register it in `providers/factory.py`; implement `stream_completion()` to yield `StreamEvent`s then a `Turn`.
- *Add a tool*: subclass `BaseTool` (declare `name`, `description`, `input_schema`, `execute`) and add it to `tools/registry.py`'s `BUILTIN_TOOLS`.
- *Tune behavior*: via `config.yaml` — no code changes for prompts, tool enablement, iteration/budget limits, or model selection.

## CLI

```
baseagent serve <config.yaml>               # run the HTTP/SSE server
baseagent run  -c <config.yaml> -p <prompt> # one-shot agent run (prints events)
baseagent config-dump -c <config.yaml>      # print the resolved config (keys redacted)
```

`python -m baseagent ...` works identically.

## Testing

```bash
python -m unittest discover -s tests -v   # 30 tests, no network required
```

The suite covers the six tools (write/read/edit/glob/grep/bash, path-sandbox
rejection), config parsing/validation, provider message conversion, the engine
loop using the mock provider (stop / max_turns / budget terminals, tool
execution, usage accumulation, reset), and the HTTP server (auth enforcement
and per-request workspace/tools overrides).

### End-to-end with a real model

The loop was also validated against a live local model via Ollama
(`qwen3:14b`): given “create `greeting.txt` with Hello world”, the model
autonomously decided to call the `Write` tool, the engine executed it (the file
was created on disk), then converged to a `terminal(reason=stop, turns=2)` — a
full tool-using agent trajectory on a real LLM, not just a unit test.

## Roadmap ideas

- Permission prompts / tool-approval hooks wired into the loop (`QueryEngine.confirm_tool`).
- History compaction and a `max_turns`-style auto-compact when the context window fills.
- Rate limiting and per-token scopes on top of the bearer auth.
- OpenTelemetry export of usage telemetry.
