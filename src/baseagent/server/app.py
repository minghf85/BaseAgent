"""FastAPI application exposing the baseagent harness over REST/SSE.

Endpoints
---------
- ``GET /health``        -> alive status + resolved config summary
- ``GET /config``        -> the fully-resolved config (keys redacted)
- ``POST /run``          -> run the agent on a prompt; streams SSE events then a
                            ``terminal`` event
- ``POST /stream``       -> one-shot run in a fresh conversation
- ``POST /reset``        -> clear a conversation (by ``session_id``)
- ``POST /abort``        -> request a clean abort of an in-flight run

Per-request overrides
---------------------
``/run`` and ``/stream`` accept (all authenticated when auth is enabled):

- ``workspace`` — override the config's workspace directory
- ``tools`` — override the enabled tool list (subset of the six built-ins)
- ``model`` — override the model name
- ``provider`` — override the provider type ("anthropic" / "openai" / ...)

``session_id`` keys the conversation; the same id reuses history across
``/run`` calls (matching the persistence semantics of a chat agent). When
overrides differ from the server defaults, the conversation is keyed by the
exprin ``session_id`` plus those overrides so contradictory settings never
share an engine.

Authentication
--------------
When ``auth.enabled`` in config is true, the mutating endpoints require either
``Authorization: Bearer <token>`` or ``X-Api-Token: <token>``. The token comes
from ``auth.token`` or the ``auth.token_env`` environment variable.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.requests import Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..config import Config
from ..engine.agent import Agent
from ..types import Terminal

logger = logging.getLogger(__name__)

_OVERRIDE_KEYS = ("workspace", "tools", "model", "provider")


class RunRequest(BaseModel):
    prompt: str = Field(..., description="The task/prompt to give the agent.")
    session_id: Optional[str] = Field(
        None, description="Conversation key; reuse to continue a conversation."
    )
    workspace: Optional[str] = Field(None, description="Override config workspace dir.")
    tools: Optional[list[str]] = Field(None, description="Override enabled tools.")
    model: Optional[str] = Field(None, description="Override the model name.")
    provider: Optional[str] = Field(None, description="Override the provider type.")
    max_iterations: Optional[int] = Field(None, description="Override iteration limit.")
    system: Optional[str] = Field(None, description="Override the system prompt.")


class ResetRequest(BaseModel):
    session_id: Optional[str] = Field(None)


def _extract_token(req: Request) -> str:
    """Return the bearer token from ``Authorization`` or ``X-Api-Token``, empty if absent."""
    auth = req.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return req.headers.get("x-api-token", "").strip()


class AgentRuntime:
    """Holds keyed conversations (config + agent + lock) for the app."""

    def __init__(self, base_config: Config):
        self.base = base_config
        # key -> (Config, Agent, asyncio.Lock)
        self._agents: dict[str, tuple[Config, Agent, asyncio.Lock]] = {}

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def auth_error(self, req: Request) -> Optional[HTTPException]:
        if not self.base.auth.enabled:
            return None
        expected = self.base.auth.resolve_token()
        if expected and _extract_token(req) == expected:
            return None
        return HTTPException(status_code=401, detail="Unauthorized")

    # ------------------------------------------------------------------
    # Conversation plumbing
    # ------------------------------------------------------------------

    def _override_config(self, req: RunRequest) -> Config:
        tools = tuple(req.tools) if req.tools is not None else None
        return self.base.overridden(
            workspace=req.workspace,
            tools=list(tools) if tools is not None else None,
            model=req.model,
            provider_type=req.provider if req.provider else None,
        )

    def _make_key(self, req: RunRequest, cfg: Config) -> str:
        sid = req.session_id or "default"
        overrides = (
            cfg.workspace,
            tuple(sorted(cfg.tools.enabled)),
            cfg.provider.model,
            cfg.provider.type,
        )
        # Only embed overrides that differ from the base so a plain /run with a
        # session_id still lands on the same engine across calls.
        base = self.base
        parts = [sid]
        if cfg.workspace != base.workspace:
            parts.append(f"ws={cfg.workspace}")
        if tuple(sorted(cfg.tools.enabled)) != tuple(sorted(base.tools.enabled)):
            parts.append(f"tools={','.join(sorted(cfg.tools.enabled))}")
        if cfg.provider.model != base.provider.model:
            parts.append(f"model={cfg.provider.model}")
        if cfg.provider.type != base.provider.type:
            parts.append(f"prov={cfg.provider.type}")
        return "|".join(parts)

    def _get(self, key: str, cfg: Config) -> Agent:
        entry = self._agents.get(key)
        if entry is None:
            entry = (cfg, Agent(cfg), asyncio.Lock())
            self._agents[key] = entry
        return entry[1]

    def _lock_of(self, key: str) -> asyncio.Lock:
        return self._agents[key][2]

    def reset(self, session_id: Optional[str]) -> None:
        if session_id:
            for key, (_, agent, _) in list(self._agents.items()):
                if key == session_id or key.startswith(session_id + "|"):
                    agent.reset()
        else:
            for _, agent, _ in self._agents.values():
                agent.reset()

    def abort(self, session_id: Optional[str]) -> None:
        for key, (_, agent, _) in self._agents.items():
            if session_id is None or key.startswith(session_id + "|") or key == session_id:
                agent.abort()

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    async def stream_run(self, req: RunRequest):
        cfg = self._override_config(req)
        key = self._make_key(req, cfg)
        agent = self._get(key, cfg)
        lock = self._lock_of(key)
        async with lock:
            async for item in agent.submit(
                req.prompt, max_iterations=req.max_iterations, system_override=req.system
            ):
                yield _encode(item)


def create_app(config: Config) -> FastAPI:
    app = FastAPI(title="baseagent", version="0.1.0", description="Modular agent harness")
    runtime = AgentRuntime(config)
    app.state.runtime = runtime

    def require_auth(req: Request):
        err = runtime.auth_error(req)
        if err is not None:
            raise err

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "model": config.provider.model,
            "provider": config.provider.type,
            "workspace": config.workspace,
            "tools": config.tools.enabled,
            "auth_required": config.auth.enabled,
        }

    @app.get("/config")
    async def get_config():
        return config.to_dict(redact_keys=True)

    @app.post("/run", dependencies=[Depends(require_auth)])
    async def run(req: RunRequest):
        if not req.prompt.strip():
            raise HTTPException(status_code=400, detail="prompt must not be empty")
        return StreamingResponse(
            runtime.stream_run(req),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/stream", dependencies=[Depends(require_auth)])
    async def stream(req: RunRequest):
        """Fresh-conversation variant of /run (always one-shot)."""
        cfg = runtime._override_config(req)
        agent = Agent(cfg)
        return StreamingResponse(
            _fresh_stream(agent, req),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/reset", dependencies=[Depends(require_auth)])
    async def reset(req: ResetRequest):
        runtime.reset(req.session_id)
        return {"status": "ok", "reset": True}

    @app.post("/abort", dependencies=[Depends(require_auth)])
    async def abort(req: ResetRequest):
        runtime.abort(req.session_id)
        return {"status": "ok", "abort_requested": True}

    return app


async def _fresh_stream(agent: Agent, req: RunRequest):
    async for item in agent.submit(
        req.prompt, max_iterations=req.max_iterations, system_override=req.system
    ):
        yield _encode(item)


def _encode(item: Any) -> str:
    """Serialize a StreamEvent or Terminal into an SSE ``data:`` frame."""
    if isinstance(item, Terminal):
        payload = item.to_event().to_dict()
    else:
        payload = item.to_dict()
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
