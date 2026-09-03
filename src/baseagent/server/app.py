"""FastAPI application exposing the baseagent harness over REST/SSE.

Endpoints
---------
- ``GET /health``        -> alive status + resolved config summary
- ``GET /config``        -> the fully-resolved config (keys redacted)
- ``POST /run``          -> run the agent on a prompt; streams SSE events then a
                            ``terminal`` event
- ``POST /reset``        -> clear the conversation history and usage
- ``POST /abort``        -> request a clean abort of the in-flight run
- ``POST /stream``       -> (one-shot path) same as /run but always starts a
                            fresh conversation for the request

One :class:`Agent` is owned per app instance; conversation state persists across
``/run`` calls so the server behaves like a persistent chat agent. A lock
serializes concurrent ``/run`` requests.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..config import Config
from ..engine.agent import Agent
from ..types import Terminal

logger = logging.getLogger(__name__)


class RunRequest(BaseModel):
    prompt: str = Field(..., description="The task/prompt to give the agent.")
    max_iterations: Optional[int] = Field(None, description="Override iteration limit.")
    system: Optional[str] = Field(None, description="Override the system prompt.")


class AgentRuntime:
    """Owns one Agent plus a lock that serializes runs.

    When ``fresh_per_run`` is False (default) the conversation is persistent;
    when True, each request gets a brand-new engine so nothing carries over.
    """

    def __init__(self, config: Config, fresh_per_run: bool = False):
        self.config = config
        self.fresh_per_run = fresh_per_run
        self._agent: Optional[Agent] = None
        self._lock = asyncio.Lock()

    def _get_agent(self) -> Agent:
        if self._agent is None:
            self._agent = Agent(self.config)
        return self._agent

    async def stream_run(self, req: RunRequest):
        async with self._lock:
            if self.fresh_per_run:
                agent = Agent(self.config)
            else:
                agent = self._get_agent()
            async for item in agent.submit(
                req.prompt, max_iterations=req.max_iterations, system_override=req.system
            ):
                yield _encode(item)
            # Final small keep-alive comment is unnecessary; the SSE stream is
            # terminated by the client closing.

    def reset(self) -> None:
        if self._agent is not None:
            self._agent.reset()

    def abort(self) -> None:
        if self._agent is not None:
            self._agent.abort()


def create_app(config: Config) -> FastAPI:
    app = FastAPI(title="baseagent", version="0.1.0", description="Modular agent harness")
    runtime = AgentRuntime(config)
    app.state.runtime = runtime

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "model": config.provider.model,
            "provider": config.provider.type,
            "workspace": config.workspace,
            "tools": config.tools.enabled,
        }

    @app.get("/config")
    async def get_config():
        return config.to_dict(redact_keys=True)

    @app.post("/run")
    async def run(req: RunRequest):
        if not req.prompt.strip():
            raise HTTPException(status_code=400, detail="prompt must not be empty")
        return StreamingResponse(
            runtime.stream_run(req),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/stream")
    async def stream(req: RunRequest):
        """Fresh-conversation variant of /run."""
        fr = AgentRuntime(config, fresh_per_run=True)
        return StreamingResponse(
            fr.stream_run(req),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/reset")
    async def reset():
        runtime.reset()
        return {"status": "ok", "reset": True}

    @app.post("/abort")
    async def abort():
        runtime.abort()
        return {"status": "ok", "abort_requested": True}

    return app


def _encode(item: Any) -> str:
    """Serialize a StreamEvent or Terminal into an SSE ``data:`` frame."""
    if isinstance(item, Terminal):
        payload = item.to_event().to_dict()
    else:
        payload = item.to_dict()
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
