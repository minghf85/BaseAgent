"""Command-line entrypoint for baseagent.

Commands
--------
- ``baseagent serve <config.yaml>``      launch the HTTP/SSE server
- ``baseagent run -c <config.yaml> -p <prompt>``   one-shot agent run (prints events)
- ``baseagent config-dump -c <config.yaml>``       print the resolved config
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from typing import Optional

from .config import Config
from .engine.agent import Agent
from .types import Terminal

LOG = logging.getLogger("baseagent")


def _load_config(path: str) -> Config:
    try:
        return Config.load(path)
    except FileNotFoundError:
        sys.exit(f"[error] config file not found: {path}")
    except Exception as exc:  # config validation errors
        sys.exit(f"[error] failed to load config {path!r}: {exc}")


def _setup_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO))


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------

def cmd_serve(args: argparse.Namespace) -> int:
    cfg = _load_config(args.config)
    _setup_logging(cfg.logging.level)

    import uvicorn

    from .server.app import create_app

    app = create_app(cfg)
    LOG.info(
        "baseagent serving at http://%s:%s using provider=%s model=%s workspace=%s",
        cfg.server.host,
        cfg.server.port,
        cfg.provider.type,
        cfg.provider.model,
        cfg.workspace,
    )
    uvicorn.run(app, host=cfg.server.host, port=cfg.server.port, log_level=cfg.logging.level.lower())
    return 0


# ---------------------------------------------------------------------------
# one-shot run
# ---------------------------------------------------------------------------

async def _run_once(cfg: Config, prompt: str, max_iterations: Optional[int], pretty: bool) -> int:
    agent = Agent(cfg)
    async for ev in agent.submit(prompt, max_iterations=max_iterations):
        if isinstance(ev, Terminal):
            print(json.dumps(ev.to_dict(), ensure_ascii=False, indent=2 if pretty else None))
        else:
            print(ev.to_dict())
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    cfg = _load_config(args.config)
    _setup_logging(cfg.logging.level)
    asyncio.run(_run_once(cfg, args.prompt, args.max_iterations, args.pretty))
    return 0


# ---------------------------------------------------------------------------
# config dump
# ---------------------------------------------------------------------------

def cmd_config_dump(args: argparse.Namespace) -> int:
    cfg = _load_config(args.config)
    print(json.dumps(cfg.to_dict(redact_keys=not args.show_keys), indent=2, ensure_ascii=False))
    return 0


# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="baseagent",
        description="Modular, config-driven agent harness (Claude-Code style).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_serve = sub.add_parser("serve", help="Launch the HTTP/SSE server.")
    p_serve.add_argument("config", help="Path to config.yaml")

    p_run = sub.add_parser("run", help="Run a one-shot agent task.")
    p_run.add_argument("-c", "--config", required=True, help="Path to config.yaml")
    p_run.add_argument("-p", "--prompt", required=True, help="The prompt/task for the agent")
    p_run.add_argument("--max-iterations", type=int, default=None)
    p_run.add_argument("--pretty", action="store_true")

    p_dump = sub.add_parser("config-dump", help="Print the resolved config.")
    p_dump.add_argument("-c", "--config", required=True, help="Path to config.yaml")
    p_dump.add_argument("--show-keys", action="store_true", help="Reveal the API key in output.")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "serve":
        sys.exit(cmd_serve(args))
    elif args.command == "run":
        sys.exit(cmd_run(args))
    elif args.command == "config-dump":
        sys.exit(cmd_config_dump(args))


if __name__ == "__main__":
    main()
