"""Configuration for the baseagent harness.

All knobs are declared as dataclasses so they can be validated programmatically,
then loaded from (and overlaid onto) a YAML file. The resolved :class:`Config`
is passed to every subsystem (providers, engine, tools, server).
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Optional

import yaml

from .types import Usage

ProviderType = Literal["anthropic", "openai", "gemini", "ollama", "openai_compatible", "mock"]

DEFAULT_SYSTEM_PROMPT = """You are a software engineering agent running inside an isolated workspace.
You have access to the following tools to inspect and modify files and run
commands in that workspace:

- Bash: run a shell command in the workspace (cwd = workspace directory).
- Read: read a file with optional offset/limit and numbered lines.
- Write: create or overwrite a file.
- Edit: perform an exact-string replacement in an existing file.
- Glob: find files by a glob pattern relative to the workspace.
- Grep: search file contents by regular expression.

Always reason about the task, use the tools as needed, and stop once the task
is complete or you cannot make further progress. If a tool fails, read the
error and recover. Prefer the dedicated tools over shell equivalents for file
operations."""


# ---------------------------------------------------------------------------
# Nested configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    # Max concurrency for /run streams (per-process semaphore).
    max_concurrency: int = 4


@dataclass
class AuthConfig:
    """API authentication for the HTTP server.

    When ``enabled``, mutating endpoints (``/run``, ``/stream``, ``/reset``,
    ``/abort``) require a bearer token in the ``Authorization`` header or an
    ``X-Api-Token`` header. The token comes from ``token`` (literal) or the
    environment variable named by ``token_env``.
    """

    enabled: bool = False
    token: str = ""
    token_env: str = "BASEAGENT_API_TOKEN"

    def resolve_token(self) -> str:
        if self.token:
            return self.token
        return os.environ.get(self.token_env, "")


@dataclass
class ProviderConfig:
    type: ProviderType = "anthropic"
    model: str = "claude-sonnet-4-6"
    # Optional base URL override (needed for openai_compatible / ollama).
    base_url: str = ""
    # Name of an environment variable holding the API key, or a literal key.
    api_key_env: str = "ANTHROPIC_API_KEY"
    api_key: str = ""  # literal fallback; takes precedence if set
    max_tokens: int = 4096
    temperature: float = 0.0
    top_p: float = 1.0
    # Anthropic extended thinking.
    thinking: bool = False
    thinking_budget_tokens: int = 1024
    # Per-request timeout in seconds.
    timeout_seconds: float = 600.0
    # Extra headers / body overrides passed through to the API (advanced).
    extra_headers: dict[str, str] = field(default_factory=dict)
    extra_body: dict[str, Any] = field(default_factory=dict)

    def resolve_api_key(self) -> str:
        if self.api_key:
            return self.api_key
        env_val = os.environ.get(self.api_key_env)
        return env_val or ""


@dataclass
class PromptConfig:
    """System prompt customization."""

    system: str = DEFAULT_SYSTEM_PROMPT
    appends: list[str] = field(default_factory=list)

    def build(self) -> str:
        # An empty ``system`` means "use the built-in default".
        base = self.system if self.system else DEFAULT_SYSTEM_PROMPT
        parts = [base, *self.appends]
        return "\n\n".join(p for p in parts if p)


@dataclass
class ToolConfig:
    enabled: list[str] = field(default_factory=lambda: ["Bash", "Edit", "Read", "Write", "Glob", "Grep"])
    # Tool-specific defaults, keyed by tool name (lower-cased keys also accepted).
    max_result_chars: int = 30000
    bash_timeout_ms: int = 120000
    bash_shell: str = ""  # empty => platform default shell
    # Extra per-tool knobs, e.g. {"bash": {"env": {...}}}.
    options: dict[str, dict[str, Any]] = field(default_factory=dict)

    def tool_option(self, name: str) -> dict[str, Any]:
        return self.options.get(name) or self.options.get(name.lower()) or {}


@dataclass
class PricingModel:
    input: float = 0.0
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0


@dataclass
class LimitsConfig:
    max_iterations: int = 20
    max_tokens_total: int = 0  # 0 => unlimited (usage.output_tokens counted)
    max_budget_usd: float = 0.0  # 0 => unlimited
    # Per-model pricing {model_name: PricingModel} used for USD budget.
    pricing: dict[str, PricingModel] = field(default_factory=dict)
    # Optional pre-call token estimate guard (context window).
    max_context_tokens: int = 0  # 0 => skip pre-call guard

    def cost_usd(self, model: str, usage: Usage) -> float:
        p = self.pricing.get(model)
        if p is None:
            return 0.0
        return (
            usage.input_tokens * p.input
            + usage.output_tokens * p.output
            + usage.cache_read_input_tokens * p.cache_read
            + (usage.cache_write_input_tokens + usage.cache_creation_input_tokens) * p.cache_write
        )


@dataclass
class LoggingConfig:
    level: str = "INFO"
    format: str = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


@dataclass
class Config:
    """The fully-resolved configuration for a baseagent server."""

    # Absolute path to the workspace directory that tools operate within.
    workspace: str = "."

    server: ServerConfig = field(default_factory=ServerConfig)
    auth: AuthConfig = field(default_factory=AuthConfig)
    provider: ProviderConfig = field(default_factory=ProviderConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    tools: ToolConfig = field(default_factory=ToolConfig)
    limits: LimitsConfig = field(default_factory=LimitsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Runtime-only extensions attached to the engine (not loaded from YAML, not
    # serialized). See ``baseagent.extension``.
    extensions: list[Any] = field(default_factory=list, repr=False)

    # Allow arbitrary extra top-level keys to be stashed (forward compatible).
    _raw: dict[str, Any] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, d: dict[str, Any], base_dir: str | None = None) -> "Config":
        provider = _build_dataclass(ProviderConfig, d.get("provider", {}))
        server = _build_dataclass(ServerConfig, d.get("server", {}))
        auth = _build_dataclass(AuthConfig, d.get("auth", {}))
        prompt = _build_dataclass(PromptConfig, d.get("prompt", {}))
        tools = _build_dataclass(ToolConfig, d.get("tools", {}))
        limits_d = d.get("limits", {})
        limits = _build_dataclass(LimitsConfig, limits_d)
        limits.pricing = _build_pricing(limits_d.get("pricing", {}))
        logging_ = _build_dataclass(LoggingConfig, d.get("logging", {}))

        workspace = _first(d, "workspace", "workspace_dir", "cwd", default=".")
        if workspace:
            workspace = os.path.expanduser(workspace)
            if not os.path.isabs(workspace):
                # Relative paths resolve against the config file's directory
                # (falling back to CWD) so setups are portable.
                origin = base_dir or os.getcwd()
                workspace = os.path.abspath(os.path.join(origin, workspace))
        workspace = os.path.abspath(workspace)

        cfg = cls(
            workspace=workspace,
            server=server,
            auth=auth,
            provider=provider,
            prompt=prompt,
            tools=tools,
            limits=limits,
            logging=logging_,
            _raw=dict(d),
        )
        cfg.validate()
        return cfg

    @classmethod
    def load(cls, path: str) -> "Config":
        path = os.path.abspath(os.path.expanduser(path))
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Config root must be a mapping: {path}")
        return cls.from_dict(data, base_dir=os.path.dirname(path))

    # ------------------------------------------------------------------
    # Validation / helpers
    # ------------------------------------------------------------------

    def validate(self) -> None:
        if self.limits.max_iterations <= 0:
            raise ValueError("limits.max_iterations must be >= 1")
        if self.provider.max_tokens <= 0:
            raise ValueError("provider.max_tokens must be >= 1")
        if self.provider.type not in {
            "anthropic",
            "openai",
            "gemini",
            "ollama",
            "openai_compatible",
            "mock",
        }:
            raise ValueError(f"Unknown provider type: {self.provider.type!r}")

    def overridden(
        self,
        *,
        workspace: str | None = None,
        tools: list[str] | None = None,
        model: str | None = None,
        provider_type: ProviderType | None = None,
        provider_options: dict[str, Any] | None = None,
    ) -> "Config":
        """Return a new Config with per-request overrides applied.

        The original config instance is never mutated; ``dataclasses.replace``
        copies each section. This is what the HTTP API uses to honour
        request-level ``workspace`` / ``tools`` / ``model`` / ``provider``
        without touching the server-wide defaults.
        """
        cfg = replace(self)
        if workspace is not None:
            ws = os.path.abspath(os.path.expanduser(workspace))
            cfg.workspace = ws
        if tools is not None:
            cfg.tools = replace(self.tools, enabled=list(tools))
        if model is not None:
            cfg.provider = replace(self.provider, model=model)
        if provider_type is not None:
            cfg.provider = replace(cfg.provider, type=provider_type)
        if provider_options:
            valid = {f for f in ProviderConfig.__dataclass_fields__}
            opts = {k: v for k, v in provider_options.items() if k in valid and v is not None}
            cfg.provider = replace(cfg.provider, **opts)
        return cfg

    def to_dict(self, redact_keys: bool = True) -> dict[str, Any]:
        d = {
            "workspace": self.workspace,
            "server": {
                "host": self.server.host,
                "port": self.server.port,
                "max_concurrency": self.server.max_concurrency,
            },
            "auth": {
                "enabled": self.auth.enabled,
                "token_env": self.auth.token_env,
                "token_set": bool(self.auth.resolve_token()),
                "token": "***" if redact_keys and self.auth.token else self.auth.token,
            },
            "provider": {
                "type": self.provider.type,
                "model": self.provider.model,
                "base_url": self.provider.base_url,
                "api_key_env": self.provider.api_key_env,
                "api_key": (
                    "***" if redact_keys and self.provider.api_key else self.provider.api_key
                ),
                "max_tokens": self.provider.max_tokens,
                "temperature": self.provider.temperature,
                "top_p": self.provider.top_p,
                "thinking": self.provider.thinking,
                "thinking_budget_tokens": self.provider.thinking_budget_tokens,
                "timeout_seconds": self.provider.timeout_seconds,
            },
            "prompt": {
                "system": self.prompt.system,
                "appends": self.prompt.appends,
            },
            "tools": {
                "enabled": self.tools.enabled,
                "max_result_chars": self.tools.max_result_chars,
                "bash_timeout_ms": self.tools.bash_timeout_ms,
                "options": self.tools.options,
            },
            "limits": {
                "max_iterations": self.limits.max_iterations,
                "max_tokens_total": self.limits.max_tokens_total,
                "max_budget_usd": self.limits.max_budget_usd,
                "max_context_tokens": self.limits.max_context_tokens,
                "pricing": {
                    m: {
                        "input": p.input,
                        "output": p.output,
                        "cache_read": p.cache_read,
                        "cache_write": p.cache_write,
                    }
                    for m, p in self.limits.pricing.items()
                },
            },
            "logging": {"level": self.logging.level},
        }
        return d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _first(d: dict[str, Any], *keys: str, default: Any) -> Any:
    for k in keys:
        if k in d:
            return d[k]
    return default


def _build_dataclass(cls: Any, d: Optional[dict[str, Any]]) -> Any:
    """Construct a dataclass from a dict, silently ignoring unknown keys."""
    if not d:
        return cls()
    valid = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
    kwargs: dict[str, Any] = {}
    for k, v in d.items():
        if k not in valid:
            continue
        # Empty YAML containers (a block with only comments) parse as None;
        # coerce those to their field default so we don't store None.
        if v is None and cls.__dataclass_fields__[k].default_factory is not dataclasses.MISSING:  # type: ignore[attr-defined]
            continue
        kwargs[k] = v
    return cls(**kwargs)


def _build_pricing(d: Optional[dict[str, Any]]) -> dict[str, PricingModel]:
    out: dict[str, PricingModel] = {}
    for model, spec in (d or {}).items():
        if isinstance(spec, (int, float)):
            out[model] = PricingModel(input=float(spec), output=float(spec))
            continue
        if isinstance(spec, dict):
            out[model] = PricingModel(
                input=float(spec.get("input", 0)),
                output=float(spec.get("output", 0)),
                cache_read=float(spec.get("cache_read", 0)),
                cache_write=float(spec.get("cache_write", 0)),
            )
    return out
