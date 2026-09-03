"""Provider factory: resolve a :class:`~baseagent.providers.base.LLMProvider`
from configuration."""

from __future__ import annotations

from ..config import Config, ProviderType
from .base import LLMProvider
from .anthropic import AnthropicProvider
from .gemini import GeminiProvider
from .mock import MockProvider
from .ollama import OllamaProvider
from .openai import OpenAIProvider

_PROVIDERS: dict[str, type[LLMProvider]] = {
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
    "openai_compatible": OpenAIProvider,
    "gemini": GeminiProvider,
    "ollama": OllamaProvider,
    "mock": MockProvider,
}


def build_provider(config: Config) -> LLMProvider:
    """Instantiate the provider selected by ``config.provider.type``.

    Also validates that the provider-specific tool schema is what the adapter
    expects — Anthropic adapters need ``input_schema`` tools, OpenAI/Gemini
    adapters need their own shapes — which the engine supplies per provider.
    """
    ptype: ProviderType = config.provider.type
    cls = _PROVIDERS.get(ptype)
    if cls is None:
        raise ValueError(f"Unknown provider type: {ptype!r}")
    return cls(config.provider)


__all__ = ["build_provider", "_PROVIDERS"]
