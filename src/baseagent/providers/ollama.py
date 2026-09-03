"""Ollama / local model adapter.

Ollama ships an OpenAI-compatible HTTP server at ``http://localhost:11434/v1``
exposing ``/chat/completions`` with function-calling support. Rather than
duplicating the protocol, this adapter reuses :class:`OpenAIProvider` and only
changes the default base URL. Set config ``provider.base_url`` to point at your
Ollama (or any other local OpenAI-compatible endpoint).
"""

from __future__ import annotations

from ..config import ProviderConfig
from .openai import OpenAIProvider

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"


class OllamaProvider(OpenAIProvider):
    name = "ollama"

    def __init__(self, config: ProviderConfig):
        if not config.base_url:
            # Avoid mutating the shared config; build a local copy.
            config = ProviderConfig(**{**config.__dict__, "base_url": DEFAULT_OLLAMA_BASE_URL})
        super().__init__(config)
