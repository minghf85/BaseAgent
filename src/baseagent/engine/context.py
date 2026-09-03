"""Optional in-process token estimation for pre-call context guarding.

Uses ``tiktoken`` (cl100k_base) as a cross-provider proxy to estimate the number
of tokens the assembled conversation will consume. This powers the
``limits.max_context_tokens`` pre-call guard so the engine can fail fast instead
of sending an over-long, expensive request. Estimation is approximate — exact
accounting comes from the provider-reported usage.
"""

from __future__ import annotations

import json
from typing import Optional

from ..types import Message

_cl100k = None


def _get_encoder():
    global _cl100k
    if _cl100k is None:
        import tiktoken

        _cl100k = tiktoken.get_encoding("cl100k_base")
    return _cl100k


def estimate_tokens_for_blocks(blocks: list) -> int:
    """Cheap estimate: ~4 chars/token for text, JSON-ish input is counted via
    its serialized length. Falls back to char/4 if tiktoken is unavailable."""
    encoder = None
    try:
        encoder = _get_encoder()
    except Exception:
        encoder = None

    total = 0
    for block in blocks:
        if block.type == "text" and block.text:
            total += _count(encoder, block.text)
        elif block.type == "tool_use":
            total += _count(encoder, f"{block.name} {json.dumps(block.input, ensure_ascii=False, separators=(',', ':'))}")
        elif block.type == "tool_result":
            content = block.content
            if isinstance(content, str):
                total += _count(encoder, content)
            elif isinstance(content, list):
                for c in content:
                    text = c.get("text", "") if isinstance(c, dict) else str(c)
                    total += _count(encoder, text)
    return total


def estimate_context_tokens(messages: list[Message], system: str) -> int:
    total = _count(_enc_or_none(), system) if system else 0
    for msg in messages:
        total += estimate_tokens_for_blocks(msg.content)
    return total


def _count(encoder, text: str) -> int:
    if encoder is not None:
        try:
            return len(encoder.encode(text))
        except Exception:
            return max(1, len(text) // 4)
    return max(1, len(text) // 4)


def _enc_or_none():
    try:
        return _get_encoder()
    except Exception:
        return None
