"""Provider factory: stub default (offline), openai_compatible when configured."""
from __future__ import annotations

from ..config import settings
from .base import LLMProvider
from .openai_provider import OpenAICompatibleProvider
from .stub import StubProvider


def get_provider() -> LLMProvider:
    if settings.llm_provider in ("openai", "openai_compatible") and settings.openai_api_key:
        return OpenAICompatibleProvider(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model=settings.openai_model,
        )
    return StubProvider()
