"""Provider factory: stub default (offline), gemini / mistral / openai_compatible when configured."""
from __future__ import annotations

from ..config import settings
from .base import LLMProvider
from .gemini_provider import GeminiProvider
from .mistral_provider import MistralProvider
from .openai_provider import OpenAICompatibleProvider
from .stub import StubProvider


def get_provider() -> LLMProvider:
    if settings.llm_provider == "gemini" and settings.gemini_api_key:
        return GeminiProvider(
            api_key=settings.gemini_api_key,
            model=settings.gemini_model,
            temperature=settings.gemini_temperature,
            timeout_s=settings.gemini_timeout_s,
            max_output_tokens=settings.gemini_max_output_tokens,
            max_retries=settings.gemini_max_retries,
            retry_base_delay_s=settings.gemini_retry_base_delay_s,
        )
    if settings.llm_provider == "mistral" and settings.mistral_api_key:
        return MistralProvider(
            api_key=settings.mistral_api_key,
            model=settings.mistral_model,
            temperature=settings.mistral_temperature,
            timeout_s=settings.mistral_timeout_s,
            max_retries=settings.mistral_max_retries,
            retry_base_delay_s=settings.mistral_retry_base_delay_s,
        )
    if settings.llm_provider in ("openai", "openai_compatible") and settings.openai_api_key:
        return OpenAICompatibleProvider(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model=settings.openai_model,
        )
    return StubProvider()
