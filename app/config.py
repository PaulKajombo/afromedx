"""Central configuration from environment. No secrets committed."""
from __future__ import annotations

import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def _get(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


class Settings:
    llm_provider: str = _get("AFROMEDX_LLM_PROVIDER", "stub").lower() or "stub"
    openai_api_key: str = _get("OPENAI_API_KEY", "")
    openai_base_url: str = _get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    openai_model: str = _get("OPENAI_MODEL", "gpt-4o-mini")
    top_k: int = int(_get("AFROMEDX_TOP_K", "6") or 6)
    min_score: float = float(_get("AFROMEDX_MIN_SCORE", "0.08") or 0.08)
    semantic_weight: float = float(_get("AFROMEDX_SEMANTIC_WEIGHT", "0.65") or 0.65)
    index_dir: str = _get("AFROMEDX_INDEX_DIR", "./data/index")
    guideline_dir: str = _get("AFROMEDX_GUIDELINE_DIR", "./data/guidelines")
    force_tfidf: bool = (_get("AFROMEDX_FORCE_TFIDF", "0") == "1")


settings = Settings()
