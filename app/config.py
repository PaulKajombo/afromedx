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
    # Native Gemini provider (google-genai SDK). Key is never committed; see .env.example.
    gemini_api_key: str = _get("GEMINI_API_KEY", "")
    gemini_model: str = _get("GEMINI_MODEL", "gemini-3.7-flash") or "gemini-3.7-flash"
    gemini_temperature: float = float(_get("GEMINI_TEMPERATURE", "0.0") or 0.0)
    gemini_timeout_s: float = float(_get("GEMINI_TIMEOUT", "60") or 60)
    gemini_max_output_tokens: int = int(_get("GEMINI_MAX_OUTPUT_TOKENS", "2048") or 2048)
    # Bounded retry for transient Gemini failures (429/5xx only): retries after
    # the initial attempt (default 2 -> at most 3 total), exponential backoff
    # from the base delay in seconds (attempt delays: base, 2*base, ...).
    gemini_max_retries: int = int(_get("GEMINI_MAX_RETRIES", "2") or 2)
    gemini_retry_base_delay_s: float = float(_get("GEMINI_RETRY_BASE_DELAY", "1.0") or 1.0)
    # Native Mistral provider (mistralai SDK). Key is never committed; see .env.example.
    mistral_api_key: str = _get("MISTRAL_API_KEY", "")
    mistral_model: str = _get("MISTRAL_MODEL", "mistral-small-2603") or "mistral-small-2603"
    mistral_temperature: float = float(_get("MISTRAL_TEMPERATURE", "0.0") or 0.0)
    mistral_timeout_s: float = float(_get("MISTRAL_TIMEOUT", "60") or 60)
    mistral_max_retries: int = int(_get("MISTRAL_MAX_RETRIES", "2") or 2)
    mistral_retry_base_delay_s: float = float(_get("MISTRAL_RETRY_BASE_DELAY", "1.0") or 1.0)
    top_k: int = int(_get("AFROMEDX_TOP_K", "6") or 6)
    min_score: float = float(_get("AFROMEDX_MIN_SCORE", "0.08") or 0.08)
    semantic_weight: float = float(_get("AFROMEDX_SEMANTIC_WEIGHT", "0.65") or 0.65)
    # P1A: weight of section/subsection/document-title matches inside the
    # keyword arm: kw = (1-section_weight)*kw_text + section_weight*kw_meta.
    # Query-time only; SBERT vectors untouched. 0 disables.
    section_weight: float = float(_get("AFROMEDX_SECTION_WEIGHT", "0.35") or 0.35)
    # P1B: clinical abbreviation/verb-form synonyms (VL, TLD, started/start, ...).
    # 1 = on (default), 0 = off (for ablation). Applies to keyword scoring and
    # the in-scope/grounding gates, which share the synonym map.
    clinical_synonyms: bool = (_get("AFROMEDX_CLINICAL_SYNONYMS", "1") == "1")
    # P1C: max chunks per document in the returned top-K (diversity). Default 2.
    # Query-time only; index untouched. 0 (or negative) disables the cap.
    max_per_doc: int = int(_get("AFROMEDX_MAX_PER_DOC", "2") or 2)
    # P1D: collapse byte-different/text-identical chunks (same content under
    # different doc ids) at query time, keeping the highest-scoring copy.
    # Query-time only; corpus untouched. 1 = on (default), 0 = off (ablation).
    collapse_dupes: bool = (_get("AFROMEDX_DEDUP_COLLAPSE", "1") == "1")
    # Edition policy (title-page evidence documented in search.py):
    # superseded documents are never retrieved; recency_weight adds a small
    # [0, weight] bonus favouring newer editions (0 disables for ablation).
    excluded_docs: frozenset = frozenset(
        d.strip() for d in _get("AFROMEDX_EXCLUDE_DOCS",
                                "paediatrics-handbook,obgyn").split(",") if d.strip())
    recency_weight: float = float(_get("AFROMEDX_RECENCY_WEIGHT", "0.02") or 0.02)
    # Answer cache (perf): seconds a repeat answer is served without
    # re-running retrieval + LLM. 0 disables. Cleared on /api/reload.
    cache_ttl: float = float(_get("AFROMEDX_CACHE_TTL", "600") or 600)
    index_dir: str = _get("AFROMEDX_INDEX_DIR", "./data/index")
    guideline_dir: str = _get("AFROMEDX_GUIDELINE_DIR", "./data/guidelines")
    force_tfidf: bool = (_get("AFROMEDX_FORCE_TFIDF", "0") == "1")


settings = Settings()
