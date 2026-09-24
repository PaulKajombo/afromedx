"""HTTP API: search (retrieve + grounded answer), documents, guideline PDFs.

Privacy: queries are not persisted; only transient in-memory processing.
"""
from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ..config import settings
from ..llm.factory import get_provider
from ..retrieval.search import search

router = APIRouter()
_store = None  # injected by main.py lifespan


def set_store(store) -> None:
    global _store
    _store = store


class SearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=500)
    top_k: int = Field(default=6, ge=1, le=10)
    document_id: str | None = None


class SearchResponse(BaseModel):
    answer: dict[str, Any]
    passages: list[dict[str, Any]]
    meta: dict[str, Any]


# ---- answer cache (perf): repeat questions skip retrieval + LLM entirely ----
# Keyed by normalized query + request shape + provider. TTL + LRU bound;
# cleared on /api/reload. Identical requests return byte-identical answers.
_CACHE_MAX = 128
_answer_cache: dict[tuple, tuple[float, dict, list]] = {}


def _cache_key(query: str, top_k: int, document_id: str | None,
               provider_name: str) -> tuple:
    from ..retrieval.search import norm_text
    return (norm_text(query), top_k, document_id or "", provider_name)


def _cache_get(key: tuple) -> tuple[dict, list] | None:
    ttl = float(getattr(settings, "cache_ttl", 600) or 0)
    if ttl <= 0:
        return None
    import time
    entry = _answer_cache.get(key)
    if entry is None:
        return None
    ts, answer, passages = entry
    if time.monotonic() - ts > ttl:
        _answer_cache.pop(key, None)
        return None
    return answer, passages


def _cache_put(key: tuple, answer: dict, passages: list) -> None:
    ttl = float(getattr(settings, "cache_ttl", 600) or 0)
    if ttl <= 0:
        return
    import time
    if key in _answer_cache:
        _answer_cache.pop(key)
    while len(_answer_cache) >= _CACHE_MAX:
        _answer_cache.pop(next(iter(_answer_cache)))
    _answer_cache[key] = (time.monotonic(), answer, passages)


@router.post("/search", response_model=SearchResponse)
def api_search(req: SearchRequest):
    assert _store is not None, "index not initialised"
    provider = get_provider()
    key = _cache_key(req.query, req.top_k or settings.top_k,
                     req.document_id, provider.name)
    hit = _cache_get(key)
    if hit is not None:
        answer, passages = hit
        cached = True
    else:
        hits = search(
            _store, req.query, top_k=req.top_k or settings.top_k,
            min_score=settings.min_score, semantic_weight=settings.semantic_weight,
            document_id=req.document_id,
        )
        ans = provider.generate(req.query, hits, _store.docs)
        passages = [
            {"text": h["chunk"].get("text", "")[:800], "score": round(h["score"], 3),
             "section": h["chunk"].get("section", ""), "page": h["chunk"].get("page"),
             "document_id": h["chunk"].get("document_id")}
            for h in hits
        ]
        answer = {"title": ans.title, "body": ans.body_markdown,
                  "key_points": ans.key_points, "citations": ans.citations,
                  "abstained": ans.abstained, "grounded": ans.grounded,
                  "provider": provider.name}
        _cache_put(key, answer, passages)
        cached = False
    return SearchResponse(
        answer=answer,
        passages=passages,
        meta={"top_k": req.top_k, "min_score": settings.min_score,
              "embedder": getattr(_store.embedder, "name", "?"),
              "chunks_indexed": _store.count(), "cached": cached},
    )


def clear_answer_cache() -> None:
    """Drop all cached answers (called on index reload)."""
    _answer_cache.clear()


@router.get("/documents")
def api_documents():
    assert _store is not None
    return {"documents": list(_store.docs.values()), "chunks": _store.count()}


def _resolve_pdf_path(stored_path: str) -> str | None:
    """Resolve a guideline PDF, tolerating relocated deployments.

    The index stores the ingest-time absolute path (e.g. a Windows path),
    which does not exist inside a Linux container. Fall back to the file's
    basename inside known guideline directories. Returns None when absent.
    """
    if stored_path and os.path.splitext(stored_path)[1].lower() == ".pdf" \
            and os.path.isfile(stored_path):
        return stored_path
    name = (stored_path or "").replace("\\", "/").split("/")[-1]
    if not name.lower().endswith(".pdf") or not name:
        return None
    for directory in ("Malawi Guidelines", "guidelines",
                      getattr(settings, "guideline_dir", "")):
        if not directory:
            continue
        candidate = os.path.join(directory, name)
        if os.path.isfile(candidate):
            return candidate
    return None


@router.get("/guideline/{document_id}/pdf")
def api_guideline_pdf(document_id: str):
    """Serve a guideline PDF, keyed by document id only (no path input).

    Lets the UI open the exact source at the cited page. Returns 404 when the
    document is unknown or its PDF is not present on this server.
    """
    assert _store is not None
    doc = (_store.docs or {}).get(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="unknown document")
    path = _resolve_pdf_path(doc.get("file_path", ""))
    if not path:
        raise HTTPException(status_code=404,
                            detail="source PDF not available on this server")
    return FileResponse(path, media_type="application/pdf",
                        headers={"Content-Disposition":
                                 f'inline; filename="{document_id}.pdf"'})


@router.get("/health")
def api_health():
    n = _store.count() if _store is not None else 0
    return {"status": "ok", "chunks": n,
            "provider": settings.llm_provider,
            "embedder": getattr(_store.embedder, "name", "?") if _store else "?"}
