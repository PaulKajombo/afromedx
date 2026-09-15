"""HTTP API: search (retrieve + grounded answer), documents, ingest status.

Privacy: queries are not persisted; only transient in-memory processing.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter
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


@router.post("/search", response_model=SearchResponse)
def api_search(req: SearchRequest):
    assert _store is not None, "index not initialised"
    hits = search(
        _store, req.query, top_k=req.top_k or settings.top_k,
        min_score=settings.min_score, semantic_weight=settings.semantic_weight,
        document_id=req.document_id,
    )
    provider = get_provider()
    ans = provider.generate(req.query, hits, _store.docs)
    passages = [
        {"text": h["chunk"].get("text", "")[:800], "score": round(h["score"], 3),
         "section": h["chunk"].get("section", ""), "page": h["chunk"].get("page"),
         "document_id": h["chunk"].get("document_id")}
        for h in hits
    ]
    return SearchResponse(
        answer={"title": ans.title, "body": ans.body_markdown,
                "key_points": ans.key_points, "citations": ans.citations,
                "abstained": ans.abstained, "grounded": ans.grounded,
                "provider": provider.name},
        passages=passages,
        meta={"top_k": req.top_k, "min_score": settings.min_score,
              "embedder": getattr(_store.embedder, "name", "?"),
              "chunks_indexed": _store.count()},
    )


@router.get("/documents")
def api_documents():
    assert _store is not None
    return {"documents": list(_store.docs.values()), "chunks": _store.count()}


@router.get("/health")
def api_health():
    n = _store.count() if _store is not None else 0
    return {"status": "ok", "chunks": n,
            "provider": settings.llm_provider,
            "embedder": getattr(_store.embedder, "name", "?") if _store else "?"}
