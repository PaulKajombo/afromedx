"""Answer cache + precomputed-structure equivalence.

- keyword_scores with store caches must equal the inline computation exactly.
- search() must work with and without caches present.
- Repeat /api/search calls must hit the TTL cache (no recompute, no LLM call);
  different queries miss; clear_answer_cache() invalidates; TTL 0 disables.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.api import routes
from app.api.routes import SearchRequest, api_search, clear_answer_cache
from app.config import settings
from app.ingestion.pdf_loader import PageText
from app.ingestion.pipeline import ingest_pages
from app.retrieval.embedder import TfidfEmbedder
from app.retrieval.search import keyword_scores, search
from app.retrieval.store import VectorStore


def _seed_store(tmp_path):
    store = VectorStore(str(tmp_path), embedder=TfidfEmbedder())
    pages = [PageText(page=12, text=(
        "SEVERE MALARIA\nSevere malaria in adults is an emergency. "
        "Give IV artesunate 2.4 mg/kg at 0, 12 and 24 hours.")),
        PageText(page=45, text=(
        "TUBERCULOSIS DIAGNOSIS\nCough for 2 weeks or more; "
        "confirm with GeneXpert MTB/RIF."))]
    res = ingest_pages(pages, doc_id="doc", title="Guideline")
    store.add_document(res.document.to_dict(), [c.to_dict() for c in res.chunks])
    return store


def test_keyword_cached_matches_inline(tmp_path):
    store = _seed_store(tmp_path)
    texts = [c.get("text", "") for c in store.chunks]
    for q in ["severe malaria artesunate dose", "GeneXpert tuberculosis",
              "bicycle tyre repair"]:
        inline = keyword_scores(q, texts)
        cached = keyword_scores(q, None, token_sets=store.chunk_token_sets,
                                df=store.df_body)
        np.testing.assert_allclose(cached, inline, rtol=1e-5, atol=1e-7)


def test_keyword_meta_cached_matches_inline(tmp_path):
    store = _seed_store(tmp_path)
    docs = store.docs
    meta = [f"{docs.get(c.get('document_id'), {}).get('title', '')} "
            f"{c.get('section', '')} {c.get('subsection', '')}" for c in store.chunks]
    inline = keyword_scores("severe malaria", meta)
    cached = keyword_scores("severe malaria", None,
                            token_sets=store.meta_token_sets, df=store.df_meta)
    np.testing.assert_allclose(cached, inline, rtol=1e-5, atol=1e-7)


def test_search_without_caches_still_works(tmp_path):
    store = _seed_store(tmp_path)
    del store.chunk_token_sets
    del store.meta_token_sets
    del store.corpus_terms
    del store.term_buckets
    del store.df_body
    del store.df_meta
    hits = search(store, "severe malaria artesunate", top_k=2, min_score=0.01)
    assert hits and "artesunate" in hits[0]["chunk"]["text"].lower()


def _use_store(store):
    prev = routes._store
    routes.set_store(store)
    clear_answer_cache()
    return prev


def test_answer_cached_second_time(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "stub")
    store = _seed_store(tmp_path)
    prev = _use_store(store)
    try:
        req = SearchRequest(query="severe malaria artesunate dose?", top_k=3)
        r1 = api_search(req)
        assert r1.meta["cached"] is False
        assert not r1.answer["abstained"]
        r2 = api_search(req)
        assert r2.meta["cached"] is True
        assert r2.answer == r1.answer
        assert r2.passages == r1.passages
    finally:
        routes.set_store(prev)
        clear_answer_cache()


def test_different_query_misses(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "stub")
    store = _seed_store(tmp_path)
    prev = _use_store(store)
    try:
        r1 = api_search(SearchRequest(query="severe malaria artesunate?", top_k=3))
        r2 = api_search(SearchRequest(query="GeneXpert tuberculosis?", top_k=3))
        assert r1.meta["cached"] is False
        assert r2.meta["cached"] is False
    finally:
        routes.set_store(prev)
        clear_answer_cache()


def test_clear_invalidates(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "stub")
    store = _seed_store(tmp_path)
    prev = _use_store(store)
    try:
        req = SearchRequest(query="severe malaria artesunate?", top_k=3)
        api_search(req)
        assert api_search(req).meta["cached"] is True
        clear_answer_cache()
        assert api_search(req).meta["cached"] is False
    finally:
        routes.set_store(prev)
        clear_answer_cache()


def test_ttl_zero_disables(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "stub")
    monkeypatch.setattr(settings, "cache_ttl", 0)
    store = _seed_store(tmp_path)
    prev = _use_store(store)
    try:
        req = SearchRequest(query="severe malaria artesunate?", top_k=3)
        assert api_search(req).meta["cached"] is False
        assert api_search(req).meta["cached"] is False
    finally:
        monkeypatch.undo()
        routes.set_store(prev)
        clear_answer_cache()
