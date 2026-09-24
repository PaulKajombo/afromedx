"""Edition policy: superseded docs excluded, newer editions preferred.

- paediatrics-handbook (3rd ed 2008) and obgyn (v3.0 2017) are superseded and
  must never be retrieved or cited (query-time exclusion; index untouched).
- Among dated docs, the newer one wins ties via a small recency bonus.
- Numbered answer items ("1. ...") survive the display sanitizer.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ingestion.pdf_loader import PageText
from app.ingestion.pipeline import ingest_pages
from app.llm.openai_provider import _clean_answer
from app.retrieval.embedder import TfidfEmbedder
from app.retrieval.search import doc_year, recency_bonus, search
from app.retrieval.store import VectorStore


def _seed_store(tmp_path, docs):
    store = VectorStore(str(tmp_path), embedder=TfidfEmbedder())
    for doc_id, title, page, text in docs:
        res = ingest_pages([PageText(page=page, text=text)],
                           doc_id=doc_id, title=title)
        store.add_document(res.document.to_dict(), [c.to_dict() for c in res.chunks])
    return store


def test_superseded_docs_excluded_by_default(tmp_path):
    store = _seed_store(tmp_path, [
        ("obgyn", "Old OBGYN", 82, "MALARIA IN PREGNANCY\nGive artesunate for severe malaria."),
        ("obs-gynae-2023", "New OBGYN", 40, "MALARIA IN PREGNANCY\nGive IV artesunate 2.4 mg/kg."),
    ])
    hits = search(store, "malaria in pregnancy artesunate", top_k=6, min_score=0.01)
    assert hits, "expected hits from the current edition"
    assert all(h["chunk"]["document_id"] != "obgyn" for h in hits)
    assert any(h["chunk"]["document_id"] == "obs-gynae-2023" for h in hits)


def test_exclusion_param_override(tmp_path):
    store = _seed_store(tmp_path, [
        ("obgyn", "Old OBGYN", 82, "MALARIA IN PREGNANCY\nGive artesunate for severe malaria."),
    ])
    assert search(store, "malaria pregnancy", top_k=3, min_score=0.01) == []
    hits = search(store, "malaria pregnancy", top_k=3, min_score=0.01,
                  exclude_doc_ids=set())
    assert hits and hits[0]["chunk"]["document_id"] == "obgyn"


def test_recency_bonus_prefers_newer_on_ties(tmp_path):
    # Note: key terms live in the BODY text here. Single-page seed headings
    # are dropped by the front-matter rule, so heading-only terms would vanish
    # (ingestion behavior, out of scope for this retrieval test).
    body = "Severe malaria treatment.\nGive IV artesunate 2.4 mg/kg for severe malaria."
    store = _seed_store(tmp_path, [
        ("doc-2020", "Old Guideline", 10, body),
        ("doc-2025", "New Guideline", 10, body),
    ])
    store.docs["doc-2020"]["publication_year"] = 2020
    store.docs["doc-2025"]["publication_year"] = 2025
    hits = search(store, "severe malaria artesunate dose", top_k=2, min_score=0.01,
                  max_per_doc=0, collapse_dupes=False)
    assert [h["chunk"]["document_id"] for h in hits] == ["doc-2025", "doc-2020"]


def test_recency_bonus_units():
    docs = {"new": {"publication_year": 2025}, "old": {"publication_year": 2020},
            "nodate": {}}
    assert recency_bonus("nodate", docs, 0.05) == 0.0
    assert recency_bonus("new", docs, 0.0) == 0.0
    assert recency_bonus("new", docs, 0.05) > recency_bonus("old", docs, 0.05) >= 0.0
    assert doc_year("missing", docs) is None


def test_numbered_items_survive_sanitizer():
    # Numbered items with DISTINCT content survive; a bullet that merely
    # echoes a body line (minus its citation tag) is dropped as duplication.
    body = "1. Give **artesunate** 2.4 mg/kg [E1].\n2. Follow with oral ACT [E1]."
    bullets = ["1. Give **artesunate** 2.4 mg/kg [E1]", "**Answer:**",
               "2. Switch to oral ACT once the patient can swallow [E1]"]
    clean_body, clean_points = _clean_answer(body, bullets)
    assert clean_body == body
    assert clean_points == ["2. Switch to oral ACT once the patient can swallow [E1]"]
