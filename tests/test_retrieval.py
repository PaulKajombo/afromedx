"""Retrieval tests: natural-language variants retrieve the same severe-malaria passage."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ingestion.pdf_loader import PageText
from app.ingestion.pipeline import ingest_pages
from app.retrieval.embedder import TfidfEmbedder
from app.retrieval.search import search
from app.retrieval.store import VectorStore


def _seed_store(tmp_path):
    store = VectorStore(str(tmp_path), embedder=TfidfEmbedder())
    pages = [PageText(page=12, text="SEVERE MALARIA. Recommended treatment: IV artesunate 2.4 mg/kg at 0, 12 and 24 hours, then daily."),
             PageText(page=45, text="TUBERCULOSIS DIAGNOSIS. Cough 2 weeks; confirm with GeneXpert MTB/RIF.")]
    res = ingest_pages(pages, doc_id="doc", title="Guideline")
    store.add_document(res.document.to_dict(), [c.to_dict() for c in res.chunks])
    return store


def test_nl_variants_retrieve(tmp_path):
    store = _seed_store(tmp_path)
    for q in ["severe malaria adult",
              "How do I manage severe malaria in an adult?",
              "adult with severe malaria treatment",
              "IV artesunate dose adult"]:
        hits = search(store, q, top_k=3, min_score=0.01)
        assert hits, f"no hits for {q!r}"
        assert "artesunate" in hits[0]["chunk"]["text"].lower()


def test_no_result_below_gate(tmp_path):
    store = _seed_store(tmp_path)
    hits = search(store, "osteosarcoma pregnancy chemotherapy", top_k=3, min_score=0.9)
    assert hits == []
