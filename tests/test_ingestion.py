"""Ingestion unit tests: chunking keeps page/section metadata, never empty output."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ingestion.pdf_loader import PageText
from app.ingestion.pipeline import ingest_pages


def test_ingest_keeps_metadata():
    pages = [
        PageText(page=1, text="SEVERE MALARIA\nRecommended treatment: IV artesunate 2.4 mg/kg at 0, 12 and 24 hours."),
        PageText(page=2, text="TUBERCULOSIS — DIAGNOSIS\nCough for 2 weeks or more; confirm with GeneXpert MTB/RIF."),
    ]
    res = ingest_pages(pages, doc_id="d1", title="Doc", edition="1e", source="MOH")
    assert res.chunks, "expected chunks"
    assert all(c.page in (1, 2) for c in res.chunks)
    assert all(c.section for c in res.chunks)
    assert res.document.checksum == ""  # pages path sets no checksum
    assert all("artesunate" in c.text or "GeneXpert" in c.text or len(c.text) > 0 for c in res.chunks)


def test_ingest_empty_raises():
    import pytest
    with pytest.raises(ValueError):
        ingest_pages([PageText(page=1, text="   ")], doc_id="d", title="T")
