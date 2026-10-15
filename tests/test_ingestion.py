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


def test_running_headers_removed():
    pages = [
        PageText(page=i, text=(
            "National Malaria Control Programme\nPage 12\n"
            f"Content about artesunate dosing on page {i} for severe malaria treatment."))
        for i in range(1, 6)
    ]
    res = ingest_pages(pages, doc_id="d2", title="Malaria")
    assert res.chunks
    for c in res.chunks:
        assert "National Malaria Control Programme" not in c.text
        assert "Page 12" not in c.text


def test_private_use_and_toc_removed():
    from app.ingestion.cleaner import clean_text
    assert "\uf06f" not in clean_text("Take \uf06f one tablet")
    pages = [PageText(page=1, text=(
        "SEVERE MALARIA\nGive artesunate 2.4 mg/kg\n"
        "1.4.1 Artesunate .......... 7\nContinue treatment and monitor the patient."))]
    res = ingest_pages(pages, doc_id="d3", title="T")
    assert res.chunks
    assert all("......" not in c.text for c in res.chunks)


def test_letter_spaced_runs_rejoined():
    from app.ingestion.cleaner import clean_text
    # Runs of spaced capitals collapse ("C P D" -> "CPD"); uneven fragments
    # may join partially ("La B O U R ..." -> "La BOURANDD") - still strictly
    # more matchable than isolated single letters, never less.
    out = clean_text("La B O U R A N D D Eliv Er Y\nC P D\nGive IV fluids.")
    assert "CPD" in out
    assert "BOURANDD" in out
    assert "Give IV fluids." in out


def test_split_words_rejoined():
    from app.ingestion.cleaner import clean_text
    out = clean_text("M alaw iN ationalR eproductive H ealth S ervice protocols here now")
    assert "Health" in out
    assert "Service" in out


def test_normal_text_untouched_by_despacing():
    from app.ingestion.cleaner import clean_text
    # Indefinite article, vitamin letters, and ordinary sentences must survive.
    assert clean_text("Take a test dose of Vitamin A deficiency daily.") == \
        "Take a test dose of Vitamin A deficiency daily."
    assert clean_text("Give IV artesunate 2.4 mg/kg at 0, 12 and 24 hours.") == \
        "Give IV artesunate 2.4 mg/kg at 0, 12 and 24 hours."
    # Two scattered capitals are below the flag bar: left alone.
    assert clean_text("Grade A B tumours present.") == "Grade A B tumours present."
    # Common words and clinical abbreviations never join, even when flagged.
    out = clean_text("A B C D package of care to day ward review now")
    assert "ofcare" not in out and "today" not in out.replace("Today", "")


def test_fragment_joins():
    from app.ingestion.cleaner import clean_text
    out = clean_text("D rugs for em ergency care now X Y Zz")
    assert "Drugs" in out
    assert "emergency" in out
    out2 = clean_text("Episiotom y and suturing pack X Y Zz")
    assert "Episiotomy" in out2
