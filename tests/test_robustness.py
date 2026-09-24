"""Robustness: phrasing convergence, spelling tolerance, question-derived titles.

- "drugs you cannot give in asthma" and "what drugs can you not give in
  asthma" must retrieve the same evidence (contraction normalization).
- A misspelled disease ("asam") must still retrieve (fuzzy bridging).
- Genuine gibberish and out-of-scope questions must still abstain (the
  NO SOURCE gate must not be weakened by tolerance).
- Answer titles come from the question, never from chunk metadata.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ingestion.pdf_loader import PageText
from app.ingestion.pipeline import ingest_pages
from app.llm.stub import StubProvider, answer_title
from app.retrieval.embedder import TfidfEmbedder
from app.retrieval.search import search, term_covered, tokenize
from app.retrieval.store import VectorStore


def _seed_store(tmp_path):
    store = VectorStore(str(tmp_path), embedder=TfidfEmbedder())
    pages = [PageText(page=22, text=(
        "ASTHMA — DRUGS TO AVOID\nAvoid non-selective beta blockers such as "
        "propranolol in asthma. Avoid NSAIDs including aspirin in "
        "aspirin-sensitive asthma. Use inhaled salbutamol for relief."))]
    res = ingest_pages(pages, doc_id="asthma-doc", title="Asthma Guideline")
    store.add_document(res.document.to_dict(), [c.to_dict() for c in res.chunks])
    return store


def test_contraction_phrasings_converge(tmp_path):
    store = _seed_store(tmp_path)
    hits_a = search(store, "drugs you cannot give in asthma", top_k=3, min_score=0.01)
    hits_b = search(store, "what drugs can you not give in asthma", top_k=3, min_score=0.01)
    assert hits_a and hits_b, "both phrasings must retrieve"
    assert hits_a[0]["chunk"]["document_id"] == "asthma-doc"
    assert hits_b[0]["chunk"]["document_id"] == "asthma-doc"


def test_tokenize_normalizes_contractions():
    assert "cannot" not in tokenize("drugs you cannot give")
    assert "not" in tokenize("drugs you cannot give")


def test_bare_topic_expansion():
    from app.retrieval.search import _is_bare_topic, expand_bare_topic
    assert _is_bare_topic("migraine headache")
    assert _is_bare_topic("drugs you cannot give in asthma")
    assert not _is_bare_topic("how do you treat migraine headache")
    assert not _is_bare_topic("What is the dose of IV artesunate?")
    assert not _is_bare_topic("First line treatment for uncomplicated malaria")
    expanded = expand_bare_topic("migraine headache")
    assert "treatment" in expanded and "migraine" in expanded


def test_bare_topic_ranks_management_content(tmp_path):
    store = VectorStore(str(tmp_path), embedder=TfidfEmbedder())
    # The treatment term lives in BODY text (headings alone never match the
    # keyword arm); expansion must promote this chunk over the definition one.
    pages = [PageText(page=10, text="MIGRAINE DEFINITION\nMigraine headache is a primary headache disorder."),
             PageText(page=11, text="MIGRAINE TREATMENT\nMigraine headache treatment: give paracetamol "
                                    "for migraine headache attacks.")]
    res = ingest_pages(pages, doc_id="mig", title="Migraine Guideline")
    store.add_document(res.document.to_dict(), [c.to_dict() for c in res.chunks])
    hits = search(store, "migraine headache", top_k=2, min_score=0.01)
    assert hits, "bare topic must retrieve"
    assert "paracetamol" in hits[0]["chunk"]["text"]


def test_misspelled_disease_retrieves(tmp_path):
    store = _seed_store(tmp_path)
    hits = search(store, "which drugs to avoid in asam", top_k=3, min_score=0.01)
    assert hits, "misspelled 'asam' should bridge to 'asthma'"
    assert hits[0]["chunk"]["document_id"] == "asthma-doc"
    assert term_covered("asam", {"asthma", "drugs"})


def test_double_deviation_retrieves(tmp_path):
    # Verb form ("avoided") + typo ("asam"): two half-bridges (0.5+0.5)
    # carry as much intent as one exact term and must pass the gate.
    store = _seed_store(tmp_path)
    hits = search(store, "which drugs should be avoided in asam", top_k=3, min_score=0.01)
    assert hits, "verb-form + typo should still retrieve"
    assert hits[0]["chunk"]["document_id"] == "asthma-doc"


def test_gibberish_still_abstains(tmp_path):
    store = _seed_store(tmp_path)
    assert search(store, "zxqvr blorpt treatment", top_k=3, min_score=0.01) == []


def test_out_of_scope_still_abstains(tmp_path):
    store = _seed_store(tmp_path)
    assert search(store, "how do i repair a bicycle tyre", top_k=3, min_score=0.01) == []


def test_title_comes_from_question():
    chunk = {"id": "d::p1::0", "document_id": "d", "page": 1,
             "section": "HOURS)", "subsection": "",
             "text": "Avoid non-selective beta blockers such as propranolol in asthma."}
    ans = StubProvider().generate("drugs you cannot give in asthma",
                                  [{"chunk": chunk, "score": 0.5}], {})
    assert ans.title == "Drugs you cannot give in asthma"
    assert "HOURS" not in ans.title


def test_answer_title_edges():
    assert answer_title("") == "Guideline evidence"
    assert answer_title("   ") == "Guideline evidence"
    assert answer_title("dose?") == "Dose?"
    long_q = "x" * 200
    assert len(answer_title(long_q)) <= 120
    assert answer_title(long_q).endswith("...")
