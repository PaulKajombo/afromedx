"""Grounding tests: stub abstains without evidence, cites with evidence, preserves doses."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.llm.stub import StubProvider


def _passages():
    chunk = {"id": "d::p12::0", "document_id": "d", "page": 12,
             "section": "Severe Malaria", "subsection": "",
             "text": "Recommended treatment: IV artesunate 2.4 mg/kg at 0, 12 and 24 hours, then daily."}
    return [{"chunk": chunk, "score": 0.8}]


def test_abstains_without_evidence():
    ans = StubProvider().generate("q", [], {})
    assert ans.abstained
    assert "couldn't find sufficient information" in ans.body_markdown


def test_abstains_when_passages_do_not_cover_question():
    # "repair" appears in the passage, but the question is about a bicycle tyre:
    # topical overlap alone must not yield an answer (NO SOURCE = NO ANSWER).
    chunk = {"id": "d::p119::0", "document_id": "d", "page": 119,
             "section": "Surgery", "subsection": "",
             "text": ("Repair is not an emergency: a lacerated extensor tendon "
                      "should be closed within 48 hours in theatre.")}
    ans = StubProvider().generate("how do i repair a bicycle tyre",
                                  [{"chunk": chunk, "score": 0.45}], {})
    assert ans.abstained


def test_cites_and_preserves_dose():
    ans = StubProvider().generate("artesunate dose?", _passages(),
                                  {"d": {"title": "Malaria Guideline", "edition": "2023", "publication_year": 2023}})
    assert not ans.abstained
    assert "2.4 mg/kg" in ans.body_markdown
    assert ans.citations and ans.citations[0]["page"] == 12
    assert ans.citations[0]["document"] == "Malaria Guideline"
