"""Display-cleanup tests: model output must not repeat UI-provided structure."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.llm.openai_provider import _clean_answer


def test_structural_labels_stripped():
    body = ("**Answer:**\nGive IV artesunate 2.4 mg/kg [E1].\n"
            "**Key Points**\n**Guideline:** Malaria Guideline, p. 34.")
    bullets = ["**Answer:**", "Give IV artesunate 2.4 mg/kg",
               "**Key Points**",
               "**Guideline:** Malaria Guideline, p. 34."]
    clean_body, clean_points = _clean_answer(body, bullets)
    assert clean_body == "Give IV artesunate 2.4 mg/kg."
    # The surviving bullet echoes the body line (minus its [E1] tag), so it is
    # dropped as duplication rather than shown twice on screen.
    assert clean_points == []
    assert "Guideline" not in clean_body


def test_clinical_content_untouched():
    body = "Give parenteral artesunate [E1].\nMonitor haemoglobin [E2]."
    bullets = ["IV artesunate 2.4 mg/kg at 0, 12 and 24 hours",
               "Contraindicated in known hypersensitivity"]
    clean_body, clean_points = _clean_answer(body, bullets)
    assert clean_body == "Give parenteral artesunate.\nMonitor haemoglobin."
    assert clean_points == bullets


def test_empty_after_clean_falls_back():
    clean_body, clean_points = _clean_answer("**Answer:**", ["**Key Points**"])
    assert clean_body == ""
    assert clean_points == []


def test_near_duplicate_bullets_dropped():
    body = "1. Give **artesunate** 2.4 mg/kg at 0, 12 and 24 hours [E1]."
    bullets = ["Give artesunate 2.4 mg/kg at 0, 12 and 24 hours [E1]",
               "Monitor for acute kidney injury [E2]"]
    clean_body, clean_points = _clean_answer(body, bullets)
    assert clean_body == "1. Give **artesunate** 2.4 mg/kg at 0, 12 and 24 hours."
    assert clean_points == ["Monitor for acute kidney injury"]


def test_topical_overlap_kept():
    body = "1. Parenteral **artesunate** is first-line for severe malaria [E1]."
    bullets = ["Artesunate dosing differs in children under 20 kg [E2]"]
    _, clean_points = _clean_answer(body, bullets)
    assert clean_points == ["Artesunate dosing differs in children under 20 kg"]


def test_numbering_renumbered_sequentially():
    from app.llm.openai_provider import _clean_answer as clean
    body = "1. Give **artesunate** [E1].\nSome continuation.\n1. Switch to ACT [E1].\n1. Monitor [E1]."
    clean_body, _ = clean(body, [])
    assert clean_body.splitlines() == [
        "1. Give **artesunate**.",
        "Some continuation.",
        "2. Switch to ACT.",
        "3. Monitor.",
    ]


def test_evidence_ids_stripped():
    from app.llm.openai_provider import _clean_answer as clean
    body = "Give **artesunate** 2.4 mg/kg [E1] and monitor [E2][E3]."
    bullets = ["Artesunate dosing [E1]", "Monitor glucose [ E9 ]"]
    clean_body, clean_points = _clean_answer(body, bullets)
    assert "[E" not in clean_body
    assert all("[E" not in p for p in clean_points)
    assert "2.4 mg/kg" in clean_body
