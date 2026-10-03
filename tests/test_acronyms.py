"""Clinical acronyms: each entry must bridge its doc (retrieval only).

One shared tiny store; each parametrised case queries the ACRONYM while the
seed doc contains only the EXPANSION. Unknown tokens pass through untouched.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ingestion.pdf_loader import PageText
from app.ingestion.pipeline import ingest_pages
from app.retrieval.acronyms import ACRONYMS, expand_acronyms
from app.retrieval.embedder import TfidfEmbedder
from app.retrieval.search import search
from app.retrieval.store import VectorStore

SEEDS = [
    ("cap", "community-acquired pneumonia CURB severity score antibiotics"),
    ("fbc", "full blood count shows haemoglobin five grams"),
    ("ors", "oral rehydration salts after each loose stool"),
    ("la", "artemether-lumefantrine four tablets twice daily"),
    ("hbv", "hepatitis b virus vaccination birth dose"),
    ("sti", "sexually transmitted infection screen with nucleic testing"),
    ("uss", "ultrasound shows free fluid abdomen"),
    ("cns", "central nervous system infection examination"),
    ("hcv", "hepatitis c virus direct acting antivirals"),
    ("ngt", "nasogastric tube feeding expressed milk"),
    ("muac", "mid-upper arm circumference below eleven point five"),
    ("lp", "lumbar puncture for headache diagnosis workup"),
    ("pcr", "polymerase chain reaction viral load testing"),
    ("rutf", "ready-to-use therapeutic food sachets appetite"),
    ("ckd", "chronic kidney disease creatinine referral criteria"),
    ("ecg", "electrocardiogram ST elevation heart attack"),
    ("ks", "kaposi sarcoma skin lesions antiretroviral therapy"),
    ("pcv", "pneumococcal conjugate vaccine infants schedule"),
    ("cxr", "chest x-ray breathing shows bilateral infiltrates"),
    ("csf", "cerebrospinal fluid protein glucose meningitis"),
    ("bp", "blood pressure cuff size measurement technique"),
    ("o2", "oxygen saturation probe finger monitoring"),
]

QUERIES = {
    "cap": "CAP adult score", "fbc": "FBC results interpretation",
    "ors": "ORS for cholera", "la": "LA malaria tablets",
    "hbv": "HBV vaccine newborn", "sti": "STI screening test",
    "uss": "USS abdomen fluid", "cns": "CNS infection signs",
    "hcv": "HCV treatment tablets", "ngt": "NGT feeding baby",
    "muac": "MUAC malnutrition cutoff", "lp": "LP headache diagnosis",
    "pcr": "PCR viral test", "rutf": "RUTF malnutrition sachets",
    "ckd": "CKD creatinine management", "ecg": "ECG chest pain",
    "ks": "KS skin lesions", "pcv": "PCV baby vaccine",
    "cxr": "CXR breathing difficulty", "csf": "CSF results meaning",
    "bp": "BP measurement technique", "o2": "O2 sats monitoring",
}


def _seed_all(tmp_path):
    store = VectorStore(str(tmp_path), embedder=TfidfEmbedder())
    for acro, text in SEEDS:
        res = ingest_pages([PageText(page=1, text=text.upper() + "\n" + text + ".")],
                           doc_id=f"doc-{acro}", title=f"Doc {acro}")
        store.add_document(res.document.to_dict(), [c.to_dict() for c in res.chunks])
    return store


@pytest.mark.parametrize("acro", sorted(ACRONYMS))
def test_each_acronym_bridges_its_doc(tmp_path, acro):
    assert acro in dict(SEEDS), f"seed missing for approved acronym {acro!r}"
    store = _seed_all(tmp_path)
    hits = search(store, QUERIES[acro], top_k=3, min_score=0.01)
    assert hits, f"no hits for {acro!r}"
    assert hits[0]["chunk"]["document_id"] == f"doc-{acro}"


def test_map_covers_approved_list():
    assert set(ACRONYMS) == set(dict(SEEDS))


def test_unknown_tokens_untouched():
    assert expand_acronyms("hello world") == "hello world"
    assert expand_acronyms("XYZQ unknown") == "XYZQ unknown"


def test_original_preserved_and_expansion_appended():
    out = expand_acronyms("management of CAP in adults")
    assert out.startswith("management of CAP in adults")
    assert "community-acquired pneumonia" in out
