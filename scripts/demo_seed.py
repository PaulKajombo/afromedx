"""Seed the local index with representative Malawi-guideline excerpts.

These are clearly-labelled SAMPLE excerpts mirroring the structure of the real
guidelines (with page/section metadata) so retrieval, grounding, citations and
evaluation work end-to-end before official PDFs are added. Replace with
ingest_pdf() of authoritative PDFs (see data/guidelines/README.md).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings
from app.ingestion.pdf_loader import PageText
from app.ingestion.pipeline import ingest_pages
from app.retrieval.embedder import get_embedder
from app.retrieval.store import VectorStore

SAMPLE_DOCS = [
    {
        "doc_id": "mw-malaria-2023",
        "title": "Malawi Guidelines for the Treatment of Malaria (SAMPLE)",
        "edition": "2023 SAMPLE",
        "publication_year": 2023,
        "source": "Malawi Ministry of Health (sample excerpt for MVP dev)",
        "pages": [
            PageText(page=12, text=(
                "SEVERE MALARIA\n"
                "Severe malaria in adults is a medical emergency. "
                "Recommended treatment: Give IV artesunate 2.4 mg/kg at 0, 12 and 24 hours, "
                "then once daily until oral medication can be taken. "
                "Follow with a full 3-day course of artemether-lumefantrine. "
                "Monitor for acute kidney injury, severe anaemia (Hb < 7 g/dL), cerebral malaria, "
                "hypoglycaemia and respiratory distress. Refer to high dependency care."
            )),
            PageText(page=13, text=(
                "SEVERE MALARIA continued\n"
                "If IV artesunate is unavailable, give IM artemether "
                "3.2 mg/kg loading dose then 1.6 mg/kg daily as interim treatment and refer urgently. "
                "Do NOT use oral monotherapy for severe malaria. Give IV fluids cautiously; avoid fluid overload. "
                "Transfuse if Hb < 7 g/dL with signs of severe anaemia."
            )),
            PageText(page=20, text=(
                "UNCOMPLICATED MALARIA\n"
                "First-line treatment in adults: artemether-lumefantrine "
                "20/120 mg, 4 tablets twice daily for 3 days. Confirm with mRDT or microscopy. "
                "Advise completion of the full course and use of insecticide-treated nets."
            )),
        ],
    },
    {
        "doc_id": "mstg-6e-2023",
        "title": "Malawi Standard Treatment Guidelines (SAMPLE)",
        "edition": "6th Edition SAMPLE",
        "publication_year": 2023,
        "source": "Malawi Ministry of Health (sample excerpt for MVP dev)",
        "pages": [
            PageText(page=123, text=(
                "MALARIA — SEVERE MALARIA, ADULT\n"
                "Recommended treatment: IV artesunate 2.4 mg/kg "
                "at 0, 12 and 24 hours then daily. Supportive care: monitor vital signs, blood glucose, "
                "urine output and haemoglobin. Treat hypoglycaemia with IV dextrose. "
                "Contraindications: known hypersensitivity to artemisinins."
            )),
            PageText(page=45, text=(
                "TUBERCULOSIS — DIAGNOSIS\n"
                "Diagnostic criteria for pulmonary TB in adults: "
                "cough for 2 weeks or more plus one of: fever, night sweats, weight loss, or haemoptysis; "
                "confirm with GeneXpert MTB/RIF or sputum smear microscopy and chest X-ray. "
                "All presumptive TB cases should be offered HIV testing."
            )),
            PageText(page=46, text=(
                "TUBERCULOSIS — TREATMENT\n"
                "New drug-susceptible pulmonary TB: 2 months of "
                "isoniazid, rifampicin, pyrazinamide and ethambutol (2HRZE) followed by 4 months "
                "of isoniazid and rifampicin (4HR). Support adherence and monitor liver function."
            )),
        ],
    },
    {
        "doc_id": "mw-hiv-2022",
        "title": "Malawi Integrated HIV Guidelines (SAMPLE)",
        "edition": "2022 SAMPLE",
        "publication_year": 2022,
        "source": "Malawi Ministry of Health (sample excerpt for MVP dev)",
        "pages": [
            PageText(page=30, text=(
                "ART INITIATION\n"
                "Start ART as soon as possible, ideally within 7 days of HIV diagnosis, "
                "in all adults regardless of CD4 count or WHO stage, after readiness assessment. "
                "First-line regimen: tenofovir + lamivudine + dolutegravir (TLD) once daily. "
                "Screen for TB and cryptococcal infection before initiation when CD4 < 200."
            )),
            PageText(page=31, text=(
                "ART MONITORING\n"
                "Monitor viral load at 6 months after initiation, then annually. "
                "Virological failure is defined as viral load > 1000 copies/mL on two consecutive "
                "measurements after adherence support. Manage drug interactions with rifampicin by "
                "adjusting dolutegravir dosing per guideline table."
            )),
        ],
    },
]


def main() -> None:
    store = VectorStore(settings.index_dir, embedder=get_embedder(force_tfidf=settings.force_tfidf))
    store.load()
    for s in SAMPLE_DOCS:
        result = ingest_pages(
            s["pages"], doc_id=s["doc_id"], title=s["title"], edition=s["edition"],
            publication_year=s["publication_year"], source=s["source"], version="1",
        )
        store.add_document(result.document.to_dict(), [c.to_dict() for c in result.chunks])
        print(f"indexed {s['doc_id']}: {len(result.chunks)} chunks")
    store.save()
    print(f"total chunks: {store.count()} (embedder={store.embedder.name})")


if __name__ == "__main__":
    main()
