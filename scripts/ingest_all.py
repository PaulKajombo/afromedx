"""Ingest every PDF in the Malawi Guidelines folder into the local index.

Replaces the SAMPLE seed with authoritative documents. Files with an identical
sha256 are ingested only once. A provenance manifest (file -> doc_id, checksum,
pages, chunk count) is written to data/guidelines/manifest.json so every future
citation can be traced back to an exact source file.

Run: uv run python scripts/ingest_all.py
     uv run python scripts/ingest_all.py --pdf-dir "Malawi Guidelines" --append
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone

logging.getLogger("pypdf").setLevel(logging.ERROR)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings
from app.ingestion.pdf_loader import extract_pages
from app.ingestion.pipeline import ingest_pages
from app.retrieval.embedder import get_embedder
from app.retrieval.store import VectorStore

DEFAULT_SOURCE = "Malawi Ministry of Health"

# filename -> (doc_id, title, edition, publication_year, source)
DOC_META: dict[str, tuple[str, str, str, int | None, str]] = {
    "2_NCD Job Aids-1 (1).pdf":
        ("ncd-job-aids", "Malawi NCD Job Aids", "", None, DEFAULT_SOURCE),
    "Blue_Book_Second_Edition.pdf":
        ("blue-book", "Malawi Integrated Guidelines for Clinical Management of HIV (Blue Book)",
         "Second Edition", None, DEFAULT_SOURCE),
    "Burns Manual 2nd-1.pdf":
        ("burns-manual", "Malawi Burns Manual", "2nd Edition", None, DEFAULT_SOURCE),
    "cancer_guidelines_v3.4_06052026.pdf":
        ("cancer-guidelines", "Malawi National Cancer Guidelines", "Version 3.4", 2026, DEFAULT_SOURCE),
    "Caring for Newborns and Children in the Community Manual for HSAs ICCM Manual April 2010.pdf":
        ("iccm-2010", "Caring for Newborns and Children in the Community (ICCM Manual for HSAs)",
         "April 2010", 2010, DEFAULT_SOURCE),
    "Clinical_Book_270906.pdf":
        ("clinical-book", "Malawi Clinical Book", "", None, DEFAULT_SOURCE),
    "COIN_Malawi_2022.pdf":
        ("coin-2022", "COIN Malawi", "", 2022, DEFAULT_SOURCE),
    "COIN_Training_Manual_2017.pdf":
        ("coin-training-2017", "COIN Training Manual", "", 2017, DEFAULT_SOURCE),
    "Guidelines_for_the_Prevention_and_Management_of_Hepatitis_B_and_C_in_Malawi_.pdf":
        ("hepatitis-bc-prevention",
         "Guidelines for the Prevention and Management of Hepatitis B and C in Malawi",
         "", None, DEFAULT_SOURCE),
    "ilide.info-malawi-surgical-handbook-pr_b2610471225e131003fc4134848b7160.pdf":
        ("surgical-handbook", "Malawi Surgical Handbook", "", None, DEFAULT_SOURCE),
    "Integrated Management of Newborn and Childhood Illness- 2021_compressed (1)_compressed.pdf":
        ("imnci-2021", "Integrated Management of Newborn and Childhood Illness (IMNCI)",
         "2021", 2021, DEFAULT_SOURCE),
    "Malaria Treatment Guidelines.pdf":
        ("malaria-treatment", "Malawi Malaria Treatment Guidelines",
         "6th Edition", 2025, DEFAULT_SOURCE),
    "Malawi Clinical HIV Guidelines 2022 edition 5 lowres.pdf":
        ("hiv-2022", "Malawi Clinical HIV Guidelines", "5th Edition", 2022, DEFAULT_SOURCE),
    "Malawi Integrated Management of Newborn and Childhood Illnesses Chartbooklet MoH August 2022.pdf":
        ("imnci-chartbooklet-2022", "Malawi IMNCI Chartbooklet", "MoH August 2022", 2022, DEFAULT_SOURCE),
    "Malawi obs and gynae 2023 protocols.pdf":
        ("obs-gynae-2023", "Malawi Obstetrics and Gynaecology Protocols", "2023", 2023, DEFAULT_SOURCE),
    "Malawi Paediatric NCD guidelines_FINAL_31.12.2024 (1).pdf":
        ("paediatric-ncd-2024", "Malawi Paediatric NCD Guidelines", "31.12.2024", 2024, DEFAULT_SOURCE),
    "Malawi Standard Treatment Guidelines (MSTG).pdf":
        ("mstg", "Malawi Standard Treatment Guidelines (MSTG)",
         "6th Edition", 2023, DEFAULT_SOURCE),
    "malawi-sti-guidelines-2025-5th-edition-version-3.pdf":
        ("sti-2025", "Malawi STI Guidelines", "5th Edition, Version 3", 2025, DEFAULT_SOURCE),
    "malawian_handbook_paediatrics.pdf":
        ("paediatrics-handbook", "Malawian Handbook of Paediatrics",
         "Third Edition (2008, SUPERSEDED)", 2008, DEFAULT_SOURCE),
    "Malawi_Paediatric_Protocols_2018.pdf":
        ("paediatric-protocols-2018", "Malawi Paediatric Protocols", "2018", 2018, DEFAULT_SOURCE),
    "Malawi_Quick_e-Guide_-_for_key_peoplepdf.pdf":
        ("quick-e-guide", "Malawi Quick e-Guide for Key People", "", None, DEFAULT_SOURCE),
    "National Breast Health Guidelines.pdf":
        ("breast-health", "Malawi National Breast Health Guidelines", "", None, DEFAULT_SOURCE),
    "National renal protocols 1st Edition 2024.pdf":
        ("renal-2024", "National Renal Protocols", "1st Edition", 2024, DEFAULT_SOURCE),
    "OBGYN Guidelines.pdf":
        ("obgyn", "Obstetrics and Gynaecology Guidelines",
         "Version 3.0 (2017, SUPERSEDED by obs-gynae-2023)", 2017, DEFAULT_SOURCE),
    "PPH_Treatment_Bundle Final A4.pdf":
        ("pph-bundle", "Postpartum Haemorrhage Treatment Bundle", "Final A4", None, DEFAULT_SOURCE),
    "Practical-manual-SOBO-Malawi-Feb-2018-final.pdf":
        ("sobo-2018", "Practical Manual SOBO Malawi", "February 2018", 2018, DEFAULT_SOURCE),
    "Revised Malaria Treatment Guidelines 5th Edition 2020 _ Final - Signed.pdf":
        ("malaria-2020", "Revised Malaria Treatment Guidelines", "5th Edition", 2020, DEFAULT_SOURCE),
    "Viral_Hepatitis_Guideline_Sept_2023.pdf":
        ("viral-hepatitis-2023", "Malawi Viral Hepatitis Guideline", "September 2023", 2023, DEFAULT_SOURCE),
    "WHO pocketbook.pdf":
        ("who-pocketbook", "WHO Pocket Book of Hospital Care for Children", "2nd Edition",
         None, "World Health Organization"),
}


def _slug(name: str) -> str:
    stem = os.path.splitext(name)[0]
    stem = re.sub(r"^ilide\.info-", "", stem, flags=re.IGNORECASE)
    slug = re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-")
    return slug or "doc"


def _meta_for(name: str) -> tuple[str, str, str, int | None, str]:
    meta = DOC_META.get(name)
    if meta is None:  # "(1)" copy of a known file -> reuse its metadata
        stripped = re.sub(r"\s*\(\d+\)(?=\.pdf$)", "", name, flags=re.IGNORECASE)
        meta = DOC_META.get(stripped)
    if meta is None:
        meta = (_slug(name), os.path.splitext(name)[0].title(), "", None, DEFAULT_SOURCE)
    return meta


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-dir", default="Malawi Guidelines")
    ap.add_argument("--append", action="store_true",
                    help="keep the existing index instead of resetting it")
    ap.add_argument("--manifest", default=os.path.join("data", "guidelines", "manifest.json"))
    args = ap.parse_args()

    files = sorted(f for f in os.listdir(args.pdf_dir) if f.lower().endswith(".pdf"))
    if not files:
        raise SystemExit(f"no PDFs found in {args.pdf_dir!r}")

    seen: dict[str, str] = {}          # checksum -> first filename
    used_ids: set[str] = set()
    items: list[tuple[dict, list[dict]]] = []
    records: list[dict] = []
    duplicates: list[dict] = []
    failures: list[dict] = []

    for name in files:
        path = os.path.join(args.pdf_dir, name)
        checksum = _sha256(path)
        if checksum in seen:
            duplicates.append({"file": name, "duplicate_of": seen[checksum],
                               "checksum": checksum})
            print(f"  duplicate  {name}  == {seen[checksum]}", flush=True)
            continue
        seen[checksum] = name

        doc_id, title, edition, year, source = _meta_for(name)
        if doc_id in used_ids:
            doc_id = f"{doc_id}-{len(records) + 1}"
        used_ids.add(doc_id)

        try:
            pages = extract_pages(path)
        except Exception as exc:  # noqa: BLE001
            failures.append({"file": name, "error": repr(exc)})
            print(f"  FAILED     {name}: {exc!r}", flush=True)
            continue

        chars = sum(len(p.text) for p in pages)
        try:
            result = ingest_pages(
                pages, doc_id=doc_id, title=title, edition=edition,
                publication_year=year, source=source,
                file_path=os.path.abspath(path), checksum=checksum, version="1",
            )
        except Exception as exc:  # noqa: BLE001
            failures.append({"file": name, "error": repr(exc)})
            print(f"  FAILED     {name}: {exc!r}", flush=True)
            continue

        items.append((result.document.to_dict(), [c.to_dict() for c in result.chunks]))
        records.append({
            "file": name, "doc_id": doc_id, "title": title, "edition": edition,
            "publication_year": year, "source": source, "checksum": checksum,
            "pages": len(pages), "chars": chars, "chunks": len(result.chunks),
            "status": "ok",
        })
        print(f"  ingested   {name}  -> {doc_id}  ({len(pages)}p, {len(result.chunks)} chunks)",
              flush=True)

    store = VectorStore(settings.index_dir, embedder=get_embedder(force_tfidf=settings.force_tfidf))
    if args.append:
        store.load()
    else:
        store.reset()
    total_new = sum(len(chunks) for _, chunks in items)
    print(f"  embedding {total_new} chunks with {getattr(store.embedder, 'name', '?')} ...",
          flush=True)
    store.add_documents(items)
    store.save()

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pdf_dir": os.path.abspath(args.pdf_dir),
        "embedder": getattr(store.embedder, "name", "unknown"),
        "total_chunks": store.count(),
        "documents_indexed": len(records),
        "documents": records,
        "duplicates_skipped": duplicates,
        "failures": failures,
    }
    os.makedirs(os.path.dirname(args.manifest), exist_ok=True)
    with open(args.manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\nindexed {len(records)} documents, {store.count()} chunks "
          f"(embedder={manifest['embedder']})")
    print(f"duplicates skipped: {len(duplicates)} | failures: {len(failures)}")
    print(f"manifest: {os.path.abspath(args.manifest)}")


if __name__ == "__main__":
    main()
