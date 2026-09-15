"""CLI: ingest a real guideline PDF into the local index.

Example:
  python scripts/ingest.py data/guidelines/mstg-6e.pdf --doc-id mstg-6e --title "Malawi Standard Treatment Guidelines" --edition "6th Edition" --year 2023 --source "Malawi Ministry of Health"
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings
from app.ingestion.pipeline import ingest_pdf
from app.retrieval.embedder import get_embedder
from app.retrieval.store import VectorStore


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", help="path to guideline PDF")
    ap.add_argument("--doc-id", required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--edition", default="")
    ap.add_argument("--year", type=int, default=None)
    ap.add_argument("--source", default="")
    ap.add_argument("--source-url", default="")
    ap.add_argument("--version", default="1")
    args = ap.parse_args()

    result = ingest_pdf(args.pdf, doc_id=args.doc_id, title=args.title,
                        edition=args.edition, publication_year=args.year,
                        source=args.source, source_url=args.source_url, version=args.version)
    store = VectorStore(settings.index_dir, embedder=get_embedder(force_tfidf=settings.force_tfidf))
    store.load()
    store.add_document(result.document.to_dict(), [c.to_dict() for c in result.chunks])
    store.save()
    print(f"ingested {args.doc_id}: {len(result.chunks)} chunks (total={store.count()})")


if __name__ == "__main__":
    main()
