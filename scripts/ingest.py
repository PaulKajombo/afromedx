"""CLI: ingest a real guideline PDF into the local index.

Example:
  python scripts/ingest.py data/guidelines/mstg-6e.pdf --doc-id mstg-6e --title "Malawi Standard Treatment Guidelines" --edition "6th Edition" --year 2023 --source "Malawi Ministry of Health"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings
from app.ingestion.pipeline import ingest_pdf
from app.retrieval.embedder import get_embedder
from app.retrieval.store import VectorStore


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def update_manifest(manifest_path: str, *, filename: str, doc_id: str,
                    title: str, edition: str, year: int | None, source: str,
                    checksum: str, pages: int, chars: int, chunks: int) -> bool:
    """Record one ingested document in manifest.json (provenance ledger).

    Returns True when a new record was added, False when the doc_id (or an
    identical checksum) was already recorded — so re-running ingest of the
    same file never duplicates provenance. Creates the manifest if missing.
    """
    manifest = None
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
    if manifest is None:
        manifest = {"documents_indexed": 0, "total_chunks": 0, "documents": [],
                    "duplicates_skipped": [], "failures": []}
    for d in manifest.get("documents", []):
        if d.get("checksum") == checksum and d.get("doc_id") != doc_id:
            return False  # identical file already recorded under another id
    record = {
        "file": filename, "doc_id": doc_id, "title": title, "edition": edition,
        "publication_year": year, "source": source, "checksum": checksum,
        "pages": pages, "chars": chars, "chunks": chunks, "status": "ok",
    }
    existing = next((d for d in manifest.get("documents", []) if d.get("doc_id") == doc_id), None)
    if existing is not None:
        existing.update(record)  # re-ingest: refresh counts, keep position
    else:
        manifest.setdefault("documents", []).append(record)
    manifest["documents_indexed"] = len(manifest["documents"])
    manifest["total_chunks"] = sum(d.get("chunks", 0) for d in manifest["documents"])
    os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return True


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
    recorded = update_manifest(
        os.path.join("data", "guidelines", "manifest.json"),
        filename=os.path.basename(args.pdf), doc_id=args.doc_id, title=args.title,
        edition=args.edition, year=args.year, source=args.source,
        checksum=result.document.checksum,
        pages=len({c.page for c in result.chunks}),
        chars=sum(len(c.text) for c in result.chunks),
        chunks=len(result.chunks))
    print(f"manifest: {'recorded' if recorded else 'already present, skipped'}")
    print("next: POST /api/reload (or restart) so the server picks up the new document")


if __name__ == "__main__":
    main()
