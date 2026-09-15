"""Reusable ingestion pipeline: PDF -> pages -> chunks + metadata -> store."""
from __future__ import annotations

import os
from dataclasses import dataclass

from ..models import Chunk, Document
from .chunker import chunk_blocks, split_into_blocks
from .pdf_loader import PageText, extract_pages


@dataclass
class IngestionResult:
    document: Document
    chunks: list[Chunk]


def ingest_pages(
    pages: list[PageText],
    *,
    doc_id: str,
    title: str,
    edition: str = "",
    publication_year: int | None = None,
    source: str = "",
    source_url: str = "",
    file_path: str = "",
    checksum: str = "",
    version: str = "1",
) -> IngestionResult:
    # Section detection needs raw line structure, so split BEFORE cleaning;
    # the chunker cleans paragraph content itself.
    usable = [p for p in pages if p.text.strip()]
    if not usable:
        raise ValueError("No extractable text found")
    blocks = split_into_blocks(usable)
    raw = chunk_blocks(blocks)
    if not raw:
        raise ValueError("Chunking produced no chunks")
    doc = Document(
        id=doc_id, title=title, edition=edition, publication_year=publication_year,
        source=source, source_url=source_url, file_path=file_path,
        checksum=checksum, version=version,
    )
    chunks = [
        Chunk(
            id=Chunk.make_id(doc_id, r["page"], i),
            document_id=doc_id, page=r["page"], section=r["section"],
            subsection=r["subsection"], text=r["text"],
            metadata={"edition": edition, "source": source},
        )
        for i, r in enumerate(raw)
    ]
    return IngestionResult(document=doc, chunks=chunks)


def ingest_pdf(
    pdf_path: str, *, doc_id: str, title: str, edition: str = "",
    publication_year: int | None = None, source: str = "",
    source_url: str = "", version: str = "1",
) -> IngestionResult:
    with open(pdf_path, "rb") as f:
        data = f.read()
    checksum = Document.checksum_for(data)
    pages = extract_pages(pdf_path)
    return ingest_pages(
        pages, doc_id=doc_id, title=title, edition=edition,
        publication_year=publication_year, source=source, source_url=source_url,
        file_path=os.path.abspath(pdf_path), checksum=checksum, version=version,
    )
