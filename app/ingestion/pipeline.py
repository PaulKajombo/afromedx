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


def _norm_exact(line: str) -> str:
    """Case/whitespace-normalised line (digits preserved)."""
    import re
    return re.sub(r"\s+", " ", line.lower()).strip()


def _norm_digits(line: str) -> str:
    """Like _norm_exact but digits collapsed to '#'. Only used on short
    lines so 'Page 12'/'Page 13' footers match without risking real content."""
    import re
    return re.sub(r"\s+", " ", re.sub(r"\d+", "#", line.lower())).strip()


def strip_running_lines(pages: list[PageText], min_pages: int = 3, ratio: float = 0.25) -> list[PageText]:
    """Remove running headers/footers: short lines repeated on many pages.

    Real guideline PDFs repeat programme names, dates and page labels on every
    page; left in, they pollute both retrieval and section detection.
    """
    exact_occ: dict[str, set[int]] = {}
    digit_occ: dict[str, set[int]] = {}
    for p in pages:
        for line in p.text.split("\n"):
            e = _norm_exact(line)
            if e:
                exact_occ.setdefault(e, set()).add(p.page)
            if 0 < len(line.strip()) <= 40:
                digit_occ.setdefault(_norm_digits(line), set()).add(p.page)
    threshold = max(min_pages, int(round(ratio * len(pages))))
    drop_exact = {k for k, where in exact_occ.items() if len(where) >= threshold and len(k) <= 80}
    drop_digit = {k for k, where in digit_occ.items() if len(where) >= threshold}
    if not drop_exact and not drop_digit:
        return pages

    def keep(line: str) -> bool:
        if _norm_exact(line) in drop_exact:
            return False
        if 0 < len(line.strip()) <= 40 and _norm_digits(line) in drop_digit:
            return False
        return True

    out: list[PageText] = []
    for p in pages:
        out.append(PageText(page=p.page, text="\n".join(ln for ln in p.text.split("\n") if keep(ln))))
    return out


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
    usable = strip_running_lines(usable)
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
