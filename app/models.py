"""Shared data model: Document and Chunk. Versioning is explicit — never mix editions."""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone


@dataclass
class Document:
    id: str
    title: str
    edition: str = ""
    publication_year: int | None = None
    source: str = ""          # provenance: e.g. "Malawi Ministry of Health"
    source_url: str = ""
    file_path: str = ""
    checksum: str = ""
    version: str = "1"        # index version for this edition; bump on re-ingest of changed file
    date_added: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def checksum_for(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()


@dataclass
class Chunk:
    id: str
    document_id: str
    page: int | None = None
    section: str = ""
    subsection: str = ""
    text: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def make_id(document_id: str, page: int | None, idx: int) -> str:
        return f"{document_id}::p{page if page is not None else 'x'}::{idx}"

    def citation(self, doc_lookup: dict | None = None) -> dict:
        """Build the citation block the UI and LLM must expose."""
        doc = (doc_lookup or {}).get(self.document_id, {})
        return {
            "document": doc.get("title", self.document_id),
            "edition": doc.get("edition", ""),
            "year": doc.get("publication_year"),
            "section": self.section,
            "subsection": self.subsection,
            "page": self.page,
            "chunk_id": self.id,
        }
