"""PDF text extraction that preserves page numbers. Never discards page metadata."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PageText:
    page: int  # 1-based
    text: str


def extract_pages(pdf_path: str) -> list[PageText]:
    """Extract text per page using pypdf. Returns [] on failure (caller decides)."""
    from pypdf import PdfReader

    reader = PdfReader(pdf_path)
    pages: list[PageText] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages.append(PageText(page=i, text=text))
    return pages


def pages_from_text(text: str, words_per_page: int = 450) -> list[PageText]:
    """Fallback: split plain text into pseudo-pages (used by seed data/tests)."""
    words = text.split()
    pages: list[PageText] = []
    for i in range(0, max(1, len(words)), words_per_page):
        pages.append(PageText(page=len(pages) + 1, text=" ".join(words[i:i + words_per_page])))
    return pages
