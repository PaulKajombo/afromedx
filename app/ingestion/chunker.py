"""Section-aware chunking.

Avoids blind N-char splits: detects headings, then packs paragraphs into
target-sized chunks with overlap. Every chunk keeps page + section metadata.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .cleaner import clean_text
from .pdf_loader import PageText

TARGET_CHARS = 900
MAX_CHARS = 1400
OVERLAP_CHARS = 150

# Numbered headings ("4.2 Severe malaria"), appendix refs, or ALL-CAPS lines.
_HEADING_RE = re.compile(
    r"^\s*(?:(?:chapter|section|part)\s+\d+|(?:\d{1,3}(?:\.\d{1,3}){0,3})\s+.+|[A-Z][A-Z0-9\s\-–/()]{4,80})$"
)
_CLINICAL_KEYWORDS = (
    "treatment", "management", "diagnosis", "dosage", "dose", "prophylaxis",
    "malaria", "tuberculosis", "hiv", "art", "pneumonia", "sepsis", "diarrhoea",
    "diarrhea", "anaemia", "anemia", "hypertension", "diabetes", "referral",
    "contraindication", "adverse", "guideline",
)
# Table-of-contents rows use dot leaders ("1.4.1 Artesunate ..... 7").
_TOC_LINE_RE = re.compile(r"\.{3,}")
# Version-stamp / document-metadata lines ("2.0 Published on 25.04.2026",
# "Revised and issued June 2024") must not become section headings.
_META_LINE_RE = re.compile(r"\b(published|issued|revised|updated|document version|date of issue)\b", re.I)
# Cover/footer/boilerplate lines that must never become headings.
_JUNK_HEADING_RE = re.compile(
    r"(avenue appia|1211 geneva|switzerland|po box|\bisbn\b|©|all rights reserved|"
    r"www\.|http[s]?://|printed in|world health organization regional|tel[:.]|e-?mail|@)",
    re.I)
# How many leading pages to treat as cover/front matter (heading detection off).
_FRONT_MATTER_PAGES = 5


def _looks_like_heading(line: str) -> bool:
    s = re.sub(r"\s+", " ", line).strip()
    if not s or len(s) > 100:
        return False
    if _JUNK_HEADING_RE.search(s):
        return False
    if _TOC_LINE_RE.search(s) or s.count(".") > 6:
        return False  # dot leaders / TOC rows are not headings
    if _META_LINE_RE.search(s) and re.search(r"\d", s) and len(s) < 60:
        return False  # "2.0 published on 25.04.2026" is metadata, not a section
    if sum(1 for ch in s if ch.isdigit()) / max(1, len(s)) > 0.4:
        return False  # numeric-layout artifacts ("61 17 52 – 27 50 200")
    if _HEADING_RE.match(s):
        low = s.lower()
        if len(s) < 80 or any(k in low for k in _CLINICAL_KEYWORDS):
            return True
    # ALL-CAPS short lines, even with a trailing period: "SEVERE MALARIA."
    if len(s) < 70 and any(k in s.lower() for k in _CLINICAL_KEYWORDS):
        head = s.rstrip(".:")
        if head.isupper() or head.istitle():
            return True
    # Title-case short lines ending without period that mention clinical terms
    if len(s) < 70 and not s.endswith(".") and any(k in s.lower() for k in _CLINICAL_KEYWORDS):
        if s.istitle() or s.isupper():
            return True
    return False


def _strong_section(line: str) -> bool:
    """TRUE for headings strong enough to (re)start a top-level section:
    numbered ("4.", "4.2"), explicitly worded (Chapter/Section/Part), or
    ALL-CAPS. Title-case lines ("Principles For Inhaled Poisons", footers)
    are treated as subsections so cover/footer junk never hijacks the section."""
    s = line.strip()
    if re.match(r"^(?:chapter|section|part|annex|appendix)\s+", s, re.I):
        return True
    if re.match(r"^\d{1,3}(?:\.\d{1,3}){0,2}\s*[-–.\s][A-Za-z]", s):
        return True
    if len(s) >= 4 and sum(1 for ch in s if ch.isdigit()) / len(s) <= 0.2:
        words = [w for w in re.split(r"[\s\-–/]+", s) if w]
        if len(words) >= 2 and all(w.isupper() and w.isalpha() for w in words):
            return True
    return False


@dataclass
class SectionBlock:
    section: str
    subsection: str
    page: int
    text: str


def split_into_blocks(pages: list[PageText]) -> list[SectionBlock]:
    """Walk pages line-by-line, tracking current section/subsection."""
    blocks: list[SectionBlock] = []
    section, subsection = "General", ""
    buf: list[str] = []
    buf_page: int | None = None

    def flush():
        nonlocal buf, buf_page
        if buf and buf_page is not None:
            text = clean_text(" ".join(buf))
            if text:
                blocks.append(SectionBlock(
                    section=section, subsection=subsection, page=buf_page, text=text))
        buf, buf_page = [], None

    for page_idx, p in enumerate(pages):
        for raw_line in p.text.split("\n"):
            line = re.sub(r"\s+", " ", raw_line).strip()
            if not line:
                continue
            if _TOC_LINE_RE.search(line):
                continue  # drop table-of-contents rows entirely
            if page_idx < _FRONT_MATTER_PAGES and _looks_like_heading(line):
                continue  # cover/front-matter "headings" (titles, addresses) don't start sections
            if _looks_like_heading(line):
                flush()
                head_raw = line
                title = re.sub(r"^[\d.\s\-–:]+", "", line).strip().rstrip(".:").title()
                if _strong_section(head_raw):
                    section, subsection = title, ""
                elif section != "General":
                    subsection = title
                buf_page = p.page if buf_page is None else buf_page
                continue
            # page change always flushes so chunk page attribution stays truthful
            if buf_page is not None and p.page != buf_page:
                flush()
            if buf_page is None:
                buf_page = p.page
            buf.append(line)
    flush()
    return blocks


def chunk_blocks(blocks: list[SectionBlock]) -> list[dict]:
    """Pack blocks into sized chunks with character overlap. Returns chunk dicts (no ids)."""
    chunks: list[dict] = []
    for b in blocks:
        text = b.text.strip()
        if not text:
            continue
        start = 0
        # split long blocks on sentence boundaries
        while start < len(text):
            end = min(start + MAX_CHARS, len(text))
            if end < len(text):
                # prefer sentence end within last 300 chars
                window = text[max(start, end - 300):end]
                m = max(window.rfind(". "), window.rfind("\n"))
                if m != -1:
                    end = max(start, end - 300) + m + 1
            piece = text[start:end].strip()
            if len(piece) >= 200 or end >= len(text):
                if piece:
                    chunks.append({
                        "page": b.page, "section": b.section,
                        "subsection": b.subsection, "text": piece,
                    })
            if end >= len(text):
                break
            start = max(start + TARGET_CHARS - OVERLAP_CHARS, end - OVERLAP_CHARS)
            if start <= 0:
                start = end
    return chunks
