"""Offline extractive provider: synthesizes ONLY from retrieved passages.

Default when no API key is set. Preserves doses/numbers verbatim by quoting
supporting sentences, always cites, and abstains when evidence is missing.
"""
from __future__ import annotations

import re

from .base import GroundedAnswer, LLMProvider
from ..retrieval.embedder import _STOP
from ..retrieval.search import GENERIC_TERMS, term_covered, tokenize

ABSTAIN_TEXT = (
    "I couldn't find sufficient information in the indexed Malawian guidelines "
    "to answer this question reliably."
)


def answer_title(question: str, fallback: str = "Guideline evidence") -> str:
    """Human title derived from the question, not from chunk metadata.

    Section headings are unreliable as titles (misdetected headings like
    "HOURS)" leak through); the question is always relevant. Truncated to
    keep the UI tidy.
    """
    q = (question or "").strip()
    if not q:
        return fallback
    q = q[0].upper() + q[1:]
    return q if len(q) <= 120 else q[:117].rstrip() + "..."

_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_RE.split(text.strip()) if s.strip()]


def _clinically_rich(sent: str) -> int:
    score = 0
    if re.search(r"\d", sent):
        score += 3  # doses, durations, thresholds matter most
    if re.search(r"\b(mg|g|ml|mmol|iu|units?|mg/kg|mcg|tablets?|daily|bd|tds|qid|weekly|hours?|days?|weeks?)\b", sent, re.I):
        score += 2
    if re.search(r"\b(recommend|first-?line|treat|give|administer|diagnos|refer|contraindicat|avoid|monitor)\b", sent, re.I):
        score += 2
    return score


def build_citations(passages: list[dict], doc_lookup: dict, limit: int = 4) -> list[dict]:
    """Build citation blocks from retrieved passages (trusted metadata only).

    Shared by all providers: a provider selects WHICH retrieved passages to
    cite, but citation metadata (title, edition, section, page, chunk_id)
    always comes from the retrieval store — never from model output.
    """
    citations = []
    seen = set()
    for p in passages[:limit]:
        chunk = p["chunk"]
        doc = doc_lookup.get(chunk.get("document_id"), {})
        key = (chunk.get("document_id"), chunk.get("page"), chunk.get("section"))
        if key in seen:
            continue
        seen.add(key)
        citations.append({
            "document": doc.get("title", chunk.get("document_id")),
            "edition": doc.get("edition", ""),
            "year": doc.get("publication_year"),
            "section": chunk.get("section", ""),
            "subsection": chunk.get("subsection", ""),
            "page": chunk.get("page"),
            "chunk_id": chunk.get("id"),
            "excerpt": chunk.get("text", "")[:400],
            "score": round(float(p.get("score", 0)), 3),
        })
    return citations


class StubProvider(LLMProvider):
    name = "stub"

    def _supported(self, question: str, passages: list[dict]) -> bool:
        """Grounding guard: a majority of the question's significant terms (or
        synonyms) must appear in the retrieved passages, else the passages are
        topical neighbours, not evidence for THIS question."""
        sig = [t for t in tokenize(question)
               if len(t) > 1 and t not in GENERIC_TERMS and t not in _STOP]
        if not sig:
            return False
        text_terms: set[str] = set()
        for p in passages[:5]:
            text_terms.update(tokenize(p["chunk"].get("text", "")))

        def bridges(t: str) -> bool:
            return term_covered(t, text_terms)

        covered = sum(1 for t in sig if bridges(t))
        return covered / len(sig) >= 0.5

    def generate(self, question: str, passages: list[dict], doc_lookup: dict) -> GroundedAnswer:
        if not passages:
            return GroundedAnswer(title="No reliable source found", body_markdown=ABSTAIN_TEXT,
                                  citations=[], abstained=True, grounded=True)
        if not self._supported(question, passages):
            return GroundedAnswer(title="No reliable source found", body_markdown=ABSTAIN_TEXT,
                                  citations=[], abstained=True, grounded=True)
        # collect candidate sentences from top passages
        cands: list[tuple[int, str, dict]] = []
        for p in passages[:6]:
            chunk = p["chunk"]
            for s in _sentences(chunk.get("text", ""))[:8]:
                if len(s) < 25:
                    continue
                cands.append((_clinically_rich(s), s, chunk))
        if not cands:
            return GroundedAnswer(title="No reliable source found", body_markdown=ABSTAIN_TEXT,
                                  citations=[], abstained=True, grounded=True)
        cands.sort(key=lambda x: -x[0])
        top = cands[:5]
        body = " ".join(s for _, s, _ in top[:3])
        key_points = [s for _, s, _ in top[:5]]
        citations = build_citations(passages, doc_lookup)
        return GroundedAnswer(title=answer_title(question), body_markdown=body, key_points=key_points,
                              citations=citations, abstained=False, grounded=True)
