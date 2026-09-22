"""OpenAI-compatible provider (OpenAI or any compatible endpoint, e.g. local Ollama).

Uses raw httpx so the app does not hard-depend on the `openai` SDK and the
base URL can be swapped via env. Falls back to abstention on any API failure
rather than inventing an answer.
"""
from __future__ import annotations

import json
import logging
import re

import httpx

from .base import GroundedAnswer, LLMProvider
from .gemini_provider import _safe_detail
from .prompts import SYSTEM_PROMPT
from .stub import ABSTAIN_TEXT, StubProvider, answer_title, build_citations

log = logging.getLogger(__name__)


def _evidence_block(passages: list[dict], doc_lookup: dict) -> str:
    lines = []
    for i, p in enumerate(passages[:6], 1):
        c = p["chunk"]
        doc = doc_lookup.get(c.get("document_id"), {})
        lines.append(
            f"[E{i}] {doc.get('title','')} {doc.get('edition','')} "
            f"| Section: {c.get('section','')} / {c.get('subsection','')} "
            f"| Page: {c.get('page')} | score={p.get('score',0):.3f}\n{c.get('text','')}"
        )
    return "\n\n".join(lines)


# Structural labels the UI already provides (title, "Key points", "Source"
# sections). The model must not repeat them inside body/key_points; strip any
# that slip through so nothing displays twice.
_STRUCTURAL_RE = re.compile(
    r"^\*{0,2}\s*(answer|key[\s\-]*points?|recommended action|heading|"
    r"guideline(\s+sources?)?)\s*:?\*{0,2}\s*$", re.IGNORECASE)
_GUIDELINE_TRAILER_RE = re.compile(r"^\*{0,2}\s*guideline", re.IGNORECASE)


def _clean_answer(body: str, bullets: list[str]) -> tuple[str, list[str]]:
    """Remove UI-duplicating structure from model output.

    Drops label-only lines (Answer:/Key Points:/Guideline:) and trailing
    Guideline source lines from the body, and drops structural/header bullets
    from key_points. Pure clinical content passes through untouched.
    """
    kept = [ln for ln in (body or "").splitlines()
            if ln.strip() and not _STRUCTURAL_RE.match(ln.strip())
            and not _GUIDELINE_TRAILER_RE.match(ln.strip())]
    clean_bullets = []
    for b in bullets or []:
        s = b.strip().lstrip("-•* ").strip()
        if not s or _STRUCTURAL_RE.match(s) or _GUIDELINE_TRAILER_RE.match(s):
            continue
        clean_bullets.append(s)
    return "\n".join(kept).strip(), clean_bullets


class OpenAICompatibleProvider(LLMProvider):
    name = "openai_compatible"

    def __init__(self, api_key: str, base_url: str, model: str,
                 timeout_s: float = 60.0) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    def generate(self, question: str, passages: list[dict], doc_lookup: dict) -> GroundedAnswer:
        if not passages:
            return StubProvider().generate(question, [], doc_lookup)
        if not self.api_key:
            return StubProvider().generate(question, passages, doc_lookup)
        user = (
            f"Clinical question: {question}\n\n"
            f"Guideline evidence (cite as [E1], [E2], ... with title + page):\n{_evidence_block(passages, doc_lookup)}\n\n"
            "Write the answer as plain clinical prose (2-4 short paragraphs at most): start directly "
            "with the recommended action, tagging supporting statements like [E2]. "
            "Do NOT add markdown headings, do NOT label sections ('Answer:', 'Key Points:', "
            "'Recommended Action:', 'Guideline:'), and do NOT append a source/Guideline trailer line — "
            "AfroMedX attaches verified citations from retrieval metadata itself. "
            "Separately list 3-6 key_points: short standalone clinical facts only, no headers or labels. "
            "If the question itself is not about clinical patient care (for example vehicle repair "
            "or sports scores), abstain even if some retrieved words overlap — topical word overlap "
            "is not clinical evidence. "
            "If evidence is insufficient, reply EXACTLY with: INSUFFICIENT EVIDENCE."
        )
        try:
            r = httpx.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model,
                      "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                   {"role": "user", "content": user}],
                      "temperature": 0.0},
                timeout=self.timeout_s,
            )
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            # Previously silent: burst traffic (e.g. back-to-back eval items)
            # can hit provider rate limits here, degrading to the stub with
            # zero visibility. Log sanitized failure like the other providers.
            log.warning("openai_compatible request failed (%s: %s); using stub fallback",
                        type(exc).__name__, _safe_detail(str(exc), self.api_key))
            return StubProvider().generate(question, passages, doc_lookup)
        if "INSUFFICIENT EVIDENCE" in text.upper():
            return GroundedAnswer(title="No reliable source found", body_markdown=ABSTAIN_TEXT,
                                  citations=_citations(passages, doc_lookup),
                                  abstained=True, grounded=True)
        # split bullets heuristically, then strip UI-duplicating structure
        lines = [ln.strip("-• ").strip() for ln in text.splitlines() if ln.strip().startswith(("-", "•", "*"))]
        clean_body, clean_points = _clean_answer(text, lines)
        if not clean_body:
            clean_body = text.strip()
        return GroundedAnswer(title=answer_title(question), body_markdown=clean_body,
                              key_points=clean_points[:6],
                              citations=_citations(passages, doc_lookup),
                              abstained=False, grounded=True)


def _citations(passages: list[dict], doc_lookup: dict) -> list[dict]:
    out, seen = [], set()
    for p in passages[:4]:
        c = p["chunk"]
        key = (c.get("document_id"), c.get("page"), c.get("section"))
        if key in seen:
            continue
        seen.add(key)
        doc = doc_lookup.get(c.get("document_id"), {})
        out.append({"document": doc.get("title", c.get("document_id")),
                    "edition": doc.get("edition", ""), "year": doc.get("publication_year"),
                    "section": c.get("section", ""), "subsection": c.get("subsection", ""),
                    "page": c.get("page"), "chunk_id": c.get("id"),
                    "excerpt": c.get("text", "")[:400], "score": round(float(p.get("score", 0)), 3)})
    return out
