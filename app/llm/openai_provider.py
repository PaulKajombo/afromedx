"""OpenAI-compatible provider (OpenAI or any compatible endpoint, e.g. local Ollama).

Uses raw httpx so the app does not hard-depend on the `openai` SDK and the
base URL can be swapped via env. Falls back to abstention on any API failure
rather than inventing an answer.
"""
from __future__ import annotations

import json

import httpx

from .base import GroundedAnswer, LLMProvider
from .prompts import SYSTEM_PROMPT
from .stub import ABSTAIN_TEXT, StubProvider


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
            "Answer concisely with: heading, recommended action, key-points bullets, "
            "then a 'Guideline:' line with title, edition, section, page. "
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
        except Exception:
            return StubProvider().generate(question, passages, doc_lookup)
        if "INSUFFICIENT EVIDENCE" in text.upper():
            return GroundedAnswer(title="No reliable source found", body_markdown=ABSTAIN_TEXT,
                                  citations=_citations(passages, doc_lookup),
                                  abstained=True, grounded=True)
        # split bullets heuristically
        lines = [ln.strip("-• ").strip() for ln in text.splitlines() if ln.strip().startswith(("-", "•", "*"))]
        first = passages[0]["chunk"]
        title = ((first.get("section") or "Answer").upper())
        return GroundedAnswer(title=title, body_markdown=text,
                              key_points=lines[:6],
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
