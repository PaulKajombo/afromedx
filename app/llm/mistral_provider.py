"""Native Mistral provider (official `mistralai` SDK, Mistral La Plateforme API).

Second production LLM alongside Gemini. Same evidence-first contract:
retrieved AfroMedX chunks are labelled [E1..En] with full provenance; Mistral
returns structured output that only SELECTS citation IDs. AfroMedX resolves
those IDs against the actually-retrieved passages and builds citations from
trusted retrieval metadata. Mistral never authors citation metadata.

No OpenAI SDK, no OpenAI-compatible routing, no tools, no web search —
retrieved evidence in, validated local citations out.
"""
from __future__ import annotations

import json
import logging
import re
import time

from pydantic import ValidationError

from .base import GroundedAnswer, LLMProvider
# Shared structured-output contract + schema sanitizer + fence stripper live
# with the Gemini provider (single source of truth for the answer contract).
from .gemini_provider import (
    GeminiStructuredAnswer,
    _response_schema,
    _strip_fences,
)
from .stub import ABSTAIN_TEXT, StubProvider, build_citations

log = logging.getLogger(__name__)

EVIDENCE_LIMIT = 6

SYSTEM_INSTRUCTION = """You are the answer-synthesis module of AfroMedX Clinical Search, \
a clinical guideline reference tool — not a general medical advice chatbot.

You receive a clinical question plus numbered guideline evidence items [E1], [E2], ... \
supplied by AfroMedX retrieval. Each item carries its source guideline title, edition, \
section, and page. These IDs refer ONLY to the evidence supplied in this request.

Rules:
- Answer using ONLY the supplied evidence. Do not use outside medical knowledge to fill gaps.
- Do NOT invent clinical recommendations, doses, contraindications, treatment durations, \
page numbers, sections, editions, or guideline names. If a number is not in the evidence, \
do not state it.
- A mere MENTION of an intervention is not a recommendation: distinguish evidence that \
RECOMMENDS an intervention from evidence that mentions it as a contraindication, an \
alternative, or background.
- Reject misleading or unsupported premises: do NOT accept a premise merely because the \
retrieved evidence contains related words. If the question assumes something the evidence \
does not support, abstain.
- If the evidence is insufficient, conflicting, ambiguous, or does not directly support the \
requested answer, set abstained=true with a short abstention_reason and leave citation_ids empty.
- Prefer the most directly relevant and specific evidence; cite every evidence item you relied on.
- When stating a recommendation, append the supporting evidence IDs in brackets, e.g. [E2].
- NEVER invent, guess, or renumber evidence IDs. Only IDs listed in the supplied evidence exist.
- Keep the answer concise and clinically useful: short heading-style summary, recommended action \
with exact doses/durations as stated, then key points.
- Respond with exactly these fields: body_markdown, key_points, citation_ids, abstained, \
abstention_reason.
"""

# Transient failures worth one bounded retry cycle: rate limiting and
# server-side faults. Everything else (auth, bad request, schema, output
# validation) fails fast to the safe fallback.
_RETRYABLE_CODES = frozenset({429, 500, 502, 503, 504})


def _status_code(exc: Exception) -> int | None:
    """HTTP status of an SDK error, duck-typed (no SDK import required).

    The mistralai SDK surfaces failures as SDKError carrying the httpx
    raw_response; anything else carrying .status_code/.code also works.
    """
    for attr in ("status_code", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    raw = getattr(exc, "raw_response", None)
    value = getattr(raw, "status_code", None)
    return value if isinstance(value, int) else None


def _is_retryable(exc: Exception) -> bool:
    return _status_code(exc) in _RETRYABLE_CODES


def _safe_detail(message: str, api_key: str, limit: int = 500) -> str:
    """Sanitized error detail for local diagnostics.

    Logs error-type/message only. Never the API key, authorization headers,
    or request payloads: any key material is redacted defensively.
    """
    text = (message or "").strip()
    if api_key and api_key in text:
        text = text.replace(api_key, "[REDACTED]")
    text = re.sub(r"(?i)(authorization[\"']?\s*[:=]\s*)\S+", r"\1[REDACTED]", text)
    text = re.sub(r"(?i)(key[\"']?\s*[:=]\s*['\"]?)[A-Za-z0-9_\-]{16,}", r"\1[REDACTED]", text)
    return text[:limit]


def _evidence_block(passages: list[dict], doc_lookup: dict,
                    limit: int = EVIDENCE_LIMIT) -> tuple[str, dict[str, dict]]:
    """Render labelled evidence and return (text, {evidence_id: passage})."""
    lines: list[str] = []
    id_map: dict[str, dict] = {}
    for i, p in enumerate(passages[:limit], 1):
        eid = f"E{i}"
        c = p["chunk"]
        doc = doc_lookup.get(c.get("document_id"), {})
        lines.append(
            f"[{eid}] document_id={c.get('document_id')} | "
            f"title={doc.get('title', '')} | edition={doc.get('edition', '')} | "
            f"year={doc.get('publication_year')} | section={c.get('section', '')} | "
            f"subsection={c.get('subsection', '')} | page={c.get('page')} | "
            f"chunk_id={c.get('id')} | retrieval_score={p.get('score', 0):.3f}\n"
            f"{c.get('text', '')}"
        )
        id_map[eid] = p
    return "\n\n".join(lines), id_map


def _abstain(passages: list[dict], doc_lookup: dict,
             valid_ids: list[str] | None = None,
             id_map: dict[str, dict] | None = None) -> GroundedAnswer:
    cites = []
    if valid_ids and id_map:
        cites = build_citations([id_map[i] for i in valid_ids], doc_lookup)
    return GroundedAnswer(title="No reliable source found", body_markdown=ABSTAIN_TEXT,
                          key_points=[], citations=cites, abstained=True, grounded=True)


class MistralProvider(LLMProvider):
    name = "mistral"

    def __init__(self, api_key: str, model: str = "mistral-small-2603",
                 temperature: float = 0.0, timeout_s: float = 60.0,
                 max_retries: int = 2, retry_base_delay_s: float = 1.0,
                 client=None, sleeper=None) -> None:
        self.api_key = api_key or ""
        self.model = model or "mistral-small-2603"
        self.temperature = temperature
        self.timeout_s = timeout_s
        # Bounded retries: max_retries after the initial attempt (default 2 →
        # at most 3 total attempts), exponential backoff from base delay.
        self.max_retries = max(0, max_retries)
        self.retry_base_delay_s = max(0.0, retry_base_delay_s)
        # Injected client (tests) or lazily-created SDK client (production).
        # The mistralai SDK is imported lazily so unit tests and offline
        # (stub) runs never require the package or a key.
        self._client = client
        # Injected sleeper (tests) or time.sleep (production).
        self._sleeper = sleeper or time.sleep

    def _get_client(self):
        if self._client is not None:
            return self._client
        from mistralai.client import Mistral
        return Mistral(api_key=self.api_key, timeout_ms=int(self.timeout_s * 1000))

    def _response_format(self):
        from mistralai.client import models
        return models.ResponseFormat(
            type="json_schema",
            json_schema=models.JSONSchema(
                name="afromedx_answer",
                schema_definition=_response_schema(),
            ),
        )

    @staticmethod
    def _message_text(response) -> str:
        """Assistant message content as plain text (defensive over SDK shapes)."""
        try:
            content = response.choices[0].message.content
        except Exception:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                else:
                    text = getattr(part, "text", None)
                    if isinstance(text, str):
                        parts.append(text)
            return "".join(parts)
        return ""

    @classmethod
    def _parse_response(cls, response) -> GeminiStructuredAnswer:
        """Parse the chat completion into the shared structured contract.

        Raises on empty/malformed responses — callers abstain.
        """
        text = (cls._message_text(response) or "").strip()
        if not text:
            raise ValueError("empty response text")
        return GeminiStructuredAnswer.model_validate(json.loads(_strip_fences(text)))

    def generate(self, question: str, passages: list[dict], doc_lookup: dict) -> GroundedAnswer:
        if not passages:
            return StubProvider().generate(question, [], doc_lookup)
        if not self.api_key and self._client is None:
            # Same contract as the other providers: without a key, degrade to
            # the offline stub rather than failing.
            return StubProvider().generate(question, passages, doc_lookup)

        evidence_text, id_map = _evidence_block(passages, doc_lookup)
        messages = [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": (
                f"Clinical question: {question}\n\n"
                f"Guideline evidence (AfroMedX retrieval — only these IDs exist):\n"
                f"{evidence_text}\n\n"
                "Respond with: body_markdown (concise answer, recommendations tagged like [E2]), "
                "key_points (short bullets), citation_ids (every [E#] you relied on, e.g. [\"E2\"]), "
                "abstained (true when the evidence is insufficient, conflicting, ambiguous, does not "
                "directly answer the question, or the question rests on a premise the evidence does "
                "not support), and abstention_reason (short, only when abstaining)."
            )},
        ]
        attempts = 1 + self.max_retries
        structured = None
        for attempt in range(1, attempts + 1):
            try:
                client = self._get_client()
                response = client.chat.complete(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    response_format=self._response_format(),
                )
                structured = self._parse_response(response)
                break
            except (ValidationError, ValueError, KeyError, TypeError, AttributeError) as exc:
                # Unusable model output (malformed/empty): the output, not the
                # transport, failed — retrying cannot help.
                log.warning("Mistral output unusable (%s); abstaining", type(exc).__name__)
                return StubProvider().generate(question, passages, doc_lookup)
            except Exception as exc:  # API / network / timeout / SDK errors
                if _is_retryable(exc) and attempt < attempts:
                    delay = self.retry_base_delay_s * (2 ** (attempt - 1))
                    log.warning(
                        "Mistral request failed (attempt %d/%d): model=%s error=%s "
                        "detail=%s; retrying in %.1fs",
                        attempt, attempts, self.model, type(exc).__name__,
                        _safe_detail(str(exc), self.api_key), delay)
                    self._sleeper(delay)
                    continue
                log.warning(
                    "Mistral request failed (attempt %d/%d): model=%s error=%s "
                    "detail=%s; abstaining",
                    attempt, attempts, self.model, type(exc).__name__,
                    _safe_detail(str(exc), self.api_key))
                return StubProvider().generate(question, passages, doc_lookup)
        assert structured is not None  # loop breaks only on success

        # Resolve citation IDs against the evidence actually supplied. Unknown
        # IDs are dropped (never exposed); only trusted metadata is cited.
        seen: set[str] = set()
        valid_ids: list[str] = []
        invalid_ids: list[str] = []
        for raw in structured.citation_ids or []:
            eid = str(raw).strip().upper()
            if eid in id_map:
                if eid not in seen:
                    seen.add(eid)
                    valid_ids.append(eid)
            elif eid:
                invalid_ids.append(str(raw))
        if invalid_ids:
            log.warning("Mistral returned unknown evidence IDs %s; dropping", invalid_ids)

        if structured.abstained:
            if structured.abstention_reason:
                log.info("Mistral abstained: %s", structured.abstention_reason)
            return _abstain(passages, doc_lookup, valid_ids, id_map)
        if not valid_ids:
            # Answered but cited nothing usable: claims would be unsupported.
            log.warning("Mistral answered without valid citation IDs; abstaining")
            return _abstain(passages, doc_lookup)
        body = (structured.body_markdown or "").strip()
        if not body:
            log.warning("Mistral returned an empty answer body; abstaining")
            return _abstain(passages, doc_lookup, valid_ids, id_map)

        selected = [id_map[i] for i in valid_ids]
        first = selected[0]["chunk"]
        title = (first.get("section") or "Guideline evidence").upper()
        return GroundedAnswer(
            title=title,
            body_markdown=body,
            key_points=list(structured.key_points or [])[:6],
            citations=build_citations(selected, doc_lookup),
            abstained=False,
            grounded=True,
        )
