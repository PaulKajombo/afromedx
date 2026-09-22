"""Native Google Gemini provider (official `google-genai` SDK, Gemini Developer API).

Evidence-first RAG: retrieved AfroMedX chunks are labelled [E1..En] with full
provenance; Gemini returns structured output that only SELECTS citation IDs.
AfroMedX resolves those IDs against the actually-retrieved passages and builds
citations from trusted retrieval metadata. Gemini never authors citation
metadata (no pages, titles, editions, or sections from the model).

No OpenAI dependency, no OpenAI-compatible routing, no Google Search
grounding, no external medical knowledge — retrieved evidence in, validated
local citations out.
"""
from __future__ import annotations

import json
import logging
import re
import time

from pydantic import BaseModel, Field, ValidationError

from .base import GroundedAnswer, LLMProvider
from .stub import ABSTAIN_TEXT, StubProvider, build_citations

log = logging.getLogger(__name__)

EVIDENCE_LIMIT = 6
MAX_OUTPUT_TOKENS = 2048

SYSTEM_INSTRUCTION = """You are the answer-synthesis module of AfroMedX Clinical Search, \
a clinical guideline reference tool — not a general medical advice chatbot.

You receive a clinical question plus numbered guideline evidence items [E1], [E2], ... \
supplied by AfroMedX retrieval. Each item carries its source guideline title, edition, \
section, and page. These IDs refer ONLY to the evidence supplied in this request.

Rules:
- Answer using ONLY the supplied evidence. Do not use outside medical knowledge to fill gaps.
- Do NOT invent recommendations, doses, contraindications, treatment durations, page numbers, \
sections, editions, or guideline names. If a number is not in the evidence, do not state it.
- Distinguish evidence that RECOMMENDS an intervention from evidence that merely MENTIONS it \
(for example as a contraindication, an alternative, or background).
- Do NOT accept a misleading or false premise merely because the retrieved evidence contains \
related words. If the question assumes something the evidence does not support, abstain.
- If the evidence is insufficient, conflicting, ambiguous, or does not directly support the \
requested answer, set abstained=true with a short abstention_reason and leave citation_ids empty.
- Prefer the most directly relevant and specific evidence; cite every evidence item you relied on.
- When stating a recommendation, append the supporting evidence IDs in brackets, e.g. [E2].
- NEVER invent, guess, or renumber evidence IDs. Only IDs listed in the supplied evidence exist.
- Keep the answer concise and clinically useful: short heading-style summary, recommended action \
with exact doses/durations as stated, then key points.
"""


class GeminiStructuredAnswer(BaseModel):
    """Structured output contract enforced via the Gemini response schema."""
    body_markdown: str = ""
    key_points: list[str] = Field(default_factory=list)
    citation_ids: list[str] = Field(default_factory=list)
    abstained: bool = False
    abstention_reason: str | None = None


def _response_schema() -> dict:
    """JSON schema for structured output, sanitized for the API's subset.

    Pydantic renders Optional[...] as anyOf, which the Gemini structured-output
    schema subset rejects; the documented nullable form is a type array
    (e.g. {"type": ["string", "null"]}). The Python-side contract
    (str | None, explicit null still parses) is unchanged.
    """
    schema = GeminiStructuredAnswer.model_json_schema()
    for prop in schema.get("properties", {}).values():
        any_of = prop.pop("anyOf", None)
        if not isinstance(any_of, list):
            continue
        types = sorted({t.get("type") for t in any_of
                        if isinstance(t, dict) and isinstance(t.get("type"), str)})
        if types:
            prop["type"] = types if len(types) > 1 else types[0]
    return schema


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


def _safe_detail(message: str, api_key: str, limit: int = 500) -> str:
    """Sanitized error detail for local diagnostics.

    Logs model/error-type/message only. Never the API key, authorization
    headers, or request payloads: any key material is redacted defensively.
    """
    text = (message or "").strip()
    if api_key and api_key in text:
        text = text.replace(api_key, "[REDACTED]")
    text = re.sub(r"(?i)(authorization[\"']?\s*[:=]\s*)\S+", r"\1[REDACTED]", text)
    text = re.sub(r"(?i)(key[\"']?\s*[:=]\s*['\"]?)[A-Za-z0-9_\-]{16,}", r"\1[REDACTED]", text)
    return text[:limit]


def _abstain(passages: list[dict], doc_lookup: dict,
             valid_ids: list[str] | None = None,
             id_map: dict[str, dict] | None = None) -> GroundedAnswer:
    cites = []
    if valid_ids and id_map:
        cites = build_citations([id_map[i] for i in valid_ids], doc_lookup)
    return GroundedAnswer(title="No reliable source found", body_markdown=ABSTAIN_TEXT,
                          key_points=[], citations=cites, abstained=True, grounded=True)


# Transient failures worth one bounded retry cycle: rate limiting and
# server-side faults. Everything else (auth, bad request, schema, output
# validation) fails fast to the safe fallback.
_RETRYABLE_CODES = frozenset({429, 500, 502, 503, 504})


def _is_retryable(exc: Exception) -> bool:
    """True only for transient provider-side failures (429 / 5xx).

    Duck-types the SDK's APIError.code so tests and SDK-less environments work;
    any ServerError subclass also qualifies as a safety net.
    """
    if getattr(exc, "code", None) in _RETRYABLE_CODES:
        return True
    try:
        from google.genai import errors as genai_errors
        return isinstance(exc, genai_errors.ServerError)
    except Exception:
        return False


def _strip_fences(text: str) -> str:
    """Remove Markdown code fences around a JSON payload (defensive only).

    With response_mime_type="application/json" the model should return raw
    JSON, but fences still occur; stripping them is a no-op otherwise.
    """
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    return m.group(1).strip() if m else text.strip()


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, api_key: str, model: str = "gemini-3.7-flash",
                 temperature: float = 0.0, timeout_s: float = 60.0,
                 max_output_tokens: int = MAX_OUTPUT_TOKENS,
                 max_retries: int = 2, retry_base_delay_s: float = 1.0,
                 client=None, sleeper=None) -> None:
        self.api_key = api_key or ""
        self.model = model or "gemini-3.7-flash"
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.max_output_tokens = max_output_tokens or MAX_OUTPUT_TOKENS
        # Bounded retries: max_retries after the initial attempt (default 2 →
        # at most 3 total attempts), exponential backoff from base delay.
        self.max_retries = max(0, max_retries)
        self.retry_base_delay_s = max(0.0, retry_base_delay_s)
        # Injected client (tests) or lazily-created SDK client (production).
        # The google-genai SDK is imported lazily so unit tests and offline
        # (stub) runs never require the package or a key.
        self._client = client
        # Injected sleeper (tests) or time.sleep (production).
        self._sleeper = sleeper or time.sleep

    def _get_client(self):
        if self._client is not None:
            return self._client
        from google import genai
        from google.genai import types
        return genai.Client(
            api_key=self.api_key,
            http_options=types.HttpOptions(timeout=int(self.timeout_s * 1000)),
        )

    def _request_config(self):
        from google.genai import types
        return types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            response_mime_type="application/json",
            response_json_schema=_response_schema(),
        )

    @staticmethod
    def _parse_response(response) -> GeminiStructuredAnswer:
        """Structured output first (response.parsed), raw-JSON fallback.

        Note: the installed SDK populates response.parsed only for the
        response_schema form; with the response_json_schema dict form used
        here, parsed is None and the text fallback is the live path.

        Raises on empty/blocked/malformed/truncated responses — callers abstain.
        """
        parsed = getattr(response, "parsed", None)
        if parsed is not None:
            return GeminiStructuredAnswer.model_validate(parsed)
        try:
            text = (response.text or "").strip()
        except Exception as exc:
            raise ValueError(f"no readable response text: {type(exc).__name__}") from exc
        if not text:
            raise ValueError("empty response text")
        return GeminiStructuredAnswer.model_validate(json.loads(_strip_fences(text)))

    def generate(self, question: str, passages: list[dict], doc_lookup: dict) -> GroundedAnswer:
        if not passages:
            return StubProvider().generate(question, [], doc_lookup)
        if not self.api_key and self._client is None:
            # Same contract as the OpenAI-compatible provider: without a key,
            # degrade to the offline stub rather than failing.
            return StubProvider().generate(question, passages, doc_lookup)

        evidence_text, id_map = _evidence_block(passages, doc_lookup)
        user = (
            f"Clinical question: {question}\n\n"
            f"Guideline evidence (AfroMedX retrieval — only these IDs exist):\n"
            f"{evidence_text}\n\n"
            "Respond with: body_markdown (concise answer, recommendations tagged like [E2]), "
            "key_points (short bullets), citation_ids (every [E#] you relied on, e.g. [\"E2\"]), "
            "abstained (true when the evidence is insufficient, conflicting, ambiguous, does not "
            "directly answer the question, or the question rests on a premise the evidence does "
            "not support), and abstention_reason (short, only when abstaining)."
        )
        attempts = 1 + self.max_retries
        structured = None
        for attempt in range(1, attempts + 1):
            try:
                client = self._get_client()
                response = client.models.generate_content(
                    model=self.model, contents=user, config=self._request_config())
                structured = self._parse_response(response)
                break
            except (ValidationError, ValueError, KeyError, TypeError, AttributeError) as exc:
                # Unusable model output (malformed/blocked/empty): the output,
                # not the transport, failed — retrying cannot help.
                log.warning("Gemini output unusable (%s); abstaining", type(exc).__name__)
                return StubProvider().generate(question, passages, doc_lookup)
            except Exception as exc:  # API / network / timeout / SDK errors
                if _is_retryable(exc) and attempt < attempts:
                    delay = self.retry_base_delay_s * (2 ** (attempt - 1))
                    log.warning(
                        "Gemini request failed (attempt %d/%d): model=%s error=%s "
                        "detail=%s; retrying in %.1fs",
                        attempt, attempts, self.model, type(exc).__name__,
                        _safe_detail(str(exc), self.api_key), delay)
                    self._sleeper(delay)
                    continue
                log.warning(
                    "Gemini request failed (attempt %d/%d): model=%s error=%s "
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
            log.warning("Gemini returned unknown evidence IDs %s; dropping", invalid_ids)

        if structured.abstained:
            if structured.abstention_reason:
                log.info("Gemini abstained: %s", structured.abstention_reason)
            return _abstain(passages, doc_lookup, valid_ids, id_map)
        if not valid_ids:
            # Answered but cited nothing usable: claims would be unsupported.
            log.warning("Gemini answered without valid citation IDs; abstaining")
            return _abstain(passages, doc_lookup)
        body = (structured.body_markdown or "").strip()
        if not body:
            log.warning("Gemini returned an empty answer body; abstaining")
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
