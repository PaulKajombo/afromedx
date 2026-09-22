"""Gemini provider tests: structured parsing, citation-ID validation, abstention, failures.

No live API key and no network: the SDK client is replaced with fakes injected
through the provider constructor. The google-genai package is never imported here.
"""
import json
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings
from app.llm.factory import get_provider
from app.llm.gemini_provider import GeminiProvider
from app.llm.stub import ABSTAIN_TEXT, StubProvider


def _doc_lookup():
    return {
        "d1": {"title": "Malaria Guideline", "edition": "6th Edition",
               "publication_year": 2025},
        "d2": {"title": "TB Guideline", "edition": "2nd Edition",
               "publication_year": 2021},
    }


def _passages():
    c1 = {"id": "d1::p12::0", "document_id": "d1", "page": 12,
          "section": "Severe Malaria", "subsection": "",
          "text": "Recommended treatment: IV artesunate 2.4 mg/kg at 0, 12 and 24 hours, then daily."}
    c2 = {"id": "d2::p45::3", "document_id": "d2", "page": 45,
          "section": "TB Diagnosis", "subsection": "",
          "text": "Confirm pulmonary TB with GeneXpert MTB/RIF."}
    return [{"chunk": c1, "score": 0.8}, {"chunk": c2, "score": 0.6}]


class _FakeModels:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls: list[dict] = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class _FakeClient:
    def __init__(self, response=None, error=None):
        self.models = _FakeModels(response, error)


def _parsed_resp(payload: dict):
    return SimpleNamespace(parsed=payload, text=json.dumps(payload))


def _payload(**over):
    d = {"body_markdown": "Give IV artesunate 2.4 mg/kg (see page 999 of Malaria Handbook).",
         "key_points": ["IV artesunate 2.4 mg/kg at 0, 12 and 24 hours"],
         "citation_ids": ["E1"],
         "abstained": False,
         "abstention_reason": None}
    d.update(over)
    return d


# ---- factory ----

def test_factory_selects_gemini(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "gemini")
    monkeypatch.setattr(settings, "gemini_api_key", "test-key")
    provider = get_provider()
    assert isinstance(provider, GeminiProvider)
    assert provider.model == settings.gemini_model


def test_factory_falls_back_to_stub_without_key(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "gemini")
    monkeypatch.setattr(settings, "gemini_api_key", "")
    assert isinstance(get_provider(), StubProvider)


# ---- missing key / empty retrieval: no API use ----

def test_missing_key_falls_back_to_stub_offline():
    ans = GeminiProvider(api_key="").generate(
        "how do i repair a bicycle tyre", _passages(), _doc_lookup())
    assert ans.abstained
    assert "couldn't find sufficient information" in ans.body_markdown


def test_empty_passages_never_calls_api():
    client = _FakeClient(error=AssertionError("must not be called"))
    ans = GeminiProvider(api_key="k", client=client).generate("q", [], _doc_lookup())
    assert ans.abstained
    assert client.models.calls == []


# ---- structured output + citation-ID validation ----

def test_valid_citation_id_resolves_to_retrieval_metadata():
    client = _FakeClient(response=_parsed_resp(_payload()))
    ans = GeminiProvider(api_key="k", client=client).generate(
        "artesunate dose?", _passages(), _doc_lookup())
    assert not ans.abstained
    assert "2.4 mg/kg" in ans.body_markdown
    # Critical rule: citation metadata comes from retrieval, not the model.
    # The model body mentions invented "page 999 / Malaria Handbook".
    assert len(ans.citations) == 1
    assert ans.citations[0]["page"] == 12
    assert ans.citations[0]["document"] == "Malaria Guideline"
    assert ans.citations[0]["chunk_id"] == "d1::p12::0"
    assert ans.key_points == ["IV artesunate 2.4 mg/kg at 0, 12 and 24 hours"]


def test_invalid_ids_dropped_case_insensitive_deduped():
    client = _FakeClient(response=_parsed_resp(_payload(citation_ids=["E9", "e2", "E2"])))
    ans = GeminiProvider(api_key="k", client=client).generate(
        "confirm TB?", _passages(), _doc_lookup())
    assert not ans.abstained
    assert len(ans.citations) == 1
    assert ans.citations[0]["chunk_id"] == "d2::p45::3"


def test_all_invalid_ids_abstains():
    client = _FakeClient(response=_parsed_resp(_payload(citation_ids=["E7"])))
    ans = GeminiProvider(api_key="k", client=client).generate(
        "artesunate dose?", _passages(), _doc_lookup())
    assert ans.abstained
    assert ans.citations == []


def test_model_abstention_flag_respected():
    client = _FakeClient(response=_parsed_resp(
        _payload(abstained=True, abstention_reason="Evidence does not cover dosage.")))
    ans = GeminiProvider(api_key="k", client=client).generate(
        "artesunate dose?", _passages(), _doc_lookup())
    assert ans.abstained
    assert ans.body_markdown == ABSTAIN_TEXT


def test_empty_body_abstains_even_with_valid_id():
    client = _FakeClient(response=_parsed_resp(_payload(body_markdown="  ")))
    ans = GeminiProvider(api_key="k", client=client).generate(
        "artesunate dose?", _passages(), _doc_lookup())
    assert ans.abstained


# ---- malformed / failure handling ----

def test_json_text_fallback_when_no_parsed_attribute():
    payload = _payload(citation_ids=["E2"], body_markdown="Confirm with GeneXpert.")
    client = _FakeClient(response=SimpleNamespace(parsed=None, text=json.dumps(payload)))
    ans = GeminiProvider(api_key="k", client=client).generate(
        "confirm TB?", _passages(), _doc_lookup())
    assert not ans.abstained
    assert ans.citations[0]["chunk_id"] == "d2::p45::3"


def test_malformed_response_falls_back_to_stub():
    # Same contract as the OpenAI-compatible provider: an unusable model
    # response degrades to the offline extractive stub, which only answers
    # from retrieved evidence (never a confabulated clinical answer).
    client = _FakeClient(response=SimpleNamespace(parsed=None, text="not json {{{"))
    ans = GeminiProvider(api_key="k", client=client).generate(
        "artesunate dose?", _passages(), _doc_lookup())
    expected = StubProvider().generate("artesunate dose?", _passages(), _doc_lookup())
    assert ans.abstained == expected.abstained
    assert ans.body_markdown == expected.body_markdown
    assert ans.citations == expected.citations


def test_missing_fields_abstains():
    client = _FakeClient(response=_parsed_resp({"unexpected": "shape"}))
    ans = GeminiProvider(api_key="k", client=client).generate(
        "artesunate dose?", _passages(), _doc_lookup())
    # No body and no valid citations -> cannot support any claim.
    assert ans.abstained


def test_api_error_falls_back_to_stub_without_leaking():
    client = _FakeClient(error=RuntimeError("boom"))
    ans = GeminiProvider(api_key="k", client=client).generate(
        "artesunate dose?", _passages(), _doc_lookup())
    expected = StubProvider().generate("artesunate dose?", _passages(), _doc_lookup())
    assert ans.abstained == expected.abstained
    assert ans.body_markdown == expected.body_markdown
    assert "boom" not in ans.body_markdown


def test_api_error_with_unsupported_question_abstains():
    client = _FakeClient(error=RuntimeError("boom"))
    ans = GeminiProvider(api_key="k", client=client).generate(
        "how do i repair a bicycle tyre", _passages(), _doc_lookup())
    assert ans.abstained
    assert "couldn't find sufficient information" in ans.body_markdown


# ---- response-schema compatibility (no anyOf for the API's schema subset) ----

def test_response_schema_has_no_anyof():
    from app.llm.gemini_provider import _response_schema
    schema = _response_schema()
    assert "anyOf" not in json.dumps(schema)
    assert schema["properties"]["abstention_reason"]["type"] == ["null", "string"]
    assert schema["properties"]["body_markdown"]["type"] == "string"


def test_nullable_contract_preserved():
    from app.llm.gemini_provider import GeminiStructuredAnswer
    ans = GeminiStructuredAnswer.model_validate({
        "body_markdown": "x", "key_points": [], "citation_ids": ["E1"],
        "abstained": True, "abstention_reason": None})
    assert ans.abstention_reason is None


# ---- discovered live failure modes (MAX_TOKENS truncation, fences) ----

def test_fenced_json_fallback_parses():
    payload = _payload(citation_ids=["E1"])
    fenced = "```json\n" + json.dumps(payload) + "\n```"
    client = _FakeClient(response=SimpleNamespace(parsed=None, text=fenced))
    ans = GeminiProvider(api_key="k", client=client).generate(
        "artesunate dose?", _passages(), _doc_lookup())
    assert not ans.abstained
    assert ans.citations[0]["chunk_id"] == "d1::p12::0"


def test_truncated_json_falls_back_to_stub():
    # Live lesson: finish_reason MAX_TOKENS yields cut-off JSON. Unrepairable
    # output must degrade to the stub, never to a partial clinical answer.
    payload = json.dumps(_payload())
    client = _FakeClient(response=SimpleNamespace(parsed=None, text=payload[:len(payload)//2]))
    ans = GeminiProvider(api_key="k", client=client).generate(
        "artesunate dose?", _passages(), _doc_lookup())
    expected = StubProvider().generate("artesunate dose?", _passages(), _doc_lookup())
    assert ans.abstained == expected.abstained
    assert ans.body_markdown == expected.body_markdown


def test_max_output_tokens_default_and_override():
    assert GeminiProvider(api_key="k").max_output_tokens == 2048
    assert GeminiProvider(api_key="k", max_output_tokens=512).max_output_tokens == 512


# ---- bounded retry (transient 429/5xx only; max 2 retries = 3 attempts) ----

class _HttpError(Exception):
    """Duck-typed stand-in for the SDK's APIError (carries int .code)."""
    def __init__(self, code, message="failure"):
        super().__init__(f"{code} {message}")
        self.code = code


class _ScriptedModels:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls: list[dict] = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        assert self.outcomes, "more attempts than scripted outcomes"
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class _ScriptedClient:
    def __init__(self, outcomes):
        self.models = _ScriptedModels(outcomes)


class _Sleeper:
    def __init__(self):
        self.delays: list[float] = []

    def __call__(self, seconds):
        self.delays.append(seconds)


def _provider(outcomes, **kw):
    sleeper = _Sleeper()
    provider = GeminiProvider(api_key="k", client=_ScriptedClient(outcomes),
                              sleeper=sleeper, **kw)
    return provider, sleeper


def test_503_then_success_succeeds_on_second_attempt():
    provider, sleeper = _provider([_HttpError(503), _parsed_resp(_payload())])
    ans = provider.generate("artesunate dose?", _passages(), _doc_lookup())
    assert not ans.abstained
    assert ans.citations[0]["chunk_id"] == "d1::p12::0"
    assert len(provider._client.models.calls) == 2
    assert sleeper.delays == [1.0]


def test_503_three_times_falls_back_after_three_attempts():
    provider, sleeper = _provider([_HttpError(503)] * 3)
    ans = provider.generate("artesunate dose?", _passages(), _doc_lookup())
    expected = StubProvider().generate("artesunate dose?", _passages(), _doc_lookup())
    assert ans.abstained == expected.abstained
    assert ans.body_markdown == expected.body_markdown
    assert len(provider._client.models.calls) == 3
    assert sleeper.delays == [1.0, 2.0]  # exponential backoff from base delay


@pytest.mark.parametrize("code", [400, 401, 403, 404])
def test_client_errors_do_not_retry(code):
    provider, sleeper = _provider([_HttpError(code, "not retryable")] * 3)
    ans = provider.generate("artesunate dose?", _passages(), _doc_lookup())
    expected = StubProvider().generate("artesunate dose?", _passages(), _doc_lookup())
    assert ans.body_markdown == expected.body_markdown
    assert len(provider._client.models.calls) == 1
    assert sleeper.delays == []


def test_malformed_output_does_not_retry():
    client = _FakeClient(response=SimpleNamespace(parsed=None, text="not json {{{"))
    provider = GeminiProvider(api_key="k", client=client, sleeper=_Sleeper())
    provider.generate("artesunate dose?", _passages(), _doc_lookup())
    assert len(client.models.calls) == 1
