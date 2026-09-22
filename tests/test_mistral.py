"""Mistral provider tests: structured parsing, citation-ID validation, abstention,
retry behavior, failure handling.

No live API key and no network: the SDK client is replaced with fakes injected
through the provider constructor. The mistralai package is never imported here.
"""
import json
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings
from app.llm.factory import get_provider
from app.llm.mistral_provider import MistralProvider
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


class _HttpError(Exception):
    """Duck-typed stand-in for the SDK error (carries int .status_code)."""
    def __init__(self, status_code, message="failure"):
        super().__init__(f"HTTP {status_code}: {message}")
        self.status_code = status_code


class _ScriptedChat:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls: list[dict] = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        assert self.outcomes, "more attempts than scripted outcomes"
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class _ScriptedClient:
    def __init__(self, outcomes):
        self.chat = _ScriptedChat(outcomes)


class _Sleeper:
    def __init__(self):
        self.delays: list[float] = []

    def __call__(self, seconds):
        self.delays.append(seconds)


def _chat_resp(payload: dict):
    return SimpleNamespace(choices=[
        SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))])


def _payload(**over):
    d = {"body_markdown": "Give IV artesunate 2.4 mg/kg (see page 999 of Malaria Handbook).",
         "key_points": ["IV artesunate 2.4 mg/kg at 0, 12 and 24 hours"],
         "citation_ids": ["E1"],
         "abstained": False,
         "abstention_reason": None}
    d.update(over)
    return d


def _provider(outcomes, api_key="k", **kw):
    sleeper = _Sleeper()
    provider = MistralProvider(api_key=api_key, client=_ScriptedClient(outcomes),
                               sleeper=sleeper, **kw)
    return provider, sleeper


# ---- 1. factory selection ----

def test_factory_selects_mistral(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "mistral")
    monkeypatch.setattr(settings, "mistral_api_key", "test-key")
    provider = get_provider()
    assert isinstance(provider, MistralProvider)
    assert provider.model == settings.mistral_model


def test_factory_falls_back_to_stub_without_key(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "mistral")
    monkeypatch.setattr(settings, "mistral_api_key", "")
    assert isinstance(get_provider(), StubProvider)


def test_gemini_still_selectable(monkeypatch):
    from app.llm.gemini_provider import GeminiProvider
    monkeypatch.setattr(settings, "llm_provider", "gemini")
    monkeypatch.setattr(settings, "gemini_api_key", "test-key")
    assert isinstance(get_provider(), GeminiProvider)


# ---- 2. missing API key ----

def test_missing_key_falls_back_to_stub_offline():
    ans = MistralProvider(api_key="").generate(
        "how do i repair a bicycle tyre", _passages(), _doc_lookup())
    assert ans.abstained
    assert "couldn't find sufficient information" in ans.body_markdown


def test_empty_passages_never_calls_api():
    client = _ScriptedClient([_HttpError(500)])
    ans = MistralProvider(api_key="k", client=client).generate("q", [], _doc_lookup())
    assert ans.abstained
    assert client.chat.calls == []


# ---- 3. structured response parsing ----

def test_structured_answer_parses():
    provider, _ = _provider([_chat_resp(_payload())])
    ans = provider.generate("artesunate dose?", _passages(), _doc_lookup())
    assert not ans.abstained
    assert "2.4 mg/kg" in ans.body_markdown
    assert ans.key_points == ["IV artesunate 2.4 mg/kg at 0, 12 and 24 hours"]


# ---- 4. valid citation IDs resolve to retrieval metadata ----

def test_valid_citation_ids_resolve_to_retrieval_metadata():
    provider, _ = _provider([_chat_resp(_payload(citation_ids=["E2"]))])
    ans = provider.generate("confirm TB?", _passages(), _doc_lookup())
    assert not ans.abstained
    # Critical rule: citation metadata comes from retrieval, not the model.
    # The model body mentions invented "page 999 / Malaria Handbook".
    assert len(ans.citations) == 1
    assert ans.citations[0]["page"] == 45
    assert ans.citations[0]["document"] == "TB Guideline"
    assert ans.citations[0]["chunk_id"] == "d2::p45::3"


# ---- 5. invalid citation IDs ----

def test_invalid_ids_dropped_case_insensitive_deduped():
    provider, _ = _provider([_chat_resp(_payload(citation_ids=["E9", "e1", "E1"]))])
    ans = provider.generate("artesunate dose?", _passages(), _doc_lookup())
    assert not ans.abstained
    assert len(ans.citations) == 1
    assert ans.citations[0]["chunk_id"] == "d1::p12::0"


def test_all_invalid_ids_abstains():
    provider, _ = _provider([_chat_resp(_payload(citation_ids=["E7"]))])
    ans = provider.generate("artesunate dose?", _passages(), _doc_lookup())
    assert ans.abstained
    assert ans.citations == []


# ---- 6. abstention ----

def test_model_abstention_flag_respected():
    provider, _ = _provider([_chat_resp(_payload(
        abstained=True, abstention_reason="Evidence does not cover dosage."))])
    ans = provider.generate("artesunate dose?", _passages(), _doc_lookup())
    assert ans.abstained
    assert ans.body_markdown == ABSTAIN_TEXT


def test_empty_body_abstains_even_with_valid_id():
    provider, _ = _provider([_chat_resp(_payload(body_markdown="  "))])
    ans = provider.generate("artesunate dose?", _passages(), _doc_lookup())
    assert ans.abstained


# ---- 7. malformed output ----

def test_malformed_output_falls_back_to_stub_without_retry():
    bad_message = SimpleNamespace(content="not json {{{")
    bad_choice = SimpleNamespace(message=bad_message)
    bad_response = SimpleNamespace(choices=[bad_choice])
    provider, sleeper = _provider([bad_response])
    ans = provider.generate("artesunate dose?", _passages(), _doc_lookup())
    expected = StubProvider().generate("artesunate dose?", _passages(), _doc_lookup())
    assert ans.body_markdown == expected.body_markdown
    assert len(provider._client.chat.calls) == 1
    assert sleeper.delays == []


def test_empty_content_abstains():
    empty_message = SimpleNamespace(content="  ")
    empty_choice = SimpleNamespace(message=empty_message)
    empty_response = SimpleNamespace(choices=[empty_choice])
    provider, _ = _provider([empty_response])
    ans = provider.generate("how do i repair a bicycle tyre", _passages(), _doc_lookup())
    assert ans.abstained
    assert "couldn't find sufficient information" in ans.body_markdown


# ---- 8. 503 then success ----

def test_503_then_success_succeeds_on_second_attempt():
    provider, sleeper = _provider([_HttpError(503), _chat_resp(_payload())])
    ans = provider.generate("artesunate dose?", _passages(), _doc_lookup())
    assert not ans.abstained
    assert ans.citations[0]["chunk_id"] == "d1::p12::0"
    assert len(provider._client.chat.calls) == 2
    assert sleeper.delays == [1.0]


# ---- 9. 503 x3 -> fallback ----

def test_503_three_times_falls_back_after_three_attempts():
    provider, sleeper = _provider([_HttpError(503)] * 3)
    ans = provider.generate("artesunate dose?", _passages(), _doc_lookup())
    expected = StubProvider().generate("artesunate dose?", _passages(), _doc_lookup())
    assert ans.abstained == expected.abstained
    assert ans.body_markdown == expected.body_markdown
    assert len(provider._client.chat.calls) == 3
    assert sleeper.delays == [1.0, 2.0]


# ---- 10. non-retryable errors do not retry ----

@pytest.mark.parametrize("code", [400, 401, 403, 404])
def test_client_errors_do_not_retry(code):
    provider, sleeper = _provider([_HttpError(code, "not retryable")] * 3)
    ans = provider.generate("artesunate dose?", _passages(), _doc_lookup())
    expected = StubProvider().generate("artesunate dose?", _passages(), _doc_lookup())
    assert ans.body_markdown == expected.body_markdown
    assert len(provider._client.chat.calls) == 1
    assert sleeper.delays == []


# ---- 11. API error does not leak secrets ----

def test_api_error_does_not_leak_key():
    secret = "SECRET-XYZ-1234567890"
    provider, _ = _provider([_HttpError(503, f"boom {secret}")] * 3,
                            api_key=secret)
    ans = provider.generate("artesunate dose?", _passages(), _doc_lookup())
    expected = StubProvider().generate("artesunate dose?", _passages(), _doc_lookup())
    assert secret not in ans.body_markdown
    assert ans.body_markdown == expected.body_markdown
