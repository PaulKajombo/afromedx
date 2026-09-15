"""Hybrid search: semantic cosine + keyword overlap, fused with metadata filtering.

Keeps it simple but satisfies: natural-language queries, keyword fallback
("IV artesunate dose adult"), metadata filter, and a min-score gate so the
answer layer can abstain (NO SOURCE = NO ANSWER).
"""
from __future__ import annotations

import re

import numpy as np

from .embedder import _STOP

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Small clinical synonym map so "management" <-> "treatment", "child" <-> paediatric, etc.
SYNONYMS: dict[str, list[str]] = {
    "treatment": ["management", "therapy", "regimen"],
    "management": ["treatment", "therapy"],
    "dose": ["dosage", "dosing"],
    "dosage": ["dose", "dosing"],
    "child": ["paediatric", "pediatric", "children"],
    "children": ["paediatric", "pediatric", "child"],
    "kidney": ["renal"],
    "renal": ["kidney"],
    "tb": ["tuberculosis"],
    "tuberculosis": ["tb"],
    "hiv": ["art", "antiretroviral"],
    "art": ["hiv", "antiretroviral"],
}


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def expand(tokens: list[str]) -> set[str]:
    out = set(tokens)
    for t in tokens:
        out.update(SYNONYMS.get(t, ()))
    return out


def keyword_scores(query: str, texts: list[str]) -> np.ndarray:
    """IDF-weighted token overlap with synonym expansion. Returns 0..1 array."""
    qtok = expand(tokenize(query))
    if not qtok:
        return np.zeros(len(texts), dtype=np.float32)
    # document frequencies over the corpus
    df: dict[str, int] = {}
    tokenized = [set(tokenize(t)) for t in texts]
    for toks in tokenized:
        for t in toks:
            df[t] = df.get(t, 0) + 1
    n = max(1, len(texts))
    import math
    idf = {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}
    qidf = sum(idf.get(t, math.log(n + 1) + 1.0) for t in qtok)
    scores = np.zeros(len(texts), dtype=np.float32)
    for i, toks in enumerate(tokenized):
        hit = qtok.intersection(toks)
        # also count synonym hits already in expanded set
        s = sum(idf.get(t, 0.0) for t in hit)
        scores[i] = s / (qidf + 1e-9)
    return scores


# Generic clinical glue terms — must NOT satisfy the in-scope coverage gate on their own.
GENERIC_TERMS = frozenset(
    "recommend recommended recommend should use used give given start started begin "
    "patient patients child children adult adults dosed dosage medications medicine "
    "medication drug drugs therapy treat treated treating management treat manage "
    "diagnose diagnosed diagnosis diagnostic identify investigation investigations".split()
)


def _term_set(texts: list[str]) -> set[str]:
    ts: set[str] = set()
    for t in texts:
        ts.update(tokenize(t))
    return ts


def _in_scope(query: str, texts: list[str], corpus_terms: set[str]) -> bool:
    """Out-of-scope gate: require at least one non-generic query term (or synonym)
    to actually appear in the indexed corpus. Zero content overlap ⇒ abstain."""
    sig = [t for t in tokenize(query) if t not in GENERIC_TERMS and t not in _STOP]
    if not sig:
        return False  # nothing but generic wording ⇒ not answerable
    expanded = expand(sig)
    return any(t in corpus_terms for t in expanded)


def search(
    store,
    query: str,
    *,
    top_k: int = 6,
    min_score: float = 0.08,
    semantic_weight: float = 0.65,
    document_id: str | None = None,
) -> list[dict]:
    """Return ranked passages: [{chunk, semantic, keyword, score}]. Empty if below gate."""
    q = (query or "").strip()
    if not q or not store.chunks:
        return []
    texts = [c.get("text", "") for c in store.chunks]
    if not _in_scope(q, texts, _term_set(texts)):
        return []

    # semantic cosine
    try:
        qv = store.embedder.encode([q]).astype(np.float32).ravel()
        qv = qv / (np.linalg.norm(qv) + 1e-9)
        sem = store.vectors @ qv if store.vectors is not None else np.zeros(len(texts))
        sem = np.clip(np.asarray(sem, dtype=np.float32).ravel(), 0, 1)
    except Exception:
        sem = np.zeros(len(texts), dtype=np.float32)

    kw = keyword_scores(q, texts)
    fused = semantic_weight * sem + (1.0 - semantic_weight) * kw

    order = np.argsort(-fused)
    results: list[dict] = []
    for idx in order[: max(top_k * 3, top_k)]:
        c = store.chunks[int(idx)]
        if document_id and c.get("document_id") != document_id:
            continue
        s = float(fused[int(idx)])
        if s < min_score:
            continue
        results.append({
            "chunk": c,
            "semantic": float(sem[int(idx)]),
            "keyword": float(kw[int(idx)]),
            "score": s,
        })
        if len(results) >= top_k:
            break
    return results
