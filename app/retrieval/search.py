"""Hybrid search: semantic cosine + keyword overlap, fused with metadata filtering.

Keeps it simple but satisfies: natural-language queries, keyword fallback
("IV artesunate dose adult"), metadata filter, and a min-score gate so the
answer layer can abstain (NO SOURCE = NO ANSWER).
"""
from __future__ import annotations

import re

import numpy as np

from .embedder import _STOP
from ..config import settings

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Negation/contraction normalization so phrasings converge BEFORE matching:
# "drugs you cannot give" and "drugs you can not give" must retrieve the same
# evidence. "cannot/can't/..." map to "not" (a stopword downstream), so the
# junk high-IDF token "cannot" can no longer swing ranking or sink the
# in-scope gate. The ORIGINAL question is always kept for answer generation —
# this only affects retrieval matching, identically on query and chunk sides.
_CONTRACTIONS = {
    "can't": "not", "cannot": "not", "cant": "not",
    "won't": "not", "don't": "not", "doesn't": "not", "didn't": "not",
    "isn't": "not", "aren't": "not", "wasn't": "not", "weren't": "not",
    "haven't": "not", "hasn't": "not", "hadn't": "not",
    "wouldn't": "not", "couldn't": "not", "shouldn't": "not",
    "mustn't": "not", "needn't": "not",
}
_CONTRACTION_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(_CONTRACTIONS)) + r")\b")

# Small clinical synonym map so "management" <-> "treatment", "child" <-> paediatric, etc.
SYNONYMS: dict[str, list[str]] = {
    "treatment": ["management", "therapy", "regimen"],
    "management": ["treatment", "therapy"],
    "dose": ["dosage", "dosing"],
    "dosage": ["dose", "dosing"],
    "child": ["paediatric", "pediatric", "children"],
    "children": ["paediatric", "pediatric", "child"],
    "kids": ["children", "child", "paediatric", "pediatric"],
    "kid": ["children", "child", "paediatric", "pediatric"],
    "tablet": ["tablets", "dose", "dosage"],
    "tablets": ["tablet", "dose", "dosage"],
    "kidney": ["renal"],
    "renal": ["kidney"],
    "tb": ["tuberculosis"],
    "tuberculosis": ["tb"],
    "hiv": ["art", "antiretroviral"],
    "art": ["hiv", "antiretroviral"],
}

# P1B: clinical abbreviation / verb-form / spelling mappings, each justified by
# terminology actually present in the indexed Malawi guidelines + benchmark:
#   VL/TLD/mg- Evaluator queries vs guideline abbreviations ("VL", "TLD");
#   started/starting/initiation (question verbs vs "Start"/"Starting" headings);
#   failed/defines (question wording vs guideline phrasing "treatment failure",
#   "defined as"); pregnant/adult/woman number and gender forms (m03/x03
#   "pregnant woman" vs "pregnancy"/"women"); diagnosed (h01 "newly diagnosed"
#   vs "Diagnosing"); haemorrhage/paracetamol en-GB vs guideline variants
#   ("hemorrhage", "Panadol"). Single tokens only: the keyword arm matches on
#   token sets, so multi-word values would never fire. Deliberately small —
#   no arbitrary dictionary.
CLINICAL_SYNONYMS: dict[str, list[str]] = {
    "vl": ["viral", "load"],
    "tld": ["tenofovir", "lamivudine", "dolutegravir"],
    "started": ["start"],
    "starting": ["start"],
    "initiation": ["start", "initiate"],
    "initiate": ["start", "initiation"],
    "failed": ["failure"],
    "failing": ["failure"],
    "defines": ["define", "defined", "definition"],
    "defined": ["define", "definition"],
    "pregnant": ["pregnancy"],
    "pregnancy": ["pregnant"],
    "adult": ["adults"],
    "adults": ["adult"],
    "woman": ["women"],
    "women": ["woman"],
    "diagnosed": ["diagnose", "diagnosis"],
    "haemorrhage": ["hemorrhage"],
    "hemorrhage": ["haemorrhage"],
    "paracetamol": ["panadol"],
    "panadol": ["paracetamol"],
}


def synonyms_of(term: str) -> tuple:
    """Active synonym list for a term: base map plus clinical map when enabled
    (settings.clinical_synonyms, default on). Single lookup point so keyword
    scoring, the in-scope gate, and the stub grounding gate stay consistent."""
    out = list(SYNONYMS.get(term, ()))
    if settings.clinical_synonyms:
        for s in CLINICAL_SYNONYMS.get(term, ()):
            if s not in out:
                out.append(s)
    return tuple(out)


def tokenize(text: str) -> list[str]:
    t = (text or "").lower()
    t = _CONTRACTION_RE.sub(lambda m: _CONTRACTIONS[m.group(1)], t)
    return _TOKEN_RE.findall(t)


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein distance (stdlib only, O(min) space)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _is_bare_topic(query: str) -> bool:
    """True for statement/bare-phrase queries with no question or action framing.

    "migraine headache" and "drugs you cannot give in asthma" name a topic
    without saying what is wanted; "how do you treat ..." does. Detected on
    normalized tokens: any question mark, wh-word, modal, or
    treatment/management/diagnosis verb keeps the query as-is.
    """
    if "?" in query:
        return False
    toks = set(tokenize(query))
    framed = {"what", "how", "when", "which", "who", "why", "can", "should",
              "is", "are", "do", "does", "did", "will", "would", "could",
              "treat", "treating", "treatment", "manage", "managing",
              "management", "diagnose", "diagnosis", "dose", "prevent"}
    return not (toks & framed)


def expand_bare_topic(query: str) -> str:
    """Append management framing to bare-topic queries (retrieval only).

    The ORIGINAL question is always kept for answer generation, titles, and
    evaluation — only ranking sees the expanded form, so a bare phrase
    retrieves management content instead of definitional fragments.
    """
    if _is_bare_topic(query):
        return query.strip() + " treatment management"
    return query


def fuzzy_match(term: str, candidates: set[str], max_dist: int = 3) -> str | None:
    """Closest candidate within max_dist edits ("asam" -> "asthma", dist 3).

    Only fires for tokens of length >= 4 against same-initial candidates of
    similar length; returns the smallest-distance match (ties: shortest).
    Returns None when nothing is close, so genuinely novel terms are never
    "corrected" into wrong content. Callers decide how much a fuzzy bridge
    counts: the in-scope gate weights it at HALF an exact term, so a lone
    near-miss ("tyre" -> "type") can never pass the gate on its own.
    """
    if len(term) < 4:
        return None
    best: str | None = None
    best_d = max_dist + 1
    for c in candidates:
        if not c or c == term or c[:1] != term[:1]:
            continue
        if abs(len(c) - len(term)) > max_dist:
            continue
        d = _edit_distance(term, c)
        if d < best_d or (d == best_d and best is not None and len(c) < len(best)):
            best, best_d = c, d
    return best if best_d <= max_dist else None


def term_covered(term: str, terms: set[str]) -> bool:
    """Does a query term bridge to a term set? Exact, synonym, then fuzzy.

    Shared by the in-scope gate and the stub grounding gate so both treat
    misspellings identically. Fuzzy applies ONLY when exact and synonym
    matching fail, so correctly-spelled queries behave exactly as before.
    """
    if term in terms:
        return True
    if any(s in terms for s in synonyms_of(term)):
        return True
    return fuzzy_match(term, terms) is not None


def norm_text(text: str) -> str:
    """Canonical form for duplicate detection: lowercase alphanumeric tokens.

    Collapses whitespace/punctuation/case differences so byte-different but
    text-identical chunks (same guideline ingested under two doc ids) compare
    equal. Query-time only.
    """
    return " ".join(_TOKEN_RE.findall((text or "").lower()))


def expand(tokens: list[str]) -> set[str]:
    out = set(tokens)
    for t in tokens:
        out.update(synonyms_of(t))
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
    # Spelling tolerance: tokens absent from the corpus are replaced by their
    # fuzzy correction ("asam" -> "asthma") so a typo neither dilutes the
    # score via max-IDF nor misses the overlap. Tokens with no close match
    # are kept as-is. Synonyms expand from the corrected forms below.
    vocab = set(df)
    corrected = set()
    for t in qtok:
        if t in df:
            corrected.add(t)
        else:
            fix = fuzzy_match(t, vocab)
            corrected.add(fix if fix else t)
    qtok = expand(corrected)
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


# Edition policy (verified from the PDFs/index themselves, never guessed):
# - paediatrics-handbook ("Malawian Handbook of Paediatrics") states on its
#   title page: Third Edition published 2008. Superseded by
#   paediatric-protocols-2018, COIN 2022 and the IMNCI materials -> EXCLUDED.
# - obgyn ("OBGYN Guidelines.pdf") states: Version 3.0, 15 December 2017.
#   Superseded by obs-gynae-2023 (Malawi Ob/Gyn Protocols 2023) -> EXCLUDED.
# Exclusion is query-time only (the index is untouched) and reversible via
# settings.excluded_docs.
EXCLUDED_DOCS = frozenset({"paediatrics-handbook", "obgyn"})

# Curated publication years, asserted ONLY where evidenced (filename edition
# markers, manifest metadata, or the documents' own title pages as extracted
# into the index). Docs not listed here are neutral: no boost, no penalty.
# Evidence: malaria-treatment 6th Ed Jan 2025 + mstg 6th Ed 2023 appear
# verbatim in their indexed chunks; the rest match manifest edition metadata.
DOC_YEARS: dict[str, int] = {
    "malaria-treatment": 2025,
    "mstg": 2023,
    "malaria-2020": 2020,
    "hiv-2022": 2022,
    "obs-gynae-2023": 2023,
    "sti-2025": 2025,
    "renal-2024": 2024,
    "paediatric-ncd-2024": 2024,
    "cancer-guidelines": 2026,
    "viral-hepatitis-2023": 2023,
    "imnci-2021": 2021,
    "imnci-chartbooklet-2022": 2022,
    "paediatric-protocols-2018": 2018,
    "sobo-2018": 2018,
    "coin-2022": 2022,
    "coin-training-2017": 2017,
    "iccm-2010": 2010,
}


def doc_year(document_id: str, docs: dict) -> int | None:
    """Best-known publication year: stored metadata first, curated map second."""
    meta_year = (docs.get(document_id, {}) or {}).get("publication_year")
    if isinstance(meta_year, int):
        return meta_year
    return DOC_YEARS.get(document_id)


def recency_bonus(document_id: str, docs: dict, weight: float) -> float:
    """Small [0, weight] bonus favouring newer guidelines.

    Newest dated doc gets the full weight, oldest dated doc gets ~0, undated
    docs get exactly 0 (neutral). Kept at tie-break scale (default 0.02):
    ablation showed 0.05 promotes merely-new docs (e.g. cancer-2026) over
    topically-correct ones, while 0.02 preserves relevance order and only
    settles near-ties. Same-topic edition conflicts are handled structurally
    by excluding superseded editions (EXCLUDED_DOCS), not by this bonus.
    """
    if not weight or weight <= 0:
        return 0.0
    year = doc_year(document_id, docs)
    if year is None:
        return 0.0
    years = list(DOC_YEARS.values())
    lo, hi = min(years), max(years)
    if hi <= lo:
        return 0.0
    return weight * (min(max(year, lo), hi) - lo) / (hi - lo)


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
    """Out-of-scope gate: abstain unless a MAJORITY of the query's significant
    tokens (or their synonyms) can be found in the indexed corpus.

    At sample scale 'any single hit term' was enough; over the full Malawi
    corpus words like "repair"/"dose" appear in medical text, so a lone generic
    clinical word bridging ("how to repair a bicycle tyre" -> "repair") is not
    evidence the question is answerable from these guidelines.
    """
    sig = [t for t in tokenize(query) if len(t) > 1 and t not in GENERIC_TERMS and t not in _STOP]
    if not sig:
        return False  # nothing but generic wording ⇒ not answerable

    def coverage(t: str) -> float:
        if t in corpus_terms:
            return 1.0
        if any(s in corpus_terms for s in synonyms_of(t)):
            return 1.0
        # A fuzzy bridge is weaker evidence of intent than an exact term: it
        # counts half, so a lone near-miss ("tyre" -> "type") can never pass
        # the gate on its own, while a typo among known terms still retrieves.
        return 0.5 if fuzzy_match(t, corpus_terms) is not None else 0.0

    covered = sum(coverage(t) for t in sig)
    # NOTE: >= (not >): two half-bridges ("avoided"+"asam") carry as much
    # intent as one exact term, and a lone typo ("asam" alone) retrieves the
    # evidently-intended content. Genuinely novel terms have no close match
    # and still score 0; non-clinical questions are caught by the model layer
    # (x01/x02/x03 verified abstaining when the provider is healthy).
    return covered / len(sig) >= 0.5


def search(
    store,
    query: str,
    *,
    top_k: int = 6,
    min_score: float = 0.08,
    semantic_weight: float = 0.80,
    document_id: str | None = None,
    section_weight: float | None = None,
    max_per_doc: int | None = None,
    collapse_dupes: bool | None = None,
    exclude_doc_ids: frozenset | set | None = None,
    recency_weight: float | None = None,
) -> list[dict]:
    """Return ranked passages: [{chunk, semantic, keyword, score}]. Empty if below gate."""
    q = (query or "").strip()
    if not q or not store.chunks:
        return []
    texts = [c.get("text", "") for c in store.chunks]
    # Section/title terms count for the scope gate too: headings carry the
    # clinical concepts ("When To Start Art") that body text may lack, and the
    # keyword arm already ranks on them (P1A) — gate and ranking must agree.
    docs = getattr(store, "docs", {}) or {}
    meta = [f"{docs.get(c.get('document_id'), {}).get('title', '')} "
            f"{c.get('section', '')} {c.get('subsection', '')}" for c in store.chunks]
    if not _in_scope(q, texts, _term_set(texts + meta)):
        return []

    # Bare-topic expansion (retrieval only): the original question is kept for
    # answer generation; ranking sees management framing so "migraine headache"
    # retrieves treatment content instead of definitional fragments.
    rq = expand_bare_topic(q)

    # semantic cosine
    try:
        qv = store.embedder.encode([rq]).astype(np.float32).ravel()
        qv = qv / (np.linalg.norm(qv) + 1e-9)
        sem = store.vectors @ qv if store.vectors is not None else np.zeros(len(texts))
        sem = np.clip(np.asarray(sem, dtype=np.float32).ravel(), 0, 1)
    except Exception:
        sem = np.zeros(len(texts), dtype=np.float32)

    kw = keyword_scores(rq, texts)
    # P1A: section/subsection/document-title matches, query-time only.
    # Headings are metadata (never embedded), so without this arm a section
    # titled exactly like the question (e.g. "When To Start Art") is invisible
    # to ranking. Blend is explicit via settings.section_weight (0 disables).
    sw = settings.section_weight if section_weight is None else section_weight
    if sw and sw > 0:
        kw_meta = keyword_scores(rq, meta)
        kw = (1.0 - sw) * kw + sw * kw_meta
    fused = semantic_weight * sem + (1.0 - semantic_weight) * kw
    # Edition policy: drop superseded documents, then favour newer editions.
    # Both are query-time only; the index is untouched.
    excluded = settings.excluded_docs if exclude_doc_ids is None else set(exclude_doc_ids)
    rw = settings.recency_weight if recency_weight is None else recency_weight
    if rw and rw > 0:
        docs = getattr(store, "docs", {}) or {}
        bonus = np.array([recency_bonus(c.get("document_id"), docs, rw)
                          for c in store.chunks], dtype=np.float32)
        fused = fused + bonus

    order = np.argsort(-fused)
    # P1C: per-document diversity cap — repeated chunks from one document must
    # not dominate top-K. Query-time only; the index is untouched.
    mpd = settings.max_per_doc if max_per_doc is None else max_per_doc
    do_dedup = settings.collapse_dupes if collapse_dupes is None else collapse_dupes
    per_doc: dict[str, int] = {}
    seen_texts: set[str] = set()
    results: list[dict] = []
    for idx in order[: max(top_k * 3, top_k)]:
        c = store.chunks[int(idx)]
        if document_id and c.get("document_id") != document_id:
            continue
        if c.get("document_id") in excluded:
            continue  # superseded edition: never retrieved, never cited
        s = float(fused[int(idx)])
        if s < min_score:
            continue
        # P1D: skip text-identical copies (keeps the highest-scoring one —
        # the loop walks score-descending). Skipped copies do not consume cap.
        if do_dedup:
            key = norm_text(c.get("text", ""))
            if key and key in seen_texts:
                continue
            seen_texts.add(key)
        if mpd and mpd > 0:
            d = c.get("document_id")
            if per_doc.get(d, 0) >= mpd:
                continue
            per_doc[d] = per_doc.get(d, 0) + 1
        results.append({
            "chunk": c,
            "semantic": float(sem[int(idx)]),
            "keyword": float(kw[int(idx)]),
            "score": s,
        })
        if len(results) >= top_k:
            break
    return results
