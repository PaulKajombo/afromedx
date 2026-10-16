"""Hybrid search: semantic cosine + keyword overlap, fused with metadata filtering.

Keeps it simple but satisfies: natural-language queries, keyword fallback
("IV artesunate dose adult"), metadata filter, and a min-score gate so the
answer layer can abstain (NO SOURCE = NO ANSWER).
"""
from __future__ import annotations

import re

import numpy as np

from .acronyms import expand_acronyms
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
    "chronic": ["persistent"],
    "persistent": ["chronic"],
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


def classify_intent(query: str) -> str:
    """Query intent: 'definition', 'management', or 'other'.

    "what is postpartum haemorrhage" asks for a DEFINITION (what it is,
    criteria, key features); bare "postpartum haemorrhage" asks for MANAGEMENT
    (what to do). Action/urgency wording ("treat", "danger", "referral",
    "dose", ...) always wins as 'other' = default behavior, so red-flag and
    treatment questions can never misroute into definitions.
    """
    q = (query or "").lower()
    toks = set(tokenize(query))
    action = {"treat", "treated", "treating", "treatment", "manage", "managing",
              "management", "dose", "dosage", "refer", "referral", "emergency",
              "urgent", "danger"}
    if toks & action:
        return "other"
    if re.search(r"\bwhat\s+(is|are)\b", q) or re.search(r"\bdefin\w*\b", q) \
            or "meaning of" in q or "causes of" in q or "symptoms of" in q \
            or "signs of" in q:
        return "definition"
    if _is_bare_topic(query):
        return "management"
    return "other"


def expand_definition(query: str) -> str:
    """Append definitional framing for definition-intent queries (retrieval only).

    The ORIGINAL question is always kept for answer generation, titles, and
    evaluation — only ranking sees the extra terms, mirroring bare-topic and
    acronym expansion.
    """
    if classify_intent(query) == "definition":
        return query.strip() + " definition defined"
    return query


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


def fuzzy_match(term: str, candidates: set[str], max_dist: int = 3,
                buckets: dict[str, list[str]] | None = None) -> str | None:
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
    if buckets is not None:
        pool = [c for c in buckets.get(term[:1], [])
                if c != term and abs(len(c) - len(term)) <= max_dist]
    else:
        pool = [c for c in candidates
                if c and c != term and c[:1] == term[:1]
                and abs(len(c) - len(term)) <= max_dist]
    if not pool:
        return None
    best: str | None = None
    best_d = max_dist + 1
    for c in pool:
        d = _edit_distance(term, c)
        if d < best_d or (d == best_d and best is not None and len(c) < len(best)):
            best, best_d = c, d
    return best if best_d <= max_dist else None


def term_covered(term: str, terms: set[str], buckets: dict[str, list[str]] | None = None) -> bool:
    """Does a query term bridge to a term set? Exact, synonym, then fuzzy.

    Shared by the in-scope gate and the stub grounding gate so both treat
    misspellings identically. Fuzzy applies ONLY when exact and synonym
    matching fail, so correctly-spelled queries behave exactly as before.
    """
    if term in terms:
        return True
    if any(s in terms for s in synonyms_of(term)):
        return True
    return fuzzy_match(term, terms, buckets=buckets) is not None


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


def keyword_scores(query: str, texts: list[str] | None = None, *,
                   token_sets: list[set[str]] | None = None,
                   df: dict[str, int] | None = None) -> np.ndarray:
    """IDF-weighted token overlap with synonym expansion. Returns 0..1 array.

    Pass precomputed token_sets (+df) from the store to skip re-tokenizing the
    corpus every query; scores are IDENTICAL to the inline computation.
    """
    qtok = expand(tokenize(query))
    n = len(token_sets) if token_sets is not None else len(texts or [])
    if not qtok or not n:
        return np.zeros(max(n, len(texts or [])), dtype=np.float32)
    if token_sets is None:
        tokenized = [set(tokenize(t)) for t in (texts or [])]
    else:
        tokenized = token_sets
    if df is None:
        df = {}
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
    n = max(1, n)
    import math
    idf = {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}
    qidf = sum(idf.get(t, math.log(n + 1) + 1.0) for t in qtok)
    scores = np.zeros(len(tokenized), dtype=np.float32)
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
# - emergency-assessment-words-only: a transcription aid that warns its doses
#   may be ambiguous — not an authoritative guideline -> EXCLUDED (the V2 wall
#   chart itself stays indexed and live).
# Exclusion is query-time only (the index is untouched) and reversible via
# settings.excluded_docs (AFROMEDX_EXCLUDE_DOCS).

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


# Authority policy: designated gold-standard sources get a fixed bonus so the
# current departmental protocol outranks older/general material on ITS topic.
# emergency-assessment is the ED's latest protocol (added 2026-10-03) but only
# has 4 thin wall-chart chunks, so it loses ties on text overlap alone.
# Weight 0.08 is decisive within-topic yet far below cross-topic score gaps,
# so it cannot promote the doc for unrelated questions. Override/disabled via
# settings.doc_boost (AFROMEDX_DOC_BOOST, "doc:weight,..." — empty disables).
DOC_BOOST: dict[str, float] = {
    "emergency-assessment": 0.08,
}


def parse_doc_boost(raw: str) -> dict[str, float]:
    """Parse "doc:weight,doc:weight" maps (env configuration)."""
    out: dict[str, float] = {}
    for part in (raw or "").split(","):
        if ":" not in part:
            continue
        key, _, value = part.partition(":")
        try:
            out[key.strip()] = float(value)
        except ValueError:
            continue
    return {k: v for k, v in out.items() if k}


# Explicit supersession pairs (curated, title-page evidenced). When a superseded
# document AND its successor are both retrieved for a question, superseded
# chunks take a small penalty so the current edition ranks first. If the
# successor is absent, older content is left alone (better than nothing).
# Evidence: obgyn v3.0 2017 vs obs-gynae-2023; paediatrics-handbook 3rd ed
# 2008 vs paediatric-protocols-2018; malaria 5th ed 2020 vs 6th ed 2025.
SUPERSEDES: dict[str, str] = {
    "obgyn": "obs-gynae-2023",
    "paediatrics-handbook": "paediatric-protocols-2018",
    "malaria-2020": "malaria-treatment",
}
SUPERSEDED_PENALTY = 0.03

# Generic-fallback content: chunks that defer to other guidelines instead of
# stating Malawi's own recommendation (exactly one such chunk corpus-wide:
# who-pocketbook p141 Tuberculosis). Deprioritized so national content leads
# but never removed — it remains legitimate background evidence.
FALLBACK_PHRASES = ("if national recommendations are not available",)
FALLBACK_PENALTY = 0.03

# Drug-susceptible vs drug-resistant TB are mutually exclusive clinical
# categories, but share most vocabulary ("drug", "tuberculosis", "regimen"),
# so DR-table chunks top DS questions on overlap alone. When the query asserts
# susceptibility and contains NO resistance markers, chunks asserting
# resistance are demoted (never removed). DR questions (MDR-TB etc.) are
# exempt by construction.
_DS_QUERY_TOKENS = frozenset({"susceptible", "sensitive"})
_DR_QUERY_TOKENS = frozenset({"resistant", "resistance", "mdr", "xdr"})
_DR_CHUNK_PHRASES = ("drug resist", "mdr", "xdr", "rr-tb", "rr tb", "dr-tb",
                     "dr tb", "resistant tb", "resistant tuberculosis",
                     "rifampicin resist")
DS_PENALTY = 0.04
# Includes question-meta nouns ("results", "meaning"): they describe the
# wanted answer, never the medicine, so they must not decide answerability.
GENERIC_TERMS = frozenset(
    "recommend recommended recommend should use used give given start started begin "
    "patient patients child children adult adults dosed dosage medications medicine "
    "medication drug drugs therapy treat treated treating management treat manage "
    "diagnose diagnosed diagnosis diagnostic identify investigation investigations "
    "results result meaning interpretation".split()
)


def _term_set(texts: list[str]) -> set[str]:
    ts: set[str] = set()
    for t in texts:
        ts.update(tokenize(t))
    return ts


def _in_scope(query: str, texts: list[str], corpus_terms: set[str],
              buckets: dict[str, list[str]] | None = None) -> bool:
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
        # Curated acronyms are known vocabulary: bridged (1.0) when any word
        # of their corpus-evidenced expansion is present. Unknown tokens fall
        # through to fuzzy matching below.
        from .acronyms import ACRONYMS
        phrases = ACRONYMS.get(t, ())
        if phrases and any(w in corpus_terms
                           for phrase in phrases for w in tokenize(phrase)):
            return 1.0
        # A fuzzy bridge is weaker evidence of intent than an exact term: it
        # counts half, so a lone near-miss ("tyre" -> "type") can never pass the
        # gate on its own, while a typo among known terms still retrieves.
        return 0.5 if fuzzy_match(t, corpus_terms, buckets=buckets) is not None else 0.0

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
    doc_boost: dict[str, float] | None = None,
    superseded_penalty: float | None = None,
    fallback_penalty: float | None = None,
    ds_penalty: float | None = None,
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
    # Precomputed corpus structures when the store provides them (identical
    # scores, milliseconds instead of seconds); otherwise compute inline.
    c_tok = getattr(store, "chunk_token_sets", None)
    m_tok = getattr(store, "meta_token_sets", None)
    c_terms = getattr(store, "corpus_terms", None)
    buckets = getattr(store, "term_buckets", None)
    df_body = getattr(store, "df_body", None)
    df_meta = getattr(store, "df_meta", None)
    cached = bool(c_tok and m_tok and c_terms is not None and len(c_tok) == len(texts))
    gate_terms = c_terms if cached else _term_set(texts + meta)
    gate_buckets = buckets if cached else None
    if not _in_scope(q, texts, gate_terms, buckets=gate_buckets):
        return []

    # Bare-topic expansion (retrieval only): the original question is kept for
    # answer generation; ranking sees management framing so "migraine headache"
    # retrieves treatment content instead of definitional fragments.
    # Acronym expansion likewise appends corpus-evidenced phrases
    # ("CAP ... community-acquired pneumonia") for ranking only.
    # Definition intent appends definitional framing the same way.
    rq = expand_acronyms(expand_bare_topic(q))
    if classify_intent(q) == "definition":
        rq = rq.rstrip() + " definition defined"

    # semantic cosine
    try:
        qv = store.embedder.encode([rq]).astype(np.float32).ravel()
        qv = qv / (np.linalg.norm(qv) + 1e-9)
        sem = store.vectors @ qv if store.vectors is not None else np.zeros(len(texts))
        sem = np.clip(np.asarray(sem, dtype=np.float32).ravel(), 0, 1)
    except Exception:
        sem = np.zeros(len(texts), dtype=np.float32)

    kw = (keyword_scores(rq, None, token_sets=c_tok, df=df_body)
          if cached else keyword_scores(rq, texts))
    # P1A: section/subsection/document-title matches, query-time only.
    # Headings are metadata (never embedded), so without this arm a section
    # titled exactly like the question (e.g. "When To Start Art") is invisible
    # to ranking. Blend is explicit via settings.section_weight (0 disables).
    sw = settings.section_weight if section_weight is None else section_weight
    if sw and sw > 0:
        kw_meta = (keyword_scores(rq, None, token_sets=m_tok, df=df_meta)
                   if cached else keyword_scores(rq, meta))
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
    # Authority policy: fixed per-document bonuses for designated
    # gold-standard sources (see DOC_BOOST). Query-time only.
    if doc_boost is None:
        raw = getattr(settings, "doc_boost_raw", "")
        boosts = DOC_BOOST if not raw else parse_doc_boost(raw)
    else:
        boosts = doc_boost
    if boosts:
        fused = fused + np.array([float(boosts.get(c.get("document_id"), 0.0))
                                  for c in store.chunks], dtype=np.float32)
    # Supersession preference: penalize superseded chunks only when their
    # successor also competes (present in the candidate window judged on
    # pre-penalty scores). Order is recomputed after. Query-time only.
    sp = settings.superseded_penalty if superseded_penalty is None else superseded_penalty
    if sp and sp > 0:
        window_docs = {store.chunks[int(i)].get("document_id")
                       for i in np.argsort(-fused)[: max(top_k * 3, top_k)]}
        penalized = {old for old, new in SUPERSEDES.items() if new in window_docs}
        if penalized:
            fused = fused - np.array(
                [sp if c.get("document_id") in penalized else 0.0
                 for c in store.chunks], dtype=np.float32)
    # Generic-fallback content defers to other guidelines; deprioritize it so
    # national content leads. Query-time only; the chunks stay citable.
    fp = settings.fallback_penalty if fallback_penalty is None else fallback_penalty
    if fp and fp > 0:
        lowered = tuple(p.lower() for p in FALLBACK_PHRASES)
        fused = fused - np.array(
            [fp if any(p in (c.get("text", "") or "").lower() for p in lowered) else 0.0
             for c in store.chunks], dtype=np.float32)
    # DS-vs-DR disambiguation: a query asserting susceptibility with no
    # resistance markers demotes resistance-asserting chunks. Query-time only.
    dp = settings.ds_penalty if ds_penalty is None else ds_penalty
    if dp and dp > 0:
        qtokens = set(tokenize(rq))
        if (qtokens & _DS_QUERY_TOKENS) and not (qtokens & _DR_QUERY_TOKENS):
            fused = fused - np.array(
                [dp if any(p in (c.get("text", "") or "").lower()
                           for p in _DR_CHUNK_PHRASES) else 0.0
                 for c in store.chunks], dtype=np.float32)

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
