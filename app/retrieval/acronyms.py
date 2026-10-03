"""Curated clinical acronym map (corpus-mined, human-approved).

Each entry was mined from the indexed guidelines (uppercase-token frequency +
"Expansion (ACRONYM)" patterns) and approved against three rules:
  1. clinical meaning with in-corpus evidence for the expansion,
  2. real retrieval need (benchmark question or observed failure),
  3. ambiguity-tolerant: applied as retrieval-side EXPANSION, never replacement,
     so e.g. English "cap" still scores alongside "community-acquired pneumonia".

Deliberately excluded: dose/route units (IV, IM, PO, BD — co-occur anyway),
noise tokens, roman numerals, already-covered synonyms (HIV, TB, ART, VL, TLD),
and ambiguous entries without a safe dominant sense (e.g. ABC = airway
protocol AND the drug abacavir).
"""
from __future__ import annotations

ACRONYMS: dict[str, list[str]] = {
    "cap": ["community-acquired pneumonia"],
    "fbc": ["full blood count"],
    "ors": ["oral rehydration salts", "oral rehydration solution"],
    "la": ["artemether-lumefantrine"],
    "hbv": ["hepatitis b virus"],
    "sti": ["sexually transmitted infection"],
    "uss": ["ultrasound"],
    "cns": ["central nervous system"],
    "hcv": ["hepatitis c virus"],
    "ngt": ["nasogastric tube"],
    "muac": ["mid-upper arm circumference"],
    "lp": ["lumbar puncture"],
    "pcr": ["polymerase chain reaction"],
    "rutf": ["ready-to-use therapeutic food"],
    "ckd": ["chronic kidney disease"],
    "ecg": ["electrocardiogram"],
    "ks": ["kaposi sarcoma"],
    "pcv": ["pneumococcal conjugate vaccine"],
    "cxr": ["chest x-ray"],
    "csf": ["cerebrospinal fluid"],
    "bp": ["blood pressure"],
    "o2": ["oxygen"],
}


def expand_acronyms(query: str) -> str:
    """Append expansion phrases for known acronyms (retrieval only).

    Returns "<query> <phrase> ..." so the original wording is preserved for
    answers, titles, citations, and evaluation; only ranking sees the extra
    terms. Unknown tokens pass through untouched.
    """
    from .search import tokenize  # lazy: search.py imports this module
    seen = set(tokenize(query))
    extra: list[str] = []
    for term in sorted(seen):
        for phrase in ACRONYMS.get(term, ()):
            if phrase not in extra:
                extra.append(phrase)
    return query if not extra else query.rstrip() + " " + " ".join(extra)
