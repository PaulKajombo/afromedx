"""Text cleaning: fix common PDF extraction artefacts, keep clinical numbers intact."""
from __future__ import annotations

import re

# Private-use area glyphs (Wingdings/Symbol bullets, checkboxes) and control chars.
_PRIVATE_USE = re.compile(r"[\ue000-\uf8ff\U000f0000-\U000ffffd]")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# TOC dot leaders / ellipses: 2+ consecutive dots are never clinical decimals.
_DOT_LEADER = re.compile(r"\.{2,}")
_WS = re.compile(r"[ \t]+")
# Letter-spaced PDF text ("L a b o u r", "H ealth", "A N D"): some PDFs emit
# every glyph separated by spaces, which pypdf preserves. Two conservative
# joins, applied ONLY to flagged lines (see _despace_line):
#  - pure single-letter runs of 3+ ("L a b o u r" -> "Labour", "C P D" -> "CPD")
#  - single CAPITAL + lowercase-start word ("H ealth" -> "Health").
# The lowercase requirement is load-bearing: it excludes the indefinite
# article ("a test" stays put) and normal "Vitamin A deficiency" lines never
# flag (far too few single-char tokens). Residual risk (e.g. an "A B C" list
# joining to "ABC") is documented and rare in guideline prose; 2-char
# fragments from uneven pre-collapsing ("iN", "ationalR") are left alone.
_SPACED_RUN_RE = re.compile(r"\b(?:[A-Za-z] ){2,}[A-Za-z]\b")
_SPLIT_WORD_RE = re.compile(r"\b([A-Z]) ([a-z][a-z]+)\b")
# Uneven pre-collapsing leaves 2-char fragments ("em ergency", "ab bing")
# and trailing singles ("Episiotom y"). Rejoin them ONLY with a flag-gated,
# stoplisted rule: the fragment must not be a common word or a clinical
# abbreviation ("of care" and "IV fluids" must never join).
_TWOCHAR_STOP = frozenset(
    "of to in on at as is it or an am pm do go we us ox ex by my up if so no "
    "iv po bp od tb hiv im la mg ml bd dr st hr nd rd th ct".split())
_FRAG_2LEAD_RE = re.compile(r"\b([A-Za-z]{2}) ([a-z][a-z]+)\b")
_FRAG_TRAIL_RE = re.compile(r"\b([A-Za-z]{2,}) ([a-z])\b")


def _despace_line(line: str) -> str:
    """Rejoin letter-spaced text on lines that are clearly affected.

    Flags lines with >= 3 single-char tokens making up >= 20% of alphabetic
    tokens: spaced lines ("C P D", "M alaw ... H ealth S ervice") trip this
    easily, while normal prose ("Take Vitamin A daily", "Grade A B tumours")
    stays below one of the two bars.
    """
    toks = line.split()
    alpha = [t for t in toks if t.isalpha()]
    singles = [t for t in alpha if len(t) == 1]
    if len(singles) < 3 or not alpha or len(singles) / len(alpha) < 0.2:
        return line
    line = _SPACED_RUN_RE.sub(lambda m: re.sub(r"\s+", "", m.group(0)), line)
    line = _SPLIT_WORD_RE.sub(r"\1\2", line)

    def _join2(m: re.Match) -> str:
        frag, rest = m.group(1), m.group(2)
        if frag.lower() in _TWOCHAR_STOP:
            return m.group(0)
        return frag + rest

    line = _FRAG_2LEAD_RE.sub(_join2, line)
    # Trailing singles ("Episiotom y"); never the article "a" ("tablets a day").
    line = re.sub(r"\b([A-Za-z]{2,}) ([a-z])\b",
                  lambda m: m.group(0) if m.group(2) == "a" else m.group(1) + m.group(2),
                  line)
    return line


def clean_text(text: str) -> str:
    if not text:
        return ""
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    # repair letter-spaced lines BEFORE structural collapsing (needs raw lines)
    t = "\n".join(_despace_line(ln) for ln in t.split("\n"))
    t = _PRIVATE_USE.sub(" ", t)
    t = _CONTROL.sub(" ", t)
    # de-hyphenate line-break splits: "artesu-\nnate" -> "artesunate"
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)
    # collapse intra-paragraph single newlines into spaces, keep blank-line breaks
    t = re.sub(r"(?<!\n)\n(?!\n)", " ", t)
    # dot leaders / ellipses -> space, then collapse whitespace
    t = _DOT_LEADER.sub(" ", t)
    t = _WS.sub(" ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    # strip running-header/footer style repeats of bare page numbers (" 123 ")
    lines = [ln.strip() for ln in t.split("\n")]
    lines = [ln for ln in lines if not re.fullmatch(r"\d{1,4}", ln)]
    t = "\n".join(lines)
    # tidy spaces before punctuation — but NEVER alter digits/units
    t = re.sub(r"\s+([.,;:%)])", r"\1", t)
    return t.strip()
