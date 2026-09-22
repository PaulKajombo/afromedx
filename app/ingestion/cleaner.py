"""Text cleaning: fix common PDF extraction artefacts, keep clinical numbers intact."""
from __future__ import annotations

import re

# Private-use area glyphs (Wingdings/Symbol bullets, checkboxes) and control chars.
_PRIVATE_USE = re.compile(r"[\ue000-\uf8ff\U000f0000-\U000ffffd]")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# TOC dot leaders / ellipses: 2+ consecutive dots are never clinical decimals.
_DOT_LEADER = re.compile(r"\.{2,}")
_WS = re.compile(r"[ \t]+")


def clean_text(text: str) -> str:
    if not text:
        return ""
    t = text.replace("\r\n", "\n").replace("\r", "\n")
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
