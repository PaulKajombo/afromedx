"""Text cleaning: fix common PDF extraction artefacts, keep clinical numbers intact."""
from __future__ import annotations

import re


def clean_text(text: str) -> str:
    if not text:
        return ""
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    # de-hyphenate line-break splits: "artesu-\nnate" -> "artesunate"
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)
    # collapse intra-paragraph single newlines into spaces, keep blank-line breaks
    t = re.sub(r"(?<!\n)\n(?!\n)", " ", t)
    # collapse excessive whitespace
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    # strip running-header/footer style repeats of bare page numbers (" 123 ")
    lines = [ln.strip() for ln in t.split("\n")]
    lines = [ln for ln in lines if not re.fullmatch(r"\d{1,4}", ln)]
    t = "\n".join(lines)
    # tidy spaces before punctuation — but NEVER alter digits/units
    t = re.sub(r"\s+([.,;:%)])", r"\1", t)
    return t.strip()
