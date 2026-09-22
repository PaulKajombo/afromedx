"""Audit PDFs: page counts, extractable text, empty-page ratio. Informs ingestion.

Run: uv run python scripts/audit_pdfs.py [--pdf-dir Malawi\ Guidelines]
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

logging.getLogger("pypdf").setLevel(logging.ERROR)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ingestion.pdf_loader import extract_pages


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-dir", default="Malawi Guidelines")
    args = ap.parse_args()

    files = sorted(
        f for f in os.listdir(args.pdf_dir) if f.lower().endswith(".pdf")
    )
    print(f"{'FILE':70} {'PAGES':>6} {'CHARS':>9} {'EMPTY%':>7}  VERDICT", flush=True)
    for name in files:
        path = os.path.join(args.pdf_dir, name)
        print(f"  scanning {name} ...", flush=True)
        pages = extract_pages(path)
        n = len(pages)
        chars = sum(len(p.text) for p in pages)
        empty = sum(1 for p in pages if len(p.text.strip()) < 40)
        empty_pct = round(100 * empty / max(1, n), 1)
        if not pages:
            verdict = "UNREADABLE"
        elif chars < 200:
            verdict = "NO TEXT (scanned? -> OCR needed)"
        elif empty_pct > 60:
            verdict = f"SPARSE TEXT ({empty_pct}% empty)"
        else:
            verdict = "OK"
        print(f"{name:70} {n:6d} {chars:9d} {empty_pct:6.1f}%  {verdict}")


if __name__ == "__main__":
    main()