"""Route tests: guideline PDF serving is doc_id-keyed (no path traversal)."""
import os
import sys

import pytest
from fastapi import HTTPException

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.api import routes


class _FakeStore:
    def __init__(self, docs):
        self.docs = docs


def _use_store(store):
    prev = routes._store
    routes.set_store(store)
    return prev


def test_pdf_serves_known_document(tmp_path):
    pdf = tmp_path / "g.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%test\n")
    prev = _use_store(_FakeStore({"d1": {"id": "d1", "title": "G",
                                         "file_path": str(pdf)}}))
    try:
        resp = routes.api_guideline_pdf("d1")
        assert resp.path == str(pdf)
        assert resp.media_type == "application/pdf"
    finally:
        routes.set_store(prev)


def test_pdf_unknown_document_404():
    prev = _use_store(_FakeStore({}))
    try:
        with pytest.raises(HTTPException) as ei:
            routes.api_guideline_pdf("nope")
        assert ei.value.status_code == 404
    finally:
        routes.set_store(prev)


def test_pdf_missing_file_404(tmp_path):
    prev = _use_store(_FakeStore({"d1": {"id": "d1",
                                         "file_path": str(tmp_path / "gone.pdf")}}))
    try:
        with pytest.raises(HTTPException) as ei:
            routes.api_guideline_pdf("d1")
        assert ei.value.status_code == 404
    finally:
        routes.set_store(prev)


def test_pdf_no_path_traversal_possible():
    # Only store keys resolve; '../../..' can never become a file path.
    prev = _use_store(_FakeStore({"d1": {"id": "d1", "file_path": ""}}))
    try:
        with pytest.raises(HTTPException):
            routes.api_guideline_pdf("../../secret")
    finally:
        routes.set_store(prev)
