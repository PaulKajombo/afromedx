"""Ingest provenance: manifest records are written once and never duplicated."""
import importlib.util
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_SPEC = importlib.util.spec_from_file_location(
    "ingest_cli", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "scripts", "ingest.py"))
_ingest = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_ingest)


def _record(**over):
    d = {"filename": "g.pdf", "doc_id": "g1", "title": "G", "edition": "1e",
         "year": 2024, "source": "MOH", "checksum": "abc", "pages": 10,
         "chars": 1000, "chunks": 5}
    d.update(over)
    return d


def test_manifest_created_and_totals(tmp_path):
    mp = str(tmp_path / "manifest.json")
    assert _ingest.update_manifest(mp, **_record()) is True
    import json
    man = json.load(open(mp, encoding="utf-8"))
    assert man["documents_indexed"] == 1
    assert man["total_chunks"] == 5
    assert man["documents"][0]["doc_id"] == "g1"


def test_manifest_skips_duplicates(tmp_path):
    mp = str(tmp_path / "manifest.json")
    assert _ingest.update_manifest(mp, **_record()) is True
    # same checksum under a different doc_id (re-ingested copy)
    assert _ingest.update_manifest(mp, **_record(doc_id="g2")) is False
    import json
    man = json.load(open(mp, encoding="utf-8"))
    assert man["documents_indexed"] == 1
    assert man["total_chunks"] == 5


def test_manifest_refresh_on_reingest(tmp_path):
    mp = str(tmp_path / "manifest.json")
    _ingest.update_manifest(mp, **_record())
    # same doc_id re-ingested (e.g. improved chunking): record refreshed
    assert _ingest.update_manifest(mp, **_record(chunks=7, pages=12)) is True
    import json
    man = json.load(open(mp, encoding="utf-8"))
    assert man["documents_indexed"] == 1
    assert man["total_chunks"] == 7
    assert man["documents"][0]["pages"] == 12


def test_manifest_appends_new_documents(tmp_path):
    mp = str(tmp_path / "manifest.json")
    _ingest.update_manifest(mp, **_record())
    assert _ingest.update_manifest(mp, **_record(doc_id="g2", checksum="def",
                                                chunks=3)) is True
    import json
    man = json.load(open(mp, encoding="utf-8"))
    assert man["documents_indexed"] == 2
    assert man["total_chunks"] == 8
