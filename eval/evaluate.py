"""Evaluation: retrieval accuracy (Recall@K), citation accuracy, grounded-answer rate.

Usage: python eval/evaluate.py [--k 3]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings
from app.llm.factory import get_provider
from app.retrieval.embedder import get_embedder
from app.retrieval.search import search
from app.retrieval.store import VectorStore


def load_benchmark(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--benchmark", default=os.path.join(os.path.dirname(__file__), "benchmark.json"))
    args = ap.parse_args()

    store = VectorStore(settings.index_dir, embedder=get_embedder(force_tfidf=settings.force_tfidf))
    if not store.load() or store.count() == 0:
        print("EMPTY INDEX — run: python scripts/demo_seed.py")
        sys.exit(2)
    bench = load_benchmark(args.benchmark)
    provider = get_provider()

    recall_hits = 0
    denom = 0
    cite_ok = 0
    grounded_ok = 0
    abstain_ok = 0
    abstain_total = 0
    rows = []

    for item in bench:
        q = item["question"]
        hits = search(store, q, top_k=max(args.k, 6), min_score=settings.min_score,
                      semantic_weight=settings.semantic_weight)
        topk_ids = [h["chunk"].get("document_id") for h in hits[:args.k]]
        ans = provider.generate(q, hits, store.docs)

        if item.get("expect_abstain"):
            abstain_total += 1
            ok = ans.abstained
            abstain_ok += ok
            rows.append((item["id"], "abstain", ok, topk_ids[:2]))
            continue
        denom += 1
        expected = set(item.get("expected_docs") or [])
        recalled = bool(expected.intersection(topk_ids)) if expected else True
        recall_hits += recalled
        # citation accuracy: top citation maps to any expected doc (page check best-effort)
        cite = (ans.citations[0].get("document", "") if ans.citations else "")
        cite_doc = next((d for d, meta in store.docs.items() if meta.get("title") == cite), cite)
        cite_match = (cite_doc in expected) if expected else False
        cite_ok += cite_match
        # grounded: answer text terms appear in retrieved passages (heuristic) or abstained-with-no-evidence
        joined = " ".join(h["chunk"].get("text", "") for h in hits[:3]).lower()
        must = [m.lower() for m in item.get("must_contain", [])]
        ground = all(m in (ans.body_markdown + " ").lower() or m in joined for m in must) if must else True
        grounded_ok += ground
        rows.append((item["id"], f"recall={recalled} cite={cite_match} grounded={ground}", recalled and cite_match, topk_ids[:2]))

    print(f"index: {store.count()} chunks, embedder={getattr(store.embedder,'name','?')}, provider={provider.name}")
    for r in rows:
        print(r)
    if denom:
        print(f"\nRecall@{args.k}: {recall_hits}/{denom} = {recall_hits/denom:.2f}")
        print(f"Citation accuracy: {cite_ok}/{denom} = {cite_ok/denom:.2f}")
        print(f"Grounded-answer rate: {grounded_ok}/{denom} = {grounded_ok/denom:.2f}")
    if abstain_total:
        print(f"Abstention on out-of-scope: {abstain_ok}/{abstain_total} = {abstain_ok/abstain_total:.2f}")


if __name__ == "__main__":
    main()
