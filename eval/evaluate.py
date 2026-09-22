"""Evaluation: retrieval recall, citation quality, and answer groundedness.

Revision (eval-only; retrieval, embeddings, ingestion, and index untouched):
- Citation recall is measured over the FULL returned citation set. The legacy
  top-1 accuracy is still reported for continuity with earlier runs.
- Citation integrity verifies every citation's chunk_id against the passages
  actually retrieved for that question, and requires page metadata. This is
  the guard that will catch invented sources once a generative LLM is added.
- Page accuracy requires (document_id, page) pairs from the benchmark's
  "expected_citations" — a document-level match alone is not enough.
- Answer groundedness ("grounded-body") requires must_contain terms in the
  ANSWER body/key_points (normalized for unit paraphrases), never in the
  retrieved passages. The previous passages-side OR conflated retrieval
  recall with answer faithfulness.
- Key-point support rate checks each key point's significant tokens against
  the union of cited excerpts (claim-to-source linkage for abstractive
  providers; the extractive stub should score near 1.0).
- The abstention suite now includes a misleading-premise (x03) and a
  false-premise (x04) clinical question, not just out-of-domain chatter.

Usage: python eval/evaluate.py [--k 3]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings
from app.llm.factory import get_provider
from app.retrieval.embedder import _STOP, get_embedder
from app.retrieval.search import GENERIC_TERMS, norm_text, search, tokenize
from app.retrieval.store import VectorStore


_UNIT_PHRASES = [
    ("milligrams per kilogram", "mg/kg"),
    ("milligram per kilogram", "mg/kg"),
    ("mg per kg", "mg/kg"),
    ("copies per ml", "copies/ml"),
    ("copies / ml", "copies/ml"),
    ("international units", "iu"),
]

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_WS_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lowercase + unit-phrase folding + punctuation stripping for matching.

    Lets "2.4 mg/kg" match "2.4mg/kg" or "2.4 milligrams per kilogram" without
    changing what counts as evidence.
    """
    t = (text or "").lower()
    for src, dst in _UNIT_PHRASES:
        t = t.replace(src, dst)
    return _WS_RE.sub(" ", _NON_ALNUM_RE.sub(" ", t)).strip()


def significant_tokens(text: str) -> list[str]:
    return [t for t in tokenize(text)
            if len(t) > 1 and t not in GENERIC_TERMS and t not in _STOP]


def citation_doc_id(cite: dict, store: VectorStore) -> str:
    """Resolve a citation to a document id.

    Prefer the chunk_id prefix (exact by construction: "<doc_id>::p<page>::<n>"),
    fall back to the legacy document-title lookup.
    """
    cid = cite.get("chunk_id", "") or ""
    if "::" in cid and cid.split("::")[0] in store.docs:
        return cid.split("::")[0]
    title = cite.get("document", "")
    for d, meta in store.docs.items():
        if meta.get("title") == title:
            return d
    return title


def load_benchmark(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--benchmark", default=os.path.join(os.path.dirname(__file__), "benchmark.json"))
    ap.add_argument("--delay-secs", type=float, default=0.0,
                    help="pause between items so burst traffic does not trip provider "
                         "rate limits (which would silently fall back to the stub).")
    args = ap.parse_args()

    store = VectorStore(settings.index_dir, embedder=get_embedder(force_tfidf=settings.force_tfidf))
    if not store.load() or store.count() == 0:
        print("EMPTY INDEX — run: python scripts/ingest_all.py")
        sys.exit(2)
    bench = load_benchmark(args.benchmark)
    provider = get_provider()

    denom = 0
    recall_hits = 0
    top1_hits = 0
    cite_any_hits = 0
    integrity_valid = 0
    integrity_total = 0
    page_denom = 0
    page_hits = 0
    grounded_ok = 0
    kpsup_sum = 0.0
    abstain_ok = 0
    abstain_total = 0
    # P6 retrieval-hygiene guards (provider-independent, all items with hits)
    uniq_items = 0
    uniq_sum = 0.0
    dupe_sum = 0.0
    rows = []

    for n, item in enumerate(bench):
        if args.delay_secs and n:
            import time
            time.sleep(args.delay_secs)
        q = item["question"]
        hits = search(store, q, top_k=max(args.k, 6), min_score=settings.min_score,
                      semantic_weight=settings.semantic_weight)
        topk_ids = [h["chunk"].get("document_id") for h in hits[:args.k]]
        passage_ids = {h["chunk"].get("id") for h in hits}
        ans = provider.generate(q, hits, store.docs)

        # P6: top-K hygiene — unique-document rate and duplicate-content rate.
        # Computed from hits for every item (answerable or abstain-expected).
        n_top = len(topk_ids)
        if n_top:
            uniq_items += 1
            uniq_sum += len(set(topk_ids)) / n_top
            seen_t: set[str] = set()
            dupes = 0
            for h in hits[:args.k]:
                t = norm_text(h["chunk"].get("text", ""))
                if t in seen_t:
                    dupes += 1
                else:
                    seen_t.add(t)
            dupe_sum += dupes / n_top

        if item.get("expect_abstain"):
            abstain_total += 1
            ok = bool(ans.abstained)
            abstain_ok += ok
            rows.append((item["id"], f"abstain={ok}", ok, topk_ids[:2]))
            continue

        denom += 1
        expected = set(item.get("expected_docs") or [])

        # 1. retrieval recall (unchanged definition)
        recalled = bool(expected.intersection(topk_ids)) if expected else True
        recall_hits += recalled

        # 2a. legacy top-1 citation accuracy (continuity with earlier runs)
        cites = ans.citations or []
        top1_doc = citation_doc_id(cites[0], store) if cites else ""
        top1_ok = (top1_doc in expected) if expected else False
        top1_hits += top1_ok

        # 2b. citation recall over the FULL returned citation set
        cited_docs = {citation_doc_id(c, store) for c in cites}
        any_ok = bool(expected.intersection(cited_docs)) if expected else False
        cite_any_hits += any_ok

        # 2c. citation integrity: every citation must point at a passage that
        # was actually retrieved for THIS question, with page metadata present
        valid = sum(1 for c in cites
                    if c.get("chunk_id") in passage_ids and c.get("page") is not None)
        integrity_valid += valid
        integrity_total += len(cites)

        # 2d. page accuracy against benchmark expected_citations (doc + page)
        exp_cites = item.get("expected_citations") or []
        if exp_cites:
            page_denom += 1
            page_ok = any(
                citation_doc_id(c, store) == e.get("document") and c.get("page") == e.get("page")
                for c in cites for e in exp_cites
            )
            page_hits += page_ok
        else:
            page_ok = True  # n/a for this item

        # 3. answer groundedness: must_contain terms must appear in the ANSWER
        # body/key_points (normalized) — retrieved passages do NOT count.
        answer_text = normalize((ans.body_markdown or "") + "\n"
                                + "\n".join(ans.key_points or []))
        must = [normalize(m) for m in item.get("must_contain", [])]
        ground = all(m and m in answer_text for m in must) if must else True
        grounded_ok += ground

        # 4. key-point support: each key point's significant tokens must be
        # traceable to the cited excerpts (claim-to-source linkage)
        excerpt_terms = set(tokenize(" ".join((c.get("excerpt") or "") for c in cites)))
        kp_supported = 0
        kp_total = 0
        for kp in (ans.key_points or []):
            sig = significant_tokens(kp)
            kp_total += 1
            if not sig:
                kp_supported += 1  # vacuous: nothing clinical to verify
            elif sum(1 for t in sig if t in excerpt_terms) / len(sig) >= 0.5:
                kp_supported += 1
        kp_frac = (kp_supported / kp_total) if kp_total else 1.0
        kpsup_sum += kp_frac

        all_ok = recalled and any_ok and page_ok and ground
        rows.append((item["id"],
                     f"recall={recalled} top1={top1_ok} cite_any={any_ok} "
                     f"page={page_ok if exp_cites else 'n/a'} grounded_body={ground} "
                     f"kpsup={kp_frac:.2f}",
                     all_ok, topk_ids[:2]))

    print(f"index: {store.count()} chunks, embedder={getattr(store.embedder,'name','?')}, provider={provider.name}")
    for r in rows:
        print(r)
    if denom:
        print(f"\nRetrieval Recall@{args.k}: {recall_hits}/{denom} = {recall_hits/denom:.2f}")
        print(f"Citation top-1 accuracy (legacy): {top1_hits}/{denom} = {top1_hits/denom:.2f}")
        print(f"Citation recall (any cited doc in expected): {cite_any_hits}/{denom} = {cite_any_hits/denom:.2f}")
    if integrity_total:
        print(f"Citation integrity (chunk cited was retrieved + has page): "
              f"{integrity_valid}/{integrity_total} = {integrity_valid/integrity_total:.2f}")
    if page_denom:
        print(f"Page accuracy (expected doc+page cited): {page_hits}/{page_denom} = {page_hits/page_denom:.2f}")
    if denom:
        print(f"Answer groundedness, body-side normalized (must_contain in answer, not passages): "
              f"{grounded_ok}/{denom} = {grounded_ok/denom:.2f}")
        print(f"Key-point support rate (key-point terms traceable to cited excerpts): "
              f"{kpsup_sum/denom:.2f} mean over {denom} items")
    if abstain_total:
        print(f"Abstention (out-of-scope + misleading/false premise): "
              f"{abstain_ok}/{abstain_total} = {abstain_ok/abstain_total:.2f}")
    if uniq_items:
        print(f"Top-{args.k} unique-document rate (mean distinct docs / K): "
              f"{uniq_sum/uniq_items:.2f} over {uniq_items} items")
        print(f"Top-{args.k} duplicate-content rate (mean text-identical repeats / K): "
              f"{dupe_sum/uniq_items:.2f} over {uniq_items} items")


if __name__ == "__main__":
    main()
