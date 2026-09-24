"""Local vector store: JSONL chunks + .npy embeddings. No server required.

Design allows swapping to Chroma/pgvector later behind the same interface.
Each edition is namespaced by document id — editions never silently mix because
chunks carry document_id (+ edition in metadata) and search can filter.
"""
from __future__ import annotations

import json
import os

import numpy as np


class VectorStore:
    def __init__(self, index_dir: str, embedder=None) -> None:
        from .embedder import get_embedder
        self.index_dir = index_dir
        self.embedder = embedder or get_embedder()
        self.chunks: list[dict] = []
        self.docs: dict[str, dict] = {}
        self.vectors: np.ndarray | None = None
        os.makedirs(index_dir, exist_ok=True)
        self._reset_text_cache()

    # ---- query-time text caches (perf: precompute once, not per query) ----
    def _reset_text_cache(self) -> None:
        self.chunk_token_sets: list[set[str]] = []
        self.meta_token_sets: list[set[str]] = []
        self.corpus_terms: set[str] = set()
        self.term_buckets: dict[str, list[str]] = {}
        self.df_body: dict[str, int] = {}
        self.df_meta: dict[str, int] = {}

    def rebuild_text_cache(self) -> None:
        """Token sets, document frequencies, and term buckets for keyword paths.

        Called after any corpus change (add_documents, load). Without this,
        every query re-tokenizes all chunks (~5 s at 13k chunks); with it,
        keyword work drops to milliseconds with IDENTICAL scores.
        """
        from collections import Counter

        from .search import tokenize
        self._reset_text_cache()
        self.chunk_token_sets = [set(tokenize(c.get("text", ""))) for c in self.chunks]
        self.meta_token_sets = [set(tokenize(
            f"{self.docs.get(c.get('document_id'), {}).get('title', '')} "
            f"{c.get('section', '')} {c.get('subsection', '')}")) for c in self.chunks]
        df: Counter[str] = Counter()
        for s in self.chunk_token_sets:
            df.update(s)
        self.df_body = dict(df)
        dfm: Counter[str] = Counter()
        for s in self.meta_token_sets:
            dfm.update(s)
        self.df_meta = dict(dfm)
        terms: set[str] = set()
        for s in self.chunk_token_sets:
            terms.update(s)
        for s in self.meta_token_sets:
            terms.update(s)
        self.corpus_terms = terms
        buckets: dict[str, list[str]] = {}
        for t in terms:
            buckets.setdefault(t[:1], []).append(t)
        for v in buckets.values():
            v.sort()  # deterministic fuzzy tie-breaks (dist, then shortest, then lexical)
        self.term_buckets = buckets

    # ---- writes ----
    def reset(self) -> None:
        """Drop all in-memory docs/chunks/vectors (call load() to re-read from disk)."""
        self.chunks = []
        self.docs = {}
        self.vectors = None
        self._reset_text_cache()

    def add_document(self, doc: dict, chunks: list[dict]) -> None:
        self.add_documents([(doc, chunks)])

    def add_documents(self, items: list[tuple[dict, list[dict]]]) -> None:
        """Add many documents then rebuild embeddings once (avoids O(n^2) re-embedding).

        Replaces any previous chunks for a document id (re-ingest of a new version
        replaces the old — editions never silently mix).
        """
        ids = {doc["id"] for doc, _ in items}
        self.chunks = [c for c in self.chunks if c.get("document_id") not in ids]
        for doc, chunks in items:
            self.docs[doc["id"]] = doc
            self.chunks.extend(chunks)
        self._rebuild_vectors()
        self.rebuild_text_cache()

    def _rebuild_vectors(self) -> None:
        texts = [c.get("text", "") for c in self.chunks]
        if not texts:
            self.vectors = None
            return
        if getattr(self.embedder, "name", "") == "tfidf":
            # Sparse CSR: memory/disk scale with non-zero terms, not n_chunks x vocab.
            self.embedder.fit(texts)
            self.vectors = self.embedder.encode_sparse(texts)
        else:
            self.vectors = self.embedder.encode(texts)
            # L2-normalize for cosine similarity
            norms = np.linalg.norm(self.vectors, axis=1, keepdims=True) + 1e-9
            self.vectors = self.vectors / norms

    # ---- persistence ----
    def save(self) -> None:
        with open(os.path.join(self.index_dir, "chunks.jsonl"), "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        with open(os.path.join(self.index_dir, "docs.json"), "w", encoding="utf-8") as f:
            json.dump(self.docs, f, ensure_ascii=False, indent=2)
        meta = {"embedder": getattr(self.embedder, "name", "unknown")}
        with open(os.path.join(self.index_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f)
        np_sparse = os.path.join(self.index_dir, "vectors.npz")
        np_dense = os.path.join(self.index_dir, "vectors.npy")
        if self.vectors is not None:
            from scipy import sparse
            if sparse.issparse(self.vectors):
                sparse.save_npz(np_sparse, self.vectors.tocsr())
                if os.path.exists(np_dense):
                    os.remove(np_dense)
            else:
                np.save(np_dense, self.vectors)
                if os.path.exists(np_sparse):
                    os.remove(np_sparse)
        self._save_embedder_state()

    def _save_embedder_state(self) -> None:
        emb = self.embedder
        if getattr(emb, "name", "") != "tfidf":
            return
        state = {"vocab": emb.vocab, "idf": emb.idf.tolist()}
        with open(os.path.join(self.index_dir, "embedder_state.json"), "w", encoding="utf-8") as f:
            json.dump(state, f)

    def _restore_embedder_state(self) -> None:
        sp = os.path.join(self.index_dir, "embedder_state.json")
        if not os.path.exists(sp) or getattr(self.embedder, "name", "") != "tfidf":
            return
        with open(sp, encoding="utf-8") as f:
            state = json.load(f)
        self.embedder.vocab = state.get("vocab", {})
        self.embedder.idf = np.asarray(state.get("idf", []), dtype=np.float32)

    def load(self) -> bool:
        cp = os.path.join(self.index_dir, "chunks.jsonl")
        if not os.path.exists(cp):
            return False
        with open(cp, encoding="utf-8") as f:
            self.chunks = [json.loads(line) for line in f if line.strip()]
        dp = os.path.join(self.index_dir, "docs.json")
        if os.path.exists(dp):
            with open(dp, encoding="utf-8") as f:
                self.docs = json.load(f)
        self._restore_embedder_state()
        vp_sparse = os.path.join(self.index_dir, "vectors.npz")
        vp_dense = os.path.join(self.index_dir, "vectors.npy")
        vecs = None
        try:
            if os.path.exists(vp_sparse):
                from scipy import sparse
                vecs = sparse.load_npz(vp_sparse)
            elif os.path.exists(vp_dense):
                vecs = np.load(vp_dense)
        except Exception:
            vecs = None
        # Rebuild if corpus size changed or dims cannot match (stale vocab)
        if vecs is not None and vecs.shape[0] == len(self.chunks) and vecs.shape[1] == self.embedder.dim:
            self.vectors = vecs
        else:
            self._rebuild_vectors()
        self.rebuild_text_cache()
        return True

    def count(self) -> int:
        return len(self.chunks)
