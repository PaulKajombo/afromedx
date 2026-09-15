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

    # ---- writes ----
    def add_document(self, doc: dict, chunks: list[dict]) -> None:
        # Replace any previous chunks for the same document id (re-ingest of new
        # version replaces old — never duplicates/mixes).
        self.chunks = [c for c in self.chunks if c.get("document_id") != doc["id"]]
        self.docs[doc["id"]] = doc
        self.chunks.extend(chunks)
        self._rebuild_vectors()

    def _rebuild_vectors(self) -> None:
        texts = [c.get("text", "") for c in self.chunks]
        if not texts:
            self.vectors = None
            return
        if getattr(self.embedder, "name", "") == "tfidf":
            self.embedder.fit(texts)
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
        if self.vectors is not None:
            np.save(os.path.join(self.index_dir, "vectors.npy"), self.vectors)
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
        vp = os.path.join(self.index_dir, "vectors.npy")
        if os.path.exists(vp):
            try:
                vecs = np.load(vp)
                # Rebuild if corpus size changed or dims cannot match (stale vocab)
                if vecs.shape[0] == len(self.chunks) and vecs.shape[1] == self.embedder.dim:
                    self.vectors = vecs
                else:
                    self._rebuild_vectors()
            except Exception:
                self._rebuild_vectors()
        else:
            self._rebuild_vectors()
        return True

    def count(self) -> int:
        return len(self.chunks)
