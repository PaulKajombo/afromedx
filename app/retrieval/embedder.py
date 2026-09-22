"""Embedder abstraction. TF-IDF (pure numpy, always works offline); SBERT when installed.

Local-first: no API key, no server, no heavy SciPy stack required for the
default path. Installing `.[semantic]` upgrades to sentence-transformers
without any code change.
"""
from __future__ import annotations

import math
import os
import re

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an the and or of to in on for with is are was were be been by as at from "
    "this that these those it its into over under after before between through "
    "what how when where which who whom do does did can could should would may "
    "might will shall have has had having not no yes if then than so such very "
    "more most other some any each per you your why".split()
)


def _tokens(text: str) -> list[str]:
    toks = [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP and len(t) > 1]
    # unigrams + bigrams capture phrases like "severe malaria", "iv artesunate"
    grams = list(toks)
    grams += [f"{a} {b}" for a, b in zip(toks, toks[1:])]
    return grams


class TfidfEmbedder:
    def __init__(self, max_features: int = 20000) -> None:
        self.max_features = max_features
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray = np.zeros(0, dtype=np.float32)

    def fit(self, texts: list[str]) -> None:
        from collections import Counter
        df: Counter[str] = Counter()
        for t in texts or ["empty"]:
            df.update(set(_tokens(t)))
        # keep most frequent terms
        top = [w for w, _ in df.most_common(self.max_features)]
        self.vocab = {w: i for i, w in enumerate(top)}
        n = max(1, len(texts))
        self.idf = np.array(
            [math.log((n + 1) / (df[w] + 1)) + 1.0 for w in top], dtype=np.float32)

    def encode(self, texts: list[str]) -> np.ndarray:
        if not self.vocab:
            self.fit(texts)
        mat = np.zeros((len(texts), len(self.vocab)), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in _tokens(t):
                j = self.vocab.get(tok)
                if j is not None:
                    mat[i, j] += 1.0
        if self.idf.shape[0] == mat.shape[1]:
            mat = mat * self.idf
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
        return (mat / norms).astype(np.float32)

    def encode_sparse(self, texts: list[str]):
        """Same TF-IDF vectors as encode(), but as a CSR matrix.

        Keeps memory/disk proportional to non-zero terms instead of
        n_chunks x vocab (which is ~1 GB for the full Malawi corpus).
        """
        from scipy import sparse

        if not self.vocab:
            self.fit(texts)
        indptr = [0]
        indices: list[int] = []
        data: list[float] = []
        for t in texts:
            counts: dict[int, float] = {}
            for tok in _tokens(t):
                j = self.vocab.get(tok)
                if j is not None:
                    counts[j] = counts.get(j, 0.0) + 1.0
            for j, c in counts.items():
                indices.append(j)
                data.append(c)
            indptr.append(len(indices))
        mat = sparse.csr_matrix(
            (np.asarray(data, dtype=np.float32), indices, np.asarray(indptr)),
            shape=(len(texts), len(self.vocab)),
        )
        if self.idf.shape[0] == mat.shape[1]:
            mat = (mat @ sparse.diags(self.idf)).tocsr()
        norms = np.sqrt(mat.multiply(mat).sum(axis=1)).A.ravel() + 1e-9
        mat = (sparse.diags(1.0 / norms) @ mat).tocsr()
        return mat.astype(np.float32)


    @property
    def dim(self) -> int:
        return len(self.vocab)

    @property
    def name(self) -> str:
        return "tfidf"


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)
        self._name = f"sbert:{model_name}"

    def fit(self, texts: list[str]) -> None:  # no fitting needed
        pass

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(texts, normalize_embeddings=True,
                                  show_progress_bar=len(texts) > 200)
        return np.asarray(vecs, dtype=np.float32)

    @property
    def dim(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    @property
    def name(self) -> str:
        return self._name


def get_embedder(force_tfidf: bool = False):
    """Auto-select: SBERT if installed and not forced off, else TF-IDF."""
    if force_tfidf or os.environ.get("AFROMEDX_FORCE_TFIDF") == "1":
        return TfidfEmbedder()
    try:
        import sentence_transformers  # noqa: F401
        return SentenceTransformerEmbedder()
    except Exception:
        return TfidfEmbedder()
