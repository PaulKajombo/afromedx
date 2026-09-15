# AfroMedX Clinical Search — MVP

> Google for Malawian clinical guidelines, with an AI layer that returns the answer directly.
> **NO RELIABLE SOURCE = NO CLINICAL ANSWER.**

## Architecture (Phase 0 assessment)

Repo was empty → greenfield MVP scaffolded:

```text
Guideline PDFs ─▶ ingestion (pypdf → clean → section-aware chunks + page metadata)
      ─▶ local vector index (data/index: chunks.jsonl + vectors.npy, TF-IDF now / SBERT optional)
      ─▶ hybrid retrieval (semantic cosine + keyword + synonyms, metadata filter, min-score gate)
      ─▶ LLM layer (Stub offline default / OpenAI-compatible when keyed, grounding-first prompt)
      ─▶ FastAPI + clinical search UI (title, key points, SOURCE block with doc/edition/section/page)
      ─▶ eval (eval/benchmark.json: Recall@K, citation accuracy, grounded rate, abstention)
```

| Decision | Choice | Why |
|---|---|---|
| Backend | FastAPI + uvicorn | simple, local, serves API + UI in one process |
| PDF extraction | pypdf | pure-Python, no system deps on Windows |
| Embeddings | TF-IDF default, sentence-transformers optional (`pip install -e .[semantic]`) | free, offline, no API key; upgrade path without code change |
| Vector store | JSONL + .npy in `data/index` | zero-server local; swap to Chroma/pgvector later behind same interface |
| LLM | `LLMProvider` interface; `stub` default, `openai_compatible` when `OPENAI_API_KEY` set | no coupling to one vendor; safe offline default that abstains |
| Frontend | vanilla HTML/JS served by FastAPI | search-engine feel, no build step |

## Quickstart (Windows, uv-managed Python)

```powershell
# 1. install deps (dev extra adds pytest). uv provisions Python if needed.
uv sync --extra dev
# or: pip install -e ".[dev]"

# 2. seed demo index (sample Malawi-style excerpts; replace with real PDFs)
uv run python scripts/demo_seed.py

# 3. run API + UI
uv run uvicorn app.main:app --reload
# open http://127.0.0.1:8000

# 4. run tests + evaluation
uv run pytest -q
uv run python eval/evaluate.py
```

Optional semantic upgrade (downloads a small local embedding model — no API key):
`uv sync --extra semantic` then set `AFROMEDX_FORCE_TFIDF=0` (default) in `.env`.
A `.env` is only needed for the LLM provider (see below); defaults run fully offline.

## Real guidelines

1. Drop PDFs in `data/guidelines/` and record provenance.
2. Ingest: `python scripts/ingest.py data/guidelines/<f>.pdf --doc-id mstg-6e --title "Malawi Standard Treatment Guidelines" --edition "6th Edition" --year 2023 --source "Malawi Ministry of Health"`
3. `POST /api/reload`, re-run eval.

## API

- `POST /api/search` `{query, top_k?, document_id?}` → `{answer{title,body,key_points,citations,abstained}, passages, meta}`
- `GET /api/documents`, `GET /api/health`

## Config (.env — never commit keys)

See `.env.example`. Key vars: `AFROMEDX_LLM_PROVIDER`, `OPENAI_API_KEY/BASE_URL/MODEL`,
`AFROMEDX_TOP_K/MIN_SCORE/SEMANTIC_WEIGHT`, `AFROMEDX_INDEX_DIR`.

## Roadmap status

- [x] Phase 1 ingestion (PDF→pages→chunks+metadata, section-aware)
- [x] Phase 2 hybrid retrieval (semantic+keyword+synonyms, out-of-scope gate)
- [x] Phase 3 grounded LLM + abstention (NO SOURCE = NO ANSWER)
- [x] Phase 4 citations (doc/edition/section/page + excerpts)
- [x] Phase 5 search UI (question → answer + source card)
- [x] Phase 6 eval benchmark + tests (Recall@K 1.00, citation 1.00, grounded 1.00, abstention 1.00)
- [ ] Phase 7: replace SAMPLE docs with authorised PDFs (provenance-logged), expand library
