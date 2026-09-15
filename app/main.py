"""FastAPI app: serves /api/* and the clinical search frontend. Runs fully offline."""
from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .api.routes import router, set_store
from .config import settings
from .retrieval.embedder import get_embedder
from .retrieval.store import VectorStore

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")

app = FastAPI(title="AfroMedX Clinical Search", version="0.1.0")
app.include_router(router, prefix="/api")

store = VectorStore(settings.index_dir, embedder=get_embedder(force_tfidf=settings.force_tfidf))
if not store.load():
    # empty index on first run; seed script or /scripts/ingest populates it
    store.save()
set_store(store)

if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/", include_in_schema=False)
def index():
    fp = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(fp):
        return FileResponse(fp)
    return {"message": "AfroMedX API running. Frontend not found.", "docs": "/docs"}


@app.post("/api/reload", include_in_schema=False)
def reload_index():
    store.load()
    set_store(store)
    return {"status": "reloaded", "chunks": store.count()}
