# AfroMedX production image: code + SBERT weights + vector index + guideline
# PDFs baked in as ONE self-contained artifact (index and PDFs can never skew).
# Secrets are NEVER baked in: configure them via env_file at runtime.
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1 \
    HF_HOME=/app/model-cache

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Dependencies first (layer cache): lockfile-respecting, no dev extras.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --extra semantic

# Pre-download the embedding model so first boot never fetches weights.
RUN uv run --no-sync --frozen python -c \
    "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Application + derived index + source PDFs (see .dockerignore for exclusions).
COPY app/ ./app/
COPY data/index/ ./data/index/
COPY "Malawi Guidelines/" "./Malawi Guidelines/"
COPY frontend/ ./frontend/
COPY scripts/ ./scripts/
COPY eval/ ./eval/

EXPOSE 8000

# Single worker is fine for pilot scale; raise --workers when needed.
CMD ["uv", "run", "--no-sync", "--frozen", "uvicorn", "app.main:app", \
     "--host", "0.0.0.0", "--port", "8000"]
