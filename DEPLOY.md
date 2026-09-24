# AfroMedX deployment

Git is the source of truth: servers only ever run committed code (`git log`
shows exactly what is live). Secrets live only in each environment's `.env`
file, which is gitignored and created once per host.

## What ships
- `Dockerfile` — one self-contained image: code + SBERT weights + vector
  index (`data/index/`) + guideline PDFs (`Malawi Guidelines/`), so citations
  can never skew from their sources. No keys inside (see `.dockerignore`).
- `docker-compose.yml` — `app` + `caddy` (automatic HTTPS).
- `Caddyfile` — serves `https://app.afromedx.com`.

## DNS
Create record: `app` → A → `<server-ip>` (apex `afromedx.com` untouched).

## First deploy (VPS with Docker + Compose plugin)
```bash
git clone <repo-url> afromedx && cd afromedx
cp .env.example .env   # then fill in: provider, *_API_KEY, models
docker compose up -d --build
sleep 90  # cold boot: SBERT model load
curl -s http://localhost:8000/api/health
```
Expected: `{"status":"ok","chunks":13072,"provider":"...","embedder":"sbert:all-MiniLM-L6-v2"}`.
Then open `https://app.afromedx.com` (Caddy provisions TLS automatically;
allow ports 80/443 through the host firewall: `ufw allow 80,443/tcp`).

## Updates
```bash
git pull
docker compose up -d --build
docker compose ps   # app (healthy) + caddy (running)
```
Restart, re-poll `/api/health`, done. Rollback: `git checkout <tag> &&
docker compose up -d --build` (code, index, and PDFs revert together).

## Pilot alternative ($0): this PC + Cloudflare Tunnel
No VPS needed for testing: run the app locally, run `cloudflared tunnel
--url http://localhost:8000`, map the resulting hostname (or a DNS CNAME for
`app.afromedx.com`) in Cloudflare Zero Trust. PC must stay on; same-origin
policy and HTTPS are handled by Cloudflare.

## Operational notes
- Single uvicorn worker is the default (see `CMD`); raise `--workers` if
  concurrent clinical use demands it (each worker reloads the ~1 GB model).
- Model downloads happen once at image build (`HF_HOME` baked in).
- Logs: `docker compose logs -f app`. Provider failures log sanitized
  warnings (never keys); repeated 429/503s mean vendor quota/capacity.
- Backups: `.env` (off-repo copy) + git history are sufficient to rebuild;
  the index regenerates via `scripts/ingest_all.py` if ever lost.
