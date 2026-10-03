# AfroMedX Runbook — operate the app without assistance

All commands run in PowerShell from the project folder
(`C:\Users\PaulT\Afromedx V1`). No code changes needed for any of this.

## Start the app
```powershell
cd "C:\Users\PaulT\Afromedx V1"
uv run uvicorn app.main:app --reload
```
- First boot downloads the embedding model (~90 MB) and takes 1–2 minutes.
  Later boots take ~40 seconds (model load). Be patient.
- The `--reload` flag restarts the backend automatically when Python files
  change. (It does NOT reload `.env` — see below.)
- Keep this window open. Closing it stops the app.

## Verify it works
Open `http://127.0.0.1:8000/api/health` — expect:
```json
{"status":"ok","chunks":13072,"provider":"...","embedder":"sbert:all-MiniLM-L6-v2"}
```
Then open `http://127.0.0.1:8000` and ask the malaria question.

## Use it from your phone (same Wi-Fi)
1. Start with network access instead:
   ```powershell
   uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```
2. Allow port 8000 once (Administrator PowerShell):
   ```powershell
   New-NetFirewallRule -DisplayName 'AfroMedX dev (8000)' -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow
   ```
3. Find this PC's address: `ipconfig` → IPv4 (e.g. `192.168.1.115`).
4. On the phone (same Wi-Fi, mobile data OFF): `http://<that-ip>:8000`.

## Stop the app
Press `Ctrl+C` in its window. If the port seems stuck afterwards:
```powershell
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force
```

## Change provider / model / keys
1. Edit `.env` in the project folder (never commit this file).
   Key lines: `AFROMEDX_LLM_PROVIDER` (`stub` | `gemini` | `mistral` |
   `openai_compatible`), `GEMINI_API_KEY`, `GEMINI_MODEL`, `MISTRAL_API_KEY`,
   `MISTRAL_MODEL`, `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL`
   (Groq runs through `openai_compatible`).
2. **Restart the server.** `.env` loads once at startup — editing it without
   restarting changes nothing (most common "my change did nothing" cause).

## Update to the latest version
```powershell
git pull
uv sync --extra dev --extra semantic
```
Then restart the server. Roll back with `git checkout <tag-or-hash>`
followed by a restart.

## Run the tests
```powershell
uv run pytest -q
```
66+ tests, a few seconds. Green means the code is healthy; a failing
*answer* with green tests is a data/quota issue, not a code issue.

## Troubleshooting
| Symptom | Likely cause → fix |
|---|---|
| `Address already in use` | Old server still running → kill python processes, start again |
| Browser shows old UI after a change | Cached files → hard refresh (`Ctrl+Shift+R`; phone: close tab, reopen) |
| Answers turn extractive/stub-like | Provider quota (Groq 429) or outage → wait, or check vendor console |
| `No reliable source found` for everything | Provider key missing/invalid in `.env` → fix key, restart |
| First boot very slow | Normal: model download + index load; watch `/api/health` until 200 |
| Phone can't reach `http://<ip>:8000` | Different Wi-Fi / mobile data on / firewall rule missing / PC asleep |

## Files you must never share or commit
`.env` (API keys), `data/index/` (rebuilt locally), `Malawi Guidelines/`
(source PDFs) — all gitignored. The repo holds code, config templates,
and docs only.
