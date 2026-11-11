# CryptoSense AI web interface

A React + TypeScript front end (`web/`) on top of a FastAPI back end (`src/cryptosense/api.py`).
The back end only arranges results from the tested `cryptosense` modules (`src/cryptosense/service.py`);
it is read-only, never downloads market data and never places trades.

## Run it on your computer

From the project root, with `.venv` active:

```
python -m pip install -r requirements.txt
uvicorn cryptosense.api:app --port 8000
```

Open http://127.0.0.1:8000. The page is served from `web/dist` (the built front end).
The API description is at http://127.0.0.1:8000/api/docs.

Data comes from the `data/` folder, exactly as for the Streamlit dashboard. Refresh the data
with the Streamlit dashboard's "Refresh data from Binance" button (or the notebooks); the web
interface picks up the new files on the next page load.

## Change the front end

Needs a current Node.js LTS release (built and checked here with Node 22).

```
cd web
npm ci            # install the pinned packages (first time only)
npm run dev       # live-reloading page on http://localhost:5173, forwarding /api to port 8000
npm run build     # type-check and rebuild web/dist
```

Keep `uvicorn cryptosense.api:app --port 8000` running in another terminal while using `npm run dev`.

## Settings (environment variables, all optional)

| Variable | Meaning |
| --- | --- |
| `CRYPTOSENSE_DATA_DIR` | Data folder (default: `data/` in the project root) |
| `CRYPTOSENSE_WEB_DIR` | Built front end (default: `web/dist`) |
| `ASSISTANT_API_URL` | Chat-completions address of an OpenAI-compatible model service |
| `ASSISTANT_API_KEY` | Its API key. Set it on the host or in `.env`; never commit it |
| `ASSISTANT_MODEL` | The model name to request |

Without the three `ASSISTANT_*` settings the assistant uses a local Ollama server if one is
running, and otherwise gives only its fixed, checked answers (buy/sell and direction questions).

## Hosting checklist

1. Build step: `pip install -r requirements.txt`, then `cd web && npm ci && npm run build`.
2. Start command: `uvicorn cryptosense.api:app --host 0.0.0.0 --port $PORT`.
3. Data: `data/` is not in git. The host needs the processed CSV files
   (`data/processed/*_1d.csv` and `*_15m.csv`), kept up to date by a scheduled refresh.
4. Assistant: set the three `ASSISTANT_*` variables in the host's settings.
5. The assistant endpoint is limited to 8 questions a minute per visitor and 500 a day in total
   (`ASK_LIMIT` and `ASK_DAILY_LIMIT` in `api.py`). Behind a proxy every visitor can appear to have
   the same address; check this on the chosen host before relying on the per-visitor limit.
