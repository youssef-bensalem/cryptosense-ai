# CryptoSense AI — instructions for Claude Code

## Project
Personal crypto research and decision-support app (BTC/USDT, ETH/USDT, SOL/USDT). Research tool, not financial advice; never place trades or connect to a trading account. See README.md and ROADMAP.md for status and findings.

- `main` branch = finished daily swing-trading project (tag v1.0). Do not modify `main` directly.
- `intraday` branch = intraday research track (Phases 8–12 in ROADMAP.md).

## Environment (Windows, PowerShell)
- Python 3.12, virtual environment in `.venv`; package installed editable from `src/cryptosense/`.
- Run tests: `python -m pytest`
- Run dashboard: `streamlit run app/main.py`
- Data lives in `data/` (git-ignored); load it with `load_processed(symbol, DATA_DIR, timeframe=...)`.

## Code rules
- Analysis logic goes in `src/cryptosense/` with pytest tests; notebooks only orchestrate; `app/` only displays results from tested modules.
- Never modify an existing test to make it pass. If an existing test fails, stop and explain why.
- Existing daily behavior (timeframe "1d") must stay identical unless I explicitly ask otherwise.
- New functions get docstrings; new modules get tests, including a truncation leakage test for anything time-based.
- Every new dependency must be explained before installing and then pinned in requirements.txt.

## Research rules (most important)
- No information from after time t may be used for any feature, label, scaler, threshold or decision at time t.
- Split data chronologically, never randomly. Purge boundaries by the outcome horizon.
- Rules, thresholds, horizons and costs are fixed BEFORE results are computed. Never change them in response to results.
- The held-out test period is evaluated once. Never re-run it after changes.
- Always compare against base rates (all periods, and the same regime) and report standard errors.
- Costs: 0.15% per side (0.10% fee + 0.05% slippage) unless I say otherwise.
- Report results as numbers only. I write the interpretation.

## Workflow rules
- Do not commit or push. I review with git diff and commit myself.
- If anything cannot be done as specified, stop and ask instead of improvising.
- At the end of each task, list every file changed, every test added, and the full pytest result.