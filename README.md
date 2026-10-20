# CryptoSense AI

A personal research application for spot swing trading in cryptocurrencies, built with Python, pandas, and machine learning.

CryptoSense AI analyzes historical market data to describe current market conditions, estimate near-term volatility, flag conditions worth investigating, and test whether strategy ideas have historical evidence behind them. The finished product will be a local Streamlit dashboard covering BTC/USDT, ETH/USDT, and SOL/USDT.

This is also a learning project: every component is built step by step, with the reasoning behind it documented in the [roadmap](ROADMAP.md).

> **Disclaimer.** This is a research and learning tool, not financial advice. It makes no profit guarantees, never places trades, and does not connect to any trading account. Historical results do not predict future performance.

## Current status

**Phases 1–4 complete: data pipeline, feature engineering, market regimes and volatility forecasting.**

**Market data (Phase 1, extended in Phase 2).** Tested functions in `src/cryptosense/data.py` fetch the full daily history for the watchlist (BTC/USDT, ETH/USDT and SOL/USDT) from Binance's public market data, with no API key required. The data is validated for duplicate dates, gaps, missing or invalid values, and inconsistent candles; the current, still-forming candle is excluded; and raw and processed datasets are saved locally and verified to reload identically.

- `notebooks/01_fetch_and_validate_btc.ipynb`: step-by-step walkthrough of the pipeline for BTC
- `notebooks/02_fetch_watchlist.ipynb`: runs the pipeline for the whole watchlist and summarizes the results

**Features (Phase 2).** `src/cryptosense/features.py` adds nine features that are comparable across time and assets: log returns, 5- and 20-day momentum, distance from the 20- and 100-day moving averages, 20- and 60-day volatility, relative volume, and a 20-day efficiency ratio for trend cleanliness. A leakage test verifies that every feature uses only past data.

- `notebooks/03_returns_eda.ipynb`: exploratory analysis of BTC returns (fat tails, volatility clustering, moving-average lag, volume behavior, and feature overlap), with written observations kept separate from interpretations

**Market regimes (Phase 3).** `src/cryptosense/regimes.py` labels each day's trend ("up", "down" or "range") and volatility ("high" or "low", relative to the past year) using transparent rules, with a leakage test confirming the labels use only past data.

- `notebooks/04_regimes.ipynb`: regime stability analysis and a K-means clustering study, which found moderate, seed-stable structure matching the rule dimensions but drifting on 2024+ data, and was therefore kept as analysis rather than adopted

**Volatility forecasting (Phase 4).** Forecasts of BTC's realized volatility over the next 5 days, evaluated with a chronological train / validation / test split, purged boundaries, and a test set used once. Scaled recent volatility beat raw recent volatility in every validation and test year (by 8–11%). A Random Forest predicting volatility relative to its recent level performed comparably to that scaled baseline (worse on validation, better on test), so neither is established as better. Gradient boosting overfit the noisy target.

- `notebooks/05_volatility_forecast.ipynb`: target construction, baselines, model comparison, overfitting diagnosis, error analysis by regime and year, and stated limitations

**Testing.** 37 automated pytest tests cover the data, feature and regime modules, including pagination edge cases tested against a simulated exchange and leakage tests for features and regimes.

Upcoming phases cover setup identification, backtesting, and the web dashboard. See [ROADMAP.md](ROADMAP.md) for the full plan and progress.

## Tech stack

- **Python 3.12**, developed in VS Code with Jupyter notebooks
- **CCXT** for market data
- **pandas** for data processing
- **Plotly** for interactive charts
- **Git and GitHub** for version control

Planned for later phases: scikit-learn, pytest, and Streamlit.

## Getting started

Requires Python 3.12 and Git. Commands are for Windows PowerShell; macOS and Linux equivalents are noted.

```
git clone https://github.com/youssef-bensalem/cryptosense-ai.git
cd cryptosense-ai
py -3.12 -m venv .venv                 # macOS/Linux: python3.12 -m venv .venv
.venv\Scripts\Activate.ps1             # macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

Open `notebooks/01_fetch_and_validate_btc.ipynb` in VS Code, select the `.venv` kernel, and use **Restart** then **Run All**. The notebook creates the `data/` folder, downloads the data, validates and saves it, and draws the charts. Data files are generated locally and are not stored in the repository.

## Project structure

```
cryptosense-ai/
├── notebooks/          # step-by-step development notebooks
├── src/
│   └── cryptosense/            # reusable package (editable install)
│       ├── __init__.py         # marks the folder as a package
│       ├── data.py             # market data fetching and preparation
│       └── features.py         # feature engineering
├── tests/
│   ├── test_data.py            # tests for the data module
│   └── test_features.py        # tests for the features module, incl. leakage
├── data/               # generated locally: raw/ and processed/ (not committed)
├── pyproject.toml      # package definition for the editable install
├── requirements.txt    # pinned dependencies
├── ROADMAP.md          # workflow, principles, and milestones
└── README.md
```
