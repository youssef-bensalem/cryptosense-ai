# CryptoSense AI

A personal research application for spot swing trading in cryptocurrencies, built with Python, pandas, and machine learning.

CryptoSense AI analyzes historical market data to describe current market conditions, estimate near-term volatility, flag conditions worth investigating, and test whether strategy ideas have historical evidence behind them. The finished product will be a local Streamlit dashboard covering BTC/USDT, ETH/USDT, and SOL/USDT.

This is also a learning project: every component is built step by step, with the reasoning behind it documented in the [roadmap](ROADMAP.md).

> **Disclaimer.** This is a research and learning tool, not financial advice. It makes no profit guarantees, never places trades, and does not connect to any trading account. Historical results do not predict future performance.

## Current status

**Phase 1 (complete): market data pipeline.** The notebook `notebooks/01_fetch_and_validate_btc.ipynb`:

- fetches the full daily BTC/USDT price history from Binance (August 2017 to the last completed day) through public market data, with no API key required
- converts it to a pandas DataFrame with UTC dates
- validates it for duplicate dates, gaps, missing or invalid values, and inconsistent candles, and excludes the current, still-forming candle
- saves raw and processed datasets locally and verifies that they reload identically
- charts closing price (linear and log scale) and trading volume

Upcoming phases cover feature engineering, market regime detection, volatility forecasting, setup identification, backtesting, and the web dashboard. See [ROADMAP.md](ROADMAP.md) for the full plan and progress.

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
