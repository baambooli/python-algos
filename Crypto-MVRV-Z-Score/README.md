# Crypto MVRV Z-Score

A local Streamlit app that charts the **MVRV Z-score** for 20 major coins in the style of
[lookintobitcoin.com](https://www.lookintobitcoin.com/charts/mvrv-zscore/): one dark chart with
market cap and realized cap on a log axis (left) and the Z-score on the right, with red
(overheated) and green (undervalued) zones.

Data comes only from free, keyless sources and is cached in `data/` inside the project:

- **Coin Metrics Community API** gives market cap and the MVRV ratio for **12 coins**, so the
  indicator is the real on-chain MVRV Z-score: BTC (from 2010), LTC, DOGE, XRP, ETH, XLM, ETC,
  LINK, BCH, ADA, UNI, ICP.
- **Yahoo Finance** gives price and volume for **8 coins** that have no complete free MVRV data:
  SOL, BNB, TRX, AVAX, SHIB, HBAR, DOT, XMR. These show a **proxy** (labelled "proxy" in the app):
  realized price is the cumulative volume-weighted average price, and history is shorter.

## Requirements

- Python 3.10+ (developed on 3.13) and an internet connection for the first download.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run

```powershell
run.bat
```

or `.venv\Scripts\python.exe -m streamlit run app.py`. The app serves at http://localhost:8502.
On first launch click **Update data** in the sidebar (about a minute). You can also download from
the command line with `.venv\Scripts\python.exe fetch_data.py`. Nothing is downloaded unless you
ask for it.

## Usage

| Sidebar field | Meaning |
|---|---|
| Coin | One of the 20 coins, labelled "on-chain" or "proxy". |
| Start / End date | `YYYY-MM-DD`. Only trims the display; the indicator always uses the full history. |
| Overheated / Undervalued | Z thresholds for the red and green zones and the status label (defaults 7 and 0). |
| Update data | Re-downloads every coin. One failing coin does not stop the others; results are listed. |

## How it calculates

- **On-chain coins:** `realized_cap = market_cap / MVRV`, then
  `Z = (market_cap - realized_cap) / expanding std(market_cap)`.
- **Proxy coins:** `realized_price = cumsum(close * volume) / cumsum(volume)`, then
  `Z = (close - realized_price) / expanding std(close)`.
- The first 30 days of Z are hidden because the expanding std is unstable there.
- Coin Metrics' free tier does not include the realized-cap metric itself, which is why it is
  derived from the MVRV ratio. Its free MVRV for BNB stops in 2019, for DOT in 2022 and is absent
  for XMR, so those use the proxy.

## Tests

```powershell
.venv\Scripts\python.exe -m pytest
```

The tests never touch the network.

## Project layout

- `mvrv.py` holds the calculations (no I/O or UI).
- `data.py` has the coin list and loads the cached CSV files.
- `fetch_data.py` downloads from Coin Metrics and Yahoo into `data/`.
- `app.py` is the Streamlit UI.
- `tests/` has the pytest suite.
- `docs/superpowers/` has the design specs and implementation plans.

This is an educational indicator, not investment advice.
