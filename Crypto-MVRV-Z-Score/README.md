# Crypto MVRV Z-Score

A local Streamlit app that charts an **MVRV-Z-score-style valuation indicator** for any coin in a
folder of [Stooq](https://stooq.com)-format daily crypto files, in the style of
[lookintobitcoin.com](https://www.lookintobitcoin.com/charts/mvrv-zscore/).

> **This is a proxy, not on-chain MVRV.** True MVRV Z needs market cap and realized cap (on-chain
> and supply data). The data here has only price and volume, so realized price is approximated by
> the cumulative volume-weighted average price. Values will differ from lookintobitcoin.com, and the
> indicator is anchored to the start of each coin's data (about 1000 days), not genesis.

## Requirements

- Python 3.10+ (developed on 3.13)
- A folder of daily files, one CSV per ticker (e.g. `BTCUSDT.CSV`) with columns
  `TICKER,PER,DATE,TIME,OPEN,HIGH,LOW,CLOSE,VOL,OPENINT`

## Setup

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The data folder defaults to `D:\share\stooq_output_crypto`. Change it in the sidebar or set the
`STOOQ_DIR` environment variable before launching.

## Run

```powershell
run.bat
```

or `.venv\Scripts\python.exe -m streamlit run app.py`. The app serves at http://localhost:8502.

## Usage

| Sidebar field | Meaning |
|---|---|
| Data folder | Folder containing the CSV files. |
| Hide leveraged tokens and stablecoins | Hides names containing BULL/BEAR and stablecoin/fiat pairs (USDC, BUSD, DAI, TUSD, PAX, USDS, EUR, GBP, AUD). On by default. |
| Search ticker / Coin | Exact match first, then prefix, then substring. |
| Start / End date | `YYYY-MM-DD`. Only trims the display; the indicator always uses the full history. |
| Overheated / Undervalued | Z thresholds for the red and green zones and the status label (defaults 7 and 0). |

The page shows price, realized price, Z-score, MVRV ratio and status, plus a chart with price
(log scale) against realized price on top and the Z-score with shaded zones below.

## How it calculates

- `realized_price = cumsum(close * volume) / cumsum(volume)` (expanding VWAP).
- `z = (close - realized_price) / expanding std of close` (all history up to each date).
- `mvrv_ratio = close / realized_price`.
- The first 30 days of Z are hidden because the expanding std is unstable there.
- Coins with 30 days of data or fewer show price only; coins with no volume report an error.

## Tests

```powershell
.venv\Scripts\python.exe -m pytest
```

## Project layout

- `mvrv.py` holds the calculation (no I/O or UI).
- `data.py` lists tickers, filters them and loads the CSV files.
- `app.py` is the Streamlit UI.
- `tests/` has the pytest suite.
- `docs/superpowers/` has the design spec and implementation plan.

This is an educational indicator, not investment advice.
