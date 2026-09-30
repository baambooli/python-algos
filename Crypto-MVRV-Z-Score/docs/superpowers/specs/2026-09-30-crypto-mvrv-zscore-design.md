# Crypto MVRV Z-Score (price/volume proxy) - Design

## Goal
A local Streamlit app (port 8502) that shows an MVRV-Z-score-style valuation indicator, in the
style of lookintobitcoin.com, for any coin in `D:\share\stooq_output_crypto`. Layout and UX follow
`D:\share\python-algos\DCA-calc`.

## Key constraint
True MVRV Z = (Market Cap - Realized Cap) / std(Market Cap) needs on-chain/supply data. The
data folder has only daily OHLC + volume (Stooq format, one `<TICKER>.CSV` per coin, about
1000 rows each; BTCUSDT runs 2024-01-05 to 2026-09-30). The app therefore computes a **proxy**
and labels it as such.

## Calculation (`mvrv.py`, pure logic, no I/O or UI)
- `realized_price = cumsum(close * vol) / cumsum(vol)` (expanding VWAP from first row; proxy for
  average holder cost basis).
- `z = (close - realized_price) / expanding_std(close)` (expanding = all history up to that date,
  no lookahead). With constant supply this equals the MVRV Z formula divided through by supply.
- First 30 days masked as NaN (unstable std).
- `mvrv_ratio = close / realized_price`.
- Bands (sidebar-adjustable): overheated >= 7, undervalued <= 0. Status label:
  Overheated / Neutral / Undervalued.
- Zero-volume rows are kept but carry zero weight; if cumulative volume is 0, the coin is
  reported as unusable.

## Data (`data.py`, adapted from DCA-calc)
- `list_tickers(folder)`, `rank_tickers(tickers, query)` as in DCA-calc (exact, prefix, substring).
- `load_ohlcv(folder, ticker)` reads `<DATE>`, `<CLOSE>`, `<VOL>`; date-indexed ascending
  DataFrame; drops duplicate dates and non-positive close.
- `is_excluded(ticker)`: true for leveraged tokens (name contains BULL or BEAR) and stablecoins
  (USDC, BUSD, DAI, TUSD, PAX, USDS, USDSB and similar). Sidebar checkbox, on by default,
  hides them.
- Default folder `D:\share\stooq_output_crypto`, overridable by `STOOQ_DIR` env var or the
  sidebar box.

## UI (`app.py`, port 8502)
- Sidebar: data folder, ticker search, selectbox (default BTCUSDT), start/end date
  (YYYY-MM-DD text inputs), band thresholds, hide-leveraged/stablecoin checkbox.
- Metrics row: price, realized price, current Z, MVRV ratio, status.
- Chart: two stacked Plotly panels, shared x-axis. Top: price (log scale) + realized price
  line. Bottom: Z-score line with red zone above the overheated threshold and green zone below
  the undervalued threshold. Unified hover.
- Expander with a data table. Caption: proxy from price and volume, not on-chain MVRV;
  anchored to the start of the data, so values differ from lookintobitcoin.
- Errors (bad date, short history, unreadable file, unusable volume) shown via `st.error`.
- `run.bat` runs `streamlit run app.py --server.port 8502`; `.streamlit/config.toml` sets port
  8502; `README.md`; `requirements.txt` (streamlit, pandas, plotly, pytest); `.gitignore`.

## Tests (pytest)
- Hand-computed VWAP and z on a tiny series; constant price gives 0/NaN; 30-day mask;
  zero-volume handling; ticker ranking; leveraged/stablecoin filter; smoke test loading the
  real BTCUSDT file (skipped if the folder is missing).

## Out of scope
On-chain data, downloading data, multi-coin comparison.
