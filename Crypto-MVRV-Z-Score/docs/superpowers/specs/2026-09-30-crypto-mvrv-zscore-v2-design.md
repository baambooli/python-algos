# Crypto MVRV Z-Score v2 - Design (supersedes the 2026-09-30 v1 spec)

## Goal
Local Streamlit app (port 8502) charting the MVRV Z-score for 20 major coins, in the style of
lookintobitcoin.com: one dark overlaid chart, real on-chain MVRV where free data exists, a
price/volume proxy otherwise. **No dependency on the user's Stooq/Binance CSV folders**: data comes
only from Coin Metrics Community API and Yahoo Finance, cached inside the project.

## Coins (verified against both APIs on 2026-09-30)
- **On-chain (Coin Metrics, free, no key)** - CoinMetrics asset id, free history start:
  btc 2010-07, ltc 2013-04, doge 2014-01, xmr 2014-05, xrp 2014-08, eth 2015-08, xlm 2015-09,
  etc 2016-07, bnb 2017-07, link 2017-09, bch 2017-08, ada 2017-12, dot 2020-08, uni 2020-09,
  icp 2021-05.
- **Proxy (Yahoo Finance, free, no key)** - Yahoo symbol: SOL-USD, TRX-USD, AVAX-USD, SHIB-USD,
  HBAR-USD (Coin Metrics does not serve these for free).
- Excluded: stablecoins, exchange/leveraged tokens, TON (no usable Yahoo series).
- The list lives in one constant (`COINS`) so adding a coin is a one-line change.

## Data layer
- `fetch_data.py` (also callable from the app): for each coin downloads data and writes
  `data/<SYMBOL>.csv`. `data/` is git-ignored.
  - On-chain: `GET https://community-api.coinmetrics.io/v4/timeseries/asset-metrics` with
    `metrics=PriceUSD,CapMrktCurUSD,CapMVRVCur`, `frequency=1d`, `paging_from=start`,
    `page_size=10000`, following `next_page_url`. Columns saved: `date, price, mcap, mvrv`.
  - Proxy: `GET https://query1.finance.yahoo.com/v8/finance/chart/<sym>?period1=0&period2=<now>&interval=1d`
    with a browser User-Agent. Columns saved: `date, close, vol`.
  - Network errors or a missing series for one coin are reported per coin and do not stop the others.
- `data.py` loads the cached CSVs: `COINS`, `cache_status()`, `load_coin(symbol) -> DataFrame`
  (date-indexed ascending, deduplicated, non-positive values dropped, NaN rows dropped).
- No network calls happen unless the user clicks **Update data** (or runs `fetch_data.py`).
  If the cache is empty the app says so and shows the button.
- The Stooq folder, `STOOQ_DIR`, the data-folder box, ticker search and the leveraged/stablecoin
  filter are removed.

## Calculation (`mvrv.py`, pure)
- **On-chain:** `realized_cap = mcap / mvrv`;
  `z = (mcap - realized_cap) / expanding population std(mcap)`; the first 30 rows of z are NaN.
  Rows with missing or non-positive `mvrv` are dropped before calculation.
- **Proxy:** `realized_price = cumsum(close*vol)/cumsum(vol)`;
  `z = (close - realized_price)/expanding std(close)`; first 30 rows NaN; all-zero volume raises.
- `classify(z, overheated, undervalued)` unchanged. Defaults 7 and 0, adjustable.

## UI (`app.py`, dark theme)
- `.streamlit/config.toml`: port 8502, address localhost, `[theme] base="dark"`.
- Sidebar: coin selectbox (20 coins, labelled "on-chain" or "proxy"), start/end date, threshold
  inputs, **Update data** button with a last-updated caption.
- Metrics row: price, realized price (cap/supply for on-chain = realized cap shown in $), Z, MVRV
  ratio, status.
- One Plotly figure, `plotly_dark` template: left log axis (market cap + realized cap for on-chain
  coins; price + realized price for proxy coins), right axis Z-score. Lines: Realized (blue),
  Market cap/Price (white), Z-score (orange). Red zone Z>=7 and green zone Z<=0 span the full plot
  with dotted borders, tied to the right axis. Unified hover, bottom legend.
- Caption states which mode the coin uses and, for proxy coins, that it is a price/volume
  proxy. Date range only trims the display; calculation always uses the full history.
- Errors (empty cache, bad/reversed dates, range outside data, too little history) are readable
  messages, not tracebacks.

## Tests (pytest, no network)
- Fetch parsers tested against small saved JSON samples (Coin Metrics incl. `next_page_url`
  paging; Yahoo incl. null entries); per-coin failure isolation.
- On-chain z hand-computed on a tiny series; realized cap derivation; MVRV <= 0 rows dropped.
- Proxy tests from v1 carry over. `load_coin` cleaning tests. Coin list sanity (20 unique symbols).
- App smoke test via `streamlit.testing.v1.AppTest` using a tiny temp cache.

## Out of scope
Scheduled refresh, coins beyond the list, any use of the old CSV folders, code review step.
