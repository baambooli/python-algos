# Crypto MVRV Z-Score v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (inline) to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Stooq-CSV data source with Coin Metrics + Yahoo downloads cached in `data/`, compute real on-chain MVRV Z for 15 coins and the price/volume proxy for 5, and show a dark overlaid chart.

**Architecture:** `mvrv.py` (pure calc), `fetch_data.py` (HTTP + parsing + cache writing, HTTP injected for tests), `data.py` (coin list + cache loading), `app.py` (Streamlit UI). Tests never touch the network.

**Tech Stack:** Python 3.13, pandas, plotly, streamlit, requests, pytest.

**Spec:** `docs/superpowers/specs/2026-09-30-crypto-mvrv-zscore-v2-design.md`

## Global Constraints

- Port 8502, address localhost, dark theme (`[theme] base="dark"`).
- Data only from Coin Metrics Community API and Yahoo Finance; no reference to `D:\share\stooq_output_crypto` or `STOOQ_DIR` anywhere in code.
- Network only on explicit user action (Update data button / `python fetch_data.py`).
- On-chain: `realized_cap = mcap/mvrv`; `z = (mcap - realized_cap)/expanding population std(mcap)`; first 30 rows NaN.
- Proxy: expanding VWAP realized price; same z form on close; first 30 rows NaN.
- 20 coins: on-chain btc ltc doge xmr xrp eth xlm etc bnb link bch ada dot uni icp; proxy SOL TRX AVAX SHIB HBAR.
- No code-review step; do not use Fable or Opus models.

## Review Focus

- Empty `data/` folder: app shows a clear message and the Update button, no traceback. (Task 3 test)
- One coin fails to download: others still saved, failure reported. (Task 2 test)
- Null/None points in Yahoo JSON and missing/zero MVRV in Coin Metrics rows: dropped, no crash or inf. (Task 2 + 1 tests)
- Coin with <= 30 rows: z all NaN, app warns. (Task 1 test, Task 3 handling)
- Reversed dates or range outside data: readable error. (Task 3 test)

---

### Task 1: On-chain calculation and unified frame

**Files:** Modify `mvrv.py`, `tests/test_mvrv.py`

**Interfaces:**
- Produces: `compute_onchain(df, min_days=MIN_DAYS)` (input cols `mcap`, `mvrv`; output adds `realized_cap`, `z`); `compute(df, kind, min_days=MIN_DAYS) -> DataFrame` with kind `"onchain"|"proxy"` returning unified columns `price, value, realized, ratio, z` (onchain: value=mcap, realized=realized_cap, ratio=mvrv, price=`price` col or mcap; proxy: value=close, realized=realized_price, ratio=mvrv_ratio, price=close). Existing `compute_mvrv` and `classify` unchanged.

- [ ] Step 1: add failing tests: hand-computed onchain (mcap [100,200,300], mvrv [1,2,3] -> rcap [100,100,100], z[1] = (200-100)/50 = 2.0, z[2] = 200/np.std([100,200,300])); first-30 mask; constant mcap gives 0/NaN; `compute` returns the unified columns for both kinds; unknown kind raises ValueError.
- [ ] Step 2: run, expect FAIL (`ImportError`/`AttributeError`).
- [ ] Step 3: implement `compute_onchain` and `compute`.
- [ ] Step 4: run `.venv/Scripts/python.exe -m pytest tests/test_mvrv.py -q`, expect all pass.
- [ ] Step 5: commit.

### Task 2: Coin list, fetching and cache

**Files:** Create `fetch_data.py`, `tests/test_fetch.py`; rewrite `data.py`, `tests/test_data.py`; modify `requirements.txt` (add `requests`), `.gitignore` (add `data/`)

**Interfaces:**
- `data.py`: `Coin(symbol, name, kind, source_id)` NamedTuple; `COINS: list[Coin]` (20); `DATA_DIR` (Path, default `<project>/data`, override env `MVRV_DATA_DIR`); `cache_status(folder=DATA_DIR) -> dict[str,bool]`; `last_updated(folder) -> datetime|None`; `load_coin(coin, folder=DATA_DIR) -> DataFrame` (onchain cols `price,mcap,mvrv`; proxy cols `close,vol`; date-indexed ascending, deduped, NaN/non-positive rows dropped, raises `ValueError` if file missing or empty).
- `fetch_data.py`: `parse_coinmetrics(pages) -> DataFrame`; `parse_yahoo(payload) -> DataFrame`; `fetch_coin(coin, get) -> DataFrame`; `fetch_all(folder=DATA_DIR, get=http_get, progress=None) -> dict[str,str]` mapping symbol to `"ok <rows> rows"` or `"failed: <reason>"`; `http_get(url) -> dict` (requests, UA header, 30 s timeout, raises on HTTP error). Running `python fetch_data.py` calls `fetch_all` and prints the results.

- [ ] Step 1: failing tests. `test_fetch.py`: parse_coinmetrics over two pages (strings to floats, `2010-07-18T00:00:00.000000000Z` to date, rows missing CapMVRVCur kept as NaN); `fetch_coin` follows `next_page_url` until absent (fake `get`); parse_yahoo drops null close and treats null volume as 0; `fetch_all` with a fake `get` that raises for one coin writes the other CSVs and reports `failed:` for the bad one; saved CSV round-trips through `load_coin`. `test_data.py`: COINS has 20 unique symbols, 15 onchain and 5 proxy; `load_coin` cleaning for both kinds (duplicates, unsorted, NaN, mvrv <= 0); missing file message; `cache_status`; `last_updated` is None for empty folder.
- [ ] Step 2: run, expect FAIL (imports).
- [ ] Step 3: implement `data.py` and `fetch_data.py`. Coin Metrics URL: `https://community-api.coinmetrics.io/v4/timeseries/asset-metrics?assets=<id>&metrics=PriceUSD,CapMrktCurUSD,CapMVRVCur&frequency=1d&paging_from=start&page_size=10000&start_time=2009-01-01`. Yahoo URL: `https://query1.finance.yahoo.com/v8/finance/chart/<sym>?period1=0&period2=<now>&interval=1d`.
- [ ] Step 4: `.venv/Scripts/python.exe -m pip install -q requests`; run `pytest -q`, expect pass.
- [ ] Step 5: live smoke: `.venv/Scripts/python.exe fetch_data.py`; expect 20 lines, BTC > 5000 rows; commit (data/ ignored).

### Task 3: Dark overlay UI, config, README

**Files:** Rewrite `app.py`; modify `.streamlit/config.toml`, `README.md`; create `tests/test_app.py`

**Interfaces:** Consumes `data.{COINS, DATA_DIR, cache_status, last_updated, load_coin}`, `fetch_data.fetch_all`, `mvrv.{compute, classify, MIN_DAYS}`.

- [ ] Step 1: failing tests with `streamlit.testing.v1.AppTest` and a temp cache set via `MVRV_DATA_DIR` (monkeypatch env): empty folder shows an info message and no exception; a synthetic 120-row BTC (onchain) cache renders 5 metrics; reversed dates shows an error; a coin with 10 rows shows the too-little-history warning.
- [ ] Step 2: run, expect FAIL.
- [ ] Step 3: implement: sidebar coin selectbox labelled "BTC - Bitcoin (on-chain)" / "(proxy)", dates, thresholds, Update data button (spinner, per-coin results in an expander) and last-updated caption; single figure with `make_subplots(specs=[[{"secondary_y": True}]])`, `plotly_dark`, left log axis (market cap + realized cap, or price + realized price), right Z axis with full-width red/green `add_hrect(secondary_y=True)` and dotted `add_hline`; colors realized `#19b5fe`, market cap/price `#ffffff`, z `#ff9f1c`; `st.plotly_chart(fig, theme=None, width="stretch")`; caption naming the mode. Theme: add `[theme] base="dark"` to config.toml.
- [ ] Step 4: run `pytest -q`, expect all pass; start the app on 8502 headless, confirm health 200 and render BTC + SOL through AppTest against the real `data/`.
- [ ] Step 5: rewrite README for the new data flow; grep the repo for `stooq` and `STOOQ_DIR` (only docs history may mention them); commit.
