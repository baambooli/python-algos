# Crypto MVRV Z-Score Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A local Streamlit app on port 8502 that charts a price/volume-proxy MVRV Z-score for any coin in `D:\share\stooq_output_crypto`.

**Architecture:** Three small modules mirroring DCA-calc: `mvrv.py` (pure calculation), `data.py` (CSV I/O, ticker search/filter), `app.py` (Streamlit UI). The indicator is always computed on the coin's full history, then sliced to the chosen date range for display.

**Tech Stack:** Python 3.10+, pandas, plotly, streamlit, pytest.

**Spec:** `docs/superpowers/specs/2026-09-30-crypto-mvrv-zscore-design.md`

## Global Constraints

- Streamlit must serve on port **8502**.
- Default data folder `D:\share\stooq_output_crypto`, overridable by `STOOQ_DIR` env var or the sidebar box.
- CSV columns are Stooq format: `<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>`; files are named `<TICKER>.CSV`.
- `realized_price = cumsum(close*vol)/cumsum(vol)`; `z = (close - realized_price)/expanding_std(close)`; first 30 rows of z are NaN.
- Bands default: overheated >= 7, undervalued <= 0, adjustable in sidebar.
- The UI must state the indicator is a price/volume proxy, not on-chain MVRV.
- Work in `D:\share\python-algos\Crypto-MVRV-Z-Score` (git repo root is `D:\share\python-algos`).

## Review Focus

- Coin with fewer than 31 rows: z is all NaN. The app shows a clear message, not an empty or crashing chart. (Task 1 test, Task 3 handling)
- Coin whose volume is all zero: clear error, not a divide-by-zero. (Task 1 test)
- CSV with blank or NaN rows, or duplicate dates: loader cleans them. (Task 2 test)
- Missing ticker file or missing data folder: readable error. (Task 2 test, Task 3)
- Start date after end date, or a range entirely outside the data: readable message, no traceback. (Task 3)

---

### Task 1: MVRV calculation

**Files:**
- Create: `requirements.txt`, `.gitignore`, `mvrv.py`, `tests/test_mvrv.py`

**Interfaces:**
- Produces: `MIN_DAYS = 30`; `compute_mvrv(df, min_days=MIN_DAYS) -> DataFrame` (input columns `close`, `vol`, date index; output adds `realized_price`, `mvrv_ratio`, `z`); `classify(z, overheated=7.0, undervalued=0.0) -> str` returning `"Overheated"`, `"Undervalued"`, `"Neutral"` or `"n/a"`.

- [ ] **Step 1: Scaffold and venv**

```bash
cd /d/share/python-algos/Crypto-MVRV-Z-Score
printf 'streamlit\npandas\nplotly\npytest\n' > requirements.txt
printf '.venv/\n__pycache__/\n.pytest_cache/\n' > .gitignore
python -m venv .venv
.venv/Scripts/python.exe -m pip install -q -r requirements.txt
mkdir -p tests
```

- [ ] **Step 2: Write the failing tests** in `tests/test_mvrv.py`

```python
import numpy as np
import pandas as pd
import pytest

from mvrv import classify, compute_mvrv


def _df(close, vol):
    idx = pd.date_range("2024-01-01", periods=len(close), freq="D")
    return pd.DataFrame({"close": close, "vol": vol}, index=idx, dtype=float)


def test_hand_computed_vwap_and_z():
    out = compute_mvrv(_df([10, 20, 30], [1, 1, 2]), min_days=0)
    assert out["realized_price"].tolist() == pytest.approx([10, 15, 22.5])
    assert np.isnan(out["z"].iloc[0])  # std is 0 on first row
    assert out["z"].iloc[1] == pytest.approx(1.0)
    assert out["z"].iloc[2] == pytest.approx((30 - 22.5) / np.std([10, 20, 30]))
    assert out["mvrv_ratio"].iloc[2] == pytest.approx(30 / 22.5)


def test_first_30_days_masked():
    out = compute_mvrv(_df(np.linspace(10, 100, 60), np.ones(60)))
    assert out["z"].iloc[:30].isna().all()
    assert out["z"].iloc[30:].notna().all()


def test_constant_price_gives_zero_or_nan():
    out = compute_mvrv(_df([5.0] * 40, [1.0] * 40))
    assert not (out["z"].dropna().abs() > 1e-9).any()


def test_short_history_is_all_nan():
    out = compute_mvrv(_df([1, 2, 3, 4], [1, 1, 1, 1]))
    assert out["z"].isna().all()


def test_zero_volume_rows_carry_no_weight():
    out = compute_mvrv(_df([10, 99, 30], [1, 0, 1]), min_days=0)
    assert out["realized_price"].tolist() == pytest.approx([10, 10, 20])


def test_leading_zero_volume_is_nan_not_inf():
    out = compute_mvrv(_df([10, 20, 30], [0, 1, 1]), min_days=0)
    assert np.isnan(out["realized_price"].iloc[0])
    assert np.isfinite(out["realized_price"].iloc[1:]).all()


def test_all_zero_volume_raises():
    with pytest.raises(ValueError, match="volume"):
        compute_mvrv(_df([1, 2, 3], [0, 0, 0]))


def test_classify():
    assert classify(7.5) == "Overheated"
    assert classify(7.0) == "Overheated"
    assert classify(0.0) == "Undervalued"
    assert classify(-1.0) == "Undervalued"
    assert classify(3.0) == "Neutral"
    assert classify(float("nan")) == "n/a"
    assert classify(3.0, overheated=2.0) == "Overheated"
```

- [ ] **Step 3: Run tests, verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_mvrv.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'mvrv'`

- [ ] **Step 4: Implement `mvrv.py`**

```python
"""MVRV-Z-score proxy computed from daily close and volume. No I/O or UI."""
import numpy as np
import pandas as pd

MIN_DAYS = 30


def compute_mvrv(df, min_days=MIN_DAYS):
    """Add realized_price (expanding VWAP), mvrv_ratio and z to a close/vol frame.

    z = (close - realized_price) / expanding population std of close. The first
    `min_days` rows of z are NaN because the expanding std is unstable there.
    """
    close, vol = df["close"], df["vol"]
    cumvol = vol.cumsum()
    if cumvol.iloc[-1] <= 0:
        raise ValueError("Volume is zero for every day, so a volume-weighted price cannot be computed.")
    realized = (close * vol).cumsum() / cumvol.where(cumvol > 0)
    std = close.expanding().std(ddof=0)
    z = (close - realized) / std.where(std > 0)
    z.iloc[:min_days] = np.nan
    out = df.copy()
    out["realized_price"] = realized
    out["mvrv_ratio"] = close / realized
    out["z"] = z
    return out


def classify(z, overheated=7.0, undervalued=0.0):
    if pd.isna(z):
        return "n/a"
    if z >= overheated:
        return "Overheated"
    if z <= undervalued:
        return "Undervalued"
    return "Neutral"
```

- [ ] **Step 5: Run tests, verify pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_mvrv.py -q`
Expected: 8 passed

- [ ] **Step 6: Commit**

```bash
git add requirements.txt .gitignore mvrv.py tests/test_mvrv.py
git commit -m "feat: MVRV z-score proxy calculation" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Data loading and ticker filtering

**Files:**
- Create: `data.py`, `tests/test_data.py`

**Interfaces:**
- Produces: `DEFAULT_FOLDER`; `list_tickers(folder) -> list[str]`; `rank_tickers(tickers, query, limit=200) -> list[str]`; `is_excluded(ticker) -> bool`; `load_ohlcv(folder, ticker) -> DataFrame` with columns `close`, `vol`, ascending unique DatetimeIndex; raises `ValueError` with a readable message for a missing or empty file.

- [ ] **Step 1: Write the failing tests** in `tests/test_data.py`

```python
from pathlib import Path

import pytest

from data import DEFAULT_FOLDER, is_excluded, list_tickers, load_ohlcv, rank_tickers

HEADER = "TICKER,PER,DATE,TIME,OPEN,HIGH,LOW,CLOSE,VOL,OPENINT\n"


def _write(tmp_path, name, rows):
    body = "".join(f"X,D,{d},00000000,1,1,1,{c},{v},0\n" for d, c, v in rows)
    (tmp_path / name).write_text(HEADER + body)


def test_load_ohlcv_cleans_data(tmp_path):
    _write(tmp_path, "AAAUSDT.CSV", [
        ("20240103", 3, 30), ("20240101", 1, 10), ("20240102", 2, 20),
        ("20240102", 2.5, 25),   # duplicate date: keep last
        ("20240104", 0, 5),      # non-positive close: dropped
        ("20240105", "", 5),     # NaN close: dropped
    ])
    df = load_ohlcv(tmp_path, "AAAUSDT")
    assert list(df.columns) == ["close", "vol"]
    assert df.index.is_monotonic_increasing and df.index.is_unique
    assert df["close"].tolist() == [1, 2.5, 3]


def test_load_ohlcv_lowercase_extension(tmp_path):
    _write(tmp_path, "AAAUSDT.csv", [("20240101", 1, 1)])
    assert len(load_ohlcv(tmp_path, "AAAUSDT")) == 1


def test_load_ohlcv_missing_file(tmp_path):
    with pytest.raises(ValueError, match="No data file"):
        load_ohlcv(tmp_path, "NOPEUSDT")


def test_load_ohlcv_empty_file(tmp_path):
    (tmp_path / "EMPTYUSDT.CSV").write_text(HEADER)
    with pytest.raises(ValueError, match="no price rows"):
        load_ohlcv(tmp_path, "EMPTYUSDT")


def test_list_tickers_sorted_csv_only(tmp_path):
    for n in ("B.CSV", "A.csv", "notes.txt"):
        (tmp_path / n).write_text("x")
    assert list_tickers(tmp_path) == ["A", "B"]


def test_list_tickers_missing_folder(tmp_path):
    with pytest.raises(OSError):
        list_tickers(tmp_path / "missing")


def test_rank_tickers_order():
    t = ["ETHUSDT", "BTCUSDT", "WBTCUSDT", "BTC"]
    assert rank_tickers(t, "btc") == ["BTC", "BTCUSDT", "WBTCUSDT"]
    assert rank_tickers(t, "") == t
    assert rank_tickers(t, "zzz") == []


@pytest.mark.parametrize("name", ["BNBBULLUSDT", "ETHBEARUSDT", "BULLUSDT", "USDCUSDT",
                                   "BUSDUSDT", "DAIUSDT", "TUSDUSDT", "PAXUSDT",
                                   "EURUSDT", "GBPUSDT", "AUDUSDT", "USDSUSDT", "USDSBUSDT"])
def test_excluded(name):
    assert is_excluded(name)


@pytest.mark.parametrize("name", ["BTCUSDT", "ETHUSDT", "PAXGUSDT", "BEAMUSDT", "AAVEUSDT"])
def test_not_excluded(name):
    assert not is_excluded(name)


@pytest.mark.skipif(not Path(DEFAULT_FOLDER).exists(), reason="real data folder not present")
def test_real_btc_file():
    df = load_ohlcv(DEFAULT_FOLDER, "BTCUSDT")
    assert len(df) > 500 and (df["close"] > 0).all()
```

- [ ] **Step 2: Run, verify fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_data.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'data'`

- [ ] **Step 3: Implement `data.py`**

```python
"""Read Stooq-format daily crypto files: one CSV per ticker."""
from pathlib import Path

import pandas as pd

DEFAULT_FOLDER = r"D:\share\stooq_output_crypto"

_STABLE_OR_FIAT = {"USDC", "BUSD", "DAI", "TUSD", "PAX", "USDS", "USDSB", "EUR", "GBP", "AUD"}


def list_tickers(folder=DEFAULT_FOLDER):
    """Sorted ticker names (file stems) found in the folder. Reads filenames only."""
    return sorted(p.stem for p in Path(folder).iterdir() if p.suffix.lower() == ".csv")


def rank_tickers(tickers, query, limit=200):
    """Tickers matching query: exact match first, then prefix, then substring."""
    q = query.strip().upper()
    if not q:
        return list(tickers)[:limit]
    exact = [t for t in tickers if t.upper() == q]
    prefix = [t for t in tickers if t.upper().startswith(q) and t.upper() != q]
    other = [t for t in tickers if q in t.upper() and not t.upper().startswith(q)]
    return (exact + prefix + other)[:limit]


def is_excluded(ticker):
    """True for leveraged tokens (BULL/BEAR) and stablecoin/fiat pairs."""
    t = ticker.upper()
    base = t[:-4] if t.endswith("USDT") else t
    return "BULL" in t or "BEAR" in t or base in _STABLE_OR_FIAT


def load_ohlcv(folder, ticker):
    """Close and volume for one ticker as a date-indexed, ascending DataFrame."""
    path = Path(folder) / f"{ticker}.CSV"
    if not path.exists():
        path = Path(folder) / f"{ticker}.csv"
    if not path.exists():
        raise ValueError(f"No data file for ticker '{ticker}'.")
    try:
        df = pd.read_csv(path, usecols=["DATE", "CLOSE", "VOL"], dtype={"DATE": str})
    except Exception as exc:
        raise ValueError(f"Could not read {path.name}: {exc}") from exc
    df = df.dropna(subset=["DATE", "CLOSE"])
    if df.empty:
        raise ValueError(f"{path.name} contains no price rows.")
    idx = pd.to_datetime(df["DATE"], format="%Y%m%d")
    out = pd.DataFrame({"close": df["CLOSE"].to_numpy(dtype=float),
                        "vol": df["VOL"].fillna(0).to_numpy(dtype=float)}, index=idx)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out[out["close"] > 0]
    if out.empty:
        raise ValueError(f"{path.name} contains no price rows.")
    return out
```

Note: the crypto files' headers have no angle brackets (`TICKER,PER,DATE,...`), unlike DCA-calc's.

- [ ] **Step 4: Run, verify pass**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: all tests pass (the real-file test runs if the folder exists)

- [ ] **Step 5: Commit**

```bash
git add data.py tests/test_data.py
git commit -m "feat: crypto CSV loader, ticker search and exclusion filter" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Streamlit app, run script, README

**Files:**
- Create: `app.py`, `run.bat`, `.streamlit/config.toml`, `README.md`

**Interfaces:**
- Consumes: `data.{DEFAULT_FOLDER, list_tickers, rank_tickers, is_excluded, load_ohlcv}`, `mvrv.{compute_mvrv, classify, MIN_DAYS}`.

- [ ] **Step 1: Write `.streamlit/config.toml` and `run.bat`**

`.streamlit/config.toml`:
```toml
[server]
port = 8502

[browser]
gatherUsageStats = false
```

`run.bat`:
```bat
@echo off
cd /d "%~dp0"
.venv\Scripts\python.exe -m streamlit run app.py --server.port 8502
```

- [ ] **Step 2: Write `app.py`**

```python
"""Streamlit UI for the crypto MVRV Z-score proxy. Run: run.bat (port 8502)."""
import os
from datetime import date, datetime

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from data import DEFAULT_FOLDER, is_excluded, list_tickers, load_ohlcv, rank_tickers
from mvrv import MIN_DAYS, classify, compute_mvrv

st.set_page_config(page_title="Crypto MVRV Z-Score", layout="wide")
st.title("Crypto MVRV Z-Score (price/volume proxy)")


@st.cache_data
def _tickers(folder):
    return list_tickers(folder)


@st.cache_data
def _mvrv(folder, ticker):
    return compute_mvrv(load_ohlcv(folder, ticker))


def _parse_date(text, label):
    try:
        return datetime.strptime(text.strip(), "%Y-%m-%d")
    except ValueError:
        raise ValueError(f"{label} date '{text}' is not valid. Use YYYY-MM-DD, e.g. 2025-01-31.")


with st.sidebar:
    folder = st.text_input("Data folder", os.environ.get("STOOQ_DIR", DEFAULT_FOLDER))
    try:
        tickers = _tickers(folder)
    except OSError as exc:
        st.error(f"Cannot read data folder: {exc}")
        st.stop()
    if not tickers:
        st.error("No CSV files found in that folder.")
        st.stop()
    hide = st.checkbox("Hide leveraged tokens and stablecoins", True)
    if hide:
        tickers = [t for t in tickers if not is_excluded(t)]
    query = st.text_input("Search ticker", "BTCUSDT")
    matches = rank_tickers(tickers, query)
    if not matches:
        st.error(f"No ticker matches '{query}'.")
        st.stop()
    ticker = st.selectbox(f"Coin ({len(matches)} shown)", matches)
    start_text = st.text_input("Start date (YYYY-MM-DD)", "2000-01-01")
    end_text = st.text_input("End date (YYYY-MM-DD)", date.today().isoformat())
    overheated = st.number_input("Overheated at Z >=", value=7.0, step=0.5)
    undervalued = st.number_input("Undervalued at Z <=", value=0.0, step=0.5)

try:
    start, end = (_parse_date(t, label) for t, label in ((start_text, "Start"), (end_text, "End")))
    if start > end:
        raise ValueError("Start date is after end date.")
    full = _mvrv(folder, ticker)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

df = full.loc[start:end]
if df.empty:
    st.error(f"No data for {ticker} between {start:%Y-%m-%d} and {end:%Y-%m-%d}. "
             f"Available: {full.index[0]:%Y-%m-%d} to {full.index[-1]:%Y-%m-%d}.")
    st.stop()
if full["z"].isna().all():
    st.warning(f"{ticker} has only {len(full)} days of data; the Z-score needs more than {MIN_DAYS}. "
               "Showing price only.")

last = df.iloc[-1]
cols = st.columns(5)
cols[0].metric("Price", f"${last['close']:,.4g}")
cols[1].metric("Realized price (proxy)", f"${last['realized_price']:,.4g}")
cols[2].metric("Z-score", "n/a" if last["z"] != last["z"] else f"{last['z']:.2f}")
cols[3].metric("MVRV ratio", f"{last['mvrv_ratio']:.2f}")
cols[4].metric("Status", classify(last["z"], overheated, undervalued))

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.55, 0.45], vertical_spacing=0.05,
                    subplot_titles=("Price vs realized price (proxy)", "MVRV Z-score (proxy)"))
fig.add_scatter(x=df.index, y=df["close"], name="Price", line=dict(color="#1f77b4"), row=1, col=1,
                hovertemplate="%{x|%Y-%m-%d}<br>Price $%{y:,.4g}<extra></extra>")
fig.add_scatter(x=df.index, y=df["realized_price"], name="Realized price (VWAP)",
                line=dict(color="#ff7f0e", dash="dash"), row=1, col=1,
                hovertemplate="%{x|%Y-%m-%d}<br>Realized $%{y:,.4g}<extra></extra>")
fig.add_scatter(x=df.index, y=df["z"], name="Z-score", line=dict(color="#333"), row=2, col=1,
                hovertemplate="%{x|%Y-%m-%d}<br>Z %{y:.2f}<extra></extra>")
zmax = max(float(df["z"].max(skipna=True) or 0), overheated + 1)
zmin = min(float(df["z"].min(skipna=True) or 0), undervalued - 1)
fig.add_hrect(y0=overheated, y1=zmax, fillcolor="red", opacity=0.15, line_width=0, row=2, col=1)
fig.add_hrect(y0=zmin, y1=undervalued, fillcolor="green", opacity=0.15, line_width=0, row=2, col=1)
fig.update_yaxes(type="log", row=1, col=1)
fig.update_layout(height=750, hovermode="x unified", legend=dict(orientation="h"),
                  title=f"{ticker}: {df.index[0]:%Y-%m-%d} to {df.index[-1]:%Y-%m-%d}")
st.plotly_chart(fig, width="stretch")
st.caption("This is a proxy built from price and volume, not on-chain MVRV: realized price is the cumulative "
           "volume-weighted average price and Z divides by the expanding std of price. It is anchored to the start "
           f"of the data ({full.index[0]:%Y-%m-%d}), so values differ from lookintobitcoin.com. "
           "The date range only trims the display; calculation always uses the full history.")

with st.expander("Data"):
    st.dataframe(df[["close", "vol", "realized_price", "mvrv_ratio", "z"]].sort_index(ascending=False))
```

- [ ] **Step 3: Write `README.md`** with: what it is, the proxy caveat, requirements, setup (`python -m venv .venv`, `pip install -r requirements.txt`), run (`run.bat`, opens http://localhost:8502), sidebar field table, how it calculates (formulas from Global Constraints), tests command, project layout, and a "not investment advice" line.

- [ ] **Step 4: Verify the app runs headless and serves on 8502**

```bash
cd /d/share/python-algos/Crypto-MVRV-Z-Score
.venv/Scripts/python.exe -m streamlit run app.py --server.headless true &
sleep 8
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8502/_stcore/health
```
Expected: `200`. Then drive the script with Streamlit's test harness:

```bash
.venv/Scripts/python.exe - <<'EOF'
from streamlit.testing.v1 import AppTest
at = AppTest.from_file("app.py", default_timeout=60).run()
assert not at.exception, at.exception
print([m.label for m in at.metric], [m.value for m in at.metric])
at.text_input[2].set_value("2030-01-01").run()   # start after available data
print("errors:", [e.value for e in at.error])
at.text_input[2].set_value("2000-01-01"); at.text_input[3].set_value("1999-01-01").run()
print("errors:", [e.value for e in at.error])
EOF
```
Expected: five metrics render for BTCUSDT; the second and third runs show readable error messages and no exception. Stop the background server afterwards.

- [ ] **Step 5: Run the full suite and commit**

Run: `.venv/Scripts/python.exe -m pytest -q` (expected: all pass)

```bash
git add app.py run.bat .streamlit README.md
git commit -m "feat: Streamlit MVRV Z-score app on port 8502" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```
