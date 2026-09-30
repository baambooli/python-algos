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


def _usd(x):
    return f"${x:,.2f}" if x >= 1 else f"${x:.6f}".rstrip("0")


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
cols[0].metric("Price", _usd(last["close"]))
cols[1].metric("Realized price (proxy)", _usd(last["realized_price"]))
cols[2].metric("Z-score", "n/a" if last["z"] != last["z"] else f"{last['z']:.2f}")
cols[3].metric("MVRV ratio", f"{last['mvrv_ratio']:.2f}")
cols[4].metric("Status", classify(last["z"], overheated, undervalued))

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.55, 0.45], vertical_spacing=0.05,
                    subplot_titles=("Price vs realized price (proxy)", "MVRV Z-score (proxy)"))
fig.add_scatter(x=df.index, y=df["close"], name="Price", line=dict(color="#1f77b4"), row=1, col=1,
                hovertemplate="%{x|%Y-%m-%d}<br>Price $%{y:,.6~g}<extra></extra>")
fig.add_scatter(x=df.index, y=df["realized_price"], name="Realized price (VWAP)",
                line=dict(color="#ff7f0e", dash="dash"), row=1, col=1,
                hovertemplate="%{x|%Y-%m-%d}<br>Realized $%{y:,.6~g}<extra></extra>")
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
