"""Streamlit UI: MVRV Z-score for 20 major coins. Run: run.bat (port 8502)."""
import math
from datetime import date, datetime

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from data import COINS, cache_status, data_dir, last_updated, load_coin
from fetch_data import fetch_all
from mvrv import MIN_DAYS, classify, compute

BLUE, WHITE, ORANGE = "#19b5fe", "#ffffff", "#ff9f1c"

st.set_page_config(page_title="Crypto MVRV Z-Score", layout="wide")
st.title("Crypto MVRV Z-Score")


@st.cache_data
def _frame(symbol, folder, stamp):
    coin = next(c for c in COINS if c.symbol == symbol)
    return compute(load_coin(coin, folder), coin.kind)


def _parse_date(text, label):
    try:
        return datetime.strptime(text.strip(), "%Y-%m-%d")
    except ValueError:
        raise ValueError(f"{label} date '{text}' is not valid. Use YYYY-MM-DD, e.g. 2025-01-31.")


def _usd(x):
    return f"${x:,.2f}" if x >= 1 else f"${x:.6f}".rstrip("0")


def _big(x):
    for div, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if x >= div:
            return f"${x / div:,.2f}{suffix}"
    return f"${x:,.0f}"


def _cap_ticks(lo, hi):
    """Log-axis ticks labelled $1M, $10B, $1T ... between lo and hi."""
    vals, text = [], []
    for p in range(max(int(math.floor(math.log10(lo))), 0), int(math.ceil(math.log10(hi))) + 1):
        vals.append(10 ** p)
        text.append(_big(10 ** p).replace(".00", ""))
    return vals, text


def _label(coin):
    return f"{coin.symbol} - {coin.name} ({'on-chain' if coin.kind == 'onchain' else 'proxy'})"


with st.sidebar:
    coin = st.selectbox("Coin", COINS, format_func=_label)
    start_text = st.text_input("Start date (YYYY-MM-DD)", "2009-01-01")
    end_text = st.text_input("End date (YYYY-MM-DD)", date.today().isoformat())
    overheated = st.number_input("Overheated at Z >=", value=7.0, step=0.5)
    undervalued = st.number_input("Undervalued at Z <=", value=0.0, step=0.5)
    st.divider()
    if st.button("Update data"):
        bar, done = st.progress(0.0), [0]

        def _progress(sym, msg):
            done[0] += 1
            bar.progress(done[0] / len(COINS), text=f"{sym}: {msg}")

        st.session_state["update_results"] = fetch_all(progress=_progress)
        st.cache_data.clear()
        st.rerun()
    stamp = last_updated()
    st.caption(f"Data last updated: {stamp:%Y-%m-%d %H:%M}" if stamp else "No data downloaded yet.")

if "update_results" in st.session_state:
    with st.expander("Last update results"):
        for sym, msg in st.session_state["update_results"].items():
            st.write(f"**{sym}**: {msg}")

if not any(cache_status().values()):
    st.info("No data downloaded yet. Click **Update data** in the sidebar to download about 20 coins "
            "from Coin Metrics and Yahoo Finance (takes a minute).")
    st.stop()

try:
    start, end = (_parse_date(t, label) for t, label in ((start_text, "Start"), (end_text, "End")))
    if start > end:
        raise ValueError("Start date is after end date.")
    full = _frame(coin.symbol, str(data_dir()), stamp)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

df = full.loc[start:end]
if df.empty:
    st.error(f"No data for {coin.symbol} between {start:%Y-%m-%d} and {end:%Y-%m-%d}. "
             f"Available: {full.index[0]:%Y-%m-%d} to {full.index[-1]:%Y-%m-%d}.")
    st.stop()
if full["z"].isna().all():
    st.warning(f"{coin.symbol} has only {len(full)} days of data; the Z-score needs more than {MIN_DAYS}.")

onchain = coin.kind == "onchain"
last = df.iloc[-1]
cols = st.columns(5)
cols[0].metric("Price", _usd(last["price"]))
cols[1].metric("Realized cap" if onchain else "Realized price (proxy)",
               _big(last["realized"]) if onchain else _usd(last["realized"]))
cols[2].metric("Z-score", "n/a" if np.isnan(last["z"]) else f"{last['z']:.2f}")
cols[3].metric("MVRV ratio", f"{last['ratio']:.2f}")
cols[4].metric("Status", classify(last["z"], overheated, undervalued))

z = df["z"]
zhi = max(float(np.nanmax(z)) if z.notna().any() else 0.0, overheated + 1)
zlo = min(float(np.nanmin(z)) if z.notna().any() else 0.0, undervalued - 0.5)

fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(x=df.index, y=df["realized"], name="Realized Cap" if onchain else "Realized Price (VWAP)",
                         line=dict(color=BLUE, width=1.5),
                         hovertemplate="%{x|%Y-%m-%d}<br>Realized $%{y:,.2f}<extra></extra>"), secondary_y=False)
fig.add_trace(go.Scatter(x=df.index, y=df["value"], name="Market Cap" if onchain else "Price",
                         line=dict(color=WHITE, width=1.5),
                         hovertemplate="%{x|%Y-%m-%d}<br>%{y:$,.2f}<extra></extra>"), secondary_y=False)
fig.add_trace(go.Scatter(x=df.index, y=z, name="Z-Score", line=dict(color=ORANGE, width=1.5),
                         hovertemplate="%{x|%Y-%m-%d}<br>Z %{y:.2f}<extra></extra>"), secondary_y=True)
fig.add_hrect(y0=overheated, y1=zhi, fillcolor="red", opacity=0.15, line_color="red", line_width=1,
              line_dash="dot", secondary_y=True)
fig.add_hrect(y0=zlo, y1=undervalued, fillcolor="green", opacity=0.15, line_color="green", line_width=1,
              line_dash="dot", secondary_y=True)
left = dict(type="log", showgrid=True, gridcolor="rgba(255,255,255,0.08)")
if onchain:
    lo, hi = float(df[["value", "realized"]].min().min()), float(df[["value", "realized"]].max().max())
    tickvals, ticktext = _cap_ticks(lo, hi)
    fig.update_yaxes(tickvals=tickvals, ticktext=ticktext, **left, secondary_y=False)
else:
    fig.update_yaxes(tickprefix="$", exponentformat="none", **left, secondary_y=False)
fig.update_yaxes(range=[zlo, zhi], title_text="MVRV Z-Score", showgrid=False, secondary_y=True)
fig.update_layout(template="plotly_dark", height=720, hovermode="x unified",
                  title=f"{coin.name}: MVRV Z-Score" + ("" if onchain else " (proxy)"),
                  legend=dict(orientation="h", y=-0.08), margin=dict(t=60))
st.plotly_chart(fig, theme=None, width="stretch")

if onchain:
    st.caption("On-chain data from Coin Metrics Community: market cap and MVRV ratio; realized cap = market cap / MVRV; "
               "Z = (market cap - realized cap) / std(market cap over all history). The date range only trims the "
               "display; the calculation always uses the full history.")
else:
    st.caption("Proxy built from Yahoo Finance price and volume, not on-chain MVRV: realized price is the cumulative "
               "volume-weighted average price and Z divides by the expanding std of price. It is anchored to the start "
               f"of the data ({full.index[0]:%Y-%m-%d}). The date range only trims the display.")

with st.expander("Data"):
    st.dataframe(df[["price", "value", "realized", "ratio", "z"]].sort_index(ascending=False))
