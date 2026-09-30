"""Download daily data from Coin Metrics (on-chain coins) and Yahoo Finance (proxy coins).

Run `python fetch_data.py` or click "Update data" in the app. HTTP is injected (`get`) so
tests never touch the network.
"""
import time
from pathlib import Path

import pandas as pd
import requests

from data import COINS, data_dir

_CM_URL = ("https://community-api.coinmetrics.io/v4/timeseries/asset-metrics?assets={id}"
           "&metrics=PriceUSD,CapMrktCurUSD,CapMVRVCur&frequency=1d&paging_from=start"
           "&page_size=10000&start_time=2009-01-01")
_YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}?period1=0&period2={now}&interval=1d"
_HEADERS = {"User-Agent": "Mozilla/5.0"}


def http_get(url):
    resp = requests.get(url, headers=_HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json()


def parse_coinmetrics(pages):
    """Coin Metrics timeseries pages -> DataFrame[price, mcap, mvrv] indexed by date."""
    rows = [r for page in pages for r in page.get("data", [])]
    df = pd.DataFrame({
        "date": pd.to_datetime([r["time"][:10] for r in rows]),
        "price": pd.to_numeric([r.get("PriceUSD") for r in rows], errors="coerce"),
        "mcap": pd.to_numeric([r.get("CapMrktCurUSD") for r in rows], errors="coerce"),
        "mvrv": pd.to_numeric([r.get("CapMVRVCur") for r in rows], errors="coerce"),
    })
    df = df.set_index("date")
    return df[~df.index.duplicated(keep="last")].sort_index()


def parse_yahoo(payload):
    """Yahoo chart JSON -> DataFrame[close, vol] indexed by date (null closes dropped)."""
    chart = payload.get("chart", {})
    result = chart.get("result")
    if not result:
        err = (chart.get("error") or {}).get("description", "no result")
        raise ValueError(f"Yahoo returned no data: {err}")
    r = result[0]
    quote = r["indicators"]["quote"][0]
    df = pd.DataFrame({
        "date": pd.to_datetime(r.get("timestamp", []), unit="s").normalize(),
        "close": pd.Series(quote["close"], dtype=float),
        "vol": pd.Series(quote["volume"], dtype=float),
    }).set_index("date")
    df = df.dropna(subset=["close"]).fillna({"vol": 0})
    return df[~df.index.duplicated(keep="last")].sort_index()


def fetch_coin(coin, get=http_get):
    """Download one coin. Returns the DataFrame that gets cached."""
    if coin.kind == "onchain":
        pages, url = [], _CM_URL.format(id=coin.source_id)
        while url:
            page = get(url)
            pages.append(page)
            url = page.get("next_page_url")
        return parse_coinmetrics(pages)
    return parse_yahoo(get(_YAHOO_URL.format(sym=coin.source_id, now=int(time.time()))))


def fetch_all(folder=None, get=http_get, progress=None):
    """Download every coin into `folder`. One failure never stops the rest.

    Returns {symbol: "ok <n> rows" | "failed: <reason>"}.
    """
    folder = Path(folder or data_dir())
    folder.mkdir(parents=True, exist_ok=True)
    results = {}
    for coin in COINS:
        try:
            df = fetch_coin(coin, get)
            if df.empty:
                raise ValueError("no data returned")
            df.to_csv(folder / f"{coin.symbol}.csv", index_label="date")
            results[coin.symbol] = f"ok {len(df)} rows"
        except Exception as exc:
            results[coin.symbol] = f"failed: {exc}"
        if progress:
            progress(coin.symbol, results[coin.symbol])
    return results


if __name__ == "__main__":
    fetch_all(progress=lambda sym, msg: print(f"{sym:5} {msg}", flush=True))
