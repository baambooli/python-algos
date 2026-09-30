import numpy as np
import pandas as pd
import pytest

from data import COINS, load_coin
from fetch_data import fetch_all, fetch_coin, parse_coinmetrics, parse_yahoo

BTC = next(c for c in COINS if c.symbol == "BTC")
SOL = next(c for c in COINS if c.symbol == "SOL")


def _row(day, price, mcap, mvrv):
    r = {"asset": "btc", "time": f"{day}T00:00:00.000000000Z"}
    if price is not None:
        r["PriceUSD"] = price
    if mcap is not None:
        r["CapMrktCurUSD"] = mcap
    if mvrv is not None:
        r["CapMVRVCur"] = mvrv
    return r


PAGE1 = {"data": [_row("2010-07-18", "0.08", "295959.15", "146.03"),
                  _row("2010-07-19", "0.09", "330000.5", None)],
         "next_page_url": "https://cm/page2"}
PAGE2 = {"data": [_row("2010-07-20", "0.1", "360000", "120.5")]}


def test_parse_coinmetrics_types_and_dates():
    df = parse_coinmetrics([PAGE1, PAGE2])
    assert list(df.columns) == ["price", "mcap", "mvrv"]
    assert df.index[0] == pd.Timestamp("2010-07-18") and df.index.tz is None
    assert df["mcap"].iloc[0] == pytest.approx(295959.15)
    assert np.isnan(df["mvrv"].iloc[1])  # missing value kept as NaN
    assert len(df) == 3


def test_fetch_coin_follows_paging():
    calls = []

    def get(url):
        calls.append(url)
        return PAGE2 if url == "https://cm/page2" else PAGE1

    df = fetch_coin(BTC, get)
    assert len(df) == 3
    assert calls[0].startswith("https://community-api.coinmetrics.io/v4/timeseries/asset-metrics")
    assert "assets=btc" in calls[0] and "CapMVRVCur" in calls[0]
    assert calls[1] == "https://cm/page2"


def _yahoo(ts, close, vol):
    return {"chart": {"result": [{"timestamp": ts, "indicators": {"quote": [{"close": close, "volume": vol}]}}],
                      "error": None}}


def test_parse_yahoo_drops_null_close_and_zero_fills_volume():
    d = 86400
    df = parse_yahoo(_yahoo([0, d, 2 * d, 3 * d], [1.0, None, 3.0, 4.0], [10, 20, None, 40]))
    assert list(df.columns) == ["close", "vol"]
    assert df["close"].tolist() == [1.0, 3.0, 4.0]
    assert df["vol"].tolist() == [10, 0, 40]
    assert df.index[0] == pd.Timestamp("1970-01-01")


def test_parse_yahoo_error_payload():
    with pytest.raises(ValueError, match="Yahoo"):
        parse_yahoo({"chart": {"result": None, "error": {"description": "Not Found"}}})


def test_fetch_coin_yahoo_uses_symbol():
    urls = []

    def get(url):
        urls.append(url)
        return _yahoo([0, 86400], [1.0, 2.0], [1, 1])

    df = fetch_coin(SOL, get)
    assert "chart/SOL-USD" in urls[0] and len(df) == 2


def test_fetch_all_isolates_failures_and_saves(tmp_path):
    def get(url):
        if "assets=eth" in url:
            raise RuntimeError("boom")
        if "coinmetrics" in url:
            return {"data": [_row("2020-01-01", "1", "100", "2"), _row("2020-01-02", "1", "110", "2")]}
        return _yahoo([0, 86400], [1.0, 2.0], [5, 5])

    seen = []
    res = fetch_all(tmp_path, get=get, progress=lambda sym, msg: seen.append(sym))
    assert len(res) == len(COINS)
    assert res["ETH"].startswith("failed") and "boom" in res["ETH"]
    assert res["BTC"].startswith("ok")
    assert res["SOL"].startswith("ok")
    assert not (tmp_path / "ETH.csv").exists()
    assert len(seen) == len(COINS)
    btc = load_coin(BTC, tmp_path)
    assert btc["mcap"].tolist() == [100, 110]


def test_fetch_all_creates_folder(tmp_path):
    target = tmp_path / "new" / "data"
    fetch_all(target, get=lambda url: (_ for _ in ()).throw(RuntimeError("down")))
    assert target.exists()


def test_fetch_all_empty_series_is_failure(tmp_path):
    res = fetch_all(tmp_path, get=lambda url: {"data": []} if "coinmetrics" in url else _yahoo([], [], []))
    assert all(v.startswith("failed") for v in res.values())
