from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _btc(path, n):
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    mcap = np.linspace(1e11, 2e11, n) * (1 + 0.1 * np.sin(np.arange(n) / 5))
    pd.DataFrame({"price": mcap / 1e7, "mcap": mcap, "mvrv": np.linspace(1.0, 2.0, n)}, index=idx) \
        .to_csv(path / "BTC.csv", index_label="date")


def _sol(path, n):
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    pd.DataFrame({"close": np.linspace(10, 50, n), "vol": np.ones(n)}, index=idx) \
        .to_csv(path / "SOL.csv", index_label="date")


def _run(tmp_path, monkeypatch):
    monkeypatch.setenv("MVRV_DATA_DIR", str(tmp_path))
    return AppTest.from_file(APP, default_timeout=60).run()


def _select(at, symbol):
    box = at.sidebar.selectbox[0]
    box.select(next(o for o in box.options if o.startswith(symbol + " ")))
    return at.run()


def test_empty_cache_shows_message(tmp_path, monkeypatch):
    at = _run(tmp_path, monkeypatch)
    assert not at.exception
    assert any("Update data" in i.value for i in at.info)
    assert len(at.metric) == 0


def test_onchain_coin_renders(tmp_path, monkeypatch):
    _btc(tmp_path, 120)
    at = _run(tmp_path, monkeypatch)
    assert not at.exception and not at.error
    assert [m.label for m in at.metric] == ["Price", "Realized cap", "Z-score", "MVRV ratio", "Status"]
    assert any("on-chain" in c.value.lower() for c in at.caption)


def test_proxy_coin_renders(tmp_path, monkeypatch):
    _btc(tmp_path, 60)
    _sol(tmp_path, 120)
    at = _select(_run(tmp_path, monkeypatch), "SOL")
    assert not at.exception and not at.error
    assert at.metric[1].label == "Realized price (proxy)"
    assert any("proxy" in c.value.lower() for c in at.caption)


def test_uncached_coin_shows_error(tmp_path, monkeypatch):
    _btc(tmp_path, 60)
    at = _select(_run(tmp_path, monkeypatch), "ETH")
    assert not at.exception
    assert any("Update data" in e.value for e in at.error)


def test_reversed_dates_error(tmp_path, monkeypatch):
    _btc(tmp_path, 120)
    at = _run(tmp_path, monkeypatch)
    at.sidebar.text_input[0].set_value("2025-01-01")
    at.sidebar.text_input[1].set_value("2024-01-01")
    at.run()
    assert not at.exception
    assert any("after end" in e.value for e in at.error)


def test_range_outside_data_error(tmp_path, monkeypatch):
    _btc(tmp_path, 120)
    at = _run(tmp_path, monkeypatch)
    at.sidebar.text_input[0].set_value("2010-01-01")
    at.sidebar.text_input[1].set_value("2010-02-01")
    at.run()
    assert not at.exception
    assert any("No data" in e.value for e in at.error)


def test_short_history_warns(tmp_path, monkeypatch):
    _btc(tmp_path, 10)
    at = _run(tmp_path, monkeypatch)
    assert not at.exception
    assert any("only 10 days" in w.value for w in at.warning)


def test_bad_date_error(tmp_path, monkeypatch):
    _btc(tmp_path, 120)
    at = _run(tmp_path, monkeypatch)
    at.sidebar.text_input[1].set_value("garbage")
    at.run()
    assert not at.exception
    assert any("not valid" in e.value for e in at.error)
