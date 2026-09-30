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
