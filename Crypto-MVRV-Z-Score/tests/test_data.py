import pytest

from data import COINS, cache_status, last_updated, load_coin

BTC = next(c for c in COINS if c.symbol == "BTC")
SOL = next(c for c in COINS if c.symbol == "SOL")


def test_coin_list():
    symbols = [c.symbol for c in COINS]
    assert len(COINS) == 20 and len(set(symbols)) == 20
    assert sum(c.kind == "onchain" for c in COINS) == 12
    assert sum(c.kind == "proxy" for c in COINS) == 8
    assert {"BTC", "ETH", "BNB", "SOL"} <= set(symbols)
    assert all(c.kind in ("onchain", "proxy") for c in COINS)


def test_load_onchain_cleans_data(tmp_path):
    (tmp_path / "BTC.csv").write_text(
        "date,price,mcap,mvrv\n"
        "2020-01-03,3,300,3\n"
        "2020-01-01,1,100,1\n"
        "2020-01-02,2,200,2\n"
        "2020-01-02,2.5,250,2.5\n"   # duplicate date: keep last
        "2020-01-04,4,400,0\n"       # mvrv <= 0: dropped
        "2020-01-05,5,500,\n"        # missing mvrv: dropped
        "2020-01-06,6,-1,2\n"        # mcap <= 0: dropped
    )
    df = load_coin(BTC, tmp_path)
    assert list(df.columns) == ["price", "mcap", "mvrv"]
    assert df.index.is_monotonic_increasing and df.index.is_unique
    assert df["mcap"].tolist() == [100, 250, 300]


def test_load_proxy_cleans_data(tmp_path):
    (tmp_path / "SOL.csv").write_text(
        "date,close,vol\n"
        "2020-01-02,2,20\n"
        "2020-01-01,1,\n"            # missing volume -> 0
        "2020-01-03,0,5\n"           # non-positive close: dropped
        "2020-01-04,,5\n"            # missing close: dropped
    )
    df = load_coin(SOL, tmp_path)
    assert list(df.columns) == ["close", "vol"]
    assert df["close"].tolist() == [1, 2]
    assert df["vol"].tolist() == [0, 20]


def test_load_missing_file(tmp_path):
    with pytest.raises(ValueError, match="Update data"):
        load_coin(BTC, tmp_path)


def test_load_missing_folder(tmp_path):
    with pytest.raises(ValueError, match="Update data"):
        load_coin(BTC, tmp_path / "nope")


def test_load_empty_file(tmp_path):
    (tmp_path / "BTC.csv").write_text("date,price,mcap,mvrv\n")
    with pytest.raises(ValueError, match="no usable rows"):
        load_coin(BTC, tmp_path)


def test_cache_status_and_last_updated(tmp_path):
    assert last_updated(tmp_path) is None
    assert last_updated(tmp_path / "nope") is None
    assert not any(cache_status(tmp_path).values())
    (tmp_path / "BTC.csv").write_text("date,price,mcap,mvrv\n2020-01-01,1,1,1\n")
    status = cache_status(tmp_path)
    assert status["BTC"] and not status["ETH"]
    assert last_updated(tmp_path) is not None
