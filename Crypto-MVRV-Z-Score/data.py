"""Coin list and the local cache of downloaded daily data (one CSV per coin in data/)."""
import os
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import pandas as pd


class Coin(NamedTuple):
    symbol: str
    name: str
    kind: str        # "onchain": Coin Metrics market cap + MVRV; "proxy": Yahoo price + volume
    source_id: str   # Coin Metrics asset id or Yahoo symbol


COINS = [
    Coin("BTC", "Bitcoin", "onchain", "btc"),
    Coin("ETH", "Ethereum", "onchain", "eth"),
    Coin("XRP", "XRP", "onchain", "xrp"),
    Coin("BNB", "BNB", "proxy", "BNB-USD"),
    Coin("ADA", "Cardano", "onchain", "ada"),
    Coin("DOGE", "Dogecoin", "onchain", "doge"),
    Coin("LINK", "Chainlink", "onchain", "link"),
    Coin("XLM", "Stellar", "onchain", "xlm"),
    Coin("BCH", "Bitcoin Cash", "onchain", "bch"),
    Coin("LTC", "Litecoin", "onchain", "ltc"),
    Coin("XMR", "Monero", "proxy", "XMR-USD"),
    Coin("ETC", "Ethereum Classic", "onchain", "etc"),
    Coin("DOT", "Polkadot", "proxy", "DOT-USD"),
    Coin("UNI", "Uniswap", "onchain", "uni"),
    Coin("ICP", "Internet Computer", "onchain", "icp"),
    Coin("SOL", "Solana", "proxy", "SOL-USD"),
    Coin("TRX", "TRON", "proxy", "TRX-USD"),
    Coin("AVAX", "Avalanche", "proxy", "AVAX-USD"),
    Coin("SHIB", "Shiba Inu", "proxy", "SHIB-USD"),
    Coin("HBAR", "Hedera", "proxy", "HBAR-USD"),
]

_COLUMNS = {"onchain": ["price", "mcap", "mvrv"], "proxy": ["close", "vol"]}


def data_dir():
    """Cache folder: MVRV_DATA_DIR if set, else <project>/data."""
    return Path(os.environ.get("MVRV_DATA_DIR") or Path(__file__).parent / "data")


def _path(symbol, folder):
    return Path(folder or data_dir()) / f"{symbol}.csv"


def cache_status(folder=None):
    """{symbol: True if a cached file exists}."""
    return {c.symbol: _path(c.symbol, folder).exists() for c in COINS}


def last_updated(folder=None):
    """Newest cache file modification time, or None if nothing is cached."""
    folder = Path(folder or data_dir())
    if not folder.is_dir():
        return None
    times = [p.stat().st_mtime for p in folder.glob("*.csv")]
    return datetime.fromtimestamp(max(times)) if times else None


def load_coin(coin, folder=None):
    """Cached data for one coin as a date-indexed, ascending, deduplicated DataFrame."""
    path = _path(coin.symbol, folder)
    if not path.exists():
        raise ValueError(f"No cached data for {coin.symbol}. Click Update data to download it.")
    try:
        df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    except Exception as exc:
        raise ValueError(f"Could not read {path.name}: {exc}") from exc
    cols = _COLUMNS[coin.kind]
    df = df[cols]
    if coin.kind == "proxy":
        df = df.assign(vol=df["vol"].fillna(0)).dropna(subset=["close"])
        df = df[df["close"] > 0]
    else:
        df = df.dropna()
        df = df[(df["mcap"] > 0) & (df["mvrv"] > 0)]
    df = df[~df.index.duplicated(keep="last")].sort_index()
    if df.empty:
        raise ValueError(f"{path.name} has no usable rows.")
    return df
