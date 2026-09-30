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
