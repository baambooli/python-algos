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


def compute_onchain(df, min_days=MIN_DAYS):
    """True MVRV Z from market cap and MVRV ratio (columns `mcap`, `mvrv`).

    realized_cap = mcap / mvrv; z = (mcap - realized_cap) / expanding population std of mcap.
    """
    mcap = df["mcap"]
    realized = mcap / df["mvrv"]
    std = mcap.expanding().std(ddof=0)
    z = (mcap - realized) / std.where(std > 0)
    z.iloc[:min_days] = np.nan
    out = df.copy()
    out["realized_cap"] = realized
    out["z"] = z
    return out


def compute(df, kind, min_days=MIN_DAYS):
    """Unified frame with columns price, value, realized, ratio, z for either kind.

    kind "onchain": df has mcap, mvrv (and optionally price). value/realized are caps.
    kind "proxy": df has close, vol. value/realized are price and VWAP.
    """
    if kind == "onchain":
        r = compute_onchain(df, min_days)
        return pd.DataFrame({"price": r["price"] if "price" in r else r["mcap"], "value": r["mcap"],
                             "realized": r["realized_cap"], "ratio": r["mvrv"], "z": r["z"]}, index=r.index)
    if kind == "proxy":
        r = compute_mvrv(df, min_days)
        return pd.DataFrame({"price": r["close"], "value": r["close"], "realized": r["realized_price"],
                             "ratio": r["mvrv_ratio"], "z": r["z"]}, index=r.index)
    raise ValueError(f"Unknown kind '{kind}'.")
