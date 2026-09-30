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
