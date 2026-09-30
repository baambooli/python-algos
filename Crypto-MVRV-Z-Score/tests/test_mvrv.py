import numpy as np
import pandas as pd
import pytest

from mvrv import classify, compute_mvrv


def _df(close, vol):
    idx = pd.date_range("2024-01-01", periods=len(close), freq="D")
    return pd.DataFrame({"close": close, "vol": vol}, index=idx, dtype=float)


def test_hand_computed_vwap_and_z():
    out = compute_mvrv(_df([10, 20, 30], [1, 1, 2]), min_days=0)
    assert out["realized_price"].tolist() == pytest.approx([10, 15, 22.5])
    assert np.isnan(out["z"].iloc[0])  # std is 0 on first row
    assert out["z"].iloc[1] == pytest.approx(1.0)
    assert out["z"].iloc[2] == pytest.approx((30 - 22.5) / np.std([10, 20, 30]))
    assert out["mvrv_ratio"].iloc[2] == pytest.approx(30 / 22.5)


def test_first_30_days_masked():
    out = compute_mvrv(_df(np.linspace(10, 100, 60), np.ones(60)))
    assert out["z"].iloc[:30].isna().all()
    assert out["z"].iloc[30:].notna().all()


def test_constant_price_gives_zero_or_nan():
    out = compute_mvrv(_df([5.0] * 40, [1.0] * 40))
    assert not (out["z"].dropna().abs() > 1e-9).any()


def test_short_history_is_all_nan():
    out = compute_mvrv(_df([1, 2, 3, 4], [1, 1, 1, 1]))
    assert out["z"].isna().all()


def test_zero_volume_rows_carry_no_weight():
    out = compute_mvrv(_df([10, 99, 30], [1, 0, 1]), min_days=0)
    assert out["realized_price"].tolist() == pytest.approx([10, 10, 20])


def test_leading_zero_volume_is_nan_not_inf():
    out = compute_mvrv(_df([10, 20, 30], [0, 1, 1]), min_days=0)
    assert np.isnan(out["realized_price"].iloc[0])
    assert np.isfinite(out["realized_price"].iloc[1:]).all()


def test_all_zero_volume_raises():
    with pytest.raises(ValueError, match="volume"):
        compute_mvrv(_df([1, 2, 3], [0, 0, 0]))


def test_classify():
    assert classify(7.5) == "Overheated"
    assert classify(7.0) == "Overheated"
    assert classify(0.0) == "Undervalued"
    assert classify(-1.0) == "Undervalued"
    assert classify(3.0) == "Neutral"
    assert classify(float("nan")) == "n/a"
    assert classify(3.0, overheated=2.0) == "Overheated"
