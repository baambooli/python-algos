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


from mvrv import compute, compute_onchain


def _oc(mcap, mvrv, price=None):
    idx = pd.date_range("2024-01-01", periods=len(mcap), freq="D")
    d = {"mcap": mcap, "mvrv": mvrv}
    if price is not None:
        d["price"] = price
    return pd.DataFrame(d, index=idx, dtype=float)


def test_onchain_hand_computed():
    out = compute_onchain(_oc([100, 200, 300], [1, 2, 3]), min_days=0)
    assert out["realized_cap"].tolist() == pytest.approx([100, 100, 100])
    assert np.isnan(out["z"].iloc[0])
    assert out["z"].iloc[1] == pytest.approx(2.0)
    assert out["z"].iloc[2] == pytest.approx(200 / np.std([100, 200, 300]))


def test_onchain_first_30_days_masked():
    out = compute_onchain(_oc(np.linspace(100, 1000, 60), np.linspace(1, 3, 60)))
    assert out["z"].iloc[:30].isna().all()
    assert out["z"].iloc[30:].notna().all()


def test_onchain_constant_mcap_gives_zero_or_nan():
    out = compute_onchain(_oc([50.0] * 40, [1.0] * 40))
    assert not (out["z"].dropna().abs() > 1e-9).any()


def test_onchain_short_history_all_nan():
    assert compute_onchain(_oc([1, 2, 3], [1, 1, 1]))["z"].isna().all()


UNIFIED = ["price", "value", "realized", "ratio", "z"]


def test_compute_unified_columns_onchain():
    out = compute(_oc([100, 200, 300], [1, 2, 3], price=[1, 2, 3]), "onchain", min_days=0)
    for c in UNIFIED:
        assert c in out.columns
    assert out["value"].tolist() == [100, 200, 300]
    assert out["realized"].tolist() == pytest.approx([100, 100, 100])
    assert out["ratio"].tolist() == [1, 2, 3]
    assert out["price"].tolist() == [1, 2, 3]


def test_compute_onchain_without_price_column_uses_mcap():
    out = compute(_oc([100, 200, 300], [1, 2, 3]), "onchain", min_days=0)
    assert out["price"].tolist() == [100, 200, 300]


def test_compute_unified_columns_proxy():
    out = compute(_df([10, 20, 30], [1, 1, 2]), "proxy", min_days=0)
    for c in UNIFIED:
        assert c in out.columns
    assert out["realized"].tolist() == pytest.approx([10, 15, 22.5])
    assert out["price"].tolist() == [10, 20, 30]


def test_compute_unknown_kind():
    with pytest.raises(ValueError, match="kind"):
        compute(_df([1, 2], [1, 1]), "bogus")
