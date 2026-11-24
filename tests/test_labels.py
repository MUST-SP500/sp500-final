import numpy as np
import pandas as pd
import pytest
from src.labels import build_labels, next_week_return


def test_hand_calculated_labels():
    dates = pd.to_datetime(['2024-01-05', '2024-01-12', '2024-01-19', '2024-01-26'])
    prices = pd.Series([100., 110., 99., 99.], index=dates)
    out = build_labels(prices)
    pd.testing.assert_index_equal(out.index, prices.index)
    assert out.label_ret.iloc[0] == pytest.approx(np.log(0.9))
    assert out.next_ret.iloc[0] == pytest.approx(-0.1)
    assert out.label_up.iloc[0] == 0
    assert out.label_up.iloc[1] == 0  # 零收益不是上涨
    assert out.next_ret.iloc[1] == 0
    assert out.tail(2).isna().all().all()
    pd.testing.assert_series_equal(next_week_return(prices), out.next_ret)
    np.testing.assert_allclose(np.expm1(out.label_ret.dropna()), out.next_ret.dropna())


def test_positive_and_missing_prices():
    out = build_labels(pd.Series([10., 20., 30., np.nan, 40.]))
    assert out.label_up.iloc[0] == 1
    assert out.iloc[1:].isna().all().all()


@pytest.mark.parametrize('values', [[1., 0., 2.], [1., -1., 2.], [1., np.inf, 2.]])
def test_invalid_prices(values):
    with pytest.raises(ValueError):
        build_labels(pd.Series(values))


def test_duplicate_or_reversed_index():
    for index in ([0, 0, 1], [2, 1, 0]):
        with pytest.raises(ValueError):
            build_labels(pd.Series([1., 2., 3.], index=index))


def test_short_sequences():
    for n in range(3):
        assert build_labels(pd.Series([100.] * n, dtype=float)).isna().all().all()


def test_current_open_does_not_affect_current_label():
    a = pd.Series([100., 110., 120., 130.])
    b = a.copy()
    b.iloc[0] = 999.
    pd.testing.assert_frame_equal(build_labels(a), build_labels(b))
