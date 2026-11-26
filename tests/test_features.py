import numpy as np
import pandas as pd
import pytest
from src.features import build_features, feature_columns


def sample(ohlc):
    df = ohlc.copy()
    df['vix'] = 20 + np.arange(len(df)) / 100
    df['dgs10'] = 4.0
    df['dtb3'] = 3.0
    return df


def test_feature_formulas(ohlc):
    df = sample(ohlc)
    out = build_features(df)
    assert list(out) == feature_columns()
    assert out.shape == (len(df), 22)
    assert out.ret_12m.iloc[:252].isna().all()
    assert out.ret_12m.iloc[252] == pytest.approx(df.close.iloc[252] / df.close.iloc[0] - 1)
    assert out.px_ma20.iloc[-1] == pytest.approx(df.close.iloc[-1] / df.close.iloc[-20:].mean() - 1)
    assert out.vol_20.iloc[-1] == pytest.approx(df.close.pct_change().iloc[-20:].std() * np.sqrt(252))
    assert out.vix_level.equals(df.vix)
    assert out.term_spread.eq(1).all()
    assert out.rsi_14.dropna().between(0, 1).all()
    assert not np.isinf(out.to_numpy()).any()


def test_truncation_and_future_mutation(ohlc):
    df = sample(ohlc)
    full = build_features(df)
    for end in (100, 220, 255):
        pd.testing.assert_frame_equal(full.iloc[:end], build_features(df.iloc[:end]))
    changed = df.copy()
    changed.loc[220:, ['close', 'high', 'low', 'volume', 'vix']] *= 2
    pd.testing.assert_frame_equal(full.iloc[:220], build_features(changed).iloc[:220])


def test_price_scale_invariance(ohlc):
    df = sample(ohlc)
    scaled = df.copy()
    scaled[['close', 'high', 'low', 'open']] *= 10
    pd.testing.assert_frame_equal(build_features(df), build_features(scaled), atol=1e-12, rtol=1e-10)


def test_bad_dates_rejected(ohlc):
    with pytest.raises(ValueError, match='日期'):
        build_features(sample(ohlc).iloc[::-1])


def test_flat_market(ohlc):
    df = sample(ohlc)
    df[['close','high','low']] = 100.0
    df['volume'] = 1000.0
    out = build_features(df)
    assert out.rsi_14.iloc[-1] == 0.5
    assert pd.isna(out.volume_z.iloc[-1])
    assert out.atr_14.iloc[-1] == 0
