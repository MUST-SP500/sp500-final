import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from src.features import make_preprocessor


def test_train_parameters_and_test_mutation():
    train = pd.DataFrame({'a': [1., 2., np.nan, 4., 100.], 'b': [5.] * 5})
    pre = make_preprocessor(0.1)
    transformed = pre.fit_transform(train)
    filled = train.fillna(train.median()).to_numpy()
    lower, upper = np.quantile(filled, [0.1, 0.9], axis=0)
    clipped = np.clip(filled, lower, upper)
    np.testing.assert_allclose(pre.named_steps['imputer'].statistics_, train.median())
    np.testing.assert_allclose(pre.named_steps['winsorizer'].lower_, lower)
    np.testing.assert_allclose(pre.named_steps['scaler'].mean_, clipped.mean(axis=0))
    np.testing.assert_allclose(transformed.mean(axis=0), 0, atol=1e-12)
    states = [pre.named_steps['imputer'].statistics_.copy(), pre.named_steps['winsorizer'].lower_.copy(), pre.named_steps['winsorizer'].upper_.copy(), pre.named_steps['scaler'].mean_.copy(), pre.named_steps['scaler'].var_.copy()]
    for value in (1e6, -1e6):
        out = pre.transform(pd.DataFrame({'a': [value, np.nan], 'b': [5., 5.]}))
        assert np.isfinite(out).all()
    after = [pre.named_steps['imputer'].statistics_, pre.named_steps['winsorizer'].lower_, pre.named_steps['winsorizer'].upper_, pre.named_steps['scaler'].mean_, pre.named_steps['scaler'].var_]
    for a, b in zip(states, after):
        np.testing.assert_array_equal(a, b)
    assert train.a.isna().sum() == 1
    assert not hasattr(clone(pre).named_steps['scaler'], 'mean_')


def test_invalid_inputs_and_column_order():
    pre = make_preprocessor()
    with pytest.raises(NotFittedError):
        pre.transform([[1., 2.]])
    for values in ([[np.nan], [np.nan]], [[1.], [np.inf]]):
        with pytest.raises(ValueError):
            pre.fit(values)
    pre.fit(pd.DataFrame({'a': [1., 2.], 'b': [3., 4.]}))
    with pytest.raises(ValueError, match='列名'):
        pre.transform(pd.DataFrame({'b': [3.], 'a': [1.]}))
    for q in (-0.01, 0.5, float('nan')):
        with pytest.raises(ValueError):
            make_preprocessor(q)


def test_minmax_for_cnn_and_autoencoder():
    pre = make_preprocessor(scaling='minmax')
    train = np.arange(60, dtype=float).reshape(20,3)
    out = pre.fit_transform(train)
    assert out.min() >= 0 and out.max() <= 1
    assert np.isfinite(pre.transform([[1000., -1000., 1.]])).all()
    with pytest.raises(ValueError):
        make_preprocessor(scaling='unknown')
