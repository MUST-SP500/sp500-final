import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from src.b3 import build_b3, fold_frames
from src.config import load_config
from src.features import feature_columns, make_preprocessor, build_features
from src.labels import build_labels


def inputs():
    dates = pd.bdate_range('1996-01-01', '2009-12-31')
    rng = np.random.default_rng(123)
    c = 100 * np.exp(rng.normal(0, .01, len(dates)).cumsum())
    daily = pd.DataFrame(dict(date=dates, open=c, high=c*1.01, low=c*.99, close=c,
                              adj_close=c, volume=rng.uniform(100, 1000, len(c)), vix=20., dgs10=4., dtb3=3.))
    weekly = daily.groupby(daily.date.dt.to_period('W-SUN')).agg(date=('date','last'),open=('open','first'),high=('high','max'),low=('low','min'),close=('close','last'),adj_close=('adj_close','last'),volume=('volume','sum'),vix=('vix','last'),dgs10=('dgs10','last'),dtb3=('dtb3','last'),n_trading_days=('date','size')).reset_index(drop=True)
    cfg = load_config()
    cfg['split']['last_test_year'] = 2009
    return daily, weekly, cfg


def test_b3_handoff_to_ml_and_pca():
    daily, weekly, cfg = inputs()
    _, _, panel, folds, val = build_b3(daily, weekly, cfg)
    assert {'open','close','vix','dtb3','execution_date','exit_date','holding_trading_days','history_ready','label_available'} <= set(panel)
    assert panel.execution_date.iloc[0] == daily.date.iloc[5]
    assert panel.exit_date.iloc[0] == daily.date.iloc[10]
    np.testing.assert_allclose(panel.next_ret.iloc[:-2], np.expm1(panel.label_ret.iloc[:-2]))
    assert panel.label_available.iloc[-2:].eq(False).all()
    assert not panel.history_ready.iloc[0]
    train, valid = fold_frames(panel, val)
    assert train.index.max() < pd.Timestamp('2006-01-01')
    assert valid.index.year.min() == 2006 and valid.index.year.max() == 2007
    all_test = []
    for fold in folds:
        train, test = fold_frames(panel, fold)
        cols = feature_columns()
        assert len(cols) == 22 and 'next_ret' not in cols
        pre = make_preprocessor()
        a = pre.fit_transform(train[cols]); b = pre.transform(test[cols])
        model = LogisticRegression(max_iter=2000).fit(a, train.label_up.astype(int))
        assert len(model.predict_proba(b)) == len(test)
        pca = PCA(n_components=.9).fit(a)
        assert pca.transform(b).shape[0] == len(test)
        all_test.extend(test.index)
    assert len(all_test) == len(set(all_test))
    assert list(fold_frames(panel, folds[-1], require_labels=False)[1].tail(2).label_up.isna()) == [True, True]


def test_daily_or_missing_week_rejected():
    dates = pd.bdate_range('2024-01-01', periods=4)
    with pytest.raises(ValueError, match='ISO'):
        build_labels(pd.Series([100.,101.,102.,103.],index=dates))
    with pytest.raises(ValueError, match='ISO'):
        build_labels(pd.Series([100.,101.,102.],index=pd.to_datetime(['2024-01-05','2024-01-12','2024-01-26'])))


def test_input_integrity():
    daily, weekly, cfg = inputs()
    bad = daily.copy(); bad.loc[100,'close'] = np.nan
    with pytest.raises(ValueError, match='缺失'):
        build_features(bad)
    with pytest.raises(ValueError):
        build_features(daily.iloc[:0])
    bad = weekly.copy(); bad.loc[0,'date'] += pd.DateOffset(days=1)
    with pytest.raises(ValueError):
        build_b3(daily,bad,cfg)
