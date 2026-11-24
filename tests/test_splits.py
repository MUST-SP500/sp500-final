import json
import pandas as pd
import pytest
from src.config import load_config
from src.splits import make_folds, purge_overlap, save_splits


def test_folds_and_label_availability(tmp_path):
    index = pd.date_range('1996-01-05', '2025-12-31', freq='W-FRI')
    cfg = load_config()
    folds = make_folds(index, cfg)
    assert [f['fold'] for f in folds] == list(range(2008, 2026))
    tests = []
    for f in folds:
        train = pd.DatetimeIndex(f['train_dates'])
        test = pd.DatetimeIndex(f['test_dates'])
        assert train[-1] + pd.DateOffset(days=14) <= pd.Timestamp(f['test_start'])
        assert (index[index.get_indexer(train) + 2] < pd.Timestamp(f['test_start'])).all()
        assert set(train).isdisjoint(test)
        assert (test.year == f['fold']).all()
        tests.extend(test)
    assert len(tests) == len(set(tests))
    assert tests == list(index[index.year >= 2008])
    assert purge_overlap(folds) == folds
    path = save_splits(folds, tmp_path / 'nested/splits.json')
    assert json.loads(path.read_text(encoding='utf-8')) == folds


def test_purge_filters_actual_dates_without_mutation():
    fold = dict(fold=2008, train_start='2007-01-01', train_end='2007-12-28',
                test_start='2008-01-01', test_end='2008-12-31', purge_weeks=2,
                train_dates=['2007-12-14', '2007-12-21', '2007-12-28'])
    result = purge_overlap([fold])[0]
    assert result['train_dates'] == ['2007-12-14']
    assert len(fold['train_dates']) == 3


def test_invalid_index_and_missing_year():
    cfg = load_config()
    with pytest.raises(ValueError):
        make_folds(pd.to_datetime(['2008-01-04', '2007-01-05']), cfg)
    with pytest.raises(ValueError):
        make_folds(pd.date_range('1996-01-05', '2008-12-31', freq='W-FRI'), cfg)


def test_validation_window_does_not_define_test_folds():
    cfg = load_config()
    assert cfg['split']['hyperparam_policy'] == 'frozen_from_2006_2007'
    folds = make_folds(pd.date_range('1996-01-05', '2025-12-31', freq='W-FRI'), cfg)
    assert all(f['fold'] >= 2008 for f in folds)
