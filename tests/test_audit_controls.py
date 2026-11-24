import numpy as np
import pandas as pd
from src.features import feature_columns
from src.leakage_audit import run_controls


def test_controls_reproducible_and_future_signal_detected():
    rng = np.random.default_rng(123)
    dates = pd.date_range('1997-01-03', periods=500, freq='W-FRI')
    panel = pd.DataFrame(rng.normal(size=(500, 22)), columns=feature_columns())
    returns = rng.normal(0, .02, 500)
    panel['date'] = dates
    panel['next_ret'] = returns
    panel['label_up'] = (returns > 0).astype(int)
    folds = [dict(fold=2005, train_dates=dates[:398].strftime('%Y-%m-%d').tolist(), test_dates=dates[400:].strftime('%Y-%m-%d').tolist())]
    detail, pred, summary = run_controls(panel, folds)
    _, again, _ = run_controls(panel, folds)
    pd.testing.assert_frame_equal(pred, again)
    assert len(detail) == 3
    assert len(pred) == 300
    assert summary.set_index('control').loc['future_return_injection', 'auc'] > .95
    assert (pred.groupby('control').date.nunique() == 100).all()
