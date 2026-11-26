"""B3 控制实验；独立审计分类器，不替代 B4 正式模型或策略回测。"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from src.features import feature_columns, make_preprocessor
from src.config import load_config, project_path


def run_controls(panel, folds, seed=42):
    """固定种子、固定模型；不根据测试 AUC 调参。"""
    panel = panel.set_index('date').copy()
    cols = feature_columns()
    eligible = panel[cols + ['label_up', 'next_ret']].notna().all(axis=1)
    random_y = pd.Series(np.nan, index=panel.index)
    random_y.loc[eligible] = np.random.default_rng(seed).permutation(panel.loc[eligible, 'label_up'].astype(int))
    rows = []
    predictions = []
    for name in ('random_labels', 'future_return_injection', 'stale_features_52w'):
        X = panel[cols].copy()
        y = panel['label_up'] if name != 'random_labels' else random_y
        if name == 'future_return_injection':
            X['FORBIDDEN_next_ret'] = panel['next_ret']
        if name == 'stale_features_52w':
            X = X.shift(52)
        good = eligible & X.notna().all(axis=1) & y.notna()
        for fold in folds:
            train = pd.DatetimeIndex(fold['train_dates'])
            test = pd.DatetimeIndex(fold['test_dates'])
            train = train[good.reindex(train).to_numpy(dtype=bool)]
            test = test[good.reindex(test).to_numpy(dtype=bool)]
            if not len(train) or not len(test):
                raise ValueError('控制实验折无有效样本')
            pre = make_preprocessor()
            a, b = pre.fit_transform(X.loc[train]), pre.transform(X.loc[test])
            model = LogisticRegression(C=1.0, max_iter=2000, random_state=seed)
            model.fit(a, y.loc[train].astype(int))
            prob = model.predict_proba(b)[:, 1]
            truth = y.loc[test].astype(int)
            auc = roc_auc_score(truth, prob) if truth.nunique() == 2 else np.nan
            rows.append(dict(control=name, fold=fold['fold'], train_rows=len(train), test_rows=len(test), auc=auc))
            predictions.append(pd.DataFrame(dict(control=name, fold=fold['fold'], date=test, y_true=truth.to_numpy(), probability=prob)))
    detail = pd.DataFrame(rows)
    predictions = pd.concat(predictions, ignore_index=True)
    summary = []
    for name, group in predictions.groupby('control', sort=False):
        auc = roc_auc_score(group.y_true, group.probability)
        passed = (0.45 <= auc <= 0.55) if name == 'random_labels' else (auc >= 0.95 if name == 'future_return_injection' else None)
        summary.append(dict(control=name, rows=len(group), auc=auc, passed=passed))
    return detail, predictions, pd.DataFrame(summary)


def main():
    cfg = load_config()
    panel = pd.read_parquet(project_path('data/processed/model_panel.parquet'))
    folds = json.loads(project_path(cfg['data']['splits']).read_text(encoding='utf-8'))
    detail, predictions, summary = run_controls(panel, folds, cfg['project']['seed'])
    for name, frame in [('control_folds', detail), ('control_predictions', predictions), ('control_summary', summary)]:
        frame.to_csv(project_path(f'data/processed/{name}.csv'), index=False)
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
