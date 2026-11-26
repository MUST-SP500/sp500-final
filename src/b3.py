"""B3 一键交接入口与统一样本读取；不修改其他组员的模型代码。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.config import load_config, project_path
from src.features import build_features, feature_columns
from src.labels import build_labels
from src.splits import make_folds, save_splits


def validation_fold(index, cfg):
    """1996–2005 训练，2006–2007 一次性验证，两周隔离。"""
    spec = cfg['split']
    local = dict(cfg, split=dict(spec, first_test_year=pd.Timestamp(spec['val_start']).year,
                                last_test_year=pd.Timestamp(spec['val_start']).year))
    fold = make_folds(index, local)[0]
    dates = index[(index >= pd.Timestamp(spec['val_start'])) & (index <= pd.Timestamp(spec['val_end']))]
    if dates.empty:
        raise ValueError('验证期无数据')
    fold.update(fold='validation', test_start=spec['val_start'], test_end=spec['val_end'],
                test_dates=dates.strftime('%Y-%m-%d').tolist())
    return fold


def build_b3(daily, weekly, cfg):
    """纯函数：保留 B1 字段，计算特征，增加实际执行/退出日期与资格标记。"""
    daily = daily.copy()
    weekly = weekly.copy()
    daily['date'] = pd.to_datetime(daily['date'])
    weekly['date'] = pd.to_datetime(weekly['date'])
    features = build_features(daily, cfg)
    features.insert(0, 'date', daily['date'])
    aligned = weekly[['date']].merge(features, on='date', how='left', validate='one_to_one', indicator=True)
    if not aligned['_merge'].eq('both').all():
        raise ValueError('周决策日期在日表中不存在')
    aligned = aligned.drop(columns='_merge')
    labels = build_labels(weekly.set_index('date')['open']).rename_axis('date').reset_index()
    panel = weekly.merge(aligned, on='date', validate='one_to_one').merge(labels, on='date', validate='one_to_one')
    daily_weeks = daily['date'].dt.to_period('W-SUN')
    weekly_weeks = weekly['date'].dt.to_period('W-SUN')
    first = daily.groupby(daily_weeks)['date'].min()
    last = daily.groupby(daily_weeks)['date'].max()
    if not np.array_equal(weekly['date'].to_numpy(), weekly_weeks.map(last).to_numpy()):
        raise ValueError('周表 date 不是对应 ISO 周的最后交易日')
    opens = weekly_weeks.map(first)
    panel['execution_date'] = opens.shift(-1)
    panel['exit_date'] = opens.shift(-2)
    panel['holding_trading_days'] = weekly['n_trading_days'].shift(-1)
    # 252 个日观测之前是历史不足，不能以训练中位数填造预热特征。
    counts = pd.Series(np.arange(len(daily)), index=daily['date'])
    panel['history_ready'] = panel['date'].map(counts) >= 252
    panel['label_available'] = panel[['label_ret', 'label_up', 'next_ret']].notna().all(axis=1)
    folds = make_folds(pd.DatetimeIndex(weekly['date']), cfg)
    val = validation_fold(pd.DatetimeIndex(weekly['date']), cfg)
    return aligned, labels, panel, folds, val


def fold_frames(panel, fold, require_labels=True):
    """B4/B5 共用：精确日期选样本；先排除预热，偶发缺失留给训练中位数。"""
    indexed = panel.set_index('date') if 'date' in panel.columns else panel
    if not indexed.index.is_unique:
        raise ValueError('建模表日期重复')
    def select(key):
        frame = indexed.loc[pd.to_datetime(fold[key])].copy()
        frame = frame.loc[frame['history_ready'].astype(bool)]
        if require_labels:
            frame = frame.loc[frame['label_available'].astype(bool)]
        return frame
    return select('train_dates'), select('test_dates')


def run(cfg=None):
    cfg = load_config() if cfg is None else cfg
    daily = pd.read_parquet(project_path(cfg['data']['daily_panel']))
    weekly = pd.read_parquet(project_path(cfg['data']['weekly_panel']))
    features, labels, panel, folds, val = build_b3(daily, weekly, cfg)
    out = project_path(cfg['data']['processed_dir'])
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in [('weekly_features', features), ('weekly_labels', labels), ('model_panel', panel)]:
        frame.to_parquet(out / f'{name}.parquet', index=False)
        frame.to_csv(out / f'{name}.csv', index=False)
    save_splits(folds, project_path(cfg['data']['splits']))
    save_splits([val], out / 'validation_split.json')
    stats = []
    for fold in [val] + folds:
        train, test = fold_frames(panel, fold)
        stats.append(dict(fold=fold['fold'], train_rows=len(train), test_rows=len(test), train_end=fold['train_end']))
    pd.DataFrame(stats).to_csv(out / 'b3_sample_counts.csv', index=False)
    rates = features[feature_columns()].isna().mean().rename('missing_fraction').rename_axis('feature')
    rates.to_csv(out / 'feature_missingness.csv')
    print(f'B3 完成：建模表 {panel.shape}，18 测试折 + 独立一次性验证切分')
    print(pd.DataFrame(stats).to_string(index=False))
    return panel, folds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None)
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == '__main__':
    main()
