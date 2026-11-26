"""B5 单元测试（PCA 部分）：只依赖 sklearn，CI 默认环境（不装 torch）也能跑。

只用合成数据，不读真实行情、不需要联网。
"""

import numpy as np
import pandas as pd
import pytest

from src.features import feature_columns, make_preprocessor
from src.models_pca import auc_table, fit_pca, pca_logreg, strategy_metrics

FEATURE_COLS = feature_columns()


@pytest.fixture
def panel() -> pd.DataFrame:
    """2003–2010 的合成面板：22 特征 + 标签 + 执行日。"""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2003-01-03", "2010-12-31", freq="W-FRI")
    features = rng.normal(0, 1, size=(len(dates), len(FEATURE_COLS)))
    label_up = (features[:, 0] + rng.normal(0, 1, len(dates)) > 0).astype(int)

    frame = pd.DataFrame(features, index=dates, columns=FEATURE_COLS)
    frame["label_up"] = label_up
    frame["execution_date"] = dates + np.timedelta64(3, "D")
    return frame


@pytest.fixture
def folds(panel: pd.DataFrame) -> list[dict]:
    """两折：2009 与 2010，训练段覆盖前几年。"""
    return [
        {
            "fold": year,
            "train_dates": panel.index[panel.index.year < year].strftime("%Y-%m-%d").tolist(),
            "test_dates": panel.index[panel.index.year == year].strftime("%Y-%m-%d").tolist(),
            "purge_weeks": 2,
        }
        for year in (2009, 2010)
    ]


def test_fit_pca_reaches_variance_target(panel):
    X_train = make_preprocessor().fit_transform(panel[FEATURE_COLS])

    pca, n_components = fit_pca(X_train, variance_target=0.90)

    assert 1 <= n_components < len(FEATURE_COLS)
    assert pca.explained_variance_ratio_.sum() >= 0.90


def test_pca_is_fitted_on_train_only(panel):
    train_rows = panel[panel.index.year < 2006]
    test_rows = panel[panel.index.year >= 2006]
    preprocessor = make_preprocessor()
    X_train = preprocessor.fit_transform(train_rows[FEATURE_COLS])

    pca, _ = fit_pca(X_train)
    full_sample_mean = np.r_[train_rows[FEATURE_COLS], test_rows[FEATURE_COLS]].mean(axis=0)

    # 训练折均值应当被 PCA 中心化掉；若用了全样本，mean_ 会等于全样本均值
    assert np.allclose(pca.mean_, 0.0, atol=1e-6)
    assert not np.allclose(pca.mean_, full_sample_mean, atol=1e-6)


def test_pca_logreg_returns_every_test_row(panel, folds):
    predictions = pca_logreg(panel, folds, {"project": {"seed": 42}})

    n_test_rows = sum(len(f["test_dates"]) for f in folds)
    assert len(predictions) == n_test_rows * 2  # pca_logreg 与 lr_full 各一套
    assert set(predictions["model"]) == {"pca_logreg", "lr_full"}
    assert predictions["y_pred"].between(0, 1).all()
    assert predictions.groupby("model")["fold"].nunique().eq(2).all()

    summary = auc_table(predictions)
    assert set(summary.columns) == {"model", "n_samples", "n_components", "auc"}
    assert summary["auc"].between(0, 1).all()


def test_pca_predictions_match_b4_contract(panel, folds):
    """P6 要能用同一段回测代码，前 7 列必须与 B4 的 predictions.parquet 同名同序。"""
    predictions = pca_logreg(panel, folds, {"project": {"seed": 42}})

    assert list(predictions.columns)[:7] == [
        "model",
        "fold",
        "decision_date",
        "execution_date",
        "y_true",
        "y_pred",
        "position",
    ]
    assert set(predictions["position"].unique()) <= {0, 1}
    assert predictions["execution_date"].notna().all()


def test_strategy_metrics_matches_manual_math():
    """收益口径要能被手算复现：空仓不参与、收益按对数收益累乘。"""
    position = np.array([1, 0, 1, 1], dtype=float)
    next_ret = np.array([0.02, -0.01, 0.03, -0.02])

    metrics = strategy_metrics(position, next_ret)

    weekly = position * next_ret
    expected_ann = np.exp(weekly.sum() * 52 / 4) - 1
    expected_sharpe = weekly.mean() / weekly.std(ddof=1) * np.sqrt(52)
    assert metrics["n_weeks"] == 4
    assert metrics["ann_return"] == pytest.approx(expected_ann, rel=1e-9)
    assert metrics["sharpe"] == pytest.approx(expected_sharpe, rel=1e-9)


def test_strategy_metrics_handles_nan_and_flat_position():
    """末尾缺 next_ret 的行要丢掉；全程空仓时夏普无定义，返回 NaN 而不是抛错。"""
    metrics = strategy_metrics([1, 1, 1], [0.01, np.nan, 0.02])
    assert metrics["n_weeks"] == 2

    flat = strategy_metrics([0, 0, 0], [0.01, 0.02, -0.01])
    assert flat["ann_return"] == pytest.approx(0.0)
    assert np.isnan(flat["sharpe"])
