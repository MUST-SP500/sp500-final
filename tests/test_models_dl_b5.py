"""B5 单元测试（深度学习部分）：DNN / CNN / Autoencoder / 滚动预测。

torch 属于可选依赖（pyproject 的 dl 组），没装就整体跳过而不是报错：
    uv sync --extra dl
"""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip(
    "torch", reason="未安装 dl 组：先跑 uv sync --extra dl", exc_type=ImportError
)

from src.features import feature_columns, make_preprocessor  # noqa: E402
from src.models_dl import (  # noqa: E402
    _output_path,
    _split_validation,
    autoencoder_bottleneck,
    autoencoder_vs_pca,
    build_cnn,
    build_dnn,
    make_windows,
    run_dl_folds,
    train_and_predict,
)

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


def test_dnn_parameter_budget():
    model = build_dnn(n_features=22)
    n_params = sum(p.numel() for p in model.parameters())

    assert n_params < 5_000
    assert model(torch.zeros(4, 22)).shape == (4, 1)


def test_cnn_parameter_budget_and_input_shape():
    model = build_cnn(window=20, n_features=22)
    n_params = sum(p.numel() for p in model.parameters())

    assert n_params < 5_000
    assert model(torch.zeros(4, 22, 20)).shape == (4, 1)


def test_make_windows_looks_backward_only(panel):
    windows, decision_dates, is_complete = make_windows(panel[FEATURE_COLS], window=20)

    assert windows.shape == (len(panel) - 19, 20, len(FEATURE_COLS))
    # 第一个样本 = 前 20 周，决策日 = 第 20 周，绝不含未来行
    assert decision_dates[0] == panel.index[19]
    assert np.allclose(windows[0], panel[FEATURE_COLS].to_numpy()[:20])
    assert is_complete.all()


def test_make_windows_flags_incomplete_windows(panel):
    features = panel[FEATURE_COLS].copy()
    features.iloc[5, 0] = np.nan

    _, decision_dates, is_complete = make_windows(features, window=20)

    # 第 5 行被污染，凡是包含它的窗口都不可用
    assert not is_complete[:6].any()
    assert is_complete[6:].all()
    assert decision_dates[is_complete][0] == panel.index[25]


def test_train_and_predict_is_reproducible(panel):
    X = panel[FEATURE_COLS].to_numpy()
    y = panel["label_up"].to_numpy()
    X_train, y_train = X[:200], y[:200]
    X_val, y_val = X[200:250], y[200:250]
    X_test = X[250:290]

    first = train_and_predict(build_dnn(22, seed=7), X_train, y_train, X_val, y_val, X_test, epochs=15)
    second = train_and_predict(build_dnn(22, seed=7), X_train, y_train, X_val, y_val, X_test, epochs=15)

    assert first.shape == (len(X_test),)
    assert np.allclose(first, second)
    assert ((first >= 0) & (first <= 1)).all()


def test_early_stopping_records_history(panel):
    X = panel[FEATURE_COLS].to_numpy()
    y = panel["label_up"].to_numpy()
    model = build_dnn(22, seed=3)

    train_and_predict(model, X[:200], y[:200], X[200:250], y[200:250], X[250:290], epochs=20)

    assert len(model.loss_history_["train"]) == len(model.loss_history_["val"])
    assert len(model.loss_history_["train"]) > 0


def test_validation_split_keeps_purge_gap(panel):
    core_rows, val_rows = _split_validation(panel)

    assert val_rows.index.min() > core_rows.index.max()
    assert (val_rows.index.min() - core_rows.index.max()).days >= 14
    assert val_rows.index.max() == panel.index.max()


def test_autoencoder_bottleneck_shapes(panel):
    X = make_preprocessor().fit_transform(panel[FEATURE_COLS])

    encoder, z_train, z_val = autoencoder_bottleneck(X[:200], X[200:250], bottleneck=6, epochs=20)

    assert z_train.shape == (200, 6)
    assert z_val.shape == (50, 6)
    assert not np.isnan(z_train).any()


def test_autoencoder_vs_pca_single_fold(panel, folds):
    table = autoencoder_vs_pca(panel, folds[0], {"project": {"seed": 42}}, bottleneck=5)

    assert len(table) == 3
    assert table["n_features"].tolist() == [5, table.loc[1, "n_features"], len(FEATURE_COLS)]
    assert table["auc"].between(0, 1).all()
    assert table["input"].str.startswith(("autoencoder_", "pca_", "full_")).all()


def test_run_dl_folds_dnn_matches_test_rows(panel, folds):
    predictions = run_dl_folds(panel, folds, {"project": {"seed": 42}}, model_kind="dnn", epochs=5)

    n_test_rows = sum(len(f["test_dates"]) for f in folds)
    assert len(predictions) == n_test_rows
    assert list(predictions.columns) == [
        "model",
        "fold",
        "decision_date",
        "execution_date",
        "y_true",
        "y_pred",
        "position",
    ]
    assert set(predictions["position"].unique()) <= {0, 1}


def test_run_dl_folds_cnn_drops_warmup_windows(panel, folds):
    predictions = run_dl_folds(panel, folds, {"project": {"seed": 42}}, model_kind="cnn", window=20, epochs=5)

    # 每折前 19 周凑不出窗口，样本数一定少于测试行数
    assert 0 < len(predictions) < sum(len(f["test_dates"]) for f in folds)
    assert predictions["decision_date"].min() > pd.Timestamp("2009-01-01")
    assert predictions["y_true"].isin([0, 1]).all()


def test_main_run_writes_canonical_filename():
    """DNN + 标准分是主口径，必须写成说明书要求的 predictions_dl.parquet。"""
    assert _output_path("dnn", "standard").endswith("predictions_dl.parquet")
    assert _output_path("cnn", "minmax").endswith("predictions_dl_cnn_minmax.parquet")
