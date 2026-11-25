"""B4 单元测试：覆盖 decision / evaluate / models_ml。"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.config import load_config, project_path
from src.decision import rule_confidence, rule_fixed, rule_kelly, rule_quantile
from src.evaluate import (
    classification_metrics,
    paired_bootstrap_diff,
    regression_metrics,
    strategy_metrics,
)
from src.models_ml import build_model, make_baselines, walk_forward


def test_rule_fixed_and_quantile():
    proba = pd.Series([0.2, 0.5, 0.51, 0.8])
    assert rule_fixed(proba).tolist() == [0.0, 0.0, 1.0, 1.0]

    train_proba = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    q_pos = rule_quantile(proba, train_proba, q=0.60)
    assert q_pos.tolist() == [0.0, 1.0, 1.0, 1.0]


def test_rule_kelly_clipping_and_r3_disabled():
    mu = pd.Series([-0.01, 0.005, 0.10])
    sigma = pd.Series([0.02, 0.05, 0.01])
    pos = rule_kelly(mu, sigma, kappa=0.1)
    assert pos.iloc[0] == 0.0
    assert np.isclose(pos.iloc[1], 0.2)
    assert pos.iloc[2] == 1.0

    with pytest.raises(NotImplementedError):
        rule_confidence(pd.Series([0.5, 0.6]))


def test_classification_and_regression_metrics():
    cls = classification_metrics([0, 1, 1, 0], [0, 1, 0, 0], [0.2, 0.8, 0.4, 0.3])
    assert cls["auc"] == 1.0
    assert cls["always_long_auc"] == 0.5
    assert np.isclose(cls["brier"], 0.1325)

    reg = regression_metrics([0.01, -0.02, 0.03], [0.02, -0.01, 0.01])
    assert reg["direction_accuracy"] == 1.0
    assert reg["rmse"] > 0.0


def test_strategy_metrics_and_bootstrap():
    idx = pd.date_range("2020-01-03", periods=40, freq="W-FRI")
    net_ret = pd.Series([0.02, -0.01, 0.015, 0.005] * 10, index=idx)
    equity = (1.0 + net_ret).cumprod()
    trades = pd.DataFrame({"net_return": net_ret, "position": 1.0, "traded": 0.0}, index=idx)
    m = strategy_metrics(equity, trades, freq=52, rf_returns=0.0)
    assert m["cagr"] > 0.0
    assert m["max_dd_weekly"] < 0.0
    assert m["avg_position"] == 1.0

    lo, hi = paired_bootstrap_diff(net_ret, net_ret * 0.5, block=4, n_boot=200, seed=42)
    assert np.isfinite(lo) and np.isfinite(hi) and lo <= hi


def test_build_model_factory():
    assert build_model("logreg", seed=42).random_state == 42
    assert build_model("ridge", seed=42).random_state == 42
    assert build_model("rf", seed=42).random_state == 42
    with pytest.raises(ValueError):
        build_model("hgb", seed=42)


def test_make_baselines_reconcile():
    panel_path = project_path("data/processed/model_panel.parquet")
    if not panel_path.exists():
        pytest.skip("model_panel.parquet 尚未生成")
    panel = pd.read_parquet(panel_path)
    cfg = load_config()
    df = make_baselines(panel, cfg)
    assert set(df["strategy"]) == {"buy_and_hold", "always_long", "const_60", "sma200"}
    bh = df.loc[df["strategy"] == "buy_and_hold", "cagr"].iloc[0]
    al = df.loc[df["strategy"] == "always_long", "cagr"].iloc[0]
    assert np.isclose(bh, al)


def test_walk_forward_two_folds_reproducible():
    panel_path = project_path("data/processed/model_panel.parquet")
    splits_path = project_path("data/processed/splits.json")
    if not panel_path.exists() or not splits_path.exists():
        pytest.skip("B3 交付文件尚未生成")
    panel = pd.read_parquet(panel_path)
    folds = json.loads(splits_path.read_text(encoding="utf-8"))[:2]
    cfg = load_config()

    p1 = walk_forward(panel, folds, "logreg", cfg, params={"C": 0.1})
    p2 = walk_forward(panel, folds, "logreg", cfg, params={"C": 0.1})

    required_cols = {"model", "fold", "decision_date", "execution_date", "y_true", "y_pred", "position"}
    assert required_cols.issubset(p1.columns)
    assert len(p1) == 105  # 2008 (52) + 2009 (53)
    assert p1["y_pred"].notna().all()
    assert p1["position"].isin([0.0, 1.0]).all()
    assert np.allclose(p1["y_pred"].to_numpy(), p2["y_pred"].to_numpy())