"""回测对账测试（B6）：净值必须能由持仓和收益逐行复算。

这是整个项目"数字可信"的底线。先让这组测试通过，再谈策略好坏。

公式（简单收益空间）：
    1 + R_p[w] = (1 + p[w]*R[w] + (1-p[w])*rf[w]) * (1 - c*|p[w]-p[w-1]|)
    E[w]       = E[w-1] * (1 + R_p[w])
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest import run_positions

INDEX = pd.Index([0, 1, 2], name="week")


def test_single_trade_hand_computed():
    """p:0→1，R=1%，rf=0.08%，c=5bps → 净值 = 1.01 * 0.9995 = 1.009495。"""
    out = run_positions(
        positions=[1.0],
        asset_returns=[0.01],
        cash_returns=[0.0008],
        cost_bps=5.0,
        initial_position=0.0,
    )
    assert out["equity"].iloc[0] == pytest.approx(1.009495, rel=1e-9)
    assert out["traded"].iloc[0] == pytest.approx(1.0)


def test_three_weeks_equity_curve():
    positions = pd.Series([1.0, 1.0, 0.0], index=INDEX)
    asset = pd.Series([0.01, -0.02, 0.015], index=INDEX)
    out = run_positions(positions, asset, cash_returns=0.0008, cost_bps=5.0)

    assert out["equity"].iloc[0] == pytest.approx(1.009495, rel=1e-9)
    assert out["equity"].iloc[1] == pytest.approx(1.009495 * 0.98, rel=1e-9)
    assert out["equity"].iloc[2] == pytest.approx(0.9896014958, rel=1e-6)


def test_cost_only_charged_when_position_changes():
    positions = pd.Series([1.0, 1.0, 1.0], index=INDEX)
    asset = pd.Series([0.01, 0.02, -0.01], index=INDEX)
    out = run_positions(positions, asset, cash_returns=0.0, cost_bps=5.0)

    assert out["cost_return"].iloc[0] > 0          # 首次建仓要收费
    assert out["cost_return"].iloc[1] == 0.0
    assert out["cost_return"].iloc[2] == 0.0


def test_cash_return_only_applies_when_flat():
    asset = pd.Series([0.05, 0.05], index=INDEX[:2])
    long_all = run_positions([1.0, 1.0], asset, cash_returns=0.10, cost_bps=0.0)
    flat_all = run_positions([0.0, 0.0], asset, cash_returns=0.10, cost_bps=0.0, initial_position=0.0)

    assert long_all["equity"].iloc[0] == pytest.approx(1.05)
    assert flat_all["equity"].iloc[0] == pytest.approx(1.10)


def test_vectorized_matches_explicit_loop():
    """向量化实现与最直白的 for 循环必须逐点一致（防止 shift 错位）。"""
    rng = np.random.default_rng(3)
    n = 120
    idx = pd.RangeIndex(n)
    pos = pd.Series(rng.integers(0, 2, n).astype(float), index=idx)
    ret = pd.Series(rng.normal(0.001, 0.02, n), index=idx)
    rf = 0.0008
    c = 5e-4

    out = run_positions(pos, ret, cash_returns=rf, cost_bps=5.0)

    equity, prev = 1.0, 0.0
    for p, r in zip(pos.to_numpy(), ret.to_numpy()):
        factor = (1 + p * r + (1 - p) * rf) * (1 - c * abs(p - prev))
        equity *= factor
        prev = p

    assert out["equity"].iloc[-1] == pytest.approx(equity, rel=1e-9)


def test_zero_cost_equals_gross():
    pos = pd.Series([1.0, 0.0, 1.0], index=INDEX)
    ret = pd.Series([0.01, 0.02, -0.01], index=INDEX)
    out = run_positions(pos, ret, cash_returns=0.0, cost_bps=0.0)

    np.testing.assert_allclose(out["net_return"], out["gross_return"])


@pytest.mark.skip(reason="待 B6 实现 src.backtest.run_backtest 后启用：端到端对账")
def test_engine_reconciles_full_backtest():
    from src.backtest import run_backtest

    result = run_backtest(pd.DataFrame(), {})
    assert {"trades", "equity_curve", "metrics"}.issubset(result)
