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


def _toy_panel(opens: list[float]) -> pd.DataFrame:
    """造一个只有 4 列（回测需要的最小集合）的周面板。"""
    return pd.DataFrame(
        {
            "date": pd.date_range("2020-01-03", periods=len(opens), freq="W-FRI"),
            "open": opens,
            "dtb3": 0.0,
            "n_trading_days": 5,
        }
    )


def test_engine_reconciles_full_backtest(tmp_path):
    """端到端：预测表 → 三张表，且净值与手算一致。

    open 序列 [100, 100, 110, 121, 121, 108.9] 意味着
    R = [+10%, +10%, 0%, -10%]（open-to-open，执行在下一周开盘）。
    """
    from src.backtest import run_backtest

    panel = _toy_panel([100.0, 100.0, 110.0, 121.0, 121.0, 108.9])
    panel_path = tmp_path / "weekly_panel.parquet"
    panel.to_parquet(panel_path, index=False)

    predictions = pd.DataFrame(
        {
            "decision_date": panel["date"].iloc[:4],
            "position": [1.0, 1.0, 0.0, 1.0],
            "model": "logreg",
        }
    )
    cfg = {
        "data": {"weekly_panel": str(panel_path), "processed_dir": str(tmp_path)},
        "backtest": {"cost_bps": 5.0, "initial_position": 0.0, "initial_equity": 1.0},
    }

    result = run_backtest(predictions, cfg, out_dir=tmp_path)

    assert {"trades", "equity_curve", "metrics"}.issubset(result)
    eq = result["equity_curve"]
    assert len(eq) == 4
    # 第一周：建仓扣 5bps，然后吃 +10%
    assert eq["equity"].iloc[0] == pytest.approx(1.10 * 0.9995, rel=1e-12)
    # 第二周：持有不动，不收费
    assert eq["equity"].iloc[1] == pytest.approx(1.10 * 0.9995 * 1.10, rel=1e-12)
    # 第三周：平仓，资产收益 0%，只扣成本
    assert eq["equity"].iloc[2] == pytest.approx(1.10 * 0.9995 * 1.10 * 0.9995, rel=1e-12)
    assert eq["drawdown"].max() <= 0

    assert len(result["trades"]) == 3
    assert len(result["metrics"]) == 1
    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "equity_curve.csv").exists()
    assert (tmp_path / "trades.csv").exists()


def test_bond_equivalent_yield_conversion():
    """DTB3 贴现率 → 债券等价收益率：d=5% 应该略大于 5%。"""
    from src.backtest import bond_equivalent_yield

    bey = bond_equivalent_yield(pd.Series([5.0])).iloc[0]
    assert bey == pytest.approx(0.05 / (1 - 0.05 * 91 / 360), rel=1e-12)
    assert bey > 0.05


def test_cash_return_scales_with_trading_days():
    """假期周（4 天）的利息应该比完整周（5 天）少 20%。"""
    from src.backtest import weekly_cash_returns

    cash = weekly_cash_returns(pd.Series([5.0, 5.0]), pd.Series([5, 4]))
    assert cash.iloc[1] == pytest.approx(cash.iloc[0] * 4 / 5, rel=1e-12)


def test_short_position_rejected():
    """本版不做空：position < 0 必须直接报错，而不是悄悄算错。"""
    from src.backtest import run_backtest

    predictions = pd.DataFrame(
        {"decision_date": pd.to_datetime(["2020-01-03"]), "position": [-0.5]}
    )
    with pytest.raises(ValueError, match="不允许做空"):
        run_backtest(predictions, {"data": {}, "backtest": {}}, write=False)
