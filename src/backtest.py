"""策略回测（B6）。

已实现：`run_positions` —— 仓位 → 逐周收益 → 净值，**这是对账测试的基准实现**，
必须先通过 `tests/test_backtest_reconcile.py`，再在它之上扩展成本敏感性等分析。

公式（与 docs/preregistration.md 一致，全部在简单收益空间）：

    1 + R_p[w] = (1 + p[w]*R[w] + (1-p[w])*rf[w]) * (1 - c*|p[w] - p[w-1]|)
    E[w]       = E[w-1] * (1 + R_p[w])

其中 w 为周索引；p 为仓位（long/cash 时取值 0 或 1）；R 为当周简单总收益；
rf 为现金周收益（按实际交易日数折算，见 2.3）；c 为单边成本（5 bps = 0.0005）。

负责人：P6（B6）
对应：SP500_分工说明书 P6 第 1–2 步
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BPS = 1e-4


def run_positions(
    positions: pd.Series | list[float],
    asset_returns: pd.Series | list[float],
    cash_returns: pd.Series | list[float] | float = 0.0,
    cost_bps: float = 5.0,
    initial_position: float = 0.0,
    initial_equity: float = 1.0,
) -> pd.DataFrame:
    """把仓位序列和收益序列对账成净值曲线。

    Parameters
    ----------
    positions : 目标仓位，索引为决策周
    asset_returns : 同一周的简单总收益 R
    cash_returns : 同一周的现金收益 rf（标量则视为每周相同）
    cost_bps : 单边成本，单位 bps

    Returns
    -------
    DataFrame，列：position, prev_position, traded, gross_return, cost_return,
    net_return, equity
    """
    pos = pd.Series(positions, dtype=float)
    asset = pd.Series(asset_returns, dtype=float).reindex(pos.index)
    if np.isscalar(cash_returns):
        cash = pd.Series(float(cash_returns), index=pos.index)
    else:
        cash = pd.Series(cash_returns, dtype=float).reindex(pos.index)

    prev = pos.shift(1).fillna(initial_position)
    traded = (pos - prev).abs()

    c = cost_bps * BPS
    gross_factor = 1.0 + pos * asset + (1.0 - pos) * cash
    net_factor = gross_factor * (1.0 - c * traded)

    equity = initial_equity * net_factor.cumprod()

    return pd.DataFrame(
        {
            "position": pos,
            "prev_position": prev,
            "traded": traded,
            "gross_return": gross_factor - 1.0,
            "cost_return": c * traded,
            "net_return": net_factor - 1.0,
            "equity": equity,
        }
    )


def run_backtest(predictions: pd.DataFrame, cfg: dict) -> dict[str, pd.DataFrame]:
    """完整回测：预测 → 决策规则 → 仓位 → 净值 → 指标。

    待实现（P6）：
      1. 按 `decision_rule` 调用 src.decision 生成仓位；
      2. 用 open-to-open 简单收益作为 asset_returns；
      3. 现金收益按 DTB3 与实际交易日数折算；
      4. 输出 trades / equity_curve / metrics 三张表。
    """
    raise NotImplementedError("B6：待实现，先让 tests/test_backtest_reconcile.py 通过")
