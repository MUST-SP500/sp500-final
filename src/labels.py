"""标签构造（B3）：下周 open-to-open 对数收益 + 二分类标签。

负责人：P3（B3）
对应：SP500_分工说明书 P3 第 4 步

约定（口径要改，先改 docs/preregistration.md 并在群里说明）：

    周 w 的决策时点  t  = 第 w 周最后一个交易日收盘
    执行时点        t+1 = 第 w+1 周第一个交易日开盘
    平仓时点        t+2 = 第 w+2 周第一个交易日开盘
    标签  r_oo      = ln(O^adj[t+2] / O^adj[t+1])     # 对数，仅用于建模与画图
    分类  y         = 1 if r_oo > 0 else 0

注意：`shift(-1)` 只能出现在**构造标签**的地方，绝不能进入特征。
回测用简单收益 R = O[t+2]/O[t+1] - 1，与这里的对数收益不可混算。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def next_week_return(open_prices: pd.Series) -> pd.Series:
    """按 open-to-open 口径计算下一持有期的简单收益（回测用）。"""
    prices = _validate_prices(open_prices)
    return (prices.shift(-2) / prices.shift(-1) - 1).rename("next_ret")


def build_labels(open_prices: pd.Series) -> pd.DataFrame:
    """输出 label_ret（对数）与 label_up（0/1），索引与周决策时点对齐。"""
    prices = _validate_prices(open_prices)
    label_ret = np.log(prices.shift(-2) / prices.shift(-1)).rename("label_ret")
    label_up = (label_ret > 0).astype("Int64").mask(label_ret.isna()).rename("label_up")
    return pd.concat([label_ret, label_up, next_week_return(prices)], axis=1)


def _validate_prices(open_prices: pd.Series) -> pd.Series:
    if not isinstance(open_prices, pd.Series):
        raise TypeError("输入必须是按周排序的复权开盘价 Series")
    if not open_prices.index.is_unique or not open_prices.index.is_monotonic_increasing:
        raise ValueError("周索引必须唯一且递增")
    if isinstance(open_prices.index, pd.DatetimeIndex):
        dates = open_prices.index
        if dates.hasnans or dates.tz is not None:
            raise ValueError("周日期必须有效且不带时区")
        weeks = dates.to_period("W-SUN").start_time
        if len(weeks) > 1 and not (np.diff(dates.to_period("W-SUN").asi8) == 1).all():
            raise ValueError("标签必须使用连续 ISO 周的开盘价，不能传入日表或删周后的表")
    prices = open_prices.astype(float)
    if np.isinf(prices.to_numpy()).any() or (prices.dropna() <= 0).any():
        raise ValueError("开盘价必须为正且有限；缺失价格保留缺失标签")
    return prices


def main() -> None:
    """统一生成完整交接表，避免旧入口覆盖回测元数据。"""
    from src.b3 import run
    run()


if __name__ == "__main__":
    main()
