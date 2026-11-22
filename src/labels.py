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

import pandas as pd


def next_week_return(open_prices: pd.Series) -> pd.Series:
    """按 open-to-open 口径计算下一持有期的简单收益（回测用）。"""
    raise NotImplementedError("B3：待实现（P3）")


def build_labels(open_prices: pd.Series) -> pd.DataFrame:
    """输出 label_ret（对数）与 label_up（0/1），索引与周决策时点对齐。"""
    raise NotImplementedError("B3：待实现（P3）")
