"""评测（B4）：预测层指标 + 策略层指标 + 配对 bootstrap。

负责人：P4（B4）
对应计划：4.6

必须报告：
  预测层  Accuracy / Balanced Accuracy / Precision / Recall / F1 / ROC-AUC
          Brier / LogLoss / 校准曲线 / 混淆矩阵（按测试年汇总）
  策略层  CAGR / 年化波动 / Sharpe（超额）/ Sortino / MaxDD / Calmar
          胜率 / 盈亏比 / 年换手 / 平均仓位 / Alpha / Beta / 信息比率
  统计    block bootstrap（block=8 周, 5000 次）给 Sharpe 与 CAGR 的 95% CI
          配对 bootstrap 给"策略 - 基准"的 Sharpe 差 CI

纪律：所有比较以"差值 + 置信区间"呈现，不做排行榜。
"""

from __future__ import annotations

import pandas as pd


def classification_metrics(y_true, y_pred, y_proba) -> dict:
    """预测层指标，同时返回 always-long 基线，便于对比。"""
    raise NotImplementedError("B4：待实现（P4）")


def strategy_metrics(equity: pd.Series, trades: pd.DataFrame, freq: int = 52) -> dict:
    """策略层指标。Sharpe 一律用超额收益，与现金计息口径一致。"""
    raise NotImplementedError("B4：待实现（P4）")


def block_bootstrap_ci(returns: pd.Series, block: int = 8, n_boot: int = 5000, seed: int = 42):
    """周频收益的 block bootstrap 置信区间。"""
    raise NotImplementedError("B4：待实现（P4）")


def paired_bootstrap_diff(returns_a: pd.Series, returns_b: pd.Series, block: int = 8, n_boot: int = 5000):
    """两条策略的 Sharpe 差值置信区间（2.4.3 功效分析的实际执行）。"""
    raise NotImplementedError("B4：待实现（P4）")
