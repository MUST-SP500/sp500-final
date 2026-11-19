"""决策规则（B4）：把"预测"变成"仓位"。

负责人：P4（B4）
对应计划：2.3.1

    R1  固定阈值   p = 1[P_hat > 0.5]                 主规则（预注册）
    R2  分位数阈值 p = 1[P_hat > 训练集 60% 分位]      核心
    R3  置信度加权 p = clip((P_hat - 0.5) / 0.5, 0, 1) 拉伸
    R4  回归+Kelly p = clip(mu_hat / sigma_hat^2 * k, 0, 1)  拉伸

纪律：阈值只能来自训练/验证区间，禁止看测试集调。
"""

from __future__ import annotations

import pandas as pd


def rule_fixed(proba: pd.Series, threshold: float = 0.5) -> pd.Series:
    """R1：固定阈值。"""
    raise NotImplementedError("B4：待实现（P4）")


def rule_quantile(proba: pd.Series, train_proba: pd.Series, q: float = 0.60) -> pd.Series:
    """R2：阈值取训练集预测概率的分位数。"""
    raise NotImplementedError("B4：待实现（P4）")


def rule_confidence(proba: pd.Series) -> pd.Series:
    """R3：置信度加权，降低换手。"""
    raise NotImplementedError("B4：拉伸项（P4）")


def rule_kelly(mu: pd.Series, sigma: pd.Series, kappa: float) -> pd.Series:
    """R4：回归预测 + Kelly 缩放。"""
    raise NotImplementedError("B4：拉伸项（P4）")
