"""决策规则（B4）：把"预测"变成"仓位"。

负责人：P4（B4）
对应：SP500_分工说明书 P4 第 4 步

    R1  固定阈值   p = 1[P_hat > 0.5]                       主规则（预注册），本版必做
    R2  分位数阈值 p = 1[P_hat > 训练集 60% 分位]            本版必做
    R4  回归+Kelly p = clip(mu_hat / sigma_hat^2 * k, 0, 1)  本版必做（核心，不是拉伸项）
    R3  置信度加权 —— 本版不做，函数留在下面只备参考，不要接进主流程

纪律：阈值只能来自训练/验证区间，禁止看测试集调。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rule_fixed(proba: pd.Series, threshold: float = 0.5) -> pd.Series:
    """R1：固定阈值。"""
    s = pd.Series(proba, dtype=float)
    pos = (s > threshold).astype(float)
    pos[s.isna()] = np.nan
    return pos


def rule_quantile(proba: pd.Series, train_proba: pd.Series, q: float = 0.60) -> pd.Series:
    """R2：阈值取训练集预测概率的分位数。"""
    s = pd.Series(proba, dtype=float)
    tr = pd.Series(train_proba, dtype=float).dropna()
    if tr.empty:
        raise ValueError("train_proba 不可为空，无法计算分位数阈值")
    threshold = float(np.quantile(tr.to_numpy(), q))
    pos = (s > threshold).astype(float)
    pos[s.isna()] = np.nan
    return pos


def rule_confidence(proba: pd.Series) -> pd.Series:
    """R3：置信度加权。本版不做，留着只为以后参考。"""
    raise NotImplementedError("B4：本版不做（R3），不要接进主流程")


def rule_kelly(mu: pd.Series, sigma: pd.Series, kappa: float) -> pd.Series:
    """R4：回归预测 + Kelly 缩放。"""
    mu_s = pd.Series(mu, dtype=float)
    sig_s = pd.Series(sigma, dtype=float).reindex(mu_s.index)
    var = sig_s ** 2
    raw = np.where((var > 0) & np.isfinite(var), (mu_s / var) * float(kappa), 0.0)
    pos = pd.Series(np.clip(raw, 0.0, 1.0), index=mu_s.index, dtype=float)
    pos[mu_s.isna() | sig_s.isna()] = np.nan
    return pos