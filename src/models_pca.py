"""PCA（B5）：折内降维 + 逻辑回归对比。

负责人：P5（B5）
注意：PCA 的 fit 必须发生在训练折内部，绝不能用全样本 fit（见 2.5 红线 1）。
图表（碎石图 / 载荷热力图）由 B2 出，本模块只出结果表。
"""

from __future__ import annotations

import pandas as pd


def fit_pca(X_train, variance_target: float = 0.90):
    """在训练折上 fit，返回 (pca, n_components)。"""
    raise NotImplementedError("B5：待实现（P5）")


def pca_logreg(panel: pd.DataFrame, folds: list[dict], cfg: dict) -> pd.DataFrame:
    """PCA + 逻辑回归，与全特征 LR 在同一 split 上对比。"""
    raise NotImplementedError("B5：待实现（P5）")
