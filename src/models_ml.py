"""传统机器学习（B4）：基线 + LR / Ridge / RF。

负责人：P4（B4）
对应：SP500_分工说明书 P4 第 1–3 步（基线 / 模型工厂 / 18 折滚动）

约束：
  - 全部使用 src.splits 生成的同一份 splits.json
  - 预处理（winsorize / scaler）只在训练折内 fit
  - 超参与阈值只在 2006–2007 上选一次，之后冻结
  - 主模型：L2 逻辑回归（预注册，不许中途替换）
  - HGB 本版不做（一周工期）；Ridge 只用来给 R4 提供回归预测
"""

from __future__ import annotations

import pandas as pd


def make_baselines(panel: pd.DataFrame) -> pd.DataFrame:
    """买入持有、恒定 60% 仓位、SMA200 —— 必须与最终模型同口径同成本。"""
    raise NotImplementedError("B4：待实现（P4）")


def build_model(name: str, seed: int = 42):
    """返回未训练的 sklearn 模型（logreg / ridge / rf）。"""
    raise NotImplementedError("B4：待实现（P4）")


def walk_forward(panel: pd.DataFrame, folds: list[dict], model_name: str, cfg: dict) -> pd.DataFrame:
    """滚动训练+预测，返回 predictions 表。"""
    raise NotImplementedError("B4：待实现（P4）")
