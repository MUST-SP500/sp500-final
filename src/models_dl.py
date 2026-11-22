"""深度学习（B5）：DNN / 1D CNN / Autoencoder。

负责人：P5（B5）
对应：SP500_分工说明书 P5 第 2–5 步

约束：
  - 早停只用本折训练段尾部 2 年，绝不用测试年
  - 全折只用 1 个固定种子（一周工期，不做多种子方差；真的需要再单独说）
  - 参数量 <= 5,000（样本只有约 1,500 周，模型不能大）
  - 单次训练 > 5 分钟的超参组合直接移出搜索空间
  - 滚动窗口的最后一个观测值必须 <= 决策时点 t，且不跨 train/test 边界
"""

from __future__ import annotations

import pandas as pd


def build_dnn(n_features: int, seed: int = 42):
    """2 隐藏层（64→32）+ Dropout 0.3 + BatchNorm，二分类 sigmoid。"""
    raise NotImplementedError("B5：待实现（P5）")


def build_cnn(window: int = 20, n_features: int = 22, seed: int = 42):
    """Conv1D(16) → Conv1D(32) → GlobalMaxPooling → Dense。"""
    raise NotImplementedError("B5：待实现（P5）")


def train_and_predict(model, X_train, y_train, X_val, y_val, X_test, epochs: int = 100):
    """训练 + 早停 + 预测，返回测试集概率。"""
    raise NotImplementedError("B5：待实现（P5）")


def autoencoder_bottleneck(X_train, X_val, bottleneck: int = 8):
    """只在 2008 一个折上跑，用来和 PCA 做定性对比；不做全 18 折。"""
    raise NotImplementedError("B5：待实现（P5），只跑 2008 一折")
