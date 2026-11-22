"""walk-forward 切分（B3）：扩展窗口 + 2 周 purge。

负责人：P3（B3）
对应：SP500_分工说明书 P3 第 5 步

协议：
  训练 1996-01-01 起，每年向前扩展
  验证（仅一次性选参）2006-01-01 – 2007-12-31，只记录在 config.yaml
  第一个测试年 2008，最后 2025
  purge = 2 周（1 周标签跨度 + 1 周缓冲）
  超参：一次性选定后全折冻结（不逐折重选）

输出必须可被 B4/B5 直接读取，且与 docs/preregistration.md 一致。
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def make_folds(index: pd.DatetimeIndex, cfg: dict) -> list[dict]:
    """按测试年生成折列表，每折含 train/test 的起止日期与 purge。"""
    raise NotImplementedError("B3：待实现（P3）")


def purge_overlap(folds: list[dict]) -> list[dict]:
    """剔除训练集末端与测试集标签窗口重叠的样本。"""
    raise NotImplementedError("B3：待实现（P3）")


def save_splits(folds: list[dict], path: str | Path) -> Path:
    """写出 splits.json，这是 B4/B5 的共同契约。"""
    raise NotImplementedError("B3：待实现（P3）")
