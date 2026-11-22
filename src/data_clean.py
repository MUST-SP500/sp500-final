"""数据清洗（B1）：复权因子、周频聚合、面板合并。

负责人：P1（B1）
对应：SP500_分工说明书 P1 第 2–4 步

口径（写死在代码里，不要各处各写一套）：
  f_t      = adj_close_t / close_t          # 复权因子
  O^adj_t  = open_t * f_t                   # 同理 high / low
  vix 统一 shift(1)；宏观序列按发布滞后处理
  周键用 ISO 年-周，取每周最后一个交易日

验收：日期唯一、升序、1996–2025 完整、SPY 与 ^GSPC 日收益相关性 > 0.99
"""

from __future__ import annotations

import pandas as pd


def add_adjusted_prices(df: pd.DataFrame) -> pd.DataFrame:
    """由 (adj_close / close) 派生复权 OHLC。"""
    raise NotImplementedError("B1：待实现（P1）")


def to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """日频 → 周频面板，索引为每周最后一个交易日。"""
    raise NotImplementedError("B1：待实现（P1）")


def build_panel(cfg: dict) -> pd.DataFrame:
    """合并价格、VIX、宏观，输出 weekly_panel.parquet。"""
    raise NotImplementedError("B1：待实现（P1）")
