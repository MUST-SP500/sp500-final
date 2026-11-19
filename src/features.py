"""特征工程（B3）：22 个特征，全部比率型。

负责人：P3（B3）
对应计划：4.3

五组特征（合计 22 个）：
  动量   1/3/6/12 个月收益、12-1 动量                              (5)
  趋势   价格/MA20、/MA50、/MA200、MA50-MA200、MA200 斜率           (5)
  波动   20/60 日已实现波动、下行波动、ATR、回撤、underwater days   (5)
  情绪   VIX 水平与变化、10Y 收益率变化、10Y-3M 期限利差            (4)
  量能   成交量 z-score、RSI14、MACD 柱                            (3)

硬性约束：
  1. 禁止使用价格绝对水平（只用比率/收益率/变化率/滚动 z-score）
  2. 所有滚动窗口只向后看，窗口末端 <= 决策时点 t
  3. 截断不变性：把原始数据截到 t 重算，结果必须与全量计算一致
"""

from __future__ import annotations

import pandas as pd

FEATURE_GROUPS = {
    "momentum": ["ret_1m", "ret_3m", "ret_6m", "ret_12m", "mom_12_1"],
    "trend": ["px_ma20", "px_ma50", "px_ma200", "ma50_ma200", "ma200_slope"],
    "volatility": ["vol_20", "vol_60", "downside_vol", "atr_14", "drawdown"],
    "sentiment": ["vix_level", "vix_chg", "dgs10_chg", "term_spread"],
    "volume": ["volume_z", "rsi_14", "macd_hist"],
}


def build_features(df: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """输入日频清洗后的数据，输出特征矩阵（索引与输入对齐）。"""
    raise NotImplementedError("B3：待实现（P3）")


def feature_columns() -> list[str]:
    """返回全部特征列名，供建模与测试统一引用。"""
    return [c for cols in FEATURE_GROUPS.values() for c in cols]
