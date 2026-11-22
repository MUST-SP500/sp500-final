"""特征工程（B3）：22 个特征，全部比率型；外加折内预处理。

负责人：P3（B3）
对应：SP500_分工说明书 P3 第 3、6 步

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
  4. 预处理（winsorize / scaler）只能在训练折上 fit，见 make_preprocessor
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


def make_preprocessor(winsor: float = 0.01) -> "Pipeline":
    """折内预处理：缺失 → winsorize（训练集 1%/99%）→ 标准化。

    必须返回一个 sklearn 风格的 Pipeline，用法固定是两步：

        pre = make_preprocessor()
        X_train = pre.fit_transform(train[feature_columns()])   # 只在训练折 fit
        X_test  = pre.transform(test[feature_columns()])        # 测试折只能 transform

    为什么要封成 Pipeline 而不是各写各的：
    只要有人在测试集上 fit 一次，整块结论就废了。封起来之后，"能不能 fit"
    由调用方式决定，测试代码也可以直接断言 scaler 的均值不等于全样本均值：

        assert not np.allclose(pre.named_steps["scaler"].mean_, 全样本均值)

    注意：
      - 树模型不需要标准化，但为了统一，仍然走这个 Pipeline；
      - 不许在这里 fillna(0)，缺失要么按字典规则处理，要么报错；
      - PCA / Autoencoder 的 fit 也只在训练折内做（各自模块里保证）。
    """
    raise NotImplementedError("B3：待实现（P3）")


def feature_columns() -> list[str]:
    """返回全部特征列名，供建模与测试统一引用。"""
    return [c for cols in FEATURE_GROUPS.values() for c in cols]
