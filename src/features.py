"""特征工程（B3）：22 个特征，全部比率型；外加折内预处理。

负责人：P3（B3）
对应：SP500_分工说明书 P3 第 3、6 步

五组特征（合计 22 个）：
  动量   1/3/6/12 个月收益、12-1 动量                              (5)
  趋势   价格/MA20、/MA50、/MA200、MA50-MA200、MA200 斜率           (5)
  波动   20/60 日已实现波动、下行波动、ATR、回撤   (5)
  情绪   VIX 水平与变化、10Y 收益率变化、10Y-3M 期限利差            (4)
  量能   成交量 z-score、RSI14、MACD 柱                            (3)

硬性约束：
  1. 禁止使用价格绝对水平（只用比率/收益率/变化率/滚动 z-score）
  2. 所有滚动窗口只向后看，窗口末端 <= 决策时点 t
  3. 截断不变性：把原始数据截到 t 重算，结果必须与全量计算一致
  4. 预处理（winsorize / scaler）只能在训练折上 fit，见 make_preprocessor
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils.validation import check_is_fitted

FEATURE_GROUPS = {
    "momentum": ["ret_1m", "ret_3m", "ret_6m", "ret_12m", "mom_12_1"],
    "trend": ["px_ma20", "px_ma50", "px_ma200", "ma50_ma200", "ma200_slope"],
    "volatility": ["vol_20", "vol_60", "downside_vol", "atr_14", "drawdown"],
    "sentiment": ["vix_level", "vix_chg", "dgs10_chg", "term_spread"],
    "volume": ["volume_z", "rsi_14", "macd_hist"],
}


def build_features(df: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """输入日频清洗后的数据，输出特征矩阵（索引与输入对齐）。"""
    if df.empty or not df.index.is_unique:
        raise ValueError("日频输入不可为空，行索引必须唯一")
    required = ["close", "high", "low", "volume", "vix", "dgs10", "dtb3"]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"缺少日频字段: {sorted(missing)}")
    dates = pd.DatetimeIndex(pd.to_datetime(df["date"])) if "date" in df else df.index
    if not isinstance(dates, pd.DatetimeIndex) or dates.hasnans or not dates.is_unique or not dates.is_monotonic_increasing:
        raise ValueError("日频日期必须有效、唯一且递增")
    x = df[required].astype(float)
    if x[["close", "high", "low", "volume"]].isna().any().any():
        raise ValueError("价格和成交量缺失，请先修复 B1 面板")
    if np.isinf(x.to_numpy()).any() or (x[["close", "high", "low"]] <= 0).any().any():
        raise ValueError("价格必须为正，输入不可包含无穷值")
    close = x["close"]
    ret = close.pct_change(fill_method=None)
    out = pd.DataFrame(index=df.index)
    for name, days in [("ret_1m", 21), ("ret_3m", 63), ("ret_6m", 126), ("ret_12m", 252)]:
        out[name] = close / close.shift(days) - 1
    out["mom_12_1"] = out["ret_12m"] - out["ret_1m"]
    ma = {n: close.rolling(n).mean() for n in (20, 50, 200)}
    for n in ma:
        out[f"px_ma{n}"] = close / ma[n] - 1
    out["ma50_ma200"] = ma[50] / ma[200] - 1
    out["ma200_slope"] = ma[200] / ma[200].shift(21) - 1
    for n in (20, 60):
        out[f"vol_{n}"] = ret.rolling(n).std() * np.sqrt(252)
    # 最近 60 个交易日内的负收益样本标准差，不压缩时间轴。
    out["downside_vol"] = ret.rolling(60).apply(
        lambda a: a[a < 0].std(ddof=1) if (a < 0).sum() >= 2 else np.nan, raw=True
    )
    prev = close.shift(1)
    tr = pd.concat([x["high"] - x["low"], (x["high"] - prev).abs(), (x["low"] - prev).abs()], axis=1).max(axis=1)
    tr.iloc[:1] = np.nan
    # Wilder 平滑：先以 n 个有效样本的均值初始化，再递推。
    def wilder(series, n):
        seed = series.rolling(n).mean().first_valid_index()
        result = pd.Series(np.nan, index=series.index, dtype=float)
        if seed is not None:
            start = series.index.get_loc(seed)
            values = series.iloc[start:].copy()
            values.iloc[0] = series.iloc[start-n+1:start+1].mean()
            result.iloc[start:] = values.ewm(alpha=1/n, adjust=False, ignore_na=False).mean()
        return result
    out["atr_14"] = wilder(tr, 14) / close
    out["drawdown"] = close / close.cummax() - 1
    out["vix_level"] = x["vix"]
    out["vix_chg"] = x["vix"] / x["vix"].shift(5) - 1
    out["dgs10_chg"] = x["dgs10"].diff(5)
    out["term_spread"] = x["dgs10"] - x["dtb3"]
    volume = x["volume"]
    out["volume_z"] = (volume - volume.rolling(20).mean()) / volume.rolling(20).std().replace(0, np.nan)
    delta = close.diff()
    gain = wilder(delta.clip(lower=0), 14)
    loss = wilder(-delta.clip(upper=0), 14)
    total = gain + loss
    out["rsi_14"] = (gain / total).where(total != 0, 0.5)
    fast = close.ewm(span=12, adjust=False, min_periods=12).mean()
    slow = close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = fast - slow
    out["macd_hist"] = (macd - macd.ewm(span=9, adjust=False, min_periods=9).mean()) / close
    return out[feature_columns()].replace([np.inf, -np.inf], np.nan)


def make_preprocessor(winsor: float = 0.01, scaling: str = "standard") -> "Pipeline":
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
    if not np.isfinite(winsor) or not 0 <= winsor < 0.5:
        raise ValueError("winsor 必须在 [0, 0.5) 内")
    if scaling not in ("standard", "minmax"):
        raise ValueError("scaling 必须为 standard 或 minmax")
    return Pipeline([
        ("imputer", TrainingMedianImputer()),
        ("winsorizer", TrainingQuantileClipper(winsor=winsor)),
        ("scaler", StandardScaler() if scaling == "standard" else MinMaxScaler(clip=True)),
    ])


class TrainingMedianImputer(TransformerMixin, BaseEstimator):
    """训练中位数填补；整列缺失、无穷值、字段错位明确报错。"""

    def _array(self, X, fitting=False):
        values = np.asarray(X, dtype=float)
        if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
            raise ValueError("特征必须是非空二维矩阵")
        if np.isinf(values).any():
            raise ValueError("特征不可包含无穷值")
        if fitting:
            self.n_features_in_ = values.shape[1]
            self.input_columns_ = list(X.columns) if isinstance(X, pd.DataFrame) else None
        else:
            if values.shape[1] != self.n_features_in_:
                raise ValueError("特征列数不一致")
            if self.input_columns_ is not None and (not isinstance(X, pd.DataFrame) or list(X.columns) != self.input_columns_):
                raise ValueError("特征列名及顺序必须与训练一致")
        return values

    def fit(self, X, y=None):
        values = self._array(X, fitting=True)
        if np.isnan(values).all(axis=0).any():
            raise ValueError("训练特征存在整列缺失，无法估计中位数")
        self.statistics_ = np.nanmedian(values, axis=0)
        return self

    def transform(self, X):
        check_is_fitted(self, "statistics_")
        values = self._array(X)
        return np.where(np.isnan(values), self.statistics_, values)


class TrainingQuantileClipper(TransformerMixin, BaseEstimator):
    """只在 fit 估计截尾阈值，transform 不更新参数。"""

    def __init__(self, winsor=0.01):
        self.winsor = winsor

    def fit(self, X, y=None):
        if not np.isfinite(self.winsor) or not 0 <= self.winsor < 0.5:
            raise ValueError("winsor 必须在 [0, 0.5) 内")
        values = np.asarray(X, dtype=float)
        if values.ndim != 2 or values.size == 0 or not np.isfinite(values).all():
            raise ValueError("截尾输入必须是非空、有限的二维矩阵")
        self.n_features_in_ = values.shape[1]
        self.lower_ = np.quantile(values, self.winsor, axis=0)
        self.upper_ = np.quantile(values, 1 - self.winsor, axis=0)
        return self

    def transform(self, X):
        check_is_fitted(self, "lower_")
        values = np.asarray(X, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.n_features_in_ or not np.isfinite(values).all():
            raise ValueError("截尾输入列数不一致或包含非有限值")
        return np.clip(values, self.lower_, self.upper_)


def feature_columns() -> list[str]:
    """返回全部特征列名，供建模与测试统一引用。"""
    return [c for cols in FEATURE_GROUPS.values() for c in cols]


def main() -> None:
    """独立 B3 特征入口：不修改 B1 面板，不拟合预处理。"""
    from src.config import load_config, project_path
    cfg = load_config()
    daily = pd.read_parquet(project_path(cfg["data"]["daily_panel"]))
    weekly = pd.read_parquet(project_path(cfg["data"]["weekly_panel"]))
    features = build_features(daily, cfg)
    features.insert(0, "date", pd.to_datetime(daily["date"]))
    aligned = weekly[["date"]].merge(features, on="date", how="left", validate="one_to_one", indicator=True)
    if not aligned["_merge"].eq("both").all():
        raise ValueError("周频日期在日频表中不存在")
    aligned = aligned.drop(columns="_merge")
    target = project_path("data/processed/weekly_features.parquet")
    aligned.to_parquet(target, index=False)
    aligned.to_csv(target.with_suffix(".csv"), index=False)
    valid = aligned[feature_columns()].notna().all(axis=1)
    print(f"周特征: {aligned.shape}; 完整行: {valid.sum()}; 首个完整决策日: {aligned.loc[valid, 'date'].min()}")
    print(f"已写入 {target} 及同名 CSV；预热缺失保留，尚未标准化。")


if __name__ == "__main__":
    main()
