"""数据质量校验（B1）。

约定：每个 `check_*` 函数返回**问题描述列表**，空列表 = 通过。
这样可以直接聚合、直接打印、直接在 pytest 里断言。

负责人：P1（B1）
"""

from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]
OHLC_COLUMNS = ["open", "high", "low", "close"]


def check_columns(df: pd.DataFrame, required: list[str] | None = None) -> list[str]:
    """必需列是否齐全。"""
    required = required or REQUIRED_COLUMNS
    return [f"缺少列: {c}" for c in required if c not in df.columns]


def check_datetime_sorted(df: pd.DataFrame, date_col: str = "date") -> list[str]:
    """日期列必须是 datetime 类型且严格升序。"""
    if date_col not in df.columns:
        return [f"缺少日期列: {date_col}"]
    issues: list[str] = []
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        issues.append(f"{date_col} 不是 datetime 类型")
    if not df[date_col].is_monotonic_increasing:
        issues.append(f"{date_col} 未按升序排列")
    return issues


def check_duplicate_dates(df: pd.DataFrame, date_col: str = "date") -> list[str]:
    """日期不允许重复。"""
    dup = df[date_col][df[date_col].duplicated()]
    if len(dup) == 0:
        return []
    sample = sorted(set(dup.astype(str)))[:5]
    return [f"重复日期 {len(dup)} 行，示例: {sample}"]


def check_ohlc(df: pd.DataFrame, tol: float = 1e-8) -> list[str]:
    """OHLC 关系：low <= min(open, close)，high >= max(open, close)。"""
    missing = [c for c in OHLC_COLUMNS if c not in df.columns]
    if missing:
        return [f"缺少 OHLC 列: {missing}"]

    sub = df[OHLC_COLUMNS].dropna()
    if sub.empty:
        return ["OHLC 全部为空"]

    body_min = sub[["open", "close"]].min(axis=1)
    body_max = sub[["open", "close"]].max(axis=1)
    bad = (sub["low"] > body_min + tol) | (sub["high"] < body_max - tol)
    if not bad.any():
        return []
    first = sub.index[bad][0]
    return [f"OHLC 关系异常 {int(bad.sum())} 行，首个位置: {first}"]


def check_missing(df: pd.DataFrame, max_ratio: float = 0.005) -> list[str]:
    """缺失率超阈值的列。注：宏观列在前向填充后再检查。"""
    ratio = df.isna().mean()
    return [f"缺失率超阈值: {col} = {r:.2%}" for col, r in ratio.items() if r > max_ratio]


def week_key(index: pd.DatetimeIndex) -> pd.Series:
    """ISO 年-周键，例如 `2020-W53`。

    必须用 ISO 年而不是自然年：2021-01-01 属于 2020 年第 53 周，
    用 `dt.year` 会把它归到 W01，造成跨年错位。
    """
    idx = pd.DatetimeIndex(index)
    iso = idx.isocalendar()
    labels = [f"{int(y)}-W{int(w):02d}" for y, w in zip(iso["year"], iso["week"])]
    return pd.Series(labels, index=idx, name="iso_week")


def last_trading_day_per_week(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """每周最后一个交易日。禁止用 `resample('W-FRI')`，假期周会错位。"""
    idx = pd.DatetimeIndex(index).sort_values()
    keys = pd.Series(week_key(idx).to_numpy(), index=idx)
    dates = pd.Series(idx, index=idx)
    last = dates.groupby(keys.to_numpy()).max().sort_values()
    return pd.DatetimeIndex(last.to_numpy())


def summarize(df: pd.DataFrame, date_col: str = "date") -> dict[str, object]:
    """一次性汇总，供数据质量报告直接写进 markdown。"""
    return {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "date_min": str(df[date_col].min()) if date_col in df.columns else None,
        "date_max": str(df[date_col].max()) if date_col in df.columns else None,
        "n_duplicate_dates": int(df[date_col].duplicated().sum()) if date_col in df.columns else None,
        "missing_ratio": {c: round(float(r), 6) for c, r in df.isna().mean().items()},
        "issues": (
            check_columns(df)
            + check_datetime_sorted(df, date_col)
            + check_duplicate_dates(df, date_col)
        ),
    }
