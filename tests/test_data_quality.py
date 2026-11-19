"""数据质量校验（B1 的验收测试）+ 周键口径测试。

这里的测试是"口径的可执行定义"：改口径必须先改测试，再改代码。
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.validate import (
    check_columns,
    check_datetime_sorted,
    check_duplicate_dates,
    check_missing,
    check_ohlc,
    last_trading_day_per_week,
    summarize,
    week_key,
)


def test_valid_frame_has_no_issues(ohlc):
    assert check_columns(ohlc) == []
    assert check_datetime_sorted(ohlc) == []
    assert check_duplicate_dates(ohlc) == []
    assert check_ohlc(ohlc) == []


def test_missing_column_detected(ohlc):
    issues = check_columns(ohlc.drop(columns=["volume"]))
    assert issues and "volume" in issues[0]


def test_duplicate_dates_detected(ohlc):
    dup = pd.concat([ohlc, ohlc.iloc[[0]]], ignore_index=True)
    assert check_duplicate_dates(dup)


def test_unsorted_dates_detected(ohlc):
    shuffled = ohlc.sample(frac=1.0, random_state=0).reset_index(drop=True)
    assert check_datetime_sorted(shuffled)


def test_ohlc_relation_violation_detected(ohlc):
    bad = ohlc.copy()
    bad.loc[bad.index[0], "high"] = bad.loc[bad.index[0], "low"] - 1.0
    issues = check_ohlc(bad)
    assert issues and "OHLC" in issues[0]


def test_missing_ratio_threshold(ohlc):
    dirty = ohlc.copy()
    dirty.loc[dirty.index[:100], "open"] = None  # 100/262 ≈ 38%
    assert check_missing(dirty, max_ratio=0.005)


def test_summarize_shape(ohlc):
    info = summarize(ohlc)
    assert info["rows"] == len(ohlc)
    assert info["n_duplicate_dates"] == 0
    assert info["issues"] == []


# ---------------- 周键口径：这是最容易出错的地方 ----------------

def test_week_key_uses_iso_year_not_calendar_year():
    """2021-01-01 属于 2020 年第 53 周，用自然年会错成 2021-W01。"""
    idx = pd.DatetimeIndex(["2020-12-31", "2021-01-01", "2021-01-04"])
    keys = week_key(idx).tolist()
    assert keys[0] == "2020-W53"
    assert keys[1] == "2020-W53", "跨年周必须归到 ISO 年 2020"
    assert keys[2] == "2021-W01"


def test_week_key_is_string_series():
    keys = week_key(pd.DatetimeIndex(["2024-03-04"]))
    assert keys.name == "iso_week"
    assert keys.iloc[0] == "2024-W10"


def test_last_trading_day_per_week_picks_friday():
    idx = pd.bdate_range("2024-11-25", "2024-11-29")  # 周一至周五
    last = last_trading_day_per_week(idx)
    assert len(last) == 1
    assert last[0] == pd.Timestamp("2024-11-29")


def test_last_trading_day_per_week_handles_holiday():
    """感恩节周只有 4 个交易日，最后一天仍是周三提前收盘后的周五（这里用周四代替）。"""
    idx = pd.DatetimeIndex(["2024-11-25", "2024-11-26", "2024-11-27", "2024-11-29"])
    last = last_trading_day_per_week(idx)
    assert list(last) == [pd.Timestamp("2024-11-29")]


def test_last_trading_day_count_matches_iso_weeks():
    idx = pd.bdate_range("2024-01-01", "2024-12-31")
    last = last_trading_day_per_week(idx)
    assert len(last) == week_key(idx).nunique()
    assert last.is_monotonic_increasing


def test_week_key_has_no_nan():
    idx = pd.bdate_range("2024-01-01", "2024-12-31")
    assert week_key(idx).notna().all()
