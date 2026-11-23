"""数据质量检查器测试（B1 闸门 3）——全部离线，用合成序列。"""

from __future__ import annotations

import pandas as pd

from src.data_quality import (
    _table,
    _yearly_table,
    find_dividend_jumps,
    find_extreme_returns,
    find_factor_drops,
    find_stale_closes,
    find_zero_volume,
)


def _daily(closes, volumes=None, factor=None) -> pd.DataFrame:
    n = len(closes)
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=n),
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "adj_close": closes,
            "close_raw": closes,
            "volume": volumes if volumes is not None else [1_000_000] * n,
            "f": factor if factor is not None else [1.0] * n,
        }
    )


def test_find_extreme_returns_only_over_threshold():
    rets = pd.Series([0.01, -0.25, 0.3, 0.05], index=pd.bdate_range("2024-01-01", periods=4))
    hit = find_extreme_returns(rets, 0.20)
    assert len(hit) == 2
    assert set(hit["value"].round(2)) == {-0.25, 0.3}


def test_find_zero_volume():
    df = _daily([1, 2, 3, 4], volumes=[100, 0, 0, 50])
    assert len(find_zero_volume(df)) == 2


def test_find_stale_closes_needs_three_in_a_row():
    # 三个相同 → 命中；两个相同 → 不命中
    df = _daily([10, 10, 10, 11, 12, 12, 13])
    hit = find_stale_closes(df, run=3)
    assert len(hit) == 1
    assert int(hit["value"].iloc[0]) == 3
    assert pd.Timestamp(hit["date"].iloc[0]) == pd.Timestamp("2024-01-01")


def test_dividend_jump_detects_upward_move():
    """除息日复权因子是**向上**跳的（f 越靠近今天越接近 1）。"""
    factor = pd.Series([0.90, 0.90, 0.9050, 0.9050], index=pd.bdate_range("2024-01-01", periods=4))
    hit = find_dividend_jumps(factor, 0.001)
    assert len(hit) == 1
    assert pd.Timestamp(hit["date"].iloc[0]) == pd.Timestamp("2024-01-03")


def test_factor_drop_flags_anomaly():
    factor = pd.Series([0.90, 0.8950, 0.8950], index=pd.bdate_range("2024-01-01", periods=3))
    assert len(find_factor_drops(factor, 0.001)) == 1
    assert len(find_dividend_jumps(factor, 0.001)) == 0


def test_yearly_table_counts_days_and_events():
    daily = _daily([100, 101, 102, 103, 104], volumes=[1, 0, 1, 1, 1])
    daily["f"] = [0.9, 0.9, 0.905, 0.905, 0.905]
    weekly = pd.DataFrame({"date": pd.to_datetime(["2024-01-05"])})
    table = _yearly_table(daily, weekly)

    row = table.iloc[0]
    assert int(row["year"]) == 2024
    assert int(row["trading_days"]) == 5
    assert int(row["weeks"]) == 1
    assert int(row["zero_volume_days"]) == 1
    assert int(row["dividend_events"]) == 1


def test_table_renders_markdown_without_breaking():
    md = _table(["a", "b"], [[1, 2], [3, 4]])
    assert md.splitlines()[0] == "| a | b |"
    assert md.splitlines()[1] == "|---|---|"
    assert md.splitlines()[2] == "| 1 | 2 |"
