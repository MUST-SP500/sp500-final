"""清洗层测试（B1）：复权因子、周频聚合、面板合并。

全部离线：用合成数据 + 临时目录，不碰 data/raw 也不打网络。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_clean import add_adjusted_prices, build_panel, to_weekly


# --------------------------------------------------------------------------- #
# 假数据
# --------------------------------------------------------------------------- #
def _daily(periods: int = 10, start: str = "2024-01-01") -> pd.DataFrame:
    """一段合成的日频 OHLCV，adj_close 恒为 close 的 0.98 倍。"""
    dates = pd.bdate_range(start, periods=periods)
    close = pd.Series(np.linspace(100, 110, periods))
    return pd.DataFrame(
        {
            "date": dates,
            "open": close - 1,
            "high": close + 2,
            "low": close - 2,
            "close": close,
            "adj_close": close * 0.98,
            "volume": np.arange(1, periods + 1) * 100,
        }
    )


def _adjusted(periods: int = 10) -> pd.DataFrame:
    return add_adjusted_prices(_daily(periods))


# --------------------------------------------------------------------------- #
# 第 2 步：复权
# --------------------------------------------------------------------------- #
def test_factor_is_shared_by_all_four_prices():
    df = _adjusted()
    # 复权价 ÷ 原始价，四个价格必须得到同一个系数
    ratios = pd.DataFrame(
        {
            "open": df["open_adj"] / df["open"],
            "high": df["high_adj"] / df["high"],
            "low": df["low_adj"] / df["low"],
            "close": df["close_adj"] / df["close"],
        }
    )
    spread = (ratios.max(axis=1) - ratios.min(axis=1)).abs().max()
    assert spread < 1e-12, "同一天的 4 个价格必须共用同一个复权因子"


def test_adjusted_close_equals_source_adj_close():
    df = _adjusted()
    assert (df["close_adj"] == df["adj_close"]).all()


def test_adjusted_ohlc_relation_still_holds():
    df = _adjusted()
    assert (df["low_adj"] <= df[["open_adj", "close_adj"]].min(axis=1)).all()
    assert (df["high_adj"] >= df[["open_adj", "close_adj"]].max(axis=1)).all()


def test_missing_column_raises():
    with pytest.raises(ValueError, match="缺少列"):
        add_adjusted_prices(_daily().drop(columns=["adj_close"]))


# --------------------------------------------------------------------------- #
# 第 3 步：周频聚合
# --------------------------------------------------------------------------- #
def test_weekly_aggregation_rules():
    """open 取首日、close 取末日、high/low 取极值、volume 求和、天数计数。"""
    df = _adjusted(periods=10)  # 2024-01-01(周一) ~ 2024-01-12(周五)，正好两周
    weekly = to_weekly(df)

    assert len(weekly) == 2
    first_week = df.iloc[0:5]
    assert weekly.loc[0, "open"] == first_week["open_adj"].iloc[0]
    assert weekly.loc[0, "close"] == first_week["close_adj"].iloc[-1]
    assert weekly.loc[0, "high"] == first_week["high_adj"].max()
    assert weekly.loc[0, "low"] == first_week["low_adj"].min()
    assert weekly.loc[0, "volume"] == first_week["volume"].sum()
    assert weekly.loc[0, "n_trading_days"] == 5
    assert weekly.loc[0, "date"] == first_week["date"].iloc[-1]


def test_weekly_columns_and_ordering():
    weekly = to_weekly(_adjusted(periods=20))
    assert list(weekly.columns) == [
        "date",
        "iso_week",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "n_trading_days",
    ]
    assert weekly["date"].is_monotonic_increasing
    assert weekly["iso_week"].is_unique


def test_holiday_week_has_four_trading_days():
    """2024 感恩节：周一~周三 + 周五（周四休市），必须仍是一行、n_trading_days=4。"""
    dates = pd.DatetimeIndex(["2024-11-25", "2024-11-26", "2024-11-27", "2024-11-29"])
    df = add_adjusted_prices(_daily(periods=4).assign(date=dates))
    weekly = to_weekly(df)

    assert len(weekly) == 1
    assert weekly.loc[0, "n_trading_days"] == 4
    assert weekly.loc[0, "date"] == pd.Timestamp("2024-11-29")


def test_year_boundary_belongs_to_iso_week_of_previous_year():
    """2020-12-31 与 2021-01-01 同属 2020-W53，不能拆成两行。"""
    dates = pd.DatetimeIndex(["2020-12-30", "2020-12-31", "2021-01-01"])
    df = add_adjusted_prices(_daily(periods=3).assign(date=dates))
    weekly = to_weekly(df)

    assert len(weekly) == 1
    assert weekly.loc[0, "iso_week"] == "2020-W53"
    assert weekly.loc[0, "date"] == pd.Timestamp("2021-01-01")


def test_three_holiday_weeks_one_row_each():
    """圣诞周 / 飓风休市周（2012-10-29、30 停市）都只能产生一行。"""
    dates = pd.DatetimeIndex(
        [
            "2012-10-31",  # 飓风后复市
            "2012-11-01",
            "2012-11-02",
            "2024-12-23",
            "2024-12-24",
            "2024-12-26",
        ]
    )
    df = add_adjusted_prices(_daily(periods=6).assign(date=dates))
    weekly = to_weekly(df)

    assert len(weekly) == 2
    assert weekly["n_trading_days"].tolist() == [3, 3]


def test_to_weekly_requires_adjusted_columns():
    with pytest.raises(ValueError, match="add_adjusted_prices"):
        to_weekly(_daily())


# --------------------------------------------------------------------------- #
# 第 4 步：面板
# --------------------------------------------------------------------------- #
def _fake_raw(root, business_days: int = 60):
    """在临时目录里造出一套 data/raw/{yahoo,fred} 结构。"""
    (root / "yahoo").mkdir(parents=True)
    (root / "fred").mkdir(parents=True)

    dates = pd.bdate_range("2024-01-01", periods=business_days)
    close = pd.Series(np.linspace(100, 120, business_days))
    spy = pd.DataFrame(
        {
            "date": dates,
            "open": close - 1,
            "high": close + 2,
            "low": close - 2,
            "close": close,
            "adj_close": close * 0.98,
            "volume": 1_000_000,
        }
    )
    spy.to_csv(root / "yahoo" / "SPY.csv", index=False)

    vix = pd.DataFrame({"date": dates, "open": 15, "high": 16, "low": 14, "close": np.arange(1, business_days + 1, dtype=float), "adj_close": 0, "volume": 0})
    vix.to_csv(root / "yahoo" / "VIX.csv", index=False)
    vix[["date", "close"]].to_csv(root / "yahoo" / "VIX_only.csv", index=False)

    pd.DataFrame({"date": dates, "dgs10": 4.0}).to_csv(root / "fred" / "DGS10.csv", index=False)
    pd.DataFrame({"date": dates, "dtb3": 5.0}).to_csv(root / "fred" / "DTB3.csv", index=False)
    return dates


def test_build_panel_shape_and_columns(tmp_path):
    root = tmp_path / "raw"
    _fake_raw(root)
    panel = build_panel({}, raw_root=root, out_path=tmp_path / "w.parquet", daily_out_path=tmp_path / "d.parquet")

    assert list(panel.columns) == [
        "date",
        "iso_week",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "n_trading_days",
        "vix",
        "dgs10",
        "dtb3",
    ]
    assert (tmp_path / "w.parquet").exists()
    assert (tmp_path / "d.parquet").exists()
    assert panel["date"].is_monotonic_increasing
    assert not panel["date"].duplicated().any()


def test_build_panel_vix_is_shifted_by_one_day(tmp_path):
    """每周的 vix 必须等于该周最后一个交易日的**前一天**的 VIX。"""
    root = tmp_path / "raw"
    dates = _fake_raw(root)
    panel = build_panel({}, raw_root=root, out_path=tmp_path / "w.parquet", daily_out_path=tmp_path / "d.parquet")

    vix_daily = pd.Series(np.arange(1, len(dates) + 1, dtype=float), index=dates)
    for _, row in panel.iterrows():
        prev_trading_day = dates[dates < row["date"]][-1]
        assert row["vix"] == vix_daily[prev_trading_day], f"{row['date']:%Y-%m-%d} 的 vix 不等于前一天"


def test_build_panel_rejects_missing_raw_file(tmp_path):
    root = tmp_path / "raw"
    _fake_raw(root)
    (root / "fred" / "DTB3.csv").unlink()

    with pytest.raises(FileNotFoundError, match="DTB3"):
        build_panel({}, raw_root=root, out_path=tmp_path / "w.parquet", daily_out_path=tmp_path / "d.parquet")


def test_daily_panel_semantics_match_weekly(tmp_path):
    """日/周两个面板的 open/high/low/close 必须是同一个口径（都是复权价）。"""
    root = tmp_path / "raw"
    _fake_raw(root)
    weekly = build_panel({}, raw_root=root, out_path=tmp_path / "w.parquet", daily_out_path=tmp_path / "d.parquet")
    daily = pd.read_parquet(tmp_path / "d.parquet")

    assert list(daily.columns) == [
        "date",
        "f",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "close_raw",
        "volume",
        "vix",
        "dgs10",
        "dtb3",
    ]
    # close 是复权价（与 adj_close 相等），close_raw 是原始价（更高）
    assert (daily["close"] == daily["adj_close"]).all()
    assert (daily["close_raw"] > daily["close"]).all()
    assert (daily["f"] == daily["close"] / daily["close_raw"]).all()

    # 周面板的 close 必须等于该周日频 close 的最后一天
    merged = weekly.merge(daily[["date", "close"]], on="date", suffixes=("_w", "_d"))
    assert len(merged) == len(weekly)
    assert (merged["close_w"] == merged["close_d"]).all()


def test_build_panel_detects_ohlc_violation(tmp_path):
    root = tmp_path / "raw"
    _fake_raw(root)
    spy = pd.read_csv(root / "yahoo" / "SPY.csv")
    spy.loc[0:4, "high"] = spy.loc[0:4, "low"] - 5  # 整周都坏，周频聚合后必须仍能抓到
    spy.to_csv(root / "yahoo" / "SPY.csv", index=False)

    with pytest.raises(AssertionError, match="OHLC"):
        build_panel({}, raw_root=root, out_path=tmp_path / "w.parquet", daily_out_path=tmp_path / "d.parquet")
