"""B2（EDA）的测试：不依赖真实行情，全部用合成数据。

覆盖三件事：
  1. 统计小工具的口径（回撤、VIF、正态分位）；
  2. 7 张图在合成数据上都能落地（防止改样式时静默坏掉）；
  3. PCA 图必须在标题上标明"全样本描述性"，防止被误用成特征选择。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import eda
from src.features import feature_columns


@pytest.fixture
def daily(trading_days: pd.DatetimeIndex) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.011, len(trading_days))))
    return pd.DataFrame({"date": trading_days, "close": close})


@pytest.fixture
def weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """每 5 个交易日取一行，模拟"周频"，避免 resample 的时区/频率告警。"""
    return daily.iloc[::5].reset_index(drop=True)


@pytest.fixture
def panel(weekly: pd.DataFrame) -> pd.DataFrame:
    """最小的 model_panel：22 个特征 + VIX + 标签 + 两个开关列。"""
    rng = np.random.default_rng(11)
    n = len(weekly)
    out = weekly.copy()
    for i, col in enumerate(feature_columns()):
        out[col] = rng.normal(0, 1, n) + 0.01 * i
    out["vix_level"] = np.abs(rng.normal(18, 5, n))
    out["label_ret"] = rng.normal(0.002, 0.02, n)
    out["label_up"] = (out["label_ret"] > 0).astype(int)
    out["history_ready"] = True
    out["label_available"] = True
    return out


@pytest.fixture
def constituents() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    sectors = ["Industrials", "Financials", "Information Technology", "Energy"]
    return pd.DataFrame(
        {
            "symbol": [f"T{i}" for i in range(40)],
            "gics_sector": rng.choice(sectors, 40),
            "date_added": pd.to_datetime("1990-01-01")
            + pd.to_timedelta(rng.integers(0, 35 * 365, 40), unit="D"),
        }
    )


def test_max_drawdown_handles_simple_series():
    s = pd.Series([100.0, 120.0, 60.0, 90.0, 130.0], index=pd.date_range("2020-01-01", periods=5))
    dd, trough, recovery = eda._max_drawdown(s)
    assert dd == pytest.approx(-0.5)
    assert trough == pd.Timestamp("2020-01-03")
    assert recovery == pd.Timestamp("2020-01-05")


def test_vif_flags_exact_linear_dependency():
    rng = np.random.default_rng(0)
    x1 = rng.normal(size=200)
    x2 = rng.normal(size=200)
    x4 = rng.normal(size=200)
    frame = pd.DataFrame({"a": x1, "b": x2, "c": x1 + x2, "d": x4})
    vif = eda._vif(frame)
    # a/b/c 互相精确线性表出 → 三者都是 inf；独立的 d 仍有限
    assert np.isinf(vif["a"]) and np.isinf(vif["b"]) and np.isinf(vif["c"])
    assert np.isfinite(vif["d"])
    assert vif["d"] == pytest.approx(1.0, abs=0.05)


def test_normal_quantiles_are_symmetric():
    q = eda._normal_quantiles(101)
    assert q[50] == pytest.approx(0.0, abs=1e-9)
    assert q[0] == pytest.approx(-q[-1], abs=1e-9)


def test_plot_market_and_distribution_write_files(daily, weekly, tmp_path):
    s1 = eda.plot_market(daily, out=tmp_path / "m.png")
    s2 = eda.plot_distribution(daily, weekly, out=tmp_path / "d.png")
    assert (tmp_path / "m.png").stat().st_size > 10_000
    assert (tmp_path / "d.png").stat().st_size > 10_000
    assert s1["max_drawdown_pct"] < 0
    assert s2["weekly_skew"] == pytest.approx(s2["weekly_skew"])  # 不是 NaN


def test_plot_volatility_and_vix(daily, panel, tmp_path):
    eda.plot_volatility(daily, out=tmp_path / "v.png")
    stats = eda.plot_vix(panel, out=tmp_path / "vix.png")
    assert (tmp_path / "v.png").exists() and (tmp_path / "vix.png").exists()
    assert len(stats["quintile_means_pct"]) == 5
    assert stats["n_weeks"] > 0


def test_plot_correlation_reports_rank(panel, tmp_path):
    stats = eda.plot_correlation(panel, out=tmp_path / "c.png")
    assert (tmp_path / "c.png").exists()
    assert stats["n_features"] == len(feature_columns())
    assert 0 < stats["matrix_rank"] <= stats["n_features"]


def test_plot_sectors_and_pca(panel, constituents, tmp_path):
    s1 = eda.plot_sectors(constituents, out=tmp_path / "s.png")
    s2 = eda.plot_pca(panel, out=tmp_path / "p.png")
    assert (tmp_path / "s.png").exists() and (tmp_path / "p.png").exists()
    assert s1["n_constituents"] == len(constituents)
    assert s2["n_components_90pct"] >= 1


def test_pca_title_marks_full_sample():
    """PCA 图必须自称 descriptive (full-sample)，否则容易被当成特征选择。"""
    src = (eda.__file__ and open(eda.__file__, encoding="utf-8").read()) or ""
    assert "descriptive (full-sample)" in src


def test_make_all_raises_readable_error_when_data_missing(monkeypatch):
    monkeypatch.setattr(eda, "load_data", lambda cfg=None: {})
    with pytest.raises(FileNotFoundError, match="pipeline"):
        eda.make_all()
