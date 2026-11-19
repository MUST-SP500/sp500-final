"""公共 fixture：合成数据，保证测试不依赖网络和真实行情。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def trading_days() -> pd.DatetimeIndex:
    """2024 年全年的工作日（够用即可，不追求真实交易日历）。"""
    return pd.bdate_range("2024-01-01", "2024-12-31")


@pytest.fixture
def ohlc(trading_days: pd.DatetimeIndex) -> pd.DataFrame:
    """一份合法的合成 OHLCV。"""
    rng = np.random.default_rng(42)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(trading_days))))
    open_ = close * (1 + rng.normal(0, 0.002, len(trading_days)))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.003, len(trading_days))))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.003, len(trading_days))))
    return pd.DataFrame(
        {
            "date": trading_days,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.integers(1_000_000, 5_000_000, len(trading_days)).astype(float),
        }
    )
