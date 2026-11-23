"""下载层测试（B1）：全部离线，用假的 yfinance / requests 验证解析、缓存、重试。

原则：测试不打真网。真网行为（限流、改版）由 manifest 的 sha256 与下载日期记录，
      不靠单元测试兜。
"""

from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

from src import data_download as dd


# --------------------------------------------------------------------------- #
# 假数据
# --------------------------------------------------------------------------- #
def _yahoo_frame(ticker: str = "SPY", periods: int = 5) -> pd.DataFrame:
    """模仿 ``Ticker.history()`` 的返回：单层列头 + Date 索引。"""
    idx = pd.bdate_range("2024-01-02", periods=periods, name="Date")
    base = [100.0 + i for i in range(periods)]
    return pd.DataFrame(
        {
            "Open": [b for b in base],
            "High": [b + 2 for b in base],
            "Low": [b - 1 for b in base],
            "Close": [b + 1 for b in base],
            "Adj Close": [b + 0.5 for b in base],
            "Volume": [1000 + i for i in range(periods)],
        },
        index=idx,
    )


def _multiindex_frame() -> pd.DataFrame:
    """老接口 ``yf.download`` 的多层列头，扁平化函数要能吃下这种。"""
    flat = _yahoo_frame()
    flat.columns = pd.MultiIndex.from_product([list(flat.columns), ["SPY"]])
    return flat


def _fake_yfinance(fn) -> types.ModuleType:
    """假 yfinance：只提供 ``Ticker(symbol).history(...)`` 这一条路径。"""
    mod = types.ModuleType("yfinance")

    class _Ticker:
        def __init__(self, symbol, *args, **kwargs):
            self.symbol = symbol

        def history(self, *args, **kwargs):
            return fn(self.symbol, *args, **kwargs)

    mod.Ticker = _Ticker  # type: ignore[attr-defined]
    return mod


# --------------------------------------------------------------------------- #
# 解析
# --------------------------------------------------------------------------- #
def test_flatten_multiindex_columns():
    flat = dd._flatten_columns(_multiindex_frame())
    assert list(flat.columns) == ["open", "high", "low", "close", "adj_close", "volume"]


def test_download_ticker_rejects_duplicate_columns(tmp_path: Path, monkeypatch):
    """多标的串表时会出重复列名，这种情况必须报错而不是悄悄写出去。"""
    bad = _yahoo_frame()
    bad = pd.concat([bad, bad], axis=1)
    monkeypatch.setitem(sys.modules, "yfinance", _fake_yfinance(lambda *a, **k: bad))

    with pytest.raises(RuntimeError, match="列名重复"):
        dd.download_ticker("SPY", "2024-01-01", "2024-01-31", tmp_path)


def test_sha256_matches_file(tmp_path: Path):
    p = tmp_path / "a.csv"
    p.write_text("date,close\n2024-01-02,100\n", encoding="utf-8")
    assert dd._sha256(p) == hashlib.sha256(p.read_bytes()).hexdigest()


def test_download_ticker_writes_normalized_csv(tmp_path: Path, monkeypatch):
    monkeypatch.setitem(sys.modules, "yfinance", _fake_yfinance(lambda *a, **k: _yahoo_frame()))
    path = dd.download_ticker("SPY", "2024-01-01", "2024-01-31", tmp_path)

    assert path.name == "SPY.csv"
    df = pd.read_csv(path)
    assert list(df.columns) == ["date", "open", "high", "low", "close", "adj_close", "volume"]
    assert df["date"].is_monotonic_increasing
    assert df["close"].notna().all()


def test_download_ticker_strips_caret_from_filename(tmp_path: Path, monkeypatch):
    monkeypatch.setitem(sys.modules, "yfinance", _fake_yfinance(lambda *a, **k: _yahoo_frame("^VIX")))
    path = dd.download_ticker("^VIX", "2024-01-01", "2024-01-31", tmp_path)
    assert path.name == "VIX.csv"


def test_fred_series_parses_dot_as_missing(tmp_path: Path, monkeypatch):
    csv = "observation_date,DTB3\n1996-01-02,5.02\n1996-01-03,.\n1996-01-04,5.10\n"

    class FakeResp:
        text = csv

        def raise_for_status(self):
            return None

    monkeypatch.setattr(dd.requests, "get", lambda *a, **k: FakeResp())
    path = dd.download_fred_series("DTB3", "1996-01-01", "1996-12-31", tmp_path)

    df = pd.read_csv(path)
    assert list(df.columns) == ["date", "dtb3"]
    assert len(df) == 2, "缺失值 '.' 必须被丢掉"
    assert df["dtb3"].tolist() == [5.02, 5.10]


# --------------------------------------------------------------------------- #
# 缓存与重试
# --------------------------------------------------------------------------- #
def test_cached_file_skips_network(tmp_path: Path, monkeypatch):
    calls = {"n": 0}

    def counting(*a, **k):
        calls["n"] += 1
        return _yahoo_frame()

    monkeypatch.setitem(sys.modules, "yfinance", _fake_yfinance(counting))
    # 区间与假数据完全一致（2024-01-02 ~ 2024-01-08），第二次才会命中缓存
    dd.download_ticker("SPY", "2024-01-02", "2024-01-08", tmp_path)
    assert calls["n"] == 1

    def boom(*a, **k):
        raise AssertionError("缓存命中时不该再发网络请求")

    monkeypatch.setitem(sys.modules, "yfinance", _fake_yfinance(boom))
    path = dd.download_ticker("SPY", "2024-01-02", "2024-01-08", tmp_path)
    assert path.exists()


def test_cache_covers_when_start_is_not_a_trading_day(tmp_path: Path, monkeypatch):
    """请求起点是元旦/周末（比首个交易日早）时，缓存仍应命中。"""
    monkeypatch.setitem(sys.modules, "yfinance", _fake_yfinance(lambda *a, **k: _yahoo_frame()))
    dd.download_ticker("SPY", "2024-01-01", "2024-01-08", tmp_path)  # 数据从 01-02 开始

    def boom(*a, **k):
        raise AssertionError("起点不是交易日不该导致重新下载")

    monkeypatch.setitem(sys.modules, "yfinance", _fake_yfinance(boom))
    assert dd.download_ticker("SPY", "2024-01-01", "2024-01-08", tmp_path).exists()


def test_refresh_forces_redownload(tmp_path: Path, monkeypatch):
    calls = {"n": 0}
    def counting(*a, **k):
        calls["n"] += 1
        return _yahoo_frame()

    monkeypatch.setitem(sys.modules, "yfinance", _fake_yfinance(counting))
    dd.download_ticker("SPY", "2024-01-01", "2024-12-31", tmp_path)
    dd.download_ticker("SPY", "2024-01-01", "2024-12-31", tmp_path, refresh=True)
    assert calls["n"] == 2


def test_retry_succeeds_on_third_attempt(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(dd, "BACKOFF_BASE", 0.0)
    attempts = {"n": 0}

    def flaky(*a, **k):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ConnectionError("模拟网络抖动")
        return _yahoo_frame()

    monkeypatch.setitem(sys.modules, "yfinance", _fake_yfinance(flaky))
    dd.download_ticker("SPY", "2024-01-01", "2024-12-31", tmp_path)
    assert attempts["n"] == 3


def test_retry_gives_up_after_three_attempts(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(dd, "BACKOFF_BASE", 0.0)
    attempts = {"n": 0}

    def always_fail(*a, **k):
        attempts["n"] += 1
        raise ConnectionError("一直失败")

    monkeypatch.setitem(sys.modules, "yfinance", _fake_yfinance(always_fail))
    with pytest.raises(RuntimeError, match="重试 3 次仍失败"):
        dd.download_ticker("SPY", "2024-01-01", "2024-12-31", tmp_path)
    assert attempts["n"] == dd.MAX_ATTEMPTS


# --------------------------------------------------------------------------- #
# manifest
# --------------------------------------------------------------------------- #
def test_write_manifest_records_every_file(tmp_path: Path):
    raw = tmp_path / "raw"
    (raw / "yahoo").mkdir(parents=True)
    (raw / "fred").mkdir(parents=True)
    pd.DataFrame({"date": ["1996-01-02", "1996-01-09"], "close": [62.1, 63.0]}).to_csv(
        raw / "yahoo" / "SPY.csv", index=False
    )
    pd.DataFrame({"date": ["1996-01-02"], "dtb3": [5.02]}).to_csv(raw / "fred" / "DTB3.csv", index=False)

    cfg = {"data": {"tickers": {"spy": "SPY", "dtb3": "DTB3"}, "raw_dir": "data/raw"}}
    path = dd.write_manifest(cfg, raw_root=raw)

    payload = json.loads(path.read_text(encoding="utf-8"))
    files = {row["file"]: row for row in payload["files"]}
    assert set(files) == {"yahoo/SPY.csv", "fred/DTB3.csv"}
    assert files["yahoo/SPY.csv"]["rows"] == 2
    assert files["yahoo/SPY.csv"]["date_range"] == "1996-01-02 ~ 1996-01-09"
    assert files["yahoo/SPY.csv"]["sha256"] == dd._sha256(raw / "yahoo" / "SPY.csv")
    assert files["fred/DTB3.csv"]["source"] == "fred"
    assert "fredgraph" in files["fred/DTB3.csv"]["url"]


def test_manifest_restores_caret_symbol(tmp_path: Path):
    raw = tmp_path / "raw" / "yahoo"
    raw.mkdir(parents=True)
    pd.DataFrame({"date": ["2024-01-02"], "close": [13.4]}).to_csv(raw / "VIX.csv", index=False)

    cfg = {"data": {"tickers": {"vix": "^VIX"}, "raw_dir": "data/raw"}}
    payload = json.loads(dd.write_manifest(cfg, raw_root=tmp_path / "raw").read_text(encoding="utf-8"))

    assert payload["files"][0]["ticker"] == "^VIX", "文件名去 ^ 但 manifest 里要还原"
    assert payload["files"][0]["file"] == "yahoo/VIX.csv"
