"""Wikipedia 成分股爬虫测试（B1 闸门 4）——全部离线，用 tests/fixtures 的静态页面。"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src import scrape_wiki as sw

FIXTURE = Path(__file__).parent / "fixtures" / "sp500_wiki.html"


@pytest.fixture()
def html() -> str:
    return FIXTURE.read_text(encoding="utf-8")


def test_parse_picks_constituent_table_not_navbox(html):
    raw = sw.parse_current_components(html)
    assert len(raw) == 3
    assert "GICS Sector" in raw.columns


def test_clean_normalizes_columns_and_strips_footnotes(html):
    df = sw.clean_components(sw.parse_current_components(html))

    assert list(df.columns) == sw.OUTPUT_COLUMNS
    assert df["symbol"].tolist() == ["AAPL", "BRK.B", "MMM"]  # 排序 + 大写 + 去 [1]
    assert df.loc[df["symbol"] == "BRK.B", "date_added"].iloc[0] == pd.Timestamp("2010-02-16")
    assert str(df["cik"].dtype) == "Int64"
    assert int(df.loc[df["symbol"] == "MMM", "cik"].iloc[0]) == 66740
    assert df["gics_sector"].notna().all()


def test_clean_rejects_duplicate_symbols(html):
    raw = sw.parse_current_components(html)
    raw.loc[len(raw)] = raw.iloc[0]
    with pytest.raises(ValueError, match="重复"):
        sw.clean_components(raw)


def test_clean_rejects_missing_column(html):
    raw = sw.parse_current_components(html).drop(columns=["CIK"])
    with pytest.raises(ValueError, match="缺少列"):
        sw.clean_components(raw)


def test_snapshot_is_reused_without_network(tmp_path, monkeypatch, html):
    """本地已有快照时不应再打网络。"""
    csv_path = tmp_path / "wiki" / "constituents.csv"
    sw.save_snapshot(sw.clean_components(sw.parse_current_components(html)), csv_path)

    def boom(*args, **kwargs):
        raise AssertionError("不该联网")

    monkeypatch.setattr(sw, "fetch_html", boom)
    cfg = {"data": {"raw_dir": str(tmp_path)}}
    df = sw.fetch_current_components(cfg, path=csv_path)
    assert len(df) == 3


def test_refresh_fetches_and_updates_manifest(tmp_path, monkeypatch, html):
    monkeypatch.setattr(sw, "fetch_html", lambda *a, **k: html)
    monkeypatch.setattr(sw, "project_path", lambda p: tmp_path / p if not str(p).startswith(str(tmp_path)) else Path(p))

    cfg = {"data": {"raw_dir": str(tmp_path)}}
    df = sw.fetch_current_components(cfg, refresh=True, path=tmp_path / "wiki" / "constituents.csv")

    assert len(df) == 3
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    entries = [f for f in manifest["files"] if f["ticker"] == "SP500_CONSTITUENTS"]
    assert len(entries) == 1
    assert entries[0]["rows"] == 3
    assert entries[0]["source"] == "wikipedia"
    assert len(entries[0]["sha256"]) == 64


def test_cache_hit_restores_missing_manifest_entry(tmp_path, monkeypatch, html):
    """下载层重写 manifest 会挤掉 wiki 条目，缓存命中时必须补登回来。"""
    csv_path = tmp_path / "wiki" / "constituents.csv"
    sw.save_snapshot(sw.clean_components(sw.parse_current_components(html)), csv_path)
    (tmp_path / "manifest.json").write_text(
        json.dumps({"generated_at": "2026-11-23T00:00:00", "files": [{"file": "yahoo/SPY.csv", "rows": 7550}]}),
        encoding="utf-8",
    )

    monkeypatch.setattr(sw, "fetch_html", lambda *a, **k: (_ for _ in ()).throw(AssertionError("不该联网")))
    cfg = {"data": {"raw_dir": str(tmp_path)}}
    sw.fetch_current_components(cfg, path=csv_path)

    files = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))["files"]
    assert any(f["file"] == "wiki/constituents.csv" for f in files)
    assert any(f["file"] == "yahoo/SPY.csv" for f in files), "已有条目不能被抹掉"
