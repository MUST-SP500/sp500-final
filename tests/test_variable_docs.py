"""变量说明文档的一致性测试：代码里的列名必须都在 docs/variable_descriptions.md 里解释过。

这样改列名却忘了改文档时，CI 会直接失败。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src import data_clean, scrape_wiki
from src.data_quality import ANOMALY_KINDS

DOC = Path(__file__).resolve().parents[1] / "docs" / "variable_descriptions.md"


@pytest.fixture(scope="module")
def text() -> str:
    return DOC.read_text(encoding="utf-8")


def test_doc_exists(text):
    assert "变量说明总表" in text


def test_weekly_panel_columns_documented(text):
    missing = [c for c in data_clean.WEEKLY_COLUMNS if f"`{c}`" not in text]
    assert not missing, f"周面板列未在变量说明里解释：{missing}"


def test_daily_panel_columns_documented(text):
    missing = [c for c in data_clean.DAILY_COLUMNS if f"`{c}`" not in text]
    assert not missing, f"日面板列未在变量说明里解释：{missing}"


def test_wiki_columns_documented(text):
    missing = [c for c in scrape_wiki.OUTPUT_COLUMNS if f"`{c}`" not in text]
    assert not missing, f"成分股列未在变量说明里解释：{missing}"


def test_anomaly_kinds_documented(text):
    missing = [k for k in ANOMALY_KINDS if k not in text]
    assert not missing, f"异常类型未在变量说明里解释：{missing}"


def test_manifest_fields_documented(text):
    missing = [f for f in ("generated_at", "retrieved_at", "sha256", "date_range", "file") if f not in text]
    assert not missing, f"manifest 字段未解释：{missing}"


def test_raw_csv_columns_documented(text):
    """雅虎原始 CSV 的 7 列、FRED 的 2 列都要有说明。"""
    for col in ("date", "open", "high", "low", "close", "adj_close", "volume", "dgs10", "dtb3"):
        assert f"`{col}`" in text, f"原始 CSV 列 {col} 未说明"
