"""复现校验器测试（B1 闸门 5）——只测纯逻辑，不跑网络、不删文件。"""

from __future__ import annotations

from src.reproduce import compare_metrics


def _metrics(rows=1566, first_close=36.30553, last_close=676.635) -> dict:
    panel = {
        "rows": rows,
        "cols": 12,
        "first_date": "1996-01-05",
        "last_date": "2025-12-31",
        "first_close": first_close,
        "last_close": last_close,
        "volume_sum": 690091047500.0,
        "na_total": 0,
    }
    return {"weekly": dict(panel), "daily": dict(panel, rows=7550), "files": {"yahoo/SPY.csv": 7550}}


def test_identical_metrics_all_pass():
    rows = compare_metrics(_metrics(), _metrics())
    assert all(r["verdict"] == "一致" for r in rows)


def test_row_count_difference_is_flagged():
    rows = compare_metrics(_metrics(), _metrics(rows=1567))
    bad = [r for r in rows if r["verdict"] != "一致"]
    assert [r["item"] for r in bad] == ["weekly.rows"]


def test_close_drift_within_tolerance_passes():
    """雅虎 adj_close 会漂 1e-4，5e-4 以内视为一致。"""
    rows = compare_metrics(_metrics(first_close=676.635), _metrics(first_close=676.635 * (1 + 1e-5)))
    assert [r["verdict"] for r in rows if r["item"] == "weekly.first_close"] == ["一致"]


def test_close_drift_beyond_tolerance_is_flagged():
    rows = compare_metrics(_metrics(last_close=676.635), _metrics(last_close=677.5))
    verdict = [r["verdict"] for r in rows if r["item"] == "weekly.last_close"][0]
    assert verdict.startswith("偏差")


def test_raw_file_count_mismatch_is_flagged():
    after = _metrics()
    after["files"]["yahoo/SPY.csv"] = 7549
    rows = compare_metrics(_metrics(), after)
    bad = [r for r in rows if r["verdict"] != "一致"]
    assert [r["item"] for r in bad] == ["raw.yahoo/SPY.csv"]
