"""超参冻结（B6 补）：保证 docs/hyperparams.md 一旦写下就不再被改写。

背景：`run()` 原本每次都会重新选参，同一份数据重跑会得到不同的 C，
导致报告里的年化/最大回撤漂移。这里把“读回已冻结值”的行为钉住。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.models_ml import read_frozen_hyperparams

TABLE = """# 冻结超参数记录表

| 模型 | 参数 | 搜索范围 | 选中值 | 验证集指标 | 选定日期 |
|---|---|---|---|---|---|
| logreg | C（L2 正则强度） | 0.01 / 0.1 / 1.0 / 10.0 | 10.0 | Val AUC = 0.5687 | 2026-11-24 |
| logreg | R1 阈值 | 预注册固定 0.50 | 0.50 | 预注册主规则 | 2026-11-24 |
| ridge | alpha（L2 正则强度） | 0.1 / 1 / 10 | 0.1 | Val RMSE = 0.01694 | 2026-11-24 |
| ridge (R4) | kappa（Kelly 缩放系数） | 0.05 / 0.1 / 0.5 | 0.1 | Val Net Sharpe = 2.46 | 2026-11-24 |
| rf | max_depth（树深） | 3 / 5 / 8 | 8 | Val AUC = 0.5171 | 2026-11-24 |
"""


def test_reads_selected_values_from_table(tmp_path: Path):
    path = tmp_path / "hyperparams.md"
    path.write_text(TABLE, encoding="utf-8")

    frozen = read_frozen_hyperparams(path)

    assert frozen is not None
    assert frozen["logreg"]["C"] == 10.0
    assert frozen["ridge"]["alpha"] == 0.1
    assert frozen["ridge"]["kappa"] == 0.1
    assert frozen["rf"]["max_depth"] == 8
    assert frozen["rf"]["n_estimators"] == 200


def test_missing_file_returns_none(tmp_path: Path):
    assert read_frozen_hyperparams(tmp_path / "nope.md") is None


def test_incomplete_table_returns_none(tmp_path: Path):
    """缺一行就退回重选，不能拿半张表去跑 18 折。"""
    path = tmp_path / "hyperparams.md"
    path.write_text(TABLE.replace("| rf | max_depth（树深） | 3 / 5 / 8 | 8 | Val AUC = 0.5171 | 2026-11-24 |\n", ""),
                    encoding="utf-8")
    assert read_frozen_hyperparams(path) is None


@pytest.mark.parametrize("path", ["docs/hyperparams.md"])
def test_repo_table_is_parseable(path: str):
    """仓库里那份表必须能被读出来，否则每次运行都会重新选参。"""
    from src.config import project_path

    frozen = read_frozen_hyperparams(project_path(path))
    assert frozen is not None
    assert frozen["logreg"]["C"] > 0
