"""冒烟测试：仓库结构、配置、模块可导入。

这一组测试的意义：任何人 clone 下来，`pytest -q` 全绿就说明环境对了。
"""

from __future__ import annotations

import sys

import pytest

from src.config import PROJECT_ROOT, load_config, project_path, set_seed

EXPECTED_MODULES = [
    "src.config",
    "src.validate",
    "src.plot_style",
    "src.backtest",
    "src.data_download",
    "src.data_clean",
    "src.scrape_wiki",
    "src.features",
    "src.labels",
    "src.splits",
    "src.models_ml",
    "src.models_pca",
    "src.models_dl",
    "src.decision",
    "src.evaluate",
    "src.pipeline",
]

# 只校验 git 一定会保存的目录；data/notebooks 等空目录 git 不保存，
# 由 src.config.ensure_dirs() 在运行时创建（见 test_pipeline_check_creates_dirs）
EXPECTED_DIRS = ["src", "tests", "docs"]


def test_python_version():
    assert sys.version_info >= (3, 10), "需要 Python 3.10+"


def test_config_loads():
    cfg = load_config()
    assert cfg["project"]["seed"] == 42
    assert cfg["data"]["tickers"]["spy"] == "SPY"
    assert cfg["split"]["purge_weeks"] == 2
    assert cfg["backtest"]["cost_bps"] == 5.0
    assert cfg["_root"] == str(PROJECT_ROOT)


def test_project_dirs_exist():
    missing = [d for d in EXPECTED_DIRS if not project_path(d).is_dir()]
    assert not missing, f"缺少目录: {missing}"


def test_ensure_dirs_is_idempotent():
    """git 不保存空目录，所以运行期目录由代码兜底创建。"""
    from src.config import RUNTIME_DIRS, ensure_dirs

    ensure_dirs()
    assert all(project_path(d).is_dir() for d in RUNTIME_DIRS)
    assert ensure_dirs() == [], "第二次调用不应该再创建任何目录"


@pytest.mark.parametrize("module", EXPECTED_MODULES)
def test_module_importable(module):
    __import__(module)


def test_set_seed_is_repeatable():
    import numpy as np

    set_seed(7)
    a = np.random.rand()
    set_seed(7)
    b = np.random.rand()
    assert a == b


def test_plot_style_returns_palette():
    from src.plot_style import apply_style

    palette = apply_style()
    assert "primary" in palette
