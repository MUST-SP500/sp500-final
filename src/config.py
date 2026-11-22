"""全局配置：读取 config.yaml、设置随机种子、提供可移植路径。

负责人：P1（B1）
其它模块一律 `from src.config import load_config, project_path`，不要各自读 yaml。
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import yaml

# 仓库根目录（src/ 的上一级），无论从哪个目录启动脚本都能定位
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """读取并返回配置字典；`_root` / `_config_path` 为解析后的绝对路径。"""
    cfg_path = Path(path) if path is not None else PROJECT_ROOT / "config.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["_root"] = str(PROJECT_ROOT)
    cfg["_config_path"] = str(cfg_path)
    return cfg


# 运行期目录：git 不保存空目录，clone 下来可能缺失，需要时自动创建
RUNTIME_DIRS = [
    "data/raw",
    "data/raw/yahoo",
    "data/raw/fred",
    "data/raw/wiki",
    "data/processed",
    "reports/figures",
    "notebooks",
    "slides",
    "output",
    "output/figures",
]


def project_path(*parts: str) -> Path:
    """相对仓库根目录拼路径，禁止在代码里写绝对路径。"""
    return PROJECT_ROOT.joinpath(*parts)


def ensure_dirs() -> list[Path]:
    """确保运行期目录存在，返回本次新建的目录。"""
    created: list[Path] = []
    for d in RUNTIME_DIRS:
        p = project_path(d)
        if not p.is_dir():
            p.mkdir(parents=True, exist_ok=True)
            created.append(p)
    return created


def set_seed(seed: int = 42) -> None:
    """固定随机种子。深度学习部分需再调用对应的 torch 种子设置。"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy 是本项目硬依赖
        pass
