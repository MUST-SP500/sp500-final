"""端到端管道（B6）：一条命令从原始数据跑到最终图表。

负责人：P6（B6）
用法：
    python -m src.pipeline --check              # 自检：配置、目录、模块状态
    python -m src.pipeline --config config.yaml # 全流程
    python -m src.pipeline --stage features     # 只跑某一阶段
"""

from __future__ import annotations

import argparse
import sys
import time

from src import __version__
from src.config import ensure_dirs, load_config, project_path, set_seed


STAGES = ["download", "clean", "features", "labels", "splits", "models", "backtest", "report"]

STAGE_MODULES = {
    "download": "src.data_download",
    "clean": "src.data_clean",
    "features": "src.features",
    "labels": "src.labels",
    "splits": "src.splits",
    "models": "src.models_ml",
    "backtest": "src.backtest",
    "report": "src.evaluate",
}

REQUIRED_DIRS = ["src", "tests", "docs", "notebooks", "reports/figures", "data/raw", "data/processed"]


def check(cfg: dict) -> int:
    """自检：不产生副作用，只报告环境是否就绪。"""
    print(f"sp500-final v{__version__}")
    print(f"config: {cfg['_config_path']}")
    print(f"root  : {cfg['_root']}")
    print(f"seed  : {cfg['project']['seed']}")
    print(f"区间  : {cfg['data']['start']} → {cfg['data']['end']}")
    print(f"成本  : {cfg['backtest']['cost_bps']} bps 单边")
    print()

    ensure_dirs()
    missing = [d for d in REQUIRED_DIRS if not (project_path(d)).is_dir()]
    if missing:
        print(f"[!] 缺少目录: {missing}")
    else:
        print("[ok] 目录结构完整")

    for name in STAGES:
        module = STAGE_MODULES[name]
        try:
            __import__(module)
            status = "可导入"
        except Exception as exc:  # pragma: no cover
            status = f"导入失败: {exc}"
        print(f"  - {name:<10} {module:<22} {status}")

    print()
    print("提示：模块能导入不代表已实现，未实现的函数会抛 NotImplementedError。")
    return 0 if not missing else 1


def run_stage(stage: str, cfg: dict) -> None:
    """按阶段调度。未实现的阶段会抛出 NotImplementedError 并指明负责人。"""
    if stage == "download":
        from src.data_download import download_all

        download_all(cfg["data"]["tickers"], cfg["data"]["start"], cfg["data"]["end"], cfg)
    elif stage == "clean":
        from src.data_clean import build_panel

        build_panel(cfg)
    elif stage == "features":
        from src.features import build_features

        build_features(None, cfg)
    elif stage == "labels":
        from src.labels import build_labels

        build_labels(None)
    elif stage == "splits":
        from src.splits import make_folds

        make_folds(None, cfg)
    elif stage == "models":
        from src.models_ml import walk_forward

        walk_forward(None, [], "logreg", cfg)
    elif stage == "backtest":
        from src.backtest import run_backtest

        run_backtest(None, cfg)
    elif stage == "report":
        raise NotImplementedError("B6：报告与图表汇总，待实现（P6）")
    else:
        raise ValueError(f"未知阶段: {stage}，可选 {STAGES}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="S&P 500 周频择时 · 端到端管道")
    parser.add_argument("--config", default=None, help="配置文件路径，默认仓库根目录 config.yaml")
    parser.add_argument("--stage", default="all", choices=STAGES + ["all"], help="只跑某一阶段")
    parser.add_argument("--check", action="store_true", help="自检环境，不跑数据")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    set_seed(int(cfg["project"]["seed"]))

    if args.check:
        return check(cfg)

    stages = STAGES if args.stage == "all" else [args.stage]
    t0 = time.time()
    for i, stage in enumerate(stages, 1):
        print(f"[{i}/{len(stages)}] {stage} ...", flush=True)
        run_stage(stage, cfg)
    print(f"完成，用时 {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
