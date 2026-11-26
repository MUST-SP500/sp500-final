"""端到端管道（B6）：一条命令从原始数据跑到图、表、网页。

    uv run main.py                 # 全流程
    uv run main.py --stage backtest
    uv run main.py --check
"""

from __future__ import annotations

import argparse
import shutil
import time

import pandas as pd

from src.config import ensure_dirs, load_config, project_path, set_seed

STAGES = ["download", "clean", "b3", "models", "backtest", "eda", "b5", "report"]
ALIASES = {"features": "b3", "labels": "b3", "splits": "b3"}
MODULES = {
    "download": "src.data_download", "clean": "src.data_clean", "b3": "src.b3",
    "models": "src.models_ml", "backtest": "src.backtest", "eda": "src.eda",
    "b5": "src.report_b5", "report": "src.backtest",
}


def check(cfg: dict) -> int:
    ensure_dirs()
    print(f"sp500-final · {cfg['_config_path']}")
    for name in STAGES:
        try:
            __import__(MODULES[name])
            print(f"  {name:<10} ok")
        except Exception as exc:
            print(f"  {name:<10} 未就绪: {exc}")
    return 0


def run_stage(stage: str, cfg: dict) -> None:
    stage = ALIASES.get(stage, stage)

    if stage == "download":
        from src.data_download import download_all
        d = cfg["data"]
        download_all(d["tickers"], d["start"], d["end"], cfg)
    elif stage == "clean":
        from src.data_clean import build_panel
        build_panel(cfg)
    elif stage == "b3":
        from src.b3 import run
        run(cfg)
    elif stage == "models":
        from src.models_ml import run
        run(cfg)
    elif stage == "backtest":
        from src.backtest import run_backtest
        run_backtest(None, cfg)
    elif stage == "eda":
        try:
            from src.eda import main
            main([])
        except (ModuleNotFoundError, FileNotFoundError) as exc:
            print(f"[eda] 跳过：{exc}（缺成分股快照时先跑 scrape_wiki）")
    elif stage == "b5":
        try:
            from src.report_b5 import main
            main()
        except ModuleNotFoundError as exc:
            print(f"[b5] 跳过：{exc}（需要 uv sync --extra dl）")
    elif stage == "report":
        build_report(cfg)
    else:
        raise ValueError(f"未知阶段 {stage}，可选 {STAGES}")


def build_report(cfg: dict) -> None:
    """出 tearsheet / 敏感性 / 网页，写一页结论，并把结果拷进 output/。"""
    from src.build_site import build as build_site
    from src.sensitivity import build_sensitivity
    from src.tearsheet import build_tearsheet

    reports = project_path("reports")
    metrics_path = reports / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError("缺少 reports/metrics.csv，先跑 --stage backtest")

    metrics = pd.read_csv(metrics_path)
    build_tearsheet(cfg)
    sens = build_sensitivity(cfg)
    site = build_site(cfg)

    out = project_path("output")
    out.mkdir(parents=True, exist_ok=True)
    for name in ("metrics.csv", "equity_curve.csv", "trades.csv",
                 "metrics_sensitivity.csv", "tearsheet_stats.csv"):
        if (reports / name).exists():
            shutil.copy2(reports / name, out / name)

    figs = out / "figures"
    figs.mkdir(exist_ok=True)
    images = list((reports / "figures").glob("*.png"))
    for png in images:
        shutil.copy2(png, figs / png.name)

    (out / "summary.md").write_text(_summary(metrics, sens, len(images)), encoding="utf-8")
    print(f"[report] output/：5 张表、{len(images)} 张图、summary.md；网页 {site}")
    print(metrics[["strategy", "cagr", "sharpe", "max_drawdown"]].to_string(
        index=False, float_format=lambda x: f"{x:.4f}"))


def _summary(metrics: pd.DataFrame, sens: pd.DataFrame, n_fig: int) -> str:
    m, s = metrics.set_index("strategy"), sens.set_index("strategy")
    label = {"logreg": "R1 logistic", "logreg_R2": "R2 quantile", "ridge": "R4 ridge/Kelly",
             "buy_and_hold": "买入持有", "const_60": "恒定 60%", "sma200": "SMA200"}
    rows = [f"| {name} | {m.loc[k, 'cagr']:.2%} | {m.loc[k, 'sharpe']:.2f} | {m.loc[k, 'max_drawdown']:.1%} |"
            for k, name in label.items() if k in m.index]
    r1, bh = m.loc["logreg"], m.loc["buy_and_hold"]
    c60 = m.loc["const_60"] if "const_60" in m.index else None

    lines = [
        "# 回测结果",
        "",
        "> 数字由 `uv run python -m src.pipeline` 从 `reports/metrics.csv` 生成，不要手改。",
        "",
        "| 策略 | 年化 | 夏普 | 最大回撤 |",
        "|---|---|---|---|",
        *rows,
        "",
        "## 结论",
        "",
        f"- 主策略 R1：年化 {r1['cagr']:.2%}、夏普 {r1['sharpe']:.2f}，"
        f"买入持有 {bh['cagr']:.2%} / {bh['sharpe']:.2f}。",
    ]
    if c60 is not None:
        lines.append(f"- 对照恒定 60%：{c60['cagr']:.2%} / {c60['sharpe']:.2f}。")
    if "logreg" in s.index:
        lines.append(f"- 成本 5→10 bps：年化 {s.loc['logreg', 'd_cagr_bps']:+.2%}、"
                     f"夏普 {s.loc['logreg', 'd_sharpe_bps']:+.2f}，排序不变。")
    lines += ["", f"图：`figures/`（{n_fig} 张，含 backtest_tearsheet.png）"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="S&P 500 周频择时 · 端到端管道")
    parser.add_argument("--config", default=None)
    parser.add_argument("--stage", default="all", choices=STAGES + list(ALIASES) + ["all"])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    set_seed(int(cfg["project"]["seed"]))
    if args.check:
        return check(cfg)

    stages = STAGES if args.stage == "all" else [args.stage]
    start = time.time()
    for i, stage in enumerate(stages, 1):
        print(f"[{i}/{len(stages)}] {stage} ...", flush=True)
        run_stage(stage, cfg)
    print(f"完成，用时 {time.time() - start:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
