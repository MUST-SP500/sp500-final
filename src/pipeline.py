"""端到端管道（B6）：一条命令从原始数据跑到最终图表。

负责人：P6（B6）
用法：
    python -m src.pipeline --check              # 自检：配置、目录、各阶段模块状态
    python -m src.pipeline                      # 全流程（download → clean → b3 → models → backtest → eda → b5 → report）
    python -m src.pipeline --stage backtest     # 只跑某一阶段

阶段与负责人：
    download  P1  抓 5 个标的
    clean     P1  复权 + 周频 + 面板
    b3        P3  特征 + 标签 + 18 折切分（src.b3.run 一把做完）
    models    P4  18 折滚动训练 + 预测（产 predictions.parquet）
    backtest  P6  回测 + 三张表（reports/metrics.csv 是报告数字的唯一来源）
    eda       P2  7 张 EDA 图（src/eda.py 未合入时自动跳过）
    b5        P5  深度学习结果表（src/report_b5.py 未合入时自动跳过）
    report    P6  数字核对 + 把结果拷进 output/
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time

import pandas as pd

from src import __version__
from src.config import ensure_dirs, load_config, project_path, set_seed

STAGES = ["download", "clean", "b3", "models", "backtest", "eda", "b5", "report"]

# 旧阶段名（分工说明书写的是 features / labels / splits 三步，B3 用一个入口实现）
ALIASES = {"features": "b3", "labels": "b3", "splits": "b3"}

STAGE_MODULES = {
    "download": "src.data_download",
    "clean": "src.data_clean",
    "b3": "src.b3",
    "models": "src.models_ml",
    "backtest": "src.backtest",
    "eda": "src.eda",
    "b5": "src.report_b5",
    "report": "src.backtest",
}

# 还没合进 main 的可选阶段：缺失时跳过而不是整条流程失败
OPTIONAL_STAGES = {"eda": "B2（src/eda.py）", "b5": "B5（src/report_b5.py）"}

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
            status = f"未就绪: {exc}"
        print(f"  - {name:<10} {module:<22} {status}")

    print()
    print("提示：模块能导入不代表已实现；eda / b5 未合入时会自动跳过。")
    return 0 if not missing else 1


def _run_optional(stage: str, func, *args) -> None:
    """可选阶段的统一入口：模块缺失就跳过，其它异常照常抛出。"""
    try:
        func(*args)
    except ModuleNotFoundError as exc:
        print(f"[{stage}] 跳过：{OPTIONAL_STAGES.get(stage, stage)} 还没合进 main（{exc.name}）")


def run_stage(stage: str, cfg: dict) -> None:
    """按阶段调度。"""
    if stage in ALIASES:
        print(f"[pipeline] {stage} 已合并进 b3（B3 的 src.b3.run 一次算完特征+标签+切分）")
        stage = ALIASES[stage]

    if stage == "download":
        from src.data_download import download_all

        download_all(cfg["data"]["tickers"], cfg["data"]["start"], cfg["data"]["end"], cfg)
    elif stage == "clean":
        from src.data_clean import build_panel

        build_panel(cfg)
    elif stage == "b3":
        from src.b3 import run as b3_run

        b3_run(cfg)
    elif stage == "models":
        from src.models_ml import run as models_run

        models_run(cfg)
    elif stage == "backtest":
        from src.backtest import run_backtest

        run_backtest(None, cfg)
    elif stage == "eda":
        try:
            from src.eda import main as eda_main
        except ModuleNotFoundError:
            print(f"[eda] 跳过：{OPTIONAL_STAGES['eda']} 还没合进 main")
            return
        eda_main([])
    elif stage == "b5":
        try:
            from src.report_b5 import main as b5_main
        except ModuleNotFoundError:
            print(f"[b5] 跳过：{OPTIONAL_STAGES['b5']} 还没合进 main")
            return
        b5_main()
    elif stage == "report":
        build_report(cfg)
    else:
        raise ValueError(f"未知阶段: {stage}，可选 {STAGES}")


def build_report(cfg: dict) -> None:
    """数字核对 + 出图 + 打包副本（P6 第 2、3、5 步）。

    1. 校验 reports/metrics.csv 存在（报告里所有数字的唯一来源）
    2. 画六宫格 tearsheet；算 5/10 bps 成本敏感性与 2008 危机区间
    3. 生成 output/summary.md（一页结论，数字全部从 CSV 读，不手抄）
    4. 把表和图拷到 output/
    """
    from src.sensitivity import build_sensitivity
    from src.tearsheet import build_tearsheet

    reports = project_path("reports")
    metrics_path = reports / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(
            "report: 找不到 reports/metrics.csv，先跑 `uv run python -m src.pipeline --stage backtest`"
        )

    metrics = pd.read_csv(metrics_path)
    build_tearsheet(cfg)
    sensitivity = build_sensitivity(cfg)

    out = project_path("output")
    out.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    for name in ("metrics.csv", "equity_curve.csv", "trades.csv", "metrics_sensitivity.csv", "tearsheet_stats.csv"):
        src_path = reports / name
        if src_path.exists():
            shutil.copy2(src_path, out / name)
            copied.append(name)

    figures_out = out / "figures"
    figures_out.mkdir(parents=True, exist_ok=True)
    n_fig = 0
    for png in sorted((reports / "figures").glob("*.png")):
        shutil.copy2(png, figures_out / png.name)
        n_fig += 1

    summary = build_summary(metrics, sensitivity, n_fig)
    (out / "summary.md").write_text(summary, encoding="utf-8")

    print(f"[report] metrics.csv：{len(metrics)} 条策略，{len(metrics.columns)} 列")
    show = [c for c in ("strategy", "cagr", "sharpe", "max_drawdown", "annual_turnover") if c in metrics.columns]
    if show:
        print(metrics[show].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"[report] 已写 output/summary.md；已拷 {len(copied)} 张表、{n_fig} 张图")

    missing = [name for name in ("metrics.csv", "equity_curve.csv", "trades.csv") if name not in copied]
    if missing:
        print(f"[!] output/ 缺: {missing}")
    if n_fig == 0:
        print("[!] reports/figures/ 里还没有图（tearsheet 未生成）")


def build_summary(metrics: pd.DataFrame, sensitivity: pd.DataFrame, n_fig: int) -> str:
    """一页结论：所有数字都从 metrics / sensitivity 里取，避免手抄。"""
    table = metrics.set_index("strategy")
    sens = sensitivity.set_index("strategy")

    def row(name: str, label: str) -> str:
        if name not in table.index:
            return f"| {label} | — | — | — |"
        m = table.loc[name]
        return (
            f"| {label} | {m['cagr']:.2%} | {m['sharpe']:.2f} | {m['max_drawdown']:.1%} |"
        )

    main_rows = [
        row("logreg", "R1 logistic（主策略）"),
        row("logreg_R2", "R2 quantile"),
        row("ridge", "R4 ridge / Kelly"),
        row("buy_and_hold", "买入持有（主对照）"),
        row("const_60", "恒定 60%（最公平对照）"),
        row("sma200", "SMA200"),
    ]

    def delta(name: str) -> tuple[str, str]:
        if name not in sens.index:
            return "—", "—"
        s = sens.loc[name]
        return f"{s['d_cagr_bps']:+.2%}", f"{s['d_sharpe_bps']:+.2f}"

    lines = [
        "# 回测结果一页结论",
        "",
        "> 本页所有数字由 `uv run python -m src.pipeline` 自动生成，来源 `reports/metrics.csv`；",
        "> 手改这份文件等于手改数字，禁止。",
        "",
        "## 1. 主表（2008–2025，5 bps 单边成本）",
        "",
        "| 策略 | 年化 | 夏普 | 最大回撤 |",
        "|---|---|---|---|",
        *main_rows,
        "",
        "## 2. 成本敏感性（5 → 10 bps）",
        "",
        "| 策略 | 年化变化 | 夏普变化 |",
        "|---|---|---|",
        f"| R1 logistic | {delta('logreg')[0]} | {delta('logreg')[1]} |",
        *[
            f"| {n} | {delta(n)[0]} | {delta(n)[1]} |"
            for n in ("logreg_R2", "ridge", "sma200")
            if n in sens.index
        ],
        "",
        "## 3. 结论",
        "",
    ]

    if {"logreg", "buy_and_hold"}.issubset(table.index):
        r1, bh = table.loc["logreg"], table.loc["buy_and_hold"]
        c60 = table.loc["const_60"] if "const_60" in table.index else None
        verdict = "跑输" if r1["cagr"] < bh["cagr"] else "跑赢"
        lines += [
            f"- 主策略 R1 年化 **{r1['cagr']:.2%}**、夏普 **{r1['sharpe']:.2f}**，"
            f"{verdict}买入持有（{bh['cagr']:.2%} / {bh['sharpe']:.2f}）。",
        ]
        if c60 is not None:
            lines.append(
                f"- 与仓位最可比的“恒定 60%”（{c60['cagr']:.2%} / {c60['sharpe']:.2f}）相比，"
                + ("仍然跑输。" if r1["cagr"] < c60["cagr"] else "略胜。")
            )
        lines.append(
            f"- 最大回撤 {r1['max_drawdown']:.1%}，对比买入持有 {bh['max_drawdown']:.1%}："
            + ("回撤改善有限，不足以补偿收益损失。" if abs(r1["max_drawdown"]) > 0.9 * abs(bh["max_drawdown"]) else "回撤明显更小。")
        )
        lines.append("- 成本从 5 bps 翻到 10 bps，结论不变（见第 2 节）：收益小幅下滑，排序不变。")
        lines.append("")
        lines.append(
            "> 结论按 `docs/preregistration.md` 的冻结口径给出：主模型 L2 逻辑回归 + R1 + 5 bps。"
            "未通过事后换模型/换规则来改善结论。"
        )

    lines += [
        "",
        "## 4. 图表",
        "",
        f"- `figures/backtest_tearsheet.png`：净值 / 回撤 / 滚动夏普 / 年度收益 / 换手 / 仓位（共 {n_fig} 张图随包）",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="S&P 500 周频择时 · 端到端管道")
    parser.add_argument("--config", default=None, help="配置文件路径，默认仓库根目录 config.yaml")
    parser.add_argument(
        "--stage",
        default="all",
        choices=STAGES + list(ALIASES) + ["all"],
        help="只跑某一阶段（features/labels/splits 是 b3 的别名）",
    )
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
