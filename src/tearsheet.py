"""回测体检表（B6）：一张六宫格 tearsheet。

对应：SP500_分工说明书 P6 第 2 步
    六张小图、一个版式：净值（对数轴）/ 回撤 / 滚动一年夏普 / 年度收益 / 换手 / 仓位。
    回撤同时给周频与日频两个数（周频会低估真实回撤）。

数据来源：只读 reports/ 下 backtest 产出的三张表 + data/processed 的两个面板，
不自己重算仓位，保证和 metrics.csv 是同一套数字。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.backtest import BPS, TRADING_DAYS
from src.config import load_config, project_path
from src.plot_style import PALETTE, apply_style, savefig

# 报告主表里的六条线（B4 的 rf 是"另一个模型"，不进六线）
MAIN_SIX: list[tuple[str, str]] = [
    ("logreg", "R1 logistic"),
    ("logreg_R2", "R2 quantile"),
    ("ridge", "R4 ridge/Kelly"),
    ("buy_and_hold", "Buy & hold"),
    ("const_60", "Constant 60%"),
    ("sma200", "SMA200"),
]

COLORS = {
    "R1 logistic": PALETTE["primary"],
    "R2 quantile": "#2e75b6",
    "R4 ridge/Kelly": "#9dc3e6",
    "Buy & hold": PALETTE["secondary"],
    "Constant 60%": "#ed7d31",
    "SMA200": PALETTE["accent"],
}


def load_results(cfg: dict | None = None) -> dict[str, pd.DataFrame]:
    """读 backtest 产出的三张表 + 预测表 + 两个面板。"""
    cfg = cfg if cfg is not None else load_config()
    reports = project_path("reports")
    processed = project_path("data", "processed")

    tables = {}
    for name in ("metrics", "equity_curve", "trades"):
        path = reports / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"tearsheet: 缺少 {path}，先跑 `uv run python -m src.pipeline --stage backtest`"
            )
        tables[name] = pd.read_csv(path, parse_dates=["date"] if name != "metrics" else None)
    tables["predictions"] = pd.read_parquet(processed / "predictions.parquet")
    tables["daily"] = pd.read_parquet(processed / "daily_panel.parquet")
    tables["weekly"] = pd.read_parquet(processed / "weekly_panel.parquet")
    return tables


def daily_equity(
    weekly_positions: pd.Series,
    exec_map: dict,
    exit_map: dict,
    daily: pd.DataFrame,
    cost_bps: float = 5.0,
    initial_equity: float = 1.0,
) -> pd.Series:
    """把周频仓位展开到日频，算日频净值（只用于量真实回撤，不改主口径）。"""
    d = daily.sort_values("date").reset_index(drop=True)
    dates = pd.DatetimeIndex(d["date"])
    close = d["close"].to_numpy(dtype=float)

    ret = np.zeros(len(d))
    ret[1:] = close[1:] / close[:-1] - 1.0

    dtb3 = d["dtb3"].fillna(0.0).to_numpy(dtype=float) / 100.0
    bey = 365.0 * dtb3 / (360.0 - 91.0 * dtb3)
    cash = bey / TRADING_DAYS

    position = np.zeros(len(d))
    cost = np.zeros(len(d))
    prev = 0.0
    for decision_date, p in weekly_positions.sort_index().items():
        start, end = exec_map.get(decision_date), exit_map.get(decision_date)
        if start is None or end is None or pd.isna(start) or pd.isna(end):
            continue
        i = int(dates.searchsorted(start, side="left"))
        j = int(dates.searchsorted(end, side="left"))
        if j <= i:
            continue
        position[i:j] = float(p)
        cost[i] += cost_bps * BPS * abs(float(p) - prev)
        prev = float(p)

    factor = (1.0 + position * ret + (1.0 - position) * cash) * (1.0 - cost)
    return pd.Series(np.cumprod(factor) * initial_equity, index=dates)


def _exec_maps(predictions: pd.DataFrame) -> tuple[dict, dict]:
    key = predictions.drop_duplicates("decision_date")[["decision_date", "execution_date", "exit_date"]]
    key["decision_date"] = pd.to_datetime(key["decision_date"])
    key["execution_date"] = pd.to_datetime(key["execution_date"])
    key["exit_date"] = pd.to_datetime(key["exit_date"])
    return (
        dict(zip(key["decision_date"], key["execution_date"])),
        dict(zip(key["decision_date"], key["exit_date"])),
    )


def _annual(frame: pd.DataFrame, value: str, how: str = "sum") -> pd.Series:
    grouped = frame.groupby(frame["date"].dt.year)[value]
    return grouped.sum() if how == "sum" else grouped.apply(lambda s: (1.0 + s).prod() - 1.0)


def build_tearsheet(cfg: dict | None = None, out_path: str | Path | None = None) -> Path:
    """画六宫格并返回图片路径。"""
    cfg = cfg if cfg is not None else load_config()
    data = load_results(cfg)

    equity_curve = data["equity_curve"]
    metrics = data["metrics"].set_index("strategy")
    exec_map, exit_map = _exec_maps(data["predictions"])
    cost_bps = float(cfg.get("backtest", {}).get("cost_bps", 5.0))

    curves = {name: equity_curve[equity_curve["strategy"] == name].reset_index(drop=True) for name, _ in MAIN_SIX}
    missing = [name for name, frame in curves.items() if frame.empty]
    if missing:
        raise ValueError(f"tearsheet: equity_curve 里没有这些策略 {missing}")

    daily_eq = {
        name: daily_equity(
            curves[name].set_index("date")["position"], exec_map, exit_map, data["daily"], cost_bps
        )
        for name, _ in MAIN_SIX
    }

    palette = apply_style()
    fig, axes = plt.subplots(3, 2, figsize=(15, 13))
    fig.suptitle("Backtest tear sheet — 2008–2025, 5 bps one-way cost", fontsize=14, y=0.995)

    # ① 净值（对数轴）
    ax = axes[0, 0]
    for name, label in MAIN_SIX:
        ax.plot(curves[name]["date"], curves[name]["equity"], label=label, color=COLORS[label])
    ax.set_yscale("log")
    ax.set_title("1. Equity curve (log scale)")
    ax.set_ylabel("Growth of $1")
    ax.legend(fontsize=8, ncol=2)

    # ② 回撤（周频面积 + 日频最大回撤标注）
    ax = axes[0, 1]
    for name, label in MAIN_SIX:
        c = curves[name]
        ax.plot(c["date"], c["drawdown"], color=COLORS[label], linewidth=1.0, label=label)
    ax.fill_between(curves["buy_and_hold"]["date"], curves["buy_and_hold"]["drawdown"], 0, alpha=0.15,
                    color=PALETTE["secondary"])
    ax.set_title("2. Drawdown (weekly)")
    ax.set_ylabel("Drawdown")
    text = "\n".join(
        f"{label}: {(daily_eq[name].div(daily_eq[name].cummax()) - 1.0).min():.1%}"
        for name, label in MAIN_SIX[:3]
    )
    ax.text(
        0.02,
        0.04,
        "max drawdown, daily basis\n" + text,
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": PALETTE["grid"], "alpha": 0.9},
    )

    # ③ 滚动一年夏普
    ax = axes[1, 0]
    for name, label in MAIN_SIX[:3] + [("buy_and_hold", "Buy & hold")]:
        r = curves[name]["net_return"]
        roll = r.rolling(52)
        sharpe = roll.mean() / roll.std(ddof=1) * np.sqrt(52)
        ax.plot(curves[name]["date"], sharpe, color=COLORS[label], label=label)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylim(-2.5, 5.0)
    ax.set_title("3. Rolling 1-year Sharpe (52 weeks, y clipped to [-2.5, 5])")
    ax.legend(fontsize=8)

    # ④ 年度收益
    ax = axes[1, 1]
    picks = [("logreg", "R1 logistic"), ("buy_and_hold", "Buy & hold"), ("const_60", "Constant 60%")]
    years = sorted(curves["logreg"]["date"].dt.year.unique())
    width = 0.8 / len(picks)
    for k, (name, label) in enumerate(picks):
        annual = _annual(curves[name], "net_return", how="compound")
        ax.bar(np.array(years) + (k - 1) * width, annual.reindex(years).fillna(0.0).to_numpy(),
               width=width, label=label, color=COLORS[label])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("4. Calendar-year return")
    ax.set_ylabel("Return")
    ax.set_xticks(years[::2])
    ax.set_xticklabels([str(y) for y in years[::2]])
    ax.legend(fontsize=8)

    # ⑤ 换手（每年）
    ax = axes[2, 0]
    turnover = {label: _annual(curves[name], "traded") for name, label in MAIN_SIX}
    for k, (name, label) in enumerate(MAIN_SIX):
        ax.bar(np.array(years) + (k - 2.5) * 0.8 / 6, turnover[label].reindex(years).fillna(0.0).to_numpy(),
               width=0.8 / 6, color=COLORS[label], label=label)
    ax.set_title("5. Turnover per year (sum of |position change|)")
    ax.set_ylabel("Turnover (x)")
    ax.set_xticks(years[::2])
    ax.set_xticklabels([str(y) for y in years[::2]])
    ax.legend(fontsize=7, ncol=2)

    # ⑥ 仓位
    ax = axes[2, 1]
    r1 = curves["logreg"]
    ax.fill_between(r1["date"], 0, r1["position"], step="post", alpha=0.35, color=COLORS["R1 logistic"],
                    label="R1 logistic (area)")
    ax.step(r1["date"], r1["position"], where="post", color=COLORS["R1 logistic"], linewidth=0.8)
    ax.axhline(0.6, color=COLORS["Constant 60%"], linewidth=1.4, label="Constant 60%")
    ax.plot(curves["sma200"]["date"], curves["sma200"]["position"], color=COLORS["SMA200"], linewidth=0.8,
            label="SMA200")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("6. Position in SPY")
    ax.set_ylabel("Weight")
    ax.legend(fontsize=8, loc="lower left")

    for ax in axes.ravel():
        ax.tick_params(labelsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.98))
    target = Path(out_path) if out_path is not None else project_path("reports", "figures", "backtest_tearsheet.png")
    path = savefig(fig, target)

    # 把日频最大回撤写回 metrics 旁边的小表，方便报告引用
    rows = []
    for name, label in MAIN_SIX:
        eq = daily_eq[name]
        rows.append(
            {
                "strategy": name,
                "label": label,
                "max_dd_weekly": float(curves[name]["drawdown"].min()),
                "max_dd_daily": float(eq.div(eq.cummax()).min() - 1.0),
                "cagr": float(metrics.loc[name, "cagr"]),
                "sharpe": float(metrics.loc[name, "sharpe"]),
            }
        )
    pd.DataFrame(rows).to_csv(project_path("reports", "tearsheet_stats.csv"), index=False, encoding="utf-8-sig")
    print(f"[tearsheet] 已写 {path} 与 reports/tearsheet_stats.csv")
    return path


if __name__ == "__main__":  # pragma: no cover
    build_tearsheet(load_config())

