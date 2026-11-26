"""回测体检表（B6）：六宫格 tearsheet。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.backtest import BPS, DAYS
from src.config import load_config, project_path
from src.plot_style import PALETTE, apply_style, savefig

SIX = [("logreg", "R1 logistic", PALETTE["primary"]),
       ("logreg_R2", "R2 quantile", "#2e75b6"),
       ("ridge", "R4 ridge/Kelly", "#9dc3e6"),
       ("buy_and_hold", "Buy & hold", PALETTE["secondary"]),
       ("const_60", "Constant 60%", "#ed7d31"),
       ("sma200", "SMA200", PALETTE["accent"])]


def _load() -> dict:
    reports = project_path("reports")
    paths = {name: reports / f"{name}.csv" for name in ("metrics", "equity_curve", "trades")}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"缺少 {missing}，先跑 --stage backtest")
    data = {k: pd.read_csv(v, parse_dates=["date"] if k != "metrics" else None) for k, v in paths.items()}
    data["daily"] = pd.read_parquet(project_path("data", "processed", "daily_panel.parquet"))
    data["pred"] = pd.read_parquet(project_path("data", "processed", "predictions.parquet"))
    return data


def _daily_equity(positions: pd.Series, exec_map: dict, exit_map: dict, daily: pd.DataFrame,
                  cost_bps: float) -> pd.Series:
    """周频仓位展开到日频，只用于量真实回撤。"""
    d = daily.sort_values("date").reset_index(drop=True)
    dates = pd.DatetimeIndex(d["date"])
    close = d["close"].to_numpy(float)
    ret = np.r_[0.0, close[1:] / close[:-1] - 1.0]

    rate = d["dtb3"].fillna(0.0).to_numpy(float) / 100.0
    cash = (365.0 * rate / (360.0 - 91.0 * rate)) / DAYS

    pos = np.zeros(len(d))
    cost = np.zeros(len(d))
    prev = 0.0
    for day, p in positions.sort_index().items():
        start, end = exec_map.get(day), exit_map.get(day)
        if pd.isna(start) or pd.isna(end):
            continue
        i, j = int(dates.searchsorted(start)), int(dates.searchsorted(end))
        pos[i:j] = float(p)
        cost[i] += cost_bps * BPS * abs(float(p) - prev)
        prev = float(p)

    return pd.Series(np.cumprod((1.0 + pos * ret + (1.0 - pos) * cash) * (1.0 - cost)), index=dates)


def build_tearsheet(cfg: dict | None = None, out_path: str | Path | None = None) -> Path:
    cfg = cfg if cfg is not None else load_config()
    data = _load()
    curve = data["equity_curve"]
    metrics = data["metrics"].set_index("strategy")
    cost = float(cfg["backtest"].get("cost_bps", 5.0))

    keys = data["pred"].drop_duplicates("decision_date")
    exec_map = dict(zip(pd.to_datetime(keys["decision_date"]), pd.to_datetime(keys["execution_date"])))
    exit_map = dict(zip(pd.to_datetime(keys["decision_date"]), pd.to_datetime(keys["exit_date"])))

    series = {k: curve[curve["strategy"] == k].reset_index(drop=True) for k, _, _ in SIX}
    daily = {k: _daily_equity(s.set_index("date")["position"], exec_map, exit_map, data["daily"], cost)
             for k, s in series.items()}

    apply_style()
    fig, ax = plt.subplots(3, 2, figsize=(15, 13))
    fig.suptitle("Backtest tear sheet — 2008–2025, 5 bps one-way cost", fontsize=14)

    for k, label, color in SIX:
        ax[0, 0].plot(series[k]["date"], series[k]["equity"], color=color, label=label)
    ax[0, 0].set_yscale("log")
    ax[0, 0].set_title("1. Equity (log)")
    ax[0, 0].set_ylabel("Growth of $1")
    ax[0, 0].legend(fontsize=8, ncol=2)

    for k, _, color in SIX:
        ax[0, 1].plot(series[k]["date"], series[k]["drawdown"], color=color, linewidth=1.0)
    ax[0, 1].set_title("2. Drawdown (weekly)")
    dd = "\n".join(f"{label}: {(daily[k].div(daily[k].cummax()) - 1).min():.1%}" for k, label, _ in SIX[:3])
    ax[0, 1].text(0.02, 0.04, "max drawdown, daily basis\n" + dd, transform=ax[0, 1].transAxes, fontsize=8,
                  va="bottom", bbox={"facecolor": "white", "edgecolor": PALETTE["grid"], "alpha": 0.9})

    for k, label, color in SIX[:4]:
        roll = series[k]["net_return"].rolling(52)
        ax[1, 0].plot(series[k]["date"], roll.mean() / roll.std(ddof=1) * np.sqrt(52), color=color, label=label)
    ax[1, 0].axhline(0, color="black", linewidth=0.8)
    ax[1, 0].set_ylim(-2.5, 5)
    ax[1, 0].set_title("3. Rolling 1-year Sharpe")
    ax[1, 0].legend(fontsize=8)

    years = sorted(series["logreg"]["date"].dt.year.unique())

    def annual(k: str, col: str, compound: bool) -> np.ndarray:
        g = series[k].groupby(series[k]["date"].dt.year)[col]
        s = g.apply(lambda x: (1 + x).prod() - 1) if compound else g.sum()
        return s.reindex(years).fillna(0.0).to_numpy()

    for i, (k, label, color) in enumerate(SIX[:3]):
        ax[1, 1].bar(np.array(years) + (i - 1) * 0.27, annual(k, "net_return", True),
                     width=0.27, color=color, label=label)
    ax[1, 1].axhline(0, color="black", linewidth=0.8)
    ax[1, 1].set_title("4. Calendar-year return")
    ax[1, 1].legend(fontsize=8)

    for i, (k, label, color) in enumerate(SIX):
        ax[2, 0].bar(np.array(years) + (i - 2.5) * 0.13, annual(k, "traded", False),
                     width=0.13, color=color, label=label)
    ax[2, 0].set_title("5. Turnover per year")
    ax[2, 0].legend(fontsize=7, ncol=2)

    r1 = series["logreg"]
    ax[2, 1].fill_between(r1["date"], 0, r1["position"], step="post", alpha=0.35, color=SIX[0][2])
    ax[2, 1].axhline(0.6, color=SIX[4][2], linewidth=1.4, label="Constant 60%")
    ax[2, 1].plot(series["sma200"]["date"], series["sma200"]["position"], color=SIX[5][2],
                  linewidth=0.8, label="SMA200")
    ax[2, 1].set_ylim(-0.05, 1.05)
    ax[2, 1].set_title("6. Position in SPY")
    ax[2, 1].legend(fontsize=8, loc="lower left")

    for a in (ax[0, 0], ax[0, 1], ax[1, 0], ax[2, 1]):
        a.tick_params(labelsize=8)
        a.xaxis.set_major_locator(mdates.YearLocator(2))
        a.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for a in (ax[1, 1], ax[2, 0]):
        a.tick_params(labelsize=8)
        a.set_xticks(years[::2])
        a.set_xticklabels([str(y) for y in years[::2]])
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    target = Path(out_path) if out_path else project_path("reports", "figures", "backtest_tearsheet.png")
    path = savefig(fig, target)

    pd.DataFrame([{
        "strategy": k,
        "label": label,
        "max_dd_weekly": float(series[k]["drawdown"].min()),
        "max_dd_daily": float(daily[k].div(daily[k].cummax()).min() - 1.0),
        "cagr": float(metrics.loc[k, "cagr"]),
        "sharpe": float(metrics.loc[k, "sharpe"]),
    } for k, label, _ in SIX]).to_csv(
        project_path("reports", "tearsheet_stats.csv"), index=False, encoding="utf-8-sig")
    print(f"[tearsheet] {path}")
    return path


if __name__ == "__main__":
    build_tearsheet(load_config())
