"""成本敏感性与危机区间（B6）。

对应：SP500_分工说明书 P6 第 3 步
    必须回答两个问题：① 扣掉手续费还赢吗？② 换个成本假设结论会不会翻？
    做法：同一套预测，分别按 5 bps / 10 bps 跑两遍回测，并排报出来。
    另外把 2008 危机区间单独拉出来看一遍。
"""

from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd

from src.backtest import run_backtest
from src.config import load_config, project_path

CRISIS_START = "2008-01-01"
CRISIS_END = "2009-12-31"
MAIN_SIX = ["logreg", "logreg_R2", "ridge", "buy_and_hold", "const_60", "sma200"]


def _crisis_return(equity_curve: pd.DataFrame, strategy: str) -> float:
    """危机区间内的累计收益（净值比）。"""
    frame = equity_curve[(equity_curve["strategy"] == strategy) & (equity_curve["date"] >= CRISIS_START)]
    frame = frame[frame["date"] <= CRISIS_END]
    if frame.empty:
        return float("nan")
    return float(frame["equity"].iloc[-1] / frame["equity"].iloc[0] - 1.0)


def build_sensitivity(cfg: dict | None = None, out_path: str | Path | None = None) -> pd.DataFrame:
    """5 bps / 10 bps 两档并排 + 2008 危机区间表现。"""
    cfg = cfg if cfg is not None else load_config()
    base_bps = float(cfg["backtest"].get("cost_bps", 5.0))
    high_bps = float(cfg["backtest"].get("cost_bps_sensitivity", 10.0))

    results = {}
    for bps in (base_bps, high_bps):
        local = copy.deepcopy(cfg)
        local["backtest"] = dict(local["backtest"], cost_bps=bps)
        results[bps] = run_backtest(None, local, write=False, include_baselines=True)

    rows = []
    for strategy in results[base_bps]["metrics"]["strategy"]:
        low = results[base_bps]["metrics"].set_index("strategy").loc[strategy]
        high = results[high_bps]["metrics"].set_index("strategy").loc[strategy]
        crisis = _crisis_return(results[base_bps]["equity_curve"], strategy)
        rows.append(
            {
                "strategy": strategy,
                f"cagr_{base_bps:g}bps": round(float(low["cagr"]), 4),
                f"sharpe_{base_bps:g}bps": round(float(low["sharpe"]), 3),
                f"maxdd_{base_bps:g}bps": round(float(low["max_drawdown"]), 4),
                f"cagr_{high_bps:g}bps": round(float(high["cagr"]), 4),
                f"sharpe_{high_bps:g}bps": round(float(high["sharpe"]), 3),
                f"maxdd_{high_bps:g}bps": round(float(high["max_drawdown"]), 4),
                "d_cagr_bps": round(float(high["cagr"]) - float(low["cagr"]), 4),
                "d_sharpe_bps": round(float(high["sharpe"]) - float(low["sharpe"]), 3),
                "crisis_2008_cum": round(crisis, 4),
            }
        )

    table = pd.DataFrame(rows)
    target = Path(out_path) if out_path is not None else project_path("reports", "metrics_sensitivity.csv")
    target.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(target, index=False, encoding="utf-8-sig")
    print(f"[sensitivity] 已写 {target}（{base_bps:g} bps vs {high_bps:g} bps）")
    return table


if __name__ == "__main__":  # pragma: no cover
    build_sensitivity(load_config())
