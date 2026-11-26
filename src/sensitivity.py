"""成本敏感性（B6）：同一套预测按 5 / 10 bps 各跑一遍，另看 2008 危机区间。"""

from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd

from src.backtest import run_backtest
from src.config import load_config, project_path

CRISIS = ("2008-01-01", "2009-12-31")


def build_sensitivity(cfg: dict | None = None, out_path: str | Path | None = None) -> pd.DataFrame:
    cfg = cfg if cfg is not None else load_config()
    low = float(cfg["backtest"].get("cost_bps", 5.0))
    high = float(cfg["backtest"].get("cost_bps_sensitivity", 10.0))

    results = {}
    for bps in (low, high):
        local = copy.deepcopy(cfg)
        local["backtest"]["cost_bps"] = bps
        results[bps] = run_backtest(None, local, write=False)

    base = results[low]["metrics"].set_index("strategy")
    other = results[high]["metrics"].set_index("strategy")
    curve = results[low]["equity_curve"]

    rows = []
    for name in base.index:
        window = curve[(curve["strategy"] == name) & curve["date"].between(*CRISIS)]
        crisis = float(window["equity"].iloc[-1] / window["equity"].iloc[0] - 1.0) if len(window) else float("nan")
        rows.append({
            "strategy": name,
            f"cagr_{low:g}bps": round(float(base.loc[name, "cagr"]), 4),
            f"sharpe_{low:g}bps": round(float(base.loc[name, "sharpe"]), 3),
            f"maxdd_{low:g}bps": round(float(base.loc[name, "max_drawdown"]), 4),
            f"cagr_{high:g}bps": round(float(other.loc[name, "cagr"]), 4),
            f"sharpe_{high:g}bps": round(float(other.loc[name, "sharpe"]), 3),
            f"maxdd_{high:g}bps": round(float(other.loc[name, "max_drawdown"]), 4),
            "d_cagr_bps": round(float(other.loc[name, "cagr"]) - float(base.loc[name, "cagr"]), 4),
            "d_sharpe_bps": round(float(other.loc[name, "sharpe"]) - float(base.loc[name, "sharpe"]), 3),
            "crisis_2008_cum": round(crisis, 4),
        })

    table = pd.DataFrame(rows)
    target = Path(out_path) if out_path else project_path("reports", "metrics_sensitivity.csv")
    table.to_csv(target, index=False, encoding="utf-8-sig")
    print(f"[sensitivity] {target}")
    return table


if __name__ == "__main__":
    build_sensitivity(load_config())
