"""回测（B6）：预测表 -> 净值 -> trades / equity_curve / metrics。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config, project_path

BPS = 1e-4
DAYS = 252
CURVE_COLS = ["date", "strategy", "position", "prev_position", "traded", "gross_return",
              "cost_return", "cash_return", "net_return", "equity", "drawdown"]
METRIC_COLS = ["strategy", "cost_bps", "n_weeks", "start", "end", "total_return", "cagr",
               "ann_vol", "sharpe", "max_drawdown", "annual_turnover", "avg_position",
               "exposure", "n_trades", "hit_rate"]


def run_positions(positions, asset_returns, cash_returns=0.0, cost_bps=5.0,
                  initial_position=0.0, initial_equity=1.0) -> pd.DataFrame:
    """仓位 + 收益 -> 净值。对账基准，测试锁定数值，不要改。"""
    pos = pd.Series(positions, dtype=float)
    asset = pd.Series(asset_returns, dtype=float).reindex(pos.index)
    cash = (pd.Series(float(cash_returns), index=pos.index) if np.isscalar(cash_returns)
            else pd.Series(cash_returns, dtype=float).reindex(pos.index))

    prev = pos.shift(1).fillna(initial_position)
    traded = (pos - prev).abs()
    gross = 1.0 + pos * asset + (1.0 - pos) * cash
    net = gross * (1.0 - cost_bps * BPS * traded)

    return pd.DataFrame({
        "position": pos,
        "prev_position": prev,
        "traded": traded,
        "gross_return": gross - 1.0,
        "cost_return": cost_bps * BPS * traded,
        "net_return": net - 1.0,
        "equity": initial_equity * net.cumprod(),
    })


def bond_equivalent_yield(rate):
    """DTB3 贴现率(%) -> 债券等价年收益率(小数)：d / (1 - d*91/360)。"""
    d = pd.Series(rate, dtype=float) / 100.0
    return d / (1.0 - d * 91.0 / 360.0)


def weekly_cash_returns(dtb3, n_days):
    """年化现金收益按当周实际交易日折算。"""
    return bond_equivalent_yield(dtb3) * pd.Series(n_days, index=pd.Series(dtb3).index, dtype=float) / DAYS


def holding_returns(panel: pd.DataFrame) -> pd.DataFrame:
    """决策周 -> 下一周开盘买入、再下周开盘卖出的收益与现金收益。"""
    p = panel.sort_values("date").reset_index(drop=True)
    o = p["open"].astype(float)
    out = pd.DataFrame({
        "decision_date": pd.to_datetime(p["date"]),
        "asset_return": (o.shift(-2) / o.shift(-1) - 1.0).to_numpy(),
        "cash_return": weekly_cash_returns(p["dtb3"], p["n_trading_days"]).shift(-1).to_numpy(),
        "holding_days": p["n_trading_days"].shift(-1).to_numpy(),
    }).dropna()
    return out.set_index("decision_date")


def _resolve(path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else project_path(str(path))


def _normalise(pred: pd.DataFrame) -> pd.DataFrame:
    """校验预测表并展开成 (strategy, decision_date, position) 长表。"""
    if not {"decision_date", "position"} <= set(pred.columns):
        raise ValueError("预测表缺 decision_date / position")
    pred = pred.assign(decision_date=pd.to_datetime(pred["decision_date"]),
                       position=pd.to_numeric(pred["position"], errors="coerce"))
    if pred["position"].isna().any() or pred["position"].min() < 0 or pred["position"].max() > 1:
        raise ValueError("position 必须是 [0, 1] 内的数值，不允许做空")

    pred = pred.assign(strategy=pred.get("rule", pred.get("model", "strategy")).astype(str))
    frames = [pred]
    if "position_r2" in pred.columns:
        frames.append(pred.assign(position=pd.to_numeric(pred["position_r2"], errors="coerce"),
                                  strategy=pred["strategy"] + "_R2").dropna(subset=["position"]))
    pred = pd.concat(frames, ignore_index=True)
    if pred.duplicated(["strategy", "decision_date"]).any():
        raise ValueError("同一策略在同一决策日有多行")
    return pred


def _strategies(pred: pd.DataFrame, cfg: dict, panel: pd.DataFrame, with_baselines: bool):
    """规范化后的预测表 + 三条基线 -> [(策略名, 仓位序列)]。"""
    jobs = [(name, sub.set_index("decision_date")["position"].astype(float).sort_index())
            for name, sub in pred.groupby("strategy", sort=True)]

    if with_baselines:
        from src.models_ml import _baseline_positions

        source = panel
        model_panel = _resolve(cfg.get("data", {}).get("model_panel", "data/processed/model_panel.parquet"))
        if model_panel.exists():
            source = pd.read_parquet(model_panel)
        low, high = pred["decision_date"].min(), pred["decision_date"].max()
        for name, pos in _baseline_positions(source.set_index("date")).items():
            s = pd.Series(pos, dtype=float)
            s.index = pd.to_datetime(s.index)
            jobs.append((name, s[~s.index.duplicated()].sort_index().loc[low:high]))
    return jobs


def _metrics(curve: pd.DataFrame, days: float, cost_bps: float) -> dict:
    net, eq = curve["net_return"], curve["equity"]
    years = days / DAYS
    total = float(eq.iloc[-1]) - 1.0
    vol = float(net.std(ddof=1)) * np.sqrt(DAYS / 5)
    cash_annual = float(curve["cash_return"].mean()) * (DAYS / 5)
    return {
        "cost_bps": cost_bps,
        "n_weeks": len(curve),
        "start": curve["date"].iloc[0].date(),
        "end": curve["date"].iloc[-1].date(),
        "total_return": total,
        "cagr": (1.0 + total) ** (1.0 / years) - 1.0 if years > 0 else np.nan,
        "ann_vol": vol,
        "sharpe": (float(net.mean()) * (DAYS / 5) - cash_annual) / vol if vol > 0 else np.nan,
        "max_drawdown": float(curve["drawdown"].min()),
        "annual_turnover": float(curve["traded"].sum()) / years,
        "avg_position": float(curve["position"].mean()),
        "exposure": float((curve["position"] > 0).mean()),
        "n_trades": int((curve["traded"] > 0).sum()),
        "hit_rate": float((net > 0).mean()),
    }


def run_backtest(predictions=None, cfg=None, *, out_dir=None, write=True,
                 include_baselines=True) -> dict[str, pd.DataFrame]:
    """预测表 -> 三张表；predictions 为 None 时读 config 里的路径。"""
    cfg = cfg if cfg is not None else load_config()
    bt = cfg.get("backtest", {})
    data = cfg.get("data", {})

    if predictions is None:
        predictions = pd.read_parquet(_resolve(data["predictions"]))
    pred = _normalise(predictions)
    panel = pd.read_parquet(_resolve(data["weekly_panel"]))
    hold = holding_returns(panel)

    cost = float(bt.get("cost_bps", 5.0))
    curves, trades, rows = [], [], []
    for name, pos in _strategies(pred, cfg, panel, include_baselines):
        joined = hold.join(pos.rename("position"), how="inner")
        if joined.empty:
            raise ValueError(f"{name} 的决策日与周面板对不上")
        res = run_positions(joined["position"], joined["asset_return"], joined["cash_return"],
                            cost_bps=cost, initial_position=float(bt.get("initial_position", 0.0)),
                            initial_equity=float(bt.get("initial_equity", 1.0)))
        res["cash_return"] = joined["cash_return"]
        res["date"] = joined.index
        res["strategy"] = name
        res["drawdown"] = res["equity"] / res["equity"].cummax() - 1.0

        curve = res[CURVE_COLS].reset_index(drop=True)
        curves.append(curve)
        trades.append(curve.loc[curve["traded"] > 0,
                                ["date", "strategy", "prev_position", "position", "traded", "cost_return"]])
        rows.append({"strategy": name, **_metrics(curve, float(joined["holding_days"].sum()), cost)})

    tables = {
        "trades": pd.concat(trades, ignore_index=True),
        "equity_curve": pd.concat(curves, ignore_index=True),
        "metrics": pd.DataFrame(rows)[METRIC_COLS],
    }

    if write:
        target = Path(out_dir) if out_dir else _resolve(cfg.get("report", {}).get("tables_dir", "reports"))
        target.mkdir(parents=True, exist_ok=True)
        for name, frame in tables.items():
            frame.to_csv(target / f"{name}.csv", index=False, encoding="utf-8-sig")
        print(f"[backtest] 写出 {target}（{len(tables['metrics'])} 条策略）")
    return tables


if __name__ == "__main__":
    run_backtest(load_config())
