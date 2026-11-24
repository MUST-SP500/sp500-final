"""策略回测（B6）。

两个函数：

- `run_positions`：仓位 → 逐周收益 → 净值。**已实现，是对账基准，不要改**。
- `run_backtest` ：预测表 → 收益对齐 → 净值 → 三张表（trades / equity_curve / metrics）。

口径（与 docs/preregistration.md 第 3、5 节一致，全部在简单收益空间）：

    决策   t   = 第 w 周最后交易日收盘            → weekly_panel.date[w]
    执行   t+1 = 第 w+1 周第一个交易日开盘        → weekly_panel.open[w+1]
    平仓   t+2 = 第 w+2 周第一个交易日开盘        → weekly_panel.open[w+2]
    资产收益 R = open[w+2] / open[w+1] - 1        # 简单收益，open-to-open
    现金收益 rf = BEY(DTB3) * n_trading_days[w+1] / 252   # 按实际交易日折算
    净值   E[w] = E[w-1] * (1 + p*R + (1-p)*rf) * (1 - c*|p[w]-p[w-1]|)

其中 BEY = 贴现率 / (1 - 贴现率 * 91/360)，把 DTB3 的"贴现率"换成"债券等价收益率"；
c = 5 bps 单边（敏感性 10 bps，见 config.yaml 的 backtest 段）。

负责人：P6（B6）
对应：SP500_分工说明书 P6 第 1 步
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ensure_dirs, load_config, project_path

BPS = 1e-4
TRADING_DAYS = 252  # 一年折算成多少个交易日
BILL_DAYS = 91  # DTB3 = 3 个月期（91 天）国库券

EQUITY_COLUMNS = [
    "date",
    "strategy",
    "position",
    "prev_position",
    "traded",
    "gross_return",
    "cost_return",
    "cash_return",
    "net_return",
    "equity",
    "drawdown",
]

REQUIRED_PREDICTION_COLUMNS = ("decision_date", "position")


# --------------------------------------------------------------------------- #
# 对账基准：仓位 → 逐周收益 → 净值（已实现，测试依赖，不要改）
# --------------------------------------------------------------------------- #
def run_positions(
    positions: pd.Series | list[float],
    asset_returns: pd.Series | list[float],
    cash_returns: pd.Series | list[float] | float = 0.0,
    cost_bps: float = 5.0,
    initial_position: float = 0.0,
    initial_equity: float = 1.0,
) -> pd.DataFrame:
    """把仓位序列和收益序列对账成净值曲线。

    Parameters
    ----------
    positions : 目标仓位，索引为决策周
    asset_returns : 同一周的简单总收益 R
    cash_returns : 同一周的现金收益 rf（标量则视为每周相同）
    cost_bps : 单边成本，单位 bps

    Returns
    -------
    DataFrame，列：position, prev_position, traded, gross_return, cost_return,
    net_return, equity
    """
    pos = pd.Series(positions, dtype=float)
    asset = pd.Series(asset_returns, dtype=float).reindex(pos.index)
    if np.isscalar(cash_returns):
        cash = pd.Series(float(cash_returns), index=pos.index)
    else:
        cash = pd.Series(cash_returns, dtype=float).reindex(pos.index)

    prev = pos.shift(1).fillna(initial_position)
    traded = (pos - prev).abs()

    c = cost_bps * BPS
    gross_factor = 1.0 + pos * asset + (1.0 - pos) * cash
    net_factor = gross_factor * (1.0 - c * traded)

    equity = initial_equity * net_factor.cumprod()

    return pd.DataFrame(
        {
            "position": pos,
            "prev_position": prev,
            "traded": traded,
            "gross_return": gross_factor - 1.0,
            "cost_return": c * traded,
            "net_return": net_factor - 1.0,
            "equity": equity,
        }
    )


# --------------------------------------------------------------------------- #
# 现金：DTB3 贴现率 → 周现金收益
# --------------------------------------------------------------------------- #
def bond_equivalent_yield(discount_rate_pct: pd.Series | np.ndarray) -> pd.Series:
    """DTB3 的贴现率（%）→ 债券等价年收益率（小数）。

    3 个月期国库券按贴现方式报价：`价格 = 100 * (1 - d * 91/360)`。
    换成"真实年化收益"：`BEY = d / (1 - d * 91/360)`。
    例：d = 5.00% → BEY ≈ 5.07%。
    """
    d = pd.Series(discount_rate_pct, dtype=float) / 100.0
    return d / (1.0 - d * BILL_DAYS / 360.0)


def weekly_cash_returns(dtb3_pct: pd.Series, n_trading_days: pd.Series) -> pd.Series:
    """按当周实际交易日数把年化现金收益率折成周收益（简单折算）。"""
    bey = bond_equivalent_yield(dtb3_pct)
    days = pd.Series(n_trading_days, dtype=float, index=bey.index)
    return bey * (days / TRADING_DAYS)


# --------------------------------------------------------------------------- #
# 把周面板变成"每一周决策对应的持有期收益"
# --------------------------------------------------------------------------- #
def holding_returns(panel: pd.DataFrame) -> pd.DataFrame:
    """周面板 → 每个决策周对应的（资产收益, 现金收益, 持有天数）。

    第 i 行是第 w 周：用它后面第 1 周的 open 买入、第 2 周的 open 卖出，
    所以资产收益 = open[i+2] / open[i+1] - 1，现金收益按第 i+1 周折算。
    末尾两行没有完整的持有期，会被丢掉（这是正常的）。
    """
    needed = ["date", "open", "dtb3", "n_trading_days"]
    missing = [c for c in needed if c not in panel.columns]
    if missing:
        raise ValueError(f"holding_returns: 周面板缺少列 {missing}")

    p = panel.sort_values("date").reset_index(drop=True)
    open_ = p["open"].astype(float)

    out = pd.DataFrame(
        {
            "decision_date": pd.to_datetime(p["date"]),
            "asset_return": (open_.shift(-2) / open_.shift(-1) - 1.0).to_numpy(),
            "cash_return": weekly_cash_returns(p["dtb3"], p["n_trading_days"]).shift(-1).to_numpy(),
            "holding_days": p["n_trading_days"].shift(-1).to_numpy(),
        }
    )
    out = out.dropna(subset=["asset_return", "cash_return", "holding_days"])
    return out.set_index("decision_date")


# --------------------------------------------------------------------------- #
# 预测表：拆成一条条策略
# --------------------------------------------------------------------------- #
def _normalise_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """校验 B4 交来的预测表，返回带 `strategy` 列的长表。"""
    missing = [c for c in REQUIRED_PREDICTION_COLUMNS if c not in predictions.columns]
    if missing:
        raise ValueError(
            "run_backtest: 预测表缺少列 "
            f"{missing}（需要 B4 的 predictions.parquet：decision_date / position）"
        )

    out = predictions.copy()
    out["decision_date"] = pd.to_datetime(out["decision_date"])
    out["position"] = pd.to_numeric(out["position"], errors="coerce")
    if out["position"].isna().any():
        raise ValueError("run_backtest: position 列有非数值或缺失")

    # 只做多 / 空仓，不做空（见 preregistration 第 5 节）
    low, high = float(out["position"].min()), float(out["position"].max())
    if low < -1e-12 or high > 1 + 1e-12:
        raise ValueError(
            f"run_backtest: position 超出 [0, 1]（[{low:.3f}, {high:.3f}]），本版不允许做空/加杠杆"
        )

    # 策略名：优先 rule（R1/R2/R4），其次 model，最后给个默认名
    if "rule" in out.columns:
        out["strategy"] = out["rule"].astype(str)
    elif "model" in out.columns:
        out["strategy"] = out["model"].astype(str)
    else:
        out["strategy"] = "strategy"

    duplicated = out.duplicated(["strategy", "decision_date"])
    if duplicated.any():
        dup = out.loc[duplicated, "decision_date"].iloc[0]
        raise ValueError(f"run_backtest: 同一策略在同一决策日有多行（例如 {dup.date()}）")
    return out


# --------------------------------------------------------------------------- #
# 指标
# --------------------------------------------------------------------------- #
def strategy_metrics(equity_curve: pd.DataFrame, holding_days: pd.Series) -> dict:
    """单条策略的体检指标。输入是 run_positions 的输出（已加 date / cash_return）。"""
    net = equity_curve["net_return"].astype(float)
    equity = equity_curve["equity"].astype(float)
    position = equity_curve["position"].astype(float)

    weeks_per_year = TRADING_DAYS / 5.0
    days = float(pd.Series(holding_days, dtype=float).sum())
    years = days / TRADING_DAYS if days > 0 else np.nan

    total_return = float(equity.iloc[-1]) - 1.0
    cagr = (1.0 + total_return) ** (1.0 / years) - 1.0 if years and years > 0 else np.nan

    ann_vol = float(net.std(ddof=1)) * np.sqrt(weeks_per_year)
    cash_annual = float(equity_curve["cash_return"].mean()) * weeks_per_year
    excess = float(net.mean()) * weeks_per_year - cash_annual
    sharpe = excess / ann_vol if ann_vol > 0 else np.nan

    return {
        "n_weeks": int(len(equity_curve)),
        "start": equity_curve["date"].iloc[0].date(),
        "end": equity_curve["date"].iloc[-1].date(),
        "total_return": total_return,
        "cagr": cagr,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": float(equity_curve["drawdown"].min()),
        "annual_turnover": float(equity_curve["traded"].sum()) / years if years else np.nan,
        "avg_position": float(position.mean()),
        "exposure": float((position > 0).mean()),
        "n_trades": int((equity_curve["traded"] > 0).sum()),
        "hit_rate": float((net > 0).mean()),
    }


# --------------------------------------------------------------------------- #
# 主函数
# --------------------------------------------------------------------------- #
def run_backtest(
    predictions: pd.DataFrame | None = None,
    cfg: dict | None = None,
    *,
    out_dir: str | Path | None = None,
    write: bool = True,
) -> dict[str, pd.DataFrame]:
    """预测表 → 回测 → 三张表。

    Parameters
    ----------
    predictions : DataFrame | None
        B4 的 `predictions.parquet`（至少 `decision_date` + `position`）。
        传 None 时从 `config.yaml` 的 `data.predictions` 读。
    cfg : dict | None
        配置字典；None 时读仓库根目录的 config.yaml。
    out_dir : 路径 | None
        三张表写到哪；默认 `data/processed`。
    write : bool
        是否真的落盘（测试里传 False 就只算不写）。

    Returns
    -------
    dict，键为 `trades` / `equity_curve` / `metrics`，值是 DataFrame。
    """
    cfg = cfg if cfg is not None else load_config()
    bt_cfg = cfg.get("backtest", {}) or {}
    data_cfg = cfg.get("data", {}) or {}

    if predictions is None:
        pred_path = _resolve(data_cfg.get("predictions", "data/processed/predictions.parquet"))
        if not pred_path.exists():
            raise FileNotFoundError(
                f"run_backtest: 找不到 {pred_path}。先跑 B4（src.models_ml.walk_forward）生成预测表。"
            )
        predictions = pd.read_parquet(pred_path)

    panel_path = _resolve(data_cfg.get("weekly_panel", "data/processed/weekly_panel.parquet"))
    if not panel_path.exists():
        raise FileNotFoundError(
            f"run_backtest: 找不到周面板 {panel_path}（先跑 python -m src.pipeline --stage clean）"
        )
    panel = pd.read_parquet(panel_path)

    hold = holding_returns(panel)
    pred = _normalise_predictions(predictions)

    cost_bps = float(bt_cfg.get("cost_bps", 5.0))
    initial_position = float(bt_cfg.get("initial_position", 0.0))
    initial_equity = float(bt_cfg.get("initial_equity", 1.0))

    curves: list[pd.DataFrame] = []
    trades: list[pd.DataFrame] = []
    rows: list[dict] = []

    for name, sub in pred.groupby("strategy", sort=True):
        positions = sub.set_index("decision_date")["position"].astype(float).sort_index()
        if positions.index.has_duplicates:
            raise ValueError(f"run_backtest: 策略 {name} 的 decision_date 有重复")
        joined = hold.join(positions.rename("position"), how="inner")
        if joined.empty:
            raise ValueError(
                f"run_backtest: 策略 {name} 的 decision_date 与周面板一个都对不上"
                "（预测表的日期口径和 weekly_panel.date 不一致）"
            )

        missed = len(positions) - len(joined)
        if missed > 0:
            print(
                f"[backtest] {name}: {missed} 行预测没有完整持有期（末尾两周无法 open-to-open 平仓），已跳过"
            )

        result = run_positions(
            positions=joined["position"],
            asset_returns=joined["asset_return"],
            cash_returns=joined["cash_return"],
            cost_bps=cost_bps,
            initial_position=initial_position,
            initial_equity=initial_equity,
        )
        result["cash_return"] = joined["cash_return"]
        result["date"] = joined.index
        result["strategy"] = name
        if initial_equity != 1.0:
            result["equity"] = result["equity"] * initial_equity
        result["drawdown"] = result["equity"] / result["equity"].cummax() - 1.0

        curve = result[EQUITY_COLUMNS].reset_index(drop=True)
        curves.append(curve)

        trades.append(
            curve.loc[
                curve["traded"] > 0,
                ["date", "strategy", "prev_position", "position", "traded", "cost_return"],
            ]
        )

        m = strategy_metrics(curve, joined["holding_days"])
        m["strategy"] = name
        m["cost_bps"] = cost_bps
        rows.append(m)

    equity_curve = pd.concat(curves, ignore_index=True)
    trades_df = (
        pd.concat(trades, ignore_index=True)
        if trades
        else pd.DataFrame(columns=["date", "strategy", "prev_position", "position", "traded", "cost_return"])
    )
    metrics = pd.DataFrame(rows)

    preferred = [
        "strategy", "cost_bps", "n_weeks", "start", "end", "total_return", "cagr",
        "ann_vol", "sharpe", "max_drawdown", "annual_turnover", "avg_position",
        "exposure", "n_trades", "hit_rate",
    ]
    metrics = metrics[[c for c in preferred if c in metrics.columns]]

    tables = {"trades": trades_df, "equity_curve": equity_curve, "metrics": metrics}

    if write:
        ensure_dirs()
        target = Path(out_dir) if out_dir is not None else _resolve(
            data_cfg.get("processed_dir", "data/processed")
        )
        target.mkdir(parents=True, exist_ok=True)
        for table_name, frame in tables.items():
            frame.to_csv(target / f"{table_name}.csv", index=False, encoding="utf-8-sig")
        print(f"[backtest] 已写 {target} 下的 trades.csv / equity_curve.csv / metrics.csv")
        for _, r in metrics.iterrows():
            print(
                f"[backtest] {r['strategy']:<10} CAGR {r['cagr']:+.2%}  "
                f"Sharpe {r['sharpe']:.2f}  MaxDD {r['max_drawdown']:.1%}  "
                f"换手 {r['annual_turnover']:.1f}/年"
            )

    return tables


def _resolve(path: str | Path) -> Path:
    """相对路径按仓库根目录解析；绝对路径原样返回。"""
    p = Path(path)
    return p if p.is_absolute() else project_path(str(path))


if __name__ == "__main__":  # pragma: no cover - 手动跑用
    run_backtest(load_config())
