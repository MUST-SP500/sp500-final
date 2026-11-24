"""评测（B4）：预测层指标 + 策略层指标 + 配对 bootstrap。

负责人：P4（B4）
对应：SP500_分工说明书 P4 第 5–6 步

本版口径（一周工期，指标是精简过的）：
  预测层  AUC / Balanced Accuracy / F1 / Brier
          回归：RMSE / MAE / Spearman IC / 方向准确率
  策略层  CAGR / 年化波动 / Sharpe（超额）/ MaxDD（周频 + 日频）/ 年换手 / 平均仓位
  统计    配对 bootstrap，给"策略 − 基准"的 Sharpe 差 95% CI
  工程    训练/推理耗时、吞吐、峰值内存 → reports/perf.csv

本版不做：校准曲线、分年混淆矩阵、LogLoss、Sortino/Calmar、单条策略的 CI
（这些留着以后补，别现在花时间）。

纪律：所有比较以"差值 + 置信区间"呈现，不做排行榜。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    root_mean_squared_error,
    roc_auc_score,
)


def classification_metrics(y_true, y_pred, y_proba) -> dict:
    """预测层指标，同时返回 always-long 基线，便于对比。"""
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)
    pr = np.asarray(y_proba, dtype=float)

    mask = np.isfinite(pr)
    yt, yp, pr = yt[mask], yp[mask], pr[mask]

    al_pred = np.ones_like(yt)
    al_proba = np.ones_like(pr, dtype=float)

    auc = float(roc_auc_score(yt, pr)) if len(np.unique(yt)) > 1 else np.nan
    bal_acc = float(balanced_accuracy_score(yt, yp))
    f1 = float(f1_score(yt, yp, zero_division=0))
    brier = float(brier_score_loss(yt, np.clip(pr, 0.0, 1.0)))

    al_bal_acc = float(balanced_accuracy_score(yt, al_pred))
    al_f1 = float(f1_score(yt, al_pred, zero_division=0))
    al_brier = float(brier_score_loss(yt, al_proba))

    return {
        "auc": auc,
        "balanced_accuracy": bal_acc,
        "f1": f1,
        "brier": brier,
        "always_long_auc": 0.5,
        "always_long_balanced_accuracy": al_bal_acc,
        "always_long_f1": al_f1,
        "always_long_brier": al_brier,
    }


def regression_metrics(y_true, y_pred) -> dict:
    """回归预测层指标：RMSE / MAE / Spearman IC / 方向准确率。"""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[mask], yp[mask]

    ic_res = spearmanr(yt, yp)
    ic = float(ic_res.statistic) if np.isfinite(ic_res.statistic) else np.nan
    dir_acc = float(np.mean((yt > 0) == (yp > 0)))

    return {
        "rmse": float(root_mean_squared_error(yt, yp)),
        "mae": float(mean_absolute_error(yt, yp)),
        "spearman_ic": ic,
        "direction_accuracy": dir_acc,
    }


def _annualized_sharpe(excess_returns: np.ndarray, freq: int = 52) -> float:
    arr = np.asarray(excess_returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        return np.nan
    std = float(np.std(arr, ddof=1))
    if std <= 0.0:
        return np.nan
    return float((np.mean(arr) / std) * np.sqrt(freq))


def strategy_metrics(
    equity: pd.Series,
    trades: pd.DataFrame,
    freq: int = 52,
    rf_returns: pd.Series | float = 0.0,
    daily_equity: pd.Series | None = None,
) -> dict:
    """策略层指标。Sharpe 一律用超额收益，与现金计息口径一致。"""
    eq = pd.Series(equity, dtype=float).dropna()
    if eq.empty:
        raise ValueError("equity 序列不可为空")

    if "net_return" in trades.columns:
        net_ret = pd.Series(trades["net_return"], dtype=float).reindex(eq.index)
    else:
        net_ret = eq.pct_change().fillna(eq.iloc[0] - 1.0)

    if "rf_return" in trades.columns:
        rf = pd.Series(trades["rf_return"], dtype=float).reindex(eq.index).fillna(0.0)
    elif np.isscalar(rf_returns):
        rf = pd.Series(float(rf_returns), index=eq.index)
    else:
        rf = pd.Series(rf_returns, dtype=float).reindex(eq.index).fillna(0.0)

    excess_ret = net_ret - rf
    n_periods = len(eq)
    total_growth = float(eq.iloc[-1])
    cagr = float(total_growth ** (freq / n_periods) - 1.0) if total_growth > 0 else np.nan
    ann_vol = float(net_ret.std(ddof=1) * np.sqrt(freq)) if n_periods > 1 else np.nan
    sharpe = _annualized_sharpe(excess_ret.to_numpy(), freq=freq)

    peak_curve = pd.concat([pd.Series([1.0]), eq])
    max_dd_weekly = float((peak_curve / peak_curve.cummax() - 1.0).min())

    if daily_equity is not None and not pd.Series(daily_equity).dropna().empty:
        deq = pd.Series(daily_equity, dtype=float).dropna()
        d_peak = pd.concat([pd.Series([1.0]), deq])
        max_dd_daily = float((d_peak / d_peak.cummax() - 1.0).min())
    else:
        max_dd_daily = np.nan

    if "traded" in trades.columns:
        ann_turnover = float(pd.Series(trades["traded"], dtype=float).mean() * freq)
    elif "position" in trades.columns:
        pos = pd.Series(trades["position"], dtype=float)
        ann_turnover = float(pos.diff().abs().fillna(pos.abs()).mean() * freq)
    else:
        ann_turnover = np.nan

    avg_pos = float(pd.Series(trades["position"], dtype=float).mean()) if "position" in trades.columns else np.nan

    return {
        "cagr": cagr,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_dd_weekly": max_dd_weekly,
        "max_dd_daily": max_dd_daily,
        "ann_turnover": ann_turnover,
        "avg_position": avg_pos,
    }


def _block_bootstrap_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    if n <= 0:
        raise ValueError("样本长度必须大于 0")
    b = max(1, min(int(block), n))
    n_blocks = int(np.ceil(n / b))
    starts = rng.integers(0, n - b + 1, size=n_blocks)
    idx = np.concatenate([np.arange(s, s + b) for s in starts])[:n]
    return idx


def block_bootstrap_ci(
    returns: pd.Series,
    block: int = 8,
    n_boot: int = 5000,
    seed: int = 42,
    freq: int = 52,
) -> tuple[float, float]:
    """周频超额收益的 block bootstrap Sharpe 95% 置信区间。"""
    arr = pd.Series(returns, dtype=float).dropna().to_numpy()
    n = len(arr)
    rng = np.random.default_rng(seed)
    sharpes = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = _block_bootstrap_indices(n, block, rng)
        sharpes[i] = _annualized_sharpe(arr[idx], freq=freq)
    valid = sharpes[np.isfinite(sharpes)]
    return (float(np.percentile(valid, 2.5)), float(np.percentile(valid, 97.5)))


def paired_bootstrap_diff(
    returns_a: pd.Series,
    returns_b: pd.Series,
    block: int = 8,
    n_boot: int = 5000,
    seed: int = 42,
    freq: int = 52,
) -> tuple[float, float]:
    """两条策略超额收益的 Sharpe 差值（A - B）95% 置信区间。"""
    sa = pd.Series(returns_a, dtype=float)
    sb = pd.Series(returns_b, dtype=float).reindex(sa.index)
    df = pd.DataFrame({"a": sa, "b": sb}).dropna()
    arr_a = df["a"].to_numpy()
    arr_b = df["b"].to_numpy()
    n = len(df)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = _block_bootstrap_indices(n, block, rng)
        diffs[i] = _annualized_sharpe(arr_a[idx], freq=freq) - _annualized_sharpe(arr_b[idx], freq=freq)
    valid = diffs[np.isfinite(diffs)]
    return (float(np.percentile(valid, 2.5)), float(np.percentile(valid, 97.5)))