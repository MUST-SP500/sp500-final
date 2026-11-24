"""EDA 与可视化叙事（B2）。

负责人：P2（赵一玮）
对应：SP500_分工说明书 P2 第 1–6 步

产出 7 张图（全部由本模块生成，禁止手工截图）：

    eda_01_market.png      30 年发生了什么、跌多深、多久恢复    正文
    eda_02_dist.png        收益是否正态（直方图 + QQ + 偏度峰度） 正文
    eda_03_vol.png         波动是否扎堆（20/60/250 日）          正文
    eda_04_corr.png        22 个指标是否重复（相关 + VIF）        正文
    eda_05_vix.png         VIX 分位 vs 未来一周收益               正文
    eda_06_sectors.png     指数成分是否稳定（板块 + 加入年份）     正文
    eda_07_pca.png         22 个指标能否压成几个主成分（附录）     附录

用法：
    uv run python -m src.eda                 # 生成全部图 + reports/figures/eda_stats.json
    uv run python -m src.eda --only market   # 只重画某一类

口径纪律：
  - 只用 `src/data_clean.py` 产出的面板，不在这里复权、不 resample；
  - 图 7 的 PCA 是**全样本描述性**分析，只用来讲故事，绝不参与特征选择；
    建模用的 PCA 由 B5 在每个训练折内重新 fit（src/models_pca.py）。
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import ensure_dirs, load_config, project_path
from src.features import feature_columns
from src.plot_style import PALETTE, apply_style, savefig

FIG_DIR = project_path("reports", "figures")
STATS_PATH = FIG_DIR / "eda_stats.json"

# NBER 官方衰退区间（硬编码：只做阴影，不作为特征，不参与建模）
RECESSIONS: list[tuple[str, str, str]] = [
    ("2001-03-01", "2001-11-01", "Dot-com"),
    ("2007-12-01", "2009-06-01", "GFC"),
    ("2020-02-01", "2020-04-01", "COVID"),
]

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# 通用小工具
# --------------------------------------------------------------------------- #
def _shade_recessions(ax: plt.Axes, label: bool = True) -> None:
    """在时间轴上画 NBER 衰退阴影。"""
    for start, end, name in RECESSIONS:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color="#c8c8c8", alpha=0.45, lw=0)
        if label:
            mid = pd.Timestamp(start) + (pd.Timestamp(end) - pd.Timestamp(start)) / 2
            ax.annotate(
                name,
                xy=(mid, 0.97),
                xycoords=("data", "axes fraction"),
                ha="center",
                va="top",
                fontsize=8,
                color="#5f5e5a",
            )


def _normal_quantiles(n: int) -> np.ndarray:
    """标准正态理论分位数（不依赖 scipy）。"""
    nd = statistics.NormalDist()
    return np.array([nd.inv_cdf((i + 0.5) / n) for i in range(n)])


def _vif(frame: pd.DataFrame) -> pd.Series:
    """方差膨胀因子：VIF_j = 1 / (1 - R²_j)，用最小二乘直接算，免装 statsmodels。"""
    x = frame.to_numpy(dtype=float)
    x = (x - x.mean(axis=0)) / x.std(axis=0, ddof=0)
    out = {}
    for j, col in enumerate(frame.columns):
        y = x[:, j]
        others = np.delete(x, j, axis=1)
        design = np.column_stack([np.ones(len(others)), others])
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        resid = y - design @ beta
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1.0 - float((resid**2).sum()) / ss_tot if ss_tot > 0 else 0.0
        out[col] = np.inf if r2 >= 1.0 else 1.0 / (1.0 - r2)
    return pd.Series(out)


def _max_drawdown(series: pd.Series) -> tuple[float, pd.Timestamp, pd.Timestamp | None]:
    """返回（最大回撤, 谷底日期, 收复前高的日期）。"""
    peak = series.cummax()
    dd = series / peak - 1.0
    trough = dd.idxmin()
    before = series.loc[:trough]
    peak_date = before.idxmax()
    after = series.loc[trough:]
    recovered = after[after >= series.loc[peak_date]]
    recovery_date = recovered.index[0] if len(recovered) else None
    return float(dd.min()), trough, recovery_date


# --------------------------------------------------------------------------- #
# 图 1：市场历史 + 回撤
# --------------------------------------------------------------------------- #
def plot_market(daily: pd.DataFrame, out: Path = FIG_DIR / "eda_01_market.png") -> dict:
    px = daily.set_index("date")["close"].astype(float)
    ma200 = px.rolling(200, min_periods=200).mean()
    dd, trough, recovery = _max_drawdown(px)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1.2]}
    )
    ax1.plot(px.index, px, color=PALETTE["primary"], lw=1.1, label="SPY total-return (log scale)")
    ax1.plot(ma200.index, ma200, color=PALETTE["secondary"], lw=1.0, label="200-day MA")
    ax1.set_yscale("log")
    ax1.set_ylabel("Adjusted price (USD, log)")
    ax1.set_title("S&P 500 (SPY) 1996–2025: price trend, 200-day MA and NBER recessions")
    _shade_recessions(ax1)
    ax1.legend(loc="lower right")

    underwater = (px / px.cummax() - 1.0) * 100.0
    ax2.fill_between(underwater.index, underwater, 0, color=PALETTE["down"], alpha=0.5, lw=0)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    ax2.set_title("Drawdown (underwater) curve", fontsize=10)
    ax2.annotate(
        f"max {dd * 100:.1f}% on {trough.date()}",
        xy=(trough, dd * 100),
        xytext=(0.02, 0.15),
        textcoords="axes fraction",
        fontsize=9,
        color=PALETTE["down"],
        arrowprops=dict(arrowstyle="->", color=PALETTE["down"], lw=0.8),
    )
    savefig(fig, out)

    years = (px.index[-1] - px.index[0]).days / 365.25
    return {
        "multiple": float(px.iloc[-1] / px.iloc[0]),
        "cagr_pct": float((px.iloc[-1] / px.iloc[0]) ** (1 / years) - 1) * 100,
        "max_drawdown_pct": dd * 100,
        "trough": str(trough.date()),
        "recovery": str(recovery.date()) if recovery is not None else None,
        "months_to_recover": (
            round((recovery - trough).days / 30.44, 1) if recovery is not None else None
        ),
    }


# --------------------------------------------------------------------------- #
# 图 2：收益分布 + QQ
# --------------------------------------------------------------------------- #
def plot_distribution(
    daily: pd.DataFrame,
    weekly: pd.DataFrame,
    out: Path = FIG_DIR / "eda_02_dist.png",
) -> dict:
    d_ret = daily.set_index("date")["close"].pct_change().dropna()
    w_ret = weekly.set_index("date")["close"].pct_change().dropna()
    r = (w_ret * 100.0).to_numpy()
    r = np.sort(r)
    q_theory = _normal_quantiles(len(r))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    axes[0].hist(d_ret * 100, bins=100, color=PALETTE["primary"], alpha=0.75)
    axes[0].set_xlim(-8, 8)
    axes[0].set_title("Daily returns (%)")
    axes[0].set_xlabel("Return (%)")
    axes[0].set_ylabel("Count")

    axes[1].hist(r, bins=45, color=PALETTE["primary"], alpha=0.75, density=True)
    mu, sd = r.mean(), r.std(ddof=1)
    grid = np.linspace(r.min(), r.max(), 300)
    axes[1].plot(grid, np.exp(-((grid - mu) ** 2) / (2 * sd**2)) / (sd * np.sqrt(2 * np.pi)),
                 color=PALETTE["secondary"], lw=1.3, label="Normal fit")
    axes[1].set_title("Weekly returns vs normal")
    axes[1].set_xlabel("Return (%)")
    axes[1].legend()

    axes[2].scatter(q_theory, r, s=8, color=PALETTE["primary"], alpha=0.7)
    lim = [min(q_theory.min(), (r - mu).min() / sd), max(q_theory.max(), (r - mu).max() / sd)]
    axes[2].plot(lim, [mu + sd * v for v in lim], color=PALETTE["secondary"], lw=1.2)
    axes[2].set_title("Normal Q–Q (weekly)")
    axes[2].set_xlabel("Theoretical quantile (std normal)")
    axes[2].set_ylabel("Observed return (%)")

    fig.suptitle(
        "Return distribution: fat tails and negative skew (weekly)", fontsize=12, y=1.02
    )
    savefig(fig, out)

    skew = float(pd.Series(r).skew())
    kurt = float(pd.Series(r).kurt())  # excess kurtosis
    return {
        "weekly_skew": skew,
        "weekly_excess_kurtosis": kurt,
        "worst_week_pct": float(r.min()),
        "worst_week": str(w_ret.idxmin().date()),
        "best_week_pct": float(r.max()),
        "pct_weeks_beyond_3sd": float((np.abs(r - mu) > 3 * sd).mean() * 100),
    }


# --------------------------------------------------------------------------- #
# 图 3：波动聚集
# --------------------------------------------------------------------------- #
def plot_volatility(daily: pd.DataFrame, out: Path = FIG_DIR / "eda_03_vol.png") -> dict:
    ret = daily.set_index("date")["close"].pct_change()
    windows = [20, 60, 250]
    colors = [PALETTE["secondary"], PALETTE["primary"], PALETTE["accent"]]
    vols = {w: ret.rolling(w).std() * np.sqrt(TRADING_DAYS) * 100 for w in windows}

    fig, ax = plt.subplots(figsize=(11, 4.6))
    for w, c in zip(windows, colors):
        ax.plot(vols[w].index, vols[w], lw=0.9, color=c, label=f"{w}-day")
    ax.set_ylabel("Annualised volatility (%)")
    ax.set_xlabel("Date")
    ax.set_title("Volatility clusters: rolling annualised volatility (20/60/250-day)")
    _shade_recessions(ax, label=False)
    ax.legend(loc="upper right")
    savefig(fig, out)

    v = vols[20].dropna()
    peak_date = v.idxmax()
    return {
        "vol20_mean_pct": float(v.mean()),
        "vol20_p95_pct": float(v.quantile(0.95)),
        "vol20_peak_pct": float(v.max()),
        "vol20_peak_date": str(peak_date.date()),
        "vol_regime_ratio_max_min": float(v.max() / max(v.quantile(0.05), 1e-9)),
    }


# --------------------------------------------------------------------------- #
# 图 4：相关性与 VIF
# --------------------------------------------------------------------------- #
def plot_correlation(panel: pd.DataFrame, out: Path = FIG_DIR / "eda_04_corr.png") -> dict:
    cols = feature_columns()
    frame = panel.loc[panel.get("history_ready", True) == True, cols].dropna()  # noqa: E712
    corr = frame.corr()

    vif = _vif(frame).sort_values(ascending=False)
    # VIF = inf 表示该特征能被其余特征线性表出（近似重复列），画图时截断到 cap 并单独标注
    cap = 50.0
    rank = int(np.linalg.matrix_rank(frame.to_numpy(dtype=float)))
    finite = vif.replace([np.inf, -np.inf], np.nan).dropna()
    vif_plot = vif.clip(upper=cap)
    dup_features = [str(k) for k in vif.index[vif == np.inf]]
    n_pairs = len(cols) * (len(cols) - 1) // 2
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    high = upper.stack()
    high_pairs = high[high.abs() > 0.8]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 6.4), gridspec_kw={"width_ratios": [1.5, 1]}, layout="constrained"
    )
    im = ax1.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax1.set_xticks(range(len(cols)), cols, rotation=90, fontsize=7)
    ax1.set_yticks(range(len(cols)), cols, fontsize=7)
    ax1.set_title(f"Feature correlation ({len(cols)} features, full sample)")
    ax1.grid(False)
    fig.colorbar(im, ax=ax1, fraction=0.042, pad=0.03)

    top = vif_plot.head(12)[::-1]
    ax2.barh(top.index, top.values, color=PALETTE["primary"], alpha=0.8)
    ax2.axvline(5, color=PALETTE["secondary"], ls="--", lw=1, label="VIF = 5")
    ax2.set_xlim(0, cap * 1.12)
    ax2.set_xlabel("VIF (capped at 50)")
    ax2.set_title("Variance inflation factor (top 12)")
    ax2.tick_params(labelsize=8)
    ax2.legend(loc="lower right")
    fig.suptitle("Redundancy among the 22 features", fontsize=12)
    if dup_features:
        fig.text(
            0.5,
            -0.03,
            "exact linear dependency: "
            + ", ".join(dup_features[:3])
            + f" — feature matrix rank {rank}/{len(cols)}",
            ha="center",
            fontsize=9,
            color=PALETTE["secondary"],
        )
    savefig(fig, out)

    return {
        "n_features": len(cols),
        "n_pairs": n_pairs,
        "n_pairs_abs_corr_gt_0.8": int(len(high_pairs)),
        "max_abs_corr": float(high.abs().max()),
        "max_abs_corr_pair": [str(v) for v in high.abs().idxmax()],
        "n_vif_gt_5": int((vif > 5).sum()),
        "max_vif": float(finite.max()),
        "max_vif_feature": str(finite.idxmax()),
        "duplicate_features_vif_inf": dup_features,
        "matrix_rank": rank,
        "rank_deficiency": len(cols) - rank,
        "exact_identity": "mom_12_1 == ret_12m - ret_1m",
    }


# --------------------------------------------------------------------------- #
# 图 5：VIX 分位 vs 未来一周收益
# --------------------------------------------------------------------------- #
def plot_vix(panel: pd.DataFrame, out: Path = FIG_DIR / "eda_05_vix.png") -> dict:
    frame = panel.dropna(subset=["vix_level", "label_ret"]).copy()
    frame = frame[frame.get("label_available", True) == True]  # noqa: E712
    frame["fwd_pct"] = frame["label_ret"] * 100.0
    frame["bucket"] = pd.qcut(frame["vix_level"], 5, labels=[f"Q{i}" for i in range(1, 6)])

    grp = frame.groupby("bucket", observed=True)["fwd_pct"]
    mean, se = grp.mean(), grp.std(ddof=1) / np.sqrt(grp.count())
    overall = frame["fwd_pct"].mean()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    ax1.scatter(frame["vix_level"], frame["fwd_pct"], s=6, alpha=0.35,
                color=PALETTE["primary"], lw=0)
    ax1.axhline(overall, color=PALETTE["accent"], ls=":", lw=1, label=f"mean {overall:.2f}%")
    ax1.set_xlabel("VIX level (lagged 1 trading day)")
    ax1.set_ylabel("Next-week log return (%)")
    ax1.set_title("Scatter: VIX vs next-week return")
    ax1.legend()

    xs = np.arange(1, 6)
    ax2.errorbar(xs, mean.values, yerr=1.96 * se.values, fmt="o-",
                 color=PALETTE["primary"], ecolor=PALETTE["accent"], capsize=4)
    ax2.axhline(overall, color=PALETTE["accent"], ls=":", lw=1)
    ax2.set_xticks(xs, [f"Q{i}\n<{frame['vix_level'].quantile(i/5):.0f}" for i in xs])
    ax2.set_xlabel("VIX quintile (low → high)")
    ax2.set_ylabel("Mean next-week return (%)")
    ax2.set_title("Binned means with 95% CI")
    fig.suptitle("Does a high VIX predict the next week? (1996–2025)", fontsize=12, y=1.02)
    savefig(fig, out)

    corr = frame["vix_level"].corr(frame["fwd_pct"], method="spearman")
    return {
        "quintile_means_pct": {k: round(float(v), 3) for k, v in mean.items()},
        "spread_q5_minus_q1_pct": float(mean.iloc[-1] - mean.iloc[0]),
        "overall_mean_pct": float(overall),
        "spearman_corr": float(corr),
        "n_weeks": int(len(frame)),
        "vix_q_cutoffs": [round(float(frame["vix_level"].quantile(q)), 1) for q in (0.2, 0.4, 0.6, 0.8)],
    }


# --------------------------------------------------------------------------- #
# 图 6：成分股稳定性（板块 + 加入年份）
# --------------------------------------------------------------------------- #
def plot_sectors(constituents: pd.DataFrame, out: Path = FIG_DIR / "eda_06_sectors.png") -> dict:
    df = constituents.copy()
    df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")
    sector = df["gics_sector"].fillna("Unknown").value_counts().sort_values()
    recent = df[df["date_added"] >= "2015-01-01"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.0), gridspec_kw={"width_ratios": [1, 1.1]})
    ax1.barh(sector.index, sector.values, color=PALETTE["primary"], alpha=0.85)
    ax1.set_xlabel("Number of constituents")
    ax1.set_title(f"Sector weights (n = {len(df)})")
    ax1.tick_params(labelsize=8)

    added = df["date_added"].dt.year.dropna()
    ax2.hist(added, bins=np.arange(1957, 2027, 3), color=PALETTE["primary"], alpha=0.8)
    ax2.axvline(2015, color=PALETTE["secondary"], ls="--", lw=1,
                label=f"added since 2015: {len(recent)} ({len(recent)/len(df)*100:.0f}%)")
    ax2.set_xlabel("Year added to the index")
    ax2.set_ylabel("Constituents (still in the index today)")
    ax2.set_title("How recent is today's index?")
    ax2.legend(loc="upper left", fontsize=8)
    fig.suptitle("Index composition is not stable: survivorship bias evidence", fontsize=12, y=1.02)
    savefig(fig, out)

    return {
        "n_constituents": int(len(df)),
        "n_sectors": int(sector.size),
        "top_sector": str(sector.index[-1]),
        "top_sector_count": int(sector.iloc[-1]),
        "pct_added_since_2015": float(len(recent) / len(df) * 100),
        "median_year_added": float(added.median()) if len(added) else None,
    }


# --------------------------------------------------------------------------- #
# 图 7：PCA 描述性（附录）
# --------------------------------------------------------------------------- #
def plot_pca(panel: pd.DataFrame, out: Path = FIG_DIR / "eda_07_pca.png") -> dict:
    cols = feature_columns()
    frame = panel.loc[panel.get("history_ready", True) == True, cols].dropna()  # noqa: E712
    x = frame.to_numpy(dtype=float)
    x = (x - x.mean(axis=0)) / x.std(axis=0, ddof=0)

    _, s, vt = np.linalg.svd(x, full_matrices=False)
    var = s**2 / (s**2).sum()
    cum = np.cumsum(var)
    n90 = int(np.searchsorted(cum, 0.90) + 1)

    n_show = 5
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.0), gridspec_kw={"width_ratios": [1, 1.25]})
    ax1.bar(np.arange(1, len(var) + 1), var * 100, color=PALETTE["primary"], alpha=0.85,
            label="explained variance")
    ax1.plot(np.arange(1, len(cum) + 1), cum * 100, color=PALETTE["secondary"], marker="o", ms=3,
             lw=1.2, label="cumulative")
    ax1.axvline(n90, color=PALETTE["accent"], ls="--", lw=1)
    ax1.set_xlabel("Principal component")
    ax1.set_ylabel("Variance explained (%)")
    ax1.set_title(f"Scree plot — {n90} PCs reach 90%")
    ax1.legend(fontsize=8)

    load = vt[:n_show].T
    im = ax2.imshow(load, cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="auto")
    ax2.set_yticks(range(len(cols)), cols, fontsize=7)
    ax2.set_xticks(range(n_show), [f"PC{i}" for i in range(1, n_show + 1)])
    ax2.set_title("Loadings (PC1–PC5)")
    ax2.grid(False)
    fig.colorbar(im, ax=ax2, fraction=0.04, pad=0.02)

    fig.suptitle(
        "PCA — descriptive (full-sample), not used for feature selection or modelling",
        fontsize=12, y=1.01, color=PALETTE["secondary"],
    )
    savefig(fig, out)

    pc1 = pd.Series(vt[0], index=cols).sort_values()
    return {
        "n_components_90pct": n90,
        "pc1_var_pct": float(var[0] * 100),
        "pc1_2_var_pct": float(var[:2].sum() * 100),
        "pc1_positive": [str(k) for k in pc1.tail(4).index],
        "pc1_negative": [str(k) for k in pc1.head(4).index],
        "n_obs": int(len(frame)),
    }


# --------------------------------------------------------------------------- #
# 编排
# --------------------------------------------------------------------------- #
def load_data(cfg: dict | None = None) -> dict[str, pd.DataFrame]:
    cfg = cfg or load_config()
    proc = project_path("data", "processed")
    out = {
        "daily": pd.read_parquet(proc / "daily_panel.parquet"),
        "weekly": pd.read_parquet(proc / "weekly_panel.parquet"),
    }
    model_path = proc / "model_panel.parquet"
    if model_path.exists():
        out["panel"] = pd.read_parquet(model_path)
    const_path = project_path("data", "raw", "wiki", "constituents.csv")
    if const_path.exists():
        out["constituents"] = pd.read_csv(const_path)
    return out


def make_all(only: str | None = None) -> dict:
    ensure_dirs()
    apply_style()
    data = load_data()
    missing = [k for k in ("daily", "weekly") if k not in data]
    if missing:
        raise FileNotFoundError(
            f"缺少 {missing}：先跑 uv run python -m src.pipeline --stage clean"
        )

    jobs = {
        "market": lambda: plot_market(data["daily"]),
        "dist": lambda: plot_distribution(data["daily"], data["weekly"]),
        "vol": lambda: plot_volatility(data["daily"]),
        "corr": lambda: plot_correlation(_need(data, "panel", "B3 的 model_panel.parquet")),
        "vix": lambda: plot_vix(_need(data, "panel", "B3 的 model_panel.parquet")),
        "sectors": lambda: plot_sectors(_need(data, "constituents", "B1 的成分股快照")),
        "pca": lambda: plot_pca(_need(data, "panel", "B3 的 model_panel.parquet")),
    }
    selected = jobs if only is None else {only: jobs[only]}

    stats: dict[str, dict] = {}
    for name, job in selected.items():
        stats[name] = job()
        print(f"[eda] {name} ok")

    if only is None:
        STATS_PATH.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[eda] 统计写入 {STATS_PATH.relative_to(project_path())}")
    return stats


def _need(data: dict, key: str, hint: str) -> pd.DataFrame:
    if key not in data:
        raise FileNotFoundError(f"缺少 {key}（{hint}），先让上游交付")
    return data[key]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="B2 EDA 出图")
    parser.add_argument("--only", choices=["market", "dist", "vol", "corr", "vix", "sectors", "pca"],
                        help="只重画某一类图")
    args = parser.parse_args(argv)
    make_all(args.only)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
