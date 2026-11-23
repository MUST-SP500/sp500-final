"""数据清洗（B1）：复权因子、周频聚合、面板合并。

负责人：P1（B1）
对应：SP500_分工说明书 P1 第 2–4 步

口径（写死在代码里，不要各处各写一套）：
  f_t      = adj_close_t / close_t            # 复权因子，同一行 4 个价格共用
  O^adj_t  = open_t * f_t                     # high / low 同理
  close_adj 直接取源字段 adj_close（不二次相乘，避免浮点误差）
  vix 统一 shift(1)；dgs10 / dtb3 滞后 1 个交易日
  周键用 ISO 年-周，取每周最后一个交易日（复用 validate.week_key）

验收：日期唯一、升序、1996–2025 不断年；SPY 与 ^GSPC 日收益相关性 > 0.99。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import ensure_dirs, project_path
from src.validate import week_key

PRICE_COLUMNS = ["open", "high", "low", "close", "adj_close", "volume"]
MACRO_COLUMNS = ["vix", "dgs10", "dtb3"]
FFILL_LIMIT = 5  # 宏观序列前向填充上限（交易日）

WEEKLY_COLUMNS = [
    "date",
    "iso_week",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "n_trading_days",
    *MACRO_COLUMNS,
]
DAILY_COLUMNS = ["date", "f", *PRICE_COLUMNS, *MACRO_COLUMNS]


# --------------------------------------------------------------------------- #
# 第 2 步：复权
# --------------------------------------------------------------------------- #
def add_adjusted_prices(df: pd.DataFrame) -> pd.DataFrame:
    """由 (adj_close / close) 派生复权 OHLC。

    - 同一行的 `f` 同时乘到 open/high/low 上（**不是**各算各的）
    - `close_adj` 直接等于源字段 `adj_close`，保证"抽查一天恰好相等"
    """
    missing = [c for c in ("open", "high", "low", "close", "adj_close") if c not in df.columns]
    if missing:
        raise ValueError(f"add_adjusted_prices: 缺少列 {missing}")

    out = df.copy()
    out["f"] = out["adj_close"] / out["close"]
    for col in ("open", "high", "low"):
        out[f"{col}_adj"] = out[col] * out["f"]
    out["close_adj"] = out["adj_close"]
    return out


# --------------------------------------------------------------------------- #
# 第 3 步：日频 → 周频
# --------------------------------------------------------------------------- #
def to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """日频 → 周频面板，索引为每周最后一个交易日。

    聚合规则（不要自创）：
      open   = 周内第一个交易日的开盘
      high   = 周内最高，low = 周内最低
      close  = 周内最后一个交易日的收盘
      volume = 周内加总
      n_trading_days = 周内交易日数（假期周可以是 4）
    """
    needed = ["date", "open_adj", "high_adj", "low_adj", "close_adj", "volume"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"to_weekly: 缺少列 {missing}（先跑 add_adjusted_prices）")

    daily = df.sort_values("date").reset_index(drop=True)
    daily["iso_week"] = week_key(pd.DatetimeIndex(daily["date"])).to_numpy()

    grouped = daily.groupby("iso_week", sort=True)
    weekly = pd.DataFrame(
        {
            "date": grouped["date"].max(),
            "open": grouped["open_adj"].first(),
            "high": grouped["high_adj"].max(),
            "low": grouped["low_adj"].min(),
            "close": grouped["close_adj"].last(),
            "adj_close": grouped["adj_close"].last(),
            "volume": grouped["volume"].sum(),
            "n_trading_days": grouped["date"].count(),
        }
    ).reset_index()

    weekly = weekly.sort_values("date").reset_index(drop=True)
    return weekly[["date", "iso_week", "open", "high", "low", "close", "adj_close", "volume", "n_trading_days"]]


# --------------------------------------------------------------------------- #
# 第 4 步：合并面板
# --------------------------------------------------------------------------- #
def _load_series(path, name: str) -> pd.Series:
    """读一条宏观/情绪序列，返回以日期为索引的 Series。

    取列优先级：与序列同名的列 → `close` → 第一个非 date 列。
    （VIX.csv 里也有 open/high/low，必须取 close；FRED 文件里列名就是 DTB3/DGS10。）
    """
    frame = pd.read_csv(path)
    if name in frame.columns:
        value_col = name
    elif "close" in frame.columns:
        value_col = "close"
    else:
        value_col = [c for c in frame.columns if c != "date"][0]
    s = pd.Series(
        pd.to_numeric(frame[value_col], errors="coerce").to_numpy(),
        index=pd.to_datetime(frame["date"]),
        name=name,
    )
    return s[~s.index.duplicated(keep="last")].sort_index()


def _align_macro(series: pd.Series, trading_days: pd.DatetimeIndex, name: str) -> pd.Series:
    """把宏观序列对齐到交易日，做发布滞后 + 限长前向填充。

    顺序不能反：先在交易日历上 ffill（补周末/假期的空缺），再 shift(1)
    表示"今天决策只能用昨天收盘前已发布的值"。
    """
    aligned = series.reindex(trading_days).ffill(limit=FFILL_LIMIT).shift(1)
    aligned.name = name
    return aligned


def build_panel(
    cfg: dict,
    *,
    raw_root=None,
    out_path=None,
    daily_out_path=None,
) -> pd.DataFrame:
    """合并价格、VIX、宏观，输出 weekly_panel.parquet（并顺带写日频面板）。"""
    ensure_dirs()
    data_cfg = cfg.get("data") or {}
    raw_root = Path(raw_root) if raw_root is not None else project_path(data_cfg.get("raw_dir", "data/raw"))
    out_path = Path(out_path) if out_path is not None else project_path(
        data_cfg.get("weekly_panel", "data/processed/weekly_panel.parquet")
    )
    daily_out_path = (
        Path(daily_out_path)
        if daily_out_path is not None
        else project_path(data_cfg.get("daily_panel", "data/processed/daily_panel.parquet"))
    )

    spy = pd.read_csv(raw_root / "yahoo" / "SPY.csv", parse_dates=["date"])
    missing = [c for c in PRICE_COLUMNS if c not in spy.columns]
    if missing:
        raise ValueError(f"build_panel: SPY 原始表缺少列 {missing}")

    daily = add_adjusted_prices(spy).sort_values("date").reset_index(drop=True)
    trading_days = pd.DatetimeIndex(daily["date"])

    sources = {
        "vix": raw_root / "yahoo" / "VIX.csv",
        "dgs10": raw_root / "fred" / "DGS10.csv",
        "dtb3": raw_root / "fred" / "DTB3.csv",
    }
    for name, path in sources.items():
        if not path.exists():
            raise FileNotFoundError(f"build_panel: 缺少原始文件 {path}（先跑 python -m src.pipeline --stage download）")
        daily[name] = _align_macro(_load_series(path, name), trading_days, name).to_numpy()

    # 日频面板（给特征工程用；周面板是建模单位）
    daily_panel = daily[DAILY_COLUMNS].copy()
    daily_out_path.parent.mkdir(parents=True, exist_ok=True)
    daily_panel.to_parquet(daily_out_path, index=False)

    weekly = to_weekly(daily)
    weekly_dates = pd.DatetimeIndex(weekly["date"])
    for name in MACRO_COLUMNS:
        weekly[name] = daily.set_index("date")[name].reindex(weekly_dates).to_numpy()

    weekly = weekly[WEEKLY_COLUMNS]

    # ---- 自查：不合格就不落盘 ----
    if weekly["date"].duplicated().any():
        raise AssertionError("面板日期重复")
    if not weekly["date"].is_monotonic_increasing:
        raise AssertionError("面板日期未升序")
    price_na = weekly[["open", "high", "low", "close", "adj_close"]].isna().sum().sum()
    if price_na:
        raise AssertionError(f"价格列出现 {price_na} 个缺失")
    macro_na = weekly[MACRO_COLUMNS].isna().mean()
    if (macro_na > 0.005).any():
        raise AssertionError(f"宏观缺失率超 0.5%：{macro_na.round(4).to_dict()}")
    bad_ohlc = (
        (weekly["low"] > weekly[["open", "close"]].min(axis=1) + 1e-9)
        | (weekly["high"] < weekly[["open", "close"]].max(axis=1) - 1e-9)
    ).sum()
    if bad_ohlc:
        raise AssertionError(f"复权后 OHLC 关系异常 {bad_ohlc} 行")
    years = weekly["date"].dt.year
    years_present = sorted(set(int(y) for y in years))
    expected_years = list(range(years_present[0], years_present[-1] + 1))
    missing_years = [y for y in expected_years if y not in years_present]
    if missing_years:
        raise AssertionError(f"面板年份不连续，缺 {missing_years}")
    counts = {int(y): int(c) for y, c in years.value_counts().sort_index().items()}
    # 只查"完整年份"：首尾年份可能只有部分周（区间起点/终点不在年初年末时不苛求）
    interior = {y: c for y, c in counts.items() if y not in (years_present[0], years_present[-1])}
    short_years = {y: c for y, c in interior.items() if c < 50}
    if short_years:
        raise AssertionError(f"某些年份周数异常（<50）：{short_years}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    weekly.to_parquet(out_path, index=False)

    print(
        f"[clean] 面板 {len(weekly)} 行 × {len(weekly.columns)} 列，"
        f"{weekly['date'].min():%Y-%m-%d} ~ {weekly['date'].max():%Y-%m-%d}；"
        f"每年 {int(years.value_counts().min())}–{int(years.value_counts().max())} 周"
    )
    print(f"[clean] 已写 {out_path.name} 与 {daily_out_path.name}")
    return weekly


if __name__ == "__main__":  # pragma: no cover - 手动跑用
    from src.config import load_config

    build_panel(load_config())
