"""数据下载（B1）：5 个标的并发拉取 + 重试 + 缓存 + manifest。

负责人：P1（B1）
对应：SP500_分工说明书 P1 第 1 步

标的：SPY（主）、^GSPC（校验）、^VIX、DTB3、DGS10（宏观走 FRED CSV，不走雅虎）

约定（改口径先改这里，不要各处各写一套）：
  - 价格：`yfinance` 拉 OHLCV + Adj Close，列名压平为 open/high/low/close/adj_close/volume
  - 宏观：FRED 的 CSV 接口 `fredgraph.csv?id=<SERIES>`，缺失值 "." 转 NaN
  - 原始文件落 `data/raw/{yahoo,fred}/{name}.csv`，**写完置只读**；缓存命中即跳过
  - `data/raw/manifest.json` 记 url / retrieved_at / rows / date_range / sha256
  - 重试 3 次（指数退避 1s/2s/4s），单请求超时 20s，线程池 max_workers=4
  - 不写死代理：需要代理的机器自己 `export HTTPS_PROXY=...`，requests/yfinance 会自动读

验收：断网重跑读缓存不报错；5 个文件 60 秒内下完；manifest 5 行齐全。
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import requests

from src.config import ensure_dirs, project_path

MAX_ATTEMPTS = 3
BACKOFF_BASE = 1.0  # 1s, 2s, 4s
TIMEOUT_S = 30
FRED_TIMEOUT_S = 60  # FRED 从本机偶发慢响应，给足时间
MAX_WORKERS = 4

PRICE_COLUMNS = ["open", "high", "low", "close", "adj_close", "volume"]
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #
def _sha256(path: Path) -> str:
    """文件内容指纹，用来判断数据源有没有在背后被改过。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _make_readonly(path: Path) -> None:
    os.chmod(path, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)


def _unlock(path: Path) -> None:
    """Windows 上只读文件不能直接覆盖/删除，先放开写权限。"""
    if path.exists():
        os.chmod(path, stat.S_IWRITE | stat.S_IREAD)


def _retry(action: Callable[[], Any], what: str) -> Any:
    """重试 3 次，指数退避；全失败抛 RuntimeError 并带上最后一次的异常。"""
    last: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return action()
        except Exception as exc:  # noqa: BLE001 - 网络层什么都可能抛
            last = exc
            if attempt == MAX_ATTEMPTS:
                break
            time.sleep(BACKOFF_BASE * 2 ** (attempt - 1))
    raise RuntimeError(f"{what}: 重试 {MAX_ATTEMPTS} 次仍失败（{last}）") from last


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """雅虎返回多层表头 ``('Close','SPY')``，压平成 ``close``。"""
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [str(col[0]) for col in out.columns]
    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]
    return out


def _normalize_dates(series: pd.Series) -> pd.Series:
    """统一成 tz-naive 的 ``datetime64``（雅虎有时给带时区的索引）。"""
    dates = pd.to_datetime(series)
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_localize(None)
    return dates.dt.normalize()


def _covers(path: Path, start: str, end: str, date_col: str = "date") -> bool:
    """缓存是否已经覆盖请求区间（只读 date 列，成本很低）。"""
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        dates = pd.read_csv(path, usecols=[date_col])[date_col]
    except (ValueError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return False
    if dates.empty:
        return False
    first, last = pd.to_datetime(dates.min()), pd.to_datetime(dates.max())
    # 两端各留 7 天缓冲：请求起点/终点常常不是交易日（元旦、周末、停更）
    return first <= pd.Timestamp(start) + pd.offsets.Day(7) and last >= pd.Timestamp(end) - pd.offsets.Day(7)


def _write_csv(df: pd.DataFrame, path: Path) -> Path:
    """先写临时文件再改名，避免半截文件被当成缓存。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    _unlock(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, date_format="%Y-%m-%d")
    tmp.replace(path)
    _make_readonly(path)
    return path


# --------------------------------------------------------------------------- #
# 单个标的
# --------------------------------------------------------------------------- #
def download_ticker(ticker: str, start: str, end: str, out_dir: str | Path, *, refresh: bool = False) -> Path:
    """下载单个价格标的（雅虎）并落盘为不可变快照，返回文件路径。"""
    out_dir = Path(out_dir)
    path = out_dir / f"{ticker.lstrip('^')}.csv"
    if not refresh and _covers(path, start, end):
        return path

    import yfinance as yf  # 延迟导入：没网/没装时也能 import 本模块

    # yfinance 的 end 是开区间，加一天才能把 end 当天算进去
    end_exclusive = (pd.Timestamp(end) + pd.offsets.Day(1)).strftime("%Y-%m-%d")

    def _fetch() -> pd.DataFrame:
        # 用 Ticker.history 而不是 yf.download：download 会做多标的批量请求，
        # 4 线程并发时不同标的的数据会串进同一张表（实测出现 3 倍重复列）。
        frame = yf.Ticker(ticker).history(
            start=start,
            end=end_exclusive,
            interval="1d",
            auto_adjust=False,  # 要同时拿到 close 和 adj_close
            actions=False,
            timeout=TIMEOUT_S,
        )
        if frame is None or frame.empty:
            raise ValueError(f"{ticker}: 雅虎返回空表")
        return frame

    raw = _retry(_fetch, f"下载 {ticker}")
    df = _flatten_columns(raw.reset_index()).rename(columns={"index": "date"})

    if df.columns.duplicated().any():
        raise RuntimeError(f"{ticker}: 列名重复 {list(df.columns)}，疑似多标的串表")

    missing = [c for c in ["date", *PRICE_COLUMNS] if c not in df.columns]
    if missing:
        raise RuntimeError(f"{ticker}: 缺少列 {missing}（实际列 {list(df.columns)}）")

    df = df[["date", *PRICE_COLUMNS]].copy()
    df["date"] = _normalize_dates(df["date"])
    df = df.dropna(subset=["close"]).sort_values("date").drop_duplicates("date", keep="last")
    df = df.reset_index(drop=True)
    return _write_csv(df, path)


def download_fred_series(series_id: str, start: str, end: str, out_dir: str | Path, *, refresh: bool = False) -> Path:
    """从 FRED 的 CSV 接口下载一条宏观序列（DTB3 / DGS10）。"""
    out_dir = Path(out_dir)
    path = out_dir / f"{series_id}.csv"
    if not refresh and _covers(path, start, end):
        return path

    url = FRED_CSV_URL.format(series_id=series_id)

    def _fetch() -> pd.DataFrame:
        # 注意：不要自定义 User-Agent。实测本机链路上 FRED 对非默认 UA 会挂起不返回，
        # 用 requests 默认 UA 反而 1.8 秒就拿到全量 CSV。
        resp = requests.get(url, timeout=FRED_TIMEOUT_S)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        if df.empty:
            raise ValueError(f"{series_id}: FRED 返回空表")
        return df

    raw = _retry(_fetch, f"下载 {series_id}")
    date_col, value_col = raw.columns[0], raw.columns[1]
    col = series_id.lower()
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(raw[date_col]),
            col: pd.to_numeric(raw[value_col], errors="coerce"),
        }
    )
    df = df.dropna(subset=[col]).sort_values("date").drop_duplicates("date", keep="last")
    df = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))].reset_index(drop=True)
    if df.empty:
        raise RuntimeError(f"{series_id}: 区间 {start}~{end} 内没有数据")
    return _write_csv(df, path)


# --------------------------------------------------------------------------- #
# 批量
# --------------------------------------------------------------------------- #
def _is_fred(name: str, cfg: dict) -> bool:
    sources = (cfg.get("data") or {}).get("sources") or {}
    fred = [str(s).upper() for s in sources.get("fred", [])]
    return str(name).upper() in fred


def download_all(
    tickers: dict[str, str],
    start: str,
    end: str,
    cfg: dict,
    *,
    refresh: bool | None = None,
) -> pd.DataFrame:
    """线程池并发下载全部标的，返回 manifest 表（同时写 manifest.json）。"""
    if refresh is None:
        refresh = os.environ.get("SP500_REFRESH", "").lower() in {"1", "true", "yes"}

    ensure_dirs()
    data_cfg = cfg.get("data") or {}
    yahoo_dir = project_path(data_cfg.get("yahoo_dir", "data/raw/yahoo"))
    fred_dir = project_path(data_cfg.get("fred_dir", "data/raw/fred"))

    jobs: list[tuple[str, Callable[[], Path]]] = []
    for name in tickers.values():
        if _is_fred(name, cfg):
            jobs.append((name, lambda n=name: download_fred_series(n, start, end, fred_dir, refresh=refresh)))
        else:
            jobs.append((name, lambda n=name: download_ticker(n, start, end, yahoo_dir, refresh=refresh)))

    t0 = time.time()
    results: dict[str, Path] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(fn): name for name, fn in jobs}
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()  # 失败直接抛，不吞异常

    manifest_path = write_manifest(cfg)
    manifest = pd.DataFrame(json.loads(manifest_path.read_text(encoding="utf-8"))["files"])

    print(f"[download] {len(results)}/{len(jobs)} 个标的就绪，用时 {time.time() - t0:.1f}s（refresh={refresh}）")
    print(manifest[["ticker", "source", "rows", "date_range", "sha256"]].to_string(index=False, max_colwidth=16))
    return manifest


def write_manifest(cfg: dict, *, raw_root: str | Path | None = None) -> Path:
    """生成 manifest.json（URL、下载时间、行数、日期范围、SHA256）。"""
    raw_root = Path(raw_root) if raw_root is not None else project_path((cfg.get("data") or {}).get("raw_dir", "data/raw"))
    raw_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    retrieved_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    # 文件名去掉了 "^"（Windows 上更安全），manifest 里还原成原始代码
    symbol_of = {str(v).lstrip("^"): str(v) for v in ((cfg.get("data") or {}).get("tickers") or {}).values()}
    for source, pattern, url_tpl in (
        ("yahoo", "yahoo/*.csv", YAHOO_CHART_URL),
        ("fred", "fred/*.csv", FRED_CSV_URL),
    ):
        for path in sorted(raw_root.glob(pattern)):
            df = pd.read_csv(path, usecols=["date"])
            dates = pd.to_datetime(df["date"])
            symbol = symbol_of.get(path.stem, path.stem)
            rows.append(
                {
                    "ticker": symbol,
                    "source": source,
                    "url": url_tpl.format(series_id=symbol, ticker=symbol),
                    "retrieved_at": retrieved_at,
                    "rows": int(len(df)),
                    "date_range": f"{dates.min():%Y-%m-%d} ~ {dates.max():%Y-%m-%d}",
                    "sha256": _sha256(path),
                    "file": path.relative_to(raw_root).as_posix(),
                }
            )

    payload = {"generated_at": retrieved_at, "files": rows}
    manifest_path = raw_root / "manifest.json"
    # 保留别的模块登记的条目（如 wiki 成分股快照），否则每次重下都会把它们抹掉
    if manifest_path.exists():
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8")).get("files", [])
        except (json.JSONDecodeError, OSError):
            previous = []
        covered = {row["file"] for row in rows}
        payload["files"] += [
            entry for entry in previous if entry.get("file") and entry["file"] not in covered
        ]
    _unlock(manifest_path)
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


if __name__ == "__main__":  # pragma: no cover - 手动跑用
    from src.config import load_config

    _cfg = load_config()
    download_all(
        _cfg["data"]["tickers"],
        _cfg["data"]["start"],
        _cfg["data"]["end"],
        _cfg,
        refresh=os.environ.get("SP500_REFRESH", "").lower() in {"1", "true", "yes"},
    )
