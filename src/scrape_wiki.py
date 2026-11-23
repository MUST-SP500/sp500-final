"""Wikipedia S&P 500 当前成分表快照（B1）。

负责人：P1（B1）
对应：SP500_分工说明书 P1 第 5 步（知识点：data scrape、正则）

决策（2026-11-23）：**只抓当前快照，不抓历史变更**。理由——
  1. 该页只有当前成分表（实测 503 行 × 8 列）；"Selected changes" 历史表已不在本页，
     另找需 Wayback Machine 逐月抓取，成本远超一周工期；
  2. 成分股数据只用于论证幸存者偏差（报告里一段文字 + 板块分布一张图），不进入回测与特征；
  3. 快照 + 抓取日期可复现，够用。
见 docs/decisions.md 的 ADR-003。

抓取纪律：
  - 带明确 User-Agent（Wikimedia 对默认 python-requests UA 直接 403）
  - 失败重试 3 次（1s/2s/4s）；仍失败则回退本地快照，并在报告中注明抓取日期
  - 解析结果写入 data/raw/wiki/constituents.csv（只读），并合并进 data/raw/manifest.json
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from src.config import project_path

URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
USER_AGENT = "sp500-final-research/0.2 (MUST G05 course project; python-requests)"
MAX_ATTEMPTS = 3
BACKOFF_BASE = 1.0
TIMEOUT_S = 30

COLUMNS = {
    "Symbol": "symbol",
    "Security": "security",
    "GICS Sector": "gics_sector",
    "GICS Sub-Industry": "gics_sub_industry",
    "Headquarters Location": "headquarters_location",
    "Date added": "date_added",
    "CIK": "cik",
    "Founded": "founded",
}
OUTPUT_COLUMNS = list(COLUMNS.values())

_FOOTNOTE = re.compile(r"\[[^\]]*\]")  # 去掉 [1]、[a] 这类角标


# --------------------------------------------------------------------------- #
# 抓取与解析
# --------------------------------------------------------------------------- #
def fetch_html(url: str = URL, timeout: int = TIMEOUT_S) -> str:
    """取页面 HTML；带 UA、重试 3 次指数退避。"""
    last: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
            resp.raise_for_status()
            return resp.text
        except Exception as exc:  # noqa: BLE001 - 网络层什么都可能抛
            last = exc
            if attempt < MAX_ATTEMPTS:
                time.sleep(BACKOFF_BASE * 2 ** (attempt - 1))
    raise RuntimeError(f"抓取 Wikipedia 失败（{MAX_ATTEMPTS} 次）：{last}") from last


def parse_current_components(html: str) -> pd.DataFrame:
    """从页面 HTML 里取出成分表（第一个同时有 Symbol 和 GICS Sector 的表）。"""
    tables = pd.read_html(io.StringIO(html), flavor="lxml")
    for table in tables:
        if "Symbol" in table.columns and "GICS Sector" in table.columns:
            return table
    raise ValueError("页面里找不到成分表（列名对不上），Wikipedia 可能改版了")


def clean_components(raw: pd.DataFrame) -> pd.DataFrame:
    """列名规范化、去角标、类型清洗。"""
    df = raw.rename(columns=COLUMNS).copy()
    missing = [c for c in OUTPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"成分表缺少列：{missing}")

    df = df[OUTPUT_COLUMNS]
    for col in ("symbol", "security", "gics_sector", "gics_sub_industry", "headquarters_location"):
        df[col] = df[col].astype(str).str.replace(_FOOTNOTE, "", regex=True).str.strip()

    df["symbol"] = df["symbol"].str.upper()
    df["date_added"] = pd.to_datetime(
        df["date_added"].astype(str).str.replace(_FOOTNOTE, "", regex=True).str.strip(),
        errors="coerce",
    )
    df["cik"] = pd.to_numeric(
        df["cik"].astype(str).str.replace(_FOOTNOTE, "", regex=True), errors="coerce"
    ).astype("Int64")
    df["founded"] = pd.to_numeric(
        df["founded"].astype(str).str.replace(_FOOTNOTE, "", regex=True), errors="coerce"
    ).astype("Int64")

    df = df.sort_values("symbol").reset_index(drop=True)
    if df["symbol"].duplicated().any():
        dup = df.loc[df["symbol"].duplicated(), "symbol"].tolist()
        raise ValueError(f"成分表里出现重复代码：{dup[:5]}")
    return df


# --------------------------------------------------------------------------- #
# 落盘
# --------------------------------------------------------------------------- #
def save_snapshot(df: pd.DataFrame, path: str | Path) -> Path:
    """保存 CSV 快照（utf-8-sig，Excel 可直接打开）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    return path


def update_manifest(cfg: dict, csv_path: Path, *, raw_root: Path, retrieved_at: str) -> Path:
    """把快照合并进 manifest.json（保留已有条目，按 file 去重）。"""
    manifest_path = raw_root / "manifest.json"
    payload = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {"generated_at": retrieved_at, "files": []}
    )
    entry = {
        "ticker": "SP500_CONSTITUENTS",
        "source": "wikipedia",
        "url": URL,
        "retrieved_at": retrieved_at,
        "rows": int(len(pd.read_csv(csv_path))),
        "date_range": "snapshot",
        "sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "file": csv_path.relative_to(raw_root).as_posix(),
    }
    files = [f for f in payload.get("files", []) if f.get("file") != entry["file"]]
    files.append(entry)
    payload["files"] = files
    payload["generated_at"] = retrieved_at
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def _ensure_manifest_entry(cfg: dict, csv_path: Path, raw_root: Path) -> None:
    """缓存命中时补登 manifest（下载层重写 manifest 可能把 wiki 条目挤掉）。"""
    manifest_path = raw_root / "manifest.json"
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        rel = csv_path.relative_to(raw_root).as_posix()
        if any(f.get("file") == rel for f in payload.get("files", [])):
            return
    stamp = datetime.fromtimestamp(csv_path.stat().st_mtime, tz=timezone.utc).astimezone()
    update_manifest(cfg, csv_path, raw_root=raw_root, retrieved_at=stamp.isoformat(timespec="seconds"))


# --------------------------------------------------------------------------- #
# 对外入口
# --------------------------------------------------------------------------- #
def fetch_current_components(cfg: dict, *, refresh: bool = False, path=None) -> pd.DataFrame:
    """当前成分表。已有快照且不要求刷新时直接读本地（断网也能跑）。"""
    data_cfg = cfg.get("data") or {}
    raw_root = project_path(data_cfg.get("raw_dir", "data/raw"))
    csv_path = Path(path) if path is not None else raw_root / "wiki" / "constituents.csv"

    if csv_path.exists() and not refresh:
        df = pd.read_csv(csv_path, parse_dates=["date_added"])
        _ensure_manifest_entry(cfg, csv_path, raw_root)
        print(f"[wiki] 命中本地快照 {csv_path.name}：{len(df)} 条（refresh=False）")
        return df

    html = fetch_html()
    df = clean_components(parse_current_components(html))
    save_snapshot(df, csv_path)
    retrieved_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    update_manifest(cfg, csv_path, raw_root=raw_root, retrieved_at=retrieved_at)

    sectors = df["gics_sector"].value_counts()
    print(f"[wiki] 抓取 {len(df)} 条成分股，{df['gics_sector'].nunique()} 个板块，抓取时间 {retrieved_at[:19]}")
    print(f"[wiki] 已写 {csv_path}")
    print("[wiki] 板块分布：" + "，".join(f"{k} {v}" for k, v in sectors.head(5).items()))
    return df


def fetch_changes(cfg: dict) -> pd.DataFrame:
    """历史变更表——按 ADR-003 决策不做。"""
    raise NotImplementedError(
        "B1：按 ADR-003 不抓历史变更（该页已无变更表，需 Wayback 逐月抓取，成本过高）"
    )


def to_turnover(changes: pd.DataFrame) -> pd.DataFrame:
    """年度换手率——依赖历史变更表，同样不做（ADR-003）。"""
    raise NotImplementedError("B1：依赖历史变更表，按 ADR-003 不做")


if __name__ == "__main__":  # pragma: no cover - 手动跑用
    import os

    from src.config import load_config

    fetch_current_components(
        load_config(), refresh=os.environ.get("SP500_REFRESH", "").lower() in {"1", "true", "yes"}
    )
