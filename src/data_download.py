"""数据下载（B1）：7 个标的并发拉取 + 重试 + 缓存 + manifest。

负责人：P1（B1）
对应计划：4.1「数据获取」、知识点「进程/线程/协程」

验收要点：
  - 线程池并发（max_workers=4），失败重试 3 次（指数退避）
  - 原始文件保存到 data/raw/{source}/{ticker}.csv，只读
  - 生成 data/raw/manifest.json：url / retrieved_at / rows / date_range / sha256
  - API 不可用时回退 Stooq CSV
  - 整条管道 < 60 秒
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def download_ticker(ticker: str, start: str, end: str, out_dir: str | Path) -> Path:
    """下载单个标的并落盘为不可变快照，返回文件路径。"""
    raise NotImplementedError("B1：待实现（P1）")


def download_all(tickers: dict[str, str], start: str, end: str, cfg: dict) -> pd.DataFrame:
    """线程池并发下载全部标的，返回 manifest 表。"""
    raise NotImplementedError("B1：待实现（P1）")


def write_manifest(cfg: dict) -> Path:
    """生成 manifest.json（URL、下载时间、行数、日期范围、SHA256）。"""
    raise NotImplementedError("B1：待实现（P1）")
