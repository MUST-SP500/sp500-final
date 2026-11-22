"""Wikipedia 成分股爬虫（B1）：当前成分表 + 历史变更表。

负责人：P1（B1）
对应：SP500_分工说明书 P1 第 5 步（知识点：data scrape、正则）

必须做到：
  - 抓取前检查 robots.txt，带明确 User-Agent，请求间隔 >= 1 秒
  - 解析失败回退到本地快照，并在报告里注明抓取日期
  - 正则清洗：ticker 用 r"(\\w{1,5})\\s*[–-]\\s*"，日期用 datetime.strptime 规范化
  - 产出 >= 200 条历史变更记录，能画年度换手图

注意：这里抓到的数据**只用于论证幸存者偏差，不进入回测**。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def fetch_current_components(cfg: dict) -> pd.DataFrame:
    """当前 S&P 500 成分表：ticker / security / sector / date_added。"""
    raise NotImplementedError("B1：待实现（P1）")


def fetch_changes(cfg: dict) -> pd.DataFrame:
    """历史变更表：date / added_ticker / removed_ticker / reason。"""
    raise NotImplementedError("B1：待实现（P1）")


def to_turnover(changes: pd.DataFrame) -> pd.DataFrame:
    """年度成分股换手率，供 B2 画 eda_component_turnover.png。"""
    raise NotImplementedError("B1：待实现（P1）")


def save_snapshot(df: pd.DataFrame, path: str | Path) -> Path:
    """保存本地快照，带抓取日期，便于解析失败时回退。"""
    raise NotImplementedError("B1：待实现（P1）")
