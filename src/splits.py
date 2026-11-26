"""walk-forward 切分（B3）：扩展窗口 + 2 周 purge。

负责人：P3（B3）
对应：SP500_分工说明书 P3 第 5 步

协议：
  训练 1996-01-01 起，每年向前扩展
  验证（仅一次性选参）2006-01-01 – 2007-12-31，只记录在 config.yaml
  第一个测试年 2008，最后 2025
  purge = 2 周（1 周标签跨度 + 1 周缓冲）
  超参：一次性选定后全折冻结（不逐折重选）

输出必须可被 B4/B5 直接读取，且与 docs/preregistration.md 一致。
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def make_folds(index: pd.DatetimeIndex, cfg: dict) -> list[dict]:
    """按测试年生成折列表，每折含 train/test 的起止日期与 purge。"""
    index = pd.DatetimeIndex(index)
    if index.hasnans or not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError("周决策日期必须有效、唯一且递增")
    spec = cfg["split"]
    weeks = spec["purge_weeks"]
    if not isinstance(weeks, int) or isinstance(weeks, bool) or weeks < 2:
        raise ValueError("purge_weeks 必须为至少 2 的整数")
    first, last = int(spec["first_test_year"]), int(spec["last_test_year"])
    if first > last:
        raise ValueError("测试年份区间无效")
    folds = []
    for year in range(first, last + 1):
        start, end = pd.Timestamp(year, 1, 1), pd.Timestamp(year, 12, 31)
        test = index[(index >= start) & (index <= end)]
        candidate = index[(index >= pd.Timestamp(spec["train_start"])) & (index < start)]
        if test.empty or len(candidate) <= weeks:
            raise ValueError(f"{year} 年训练或测试样本不足")
        # 同时剔除末端两个决策周、满足 14 个日历日间隔。
        train = candidate[:-weeks]
        train = train[train <= start - pd.DateOffset(days=7 * weeks)]
        if train.empty:
            raise ValueError(f"{year} 年 purge 后无训练样本")
        # w+2 周末是卖出开盘可获得时间的保守上界。
        positions = index.get_indexer(train)
        if ((positions + 2) >= len(index)).any() or not (index[positions + 2] < start).all():
            raise ValueError("训练标签结束时间进入测试区间")
        folds.append({
            "fold": year, "train_start": pd.Timestamp(spec["train_start"]).date().isoformat(),
            "train_end": train[-1].date().isoformat(),
            "test_start": start.date().isoformat(), "test_end": end.date().isoformat(),
            "purge_weeks": weeks,
            "train_dates": train.strftime("%Y-%m-%d").tolist(),
            "test_dates": test.strftime("%Y-%m-%d").tolist(),
        })
    return purge_overlap(folds)


def purge_overlap(folds: list[dict]) -> list[dict]:
    """剔除训练集末端与测试集标签窗口重叠的样本。"""
    result = []
    for original in folds:
        fold = dict(original)
        weeks = fold["purge_weeks"]
        if not isinstance(weeks, int) or isinstance(weeks, bool) or weeks < 2:
            raise ValueError("purge_weeks 必须为至少 2 的整数")
        cutoff = pd.Timestamp(fold["test_start"]) - pd.DateOffset(days=7 * weeks)
        if "train_dates" not in fold:
            raise ValueError("purge 需要实际 train_dates，不能猜测交易日期")
        dates = pd.DatetimeIndex(pd.to_datetime(fold["train_dates"]))
        if not dates.is_unique or not dates.is_monotonic_increasing or dates.hasnans:
            raise ValueError("训练日期必须唯一且递增")
        dates = dates[(dates >= pd.Timestamp(fold["train_start"])) & (dates <= cutoff)]
        if dates.empty:
            raise ValueError("purge 后无训练样本")
        fold["train_dates"] = dates.strftime("%Y-%m-%d").tolist()
        fold["train_end"] = dates[-1].date().isoformat()
        result.append(fold)
    return result


def save_splits(folds: list[dict], path: str | Path) -> Path:
    """写出 splits.json，这是 B4/B5 的共同契约。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(folds, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> None:
    from src.config import load_config, project_path
    cfg = load_config()
    weekly = pd.read_parquet(project_path(cfg["data"]["weekly_panel"]))
    folds = make_folds(pd.DatetimeIndex(weekly["date"]), cfg)
    target = save_splits(folds, project_path(cfg["data"]["splits"]))
    print(f"已生成 {len(folds)} 折: {folds[0]['fold']}–{folds[-1]['fold']}，purge={folds[0]['purge_weeks']} 周")
    for fold in (folds[0], folds[-1]):
        print(f"{fold['fold']}: train_end={fold['train_end']}, test={fold['test_start']} ~ {fold['test_end']}, 训练/测试周数={len(fold['train_dates'])}/{len(fold['test_dates'])}")
    print(f"已写入 {target}；训练日期包含预热期，建模时仍需过滤未知标签及特征预热。")


if __name__ == "__main__":
    main()
