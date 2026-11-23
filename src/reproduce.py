"""复现验证与日志（B1 闸门 5）。

做两件事：
  1. 按顺序跑 download → clean → quality → wiki → export，记录每步耗时与产出
  2. （--full-rebuild）删掉所有生成的数据文件，从零重跑一遍，与重跑前逐项对齐，
     把结果写进 docs/reproduce_log.md —— 这是"换台电脑能不能复现"的证据

用法：
    uv run python -m src.reproduce                 # 只跑一遍并记日志
    uv run python -m src.reproduce --full-rebuild  # 删数据重跑 + 对齐校验（需要网络）
"""

from __future__ import annotations

import argparse
import os
import stat
import json
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.config import load_config, project_path

LOG_PATH = "docs/reproduce_log.md"
# 复权价每次都由雅虎重算，允许 1e-4 量级的漂移（见 docs/data_dictionary.md）
PRICE_RTOL = 5e-4


# --------------------------------------------------------------------------- #
# 指标采集与比对
# --------------------------------------------------------------------------- #
def collect_metrics(cfg: dict) -> dict:
    """采集面板关键指标，用于重跑前后比对。"""
    data_cfg = cfg.get("data") or {}
    weekly_path = project_path(data_cfg["weekly_panel"])
    daily_path = project_path(data_cfg["daily_panel"])
    raw_root = project_path(data_cfg.get("raw_dir", "data/raw"))

    metrics: dict = {"files": {}}
    for name, path in (("weekly", weekly_path), ("daily", daily_path)):
        if not path.exists():
            metrics[name] = None
            continue
        df = pd.read_parquet(path)
        metrics[name] = {
            "rows": int(len(df)),
            "cols": int(df.shape[1]),
            "first_date": str(df["date"].min().date()),
            "last_date": str(df["date"].max().date()),
            "first_close": float(df["close"].iloc[0]),
            "last_close": float(df["close"].iloc[-1]),
            "volume_sum": float(df["volume"].sum()),
            "na_total": int(df.isna().sum().sum()),
        }

    manifest_path = raw_root / "manifest.json"
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        metrics["files"] = {f["file"]: int(f["rows"]) for f in payload.get("files", [])}
    return metrics


def compare_metrics(before: dict, after: dict) -> list[dict]:
    """逐项比对，返回 {指标, 重跑前, 重跑后, 判定}。"""
    rows: list[dict] = []
    for name in ("weekly", "daily"):
        b, a = before.get(name), after.get(name)
        if b is None or a is None:
            rows.append({"item": f"{name}_panel", "before": "缺失", "after": "缺失", "verdict": "FAIL"})
            continue
        for key in ("rows", "cols", "first_date", "last_date", "volume_sum", "na_total"):
            same = b[key] == a[key]
            rows.append({"item": f"{name}.{key}", "before": b[key], "after": a[key],
                         "verdict": "一致" if same else "不一致"})
        for key in ("first_close", "last_close"):
            rel = abs(a[key] - b[key]) / max(abs(b[key]), 1e-9)
            rows.append({"item": f"{name}.{key}", "before": round(b[key], 4), "after": round(a[key], 4),
                         "verdict": "一致" if rel <= PRICE_RTOL else f"偏差 {rel:.2e}"})

    for file, count in before.get("files", {}).items():
        after_count = after.get("files", {}).get(file)
        rows.append({"item": f"raw.{file}", "before": count, "after": after_count,
                     "verdict": "一致" if after_count == count else "不一致"})
    return rows


# --------------------------------------------------------------------------- #
# 环境信息
# --------------------------------------------------------------------------- #
def _cmd(args: list[str]) -> str:
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=30, check=False).stdout.strip()
    except Exception:  # pragma: no cover
        return "n/a"


def env_table() -> list[list[str]]:
    from src import __version__

    import numpy
    import pandas

    return [
        ["系统", f"{platform.system()} {platform.release()}（{platform.machine()}）"],
        ["Python", f"{sys.version.split()[0]}（{Path(sys.executable).parent}）"],
        ["uv", _cmd(["uv", "--version"]) or "n/a"],
        ["项目版本", f"v{__version__}"],
        ["git commit", _cmd(["git", "rev-parse", "--short", "HEAD"]) or "n/a"],
        ["pandas / numpy", f"{pandas.__version__} / {numpy.__version__}"],
        ["随机种子", "42（config.yaml）"],
    ]


# --------------------------------------------------------------------------- #
# 跑流程
# --------------------------------------------------------------------------- #
def run_all(cfg: dict, *, refresh_download: bool = False) -> list[dict]:
    """按顺序执行，每步记录耗时与产出摘要。"""
    from src.data_clean import build_panel
    from src.data_download import download_all
    from src.data_quality import build_report
    from src.export_csv import export_csv
    from src.scrape_wiki import fetch_current_components

    steps: list[dict] = []

    def timed(name: str, func, note: str) -> None:
        t0 = time.perf_counter()
        func()
        steps.append({"step": name, "seconds": round(time.perf_counter() - t0, 2), "note": note})

    data = cfg["data"]
    timed(
        "download",
        lambda: download_all(data["tickers"], data["start"], data["end"], cfg, refresh=refresh_download),
        "5 个标的（雅虎 3 + FRED 2）",
    )
    timed("clean", lambda: build_panel(cfg), "复权 + 周频 + 面板")
    timed("quality", lambda: build_report(cfg), "质量报告 + 异常清单")
    timed("wiki", lambda: fetch_current_components(cfg), "当前成分表快照")
    timed("export", lambda: export_csv(cfg), "面板导出 CSV 供核对")
    return steps


def wipe_generated(cfg: dict) -> list[str]:
    """删掉所有"生成的数据"，保留 .gitkeep。"""
    data_cfg = cfg.get("data") or {}
    raw_root = project_path(data_cfg.get("raw_dir", "data/raw"))
    processed = project_path(data_cfg.get("processed_dir", "data/processed"))

    removed: list[str] = []
    for pattern, root in (
        ("yahoo/*.csv", raw_root),
        ("fred/*.csv", raw_root),
        ("wiki/*.csv", raw_root),
        ("manifest.json", raw_root),
        ("*.parquet", processed),
        ("anomalies.csv", processed),
        ("csv/*.csv", processed),
    ):
        for path in root.glob(pattern):
            removed.append(path.relative_to(project_path(".")).as_posix())
            os.chmod(path, stat.S_IWRITE)  # 原始数据写后置了只读
            path.unlink()
    shutil.rmtree(processed / "csv", ignore_errors=True)
    return removed


# --------------------------------------------------------------------------- #
# 写日志
# --------------------------------------------------------------------------- #
def _table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def write_log(cfg: dict, steps: list[dict], comparison: list[dict] | None, removed: list[str]) -> Path:
    out = project_path(LOG_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    metrics = collect_metrics(cfg)
    total = sum(s["seconds"] for s in steps)

    md: list[str] = []
    md.append("# 复现日志（B1）")
    md.append("")
    md.append(f"> 生成时间：{datetime.now():%Y-%m-%d %H:%M}　|　由 `uv run python -m src.reproduce` 自动生成")
    md.append("")
    md.append("## 1. 环境")
    md.append("")
    md.append(_table(["项目", "值"], env_table()))
    md.append("")
    md.append("## 2. 一键复现命令")
    md.append("")
    md.append("```bash")
    md.append("uv sync                                              # 按 uv.lock 装依赖")
    md.append("uv run python -m src.pipeline --stage download        # 抓 5 个标的（约 3 秒）")
    md.append("uv run python -m src.pipeline --stage clean           # 复权 + 周频 + 面板（约 0.5 秒）")
    md.append("uv run python -m src.data_quality                     # 质量报告 + 异常清单")
    md.append("uv run python -m src.scrape_wiki                      # 成分股快照")
    md.append("uv run python -m src.export_csv                       # 导出 CSV 供核对")
    md.append("uv run pytest                                         # 全部测试")
    md.append("```")
    md.append("")
    md.append("或一条命令跑完前五步：`uv run python -m src.reproduce`")
    md.append("")
    md.append("## 3. 本次运行")
    md.append("")
    md.append(_table(
        ["步骤", "耗时 (s)", "产出"],
        [[s["step"], s["seconds"], s["note"]] for s in steps] + [["**合计**", f"**{total:.2f}**", ""]],
    ))
    md.append("")
    if metrics.get("weekly"):
        w = metrics["weekly"]
        md.append(f"- 周面板：{w['rows']:,} 行 × {w['cols']} 列，{w['first_date']} ~ {w['last_date']}")
        md.append(f"- 日面板：{metrics['daily']['rows']:,} 行 × {metrics['daily']['cols']} 列")
    if removed:
        md.append(f"- 本次重跑前清除了 {len(removed)} 个生成文件（原始数据、面板、CSV、manifest）")
    md.append("")
    if comparison is not None:
        md.append("## 4. 删数据重跑的一致性校验")
        md.append("")
        md.append(_table(
            ["指标", "重跑前", "重跑后", "判定"],
            [[c["item"], c["before"], c["after"], c["verdict"]] for c in comparison],
        ))
        md.append("")
        bad = [c for c in comparison if c["verdict"] not in ("一致",)]
        md.append(
            "**结论**：" + ("全部一致，换台电脑可复现。" if not bad else f"{len(bad)} 项不一致，见上表。")
        )
        md.append("")
        md.append(
            "> `first_close` / `last_close` 允许 5e-4 的相对偏差：雅虎每次请求会重算复权因子，"
            "`adj_close` 天然有 1e-4 量级漂移（见数据字典）。行数、日期、成交量和缺失数必须逐位相同。"
        )
        md.append("")
    md.append("## 5. 已知限制")
    md.append("")
    md.append("1. 雅虎 `adj_close` 每次重下会漂移约 1e-4，面板的 `close` 随之漂移；跨机核对用「行数 + 日期 + close 允许 5e-4」。")
    md.append("2. FRED 的 DGS10/DTB3 在 H.15 假期不出数，成交日与交易所差 45 天，已在前向填充上限 5 天内补齐。")
    md.append("3. 成分股只有当前快照（503 只），没有历史变更，无法画年度换手图（ADR-003）。")
    md.append("4. 末尾一周是不完整周（数据截至 2025-12-31），建模时该行目标值为空。")
    md.append("")

    out.write_text("\n".join(md), encoding="utf-8")
    print(f"[reproduce] 日志已写 {out}")
    return out


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="复现验证")
    parser.add_argument("--config", default=None)
    parser.add_argument("--full-rebuild", action="store_true", help="删掉生成数据后从零重跑并比对")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    before = collect_metrics(cfg) if args.full_rebuild else None
    removed = wipe_generated(cfg) if args.full_rebuild else []
    if removed:
        print(f"[reproduce] 已清除 {len(removed)} 个生成文件")

    steps = run_all(cfg, refresh_download=args.full_rebuild)
    comparison = compare_metrics(before, collect_metrics(cfg)) if before is not None else None
    if comparison is not None:
        bad = [c for c in comparison if c["verdict"] != "一致"]
        print(f"[reproduce] 一致性校验：{len(comparison) - len(bad)}/{len(comparison)} 项一致")
        for c in bad:
            print(f"  [!] {c['item']}: {c['before']} -> {c['after']}")

    write_log(cfg, steps, comparison, removed)
    for s in steps:
        print(f"  {s['step']:<10} {s['seconds']:>5.2f}s  {s['note']}")
    return 1 if comparison and any(c["verdict"] != "一致" for c in comparison) else 0


if __name__ == "__main__":
    raise SystemExit(main())
