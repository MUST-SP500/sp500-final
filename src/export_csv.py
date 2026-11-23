"""把周/日面板导出成 CSV，供肉眼核对（Excel 直接双击打开）。

- 数据文件不进 git：CSV 落在 `data/processed/csv/`，已被 `.gitignore` 排除
- 建模主表仍是 parquet；CSV 只是给人看的副本，不要拿它当输入
- 编码用 `utf-8-sig`（带 BOM），Excel 打开不会乱码

用法：
    uv run python -m src.export_csv              # 导出到 data/processed/csv/
    uv run python -m src.export_csv --dir out/csv
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from src.config import load_config, project_path

PANELS = [("weekly_panel", "weekly_panel.csv"), ("daily_panel", "daily_panel.csv")]


def _tidy(df: pd.DataFrame) -> pd.DataFrame:
    """价格/收益率保留 6 位小数，成交量转整数——只为在 Excel 里看得舒服。"""
    out = df.copy()
    numeric = out.select_dtypes("number").columns
    rounded = [c for c in numeric if c != "volume"]
    out[rounded] = out[rounded].round(6)
    if "volume" in out.columns:
        out["volume"] = out["volume"].fillna(0).astype("int64")
    if "n_trading_days" in out.columns:
        out["n_trading_days"] = out["n_trading_days"].astype("int64")
    return out


def export_csv(cfg: dict, out_dir=None) -> list[Path]:
    data_cfg = cfg.get("data") or {}
    out_dir = Path(out_dir) if out_dir is not None else project_path("data/processed/csv")
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for key, filename in PANELS:
        src_path = project_path(data_cfg.get(key, f"data/processed/{key}.parquet"))
        if not src_path.exists():
            raise FileNotFoundError(f"找不到 {src_path}，先跑 --stage download 和 --stage clean")
        frame = _tidy(pd.read_parquet(src_path))
        dst = out_dir / filename
        frame.to_csv(dst, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")

        digest = hashlib.sha256(dst.read_bytes()).hexdigest()[:12]
        size_kb = dst.stat().st_size / 1024
        print(f"[csv] {filename:<18} {len(frame):>5} 行 × {len(frame.columns):>2} 列  {size_kb:6.1f} KB  sha256={digest}")
        written.append(dst)

    weekly = pd.read_csv(written[0], nrows=0)
    print(f"[csv] 周面板列: {', '.join(weekly.columns)}")
    print(f"[csv] 导出目录: {out_dir}")
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="导出面板 CSV（供核对，不参与建模）")
    parser.add_argument("--config", default=None)
    parser.add_argument("--dir", default=None, help="输出目录，默认 data/processed/csv")
    args = parser.parse_args(argv)
    export_csv(load_config(args.config), args.dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
