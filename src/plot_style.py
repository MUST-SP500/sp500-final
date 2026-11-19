"""统一图表风格（B2）。

所有图必须走这里，避免六个人画出六种配色。

负责人：P2（B2）
用法：
    from src.plot_style import apply_style, savefig
    apply_style()
    ...
    savefig(fig, "reports/figures/eda_01_price.png")
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无显示器环境（CI / 服务器）也能出图

import matplotlib.pyplot as plt  # noqa: E402

# 统一配色（色盲友好）
PALETTE = {
    "primary": "#1f4e79",
    "secondary": "#c00000",
    "accent": "#7f7f7f",
    "grid": "#d9d9d9",
    "up": "#2e7d32",
    "down": "#c62828",
}

# 报告是中文正文 + 英文图表，所以字体优先英文字体，避免中文字形缺失
RC_PARAMS = {
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "figure.figsize": (10, 5),
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.grid": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": PALETTE["grid"],
    "grid.linewidth": 0.6,
    "lines.linewidth": 1.4,
    "legend.frameon": False,
    "figure.autolayout": False,
}


def apply_style() -> dict[str, str]:
    """应用全局 rcParams，返回配色字典以便复用。"""
    plt.rcParams.update(RC_PARAMS)
    return dict(PALETTE)


def savefig(fig: plt.Figure, path: str | Path, close: bool = True) -> Path:
    """保存到指定路径，自动建目录。"""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    if close:
        plt.close(fig)
    return out
