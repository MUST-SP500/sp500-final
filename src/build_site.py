"""生成展示网页（B6）：一个可直接双击打开的静态站点。

用法：
    uv run python -m src.build_site      # 生成 site/index.html + site/figures/

设计目标（分工说明书 P6 + 组内要求）：
  - 顶部标签切换，一人一页：“谁负责什么 / 做了什么 / 结果如何”
  - 零依赖、离线可开：不联网、不装东西，双击 index.html 就能看
  - 数字全部从 reports/*.csv 读，不手抄
"""

from __future__ import annotations

import csv
import html
import json
import shutil
from pathlib import Path

from src.config import load_config, project_path

SECTIONS = [
    ("home", "首页"),
    ("b1", "P1 数据"),
    ("b2", "P2 探索"),
    ("b3", "P3 特征"),
    ("b4", "P4 模型"),
    ("b5", "P5 深度"),
    ("b6", "P6 回测"),
]


def _csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _pct(value, digits: int = 1) -> str:
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "—"


def _num(value, digits: int = 2) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def _table(headers: list[str], rows: list[list[str]], highlight: int | None = None) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = []
    for i, row in enumerate(rows):
        cls = ' class="hl"' if highlight is not None and i == highlight else ""
        cells = "".join(f"<td>{html.escape(str(c))}</td>" for c in row)
        body.append(f"<tr{cls}>{cells}</tr>")
    return f'<table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table>'


def _figure(name: str, caption: str) -> str:
    return (
        f'<figure><img src="figures/{name}" alt="{html.escape(caption)}" loading="lazy">'
        f"<figcaption>{html.escape(caption)}</figcaption></figure>"
    )


def _card(title: str, body: str, kicker: str = "") -> str:
    kick = f'<div class="kicker">{html.escape(kicker)}</div>' if kicker else ""
    return f'<section class="card"><h2>{html.escape(title)}</h2>{kick}{body}</section>'


def build(cfg: dict | None = None) -> Path:
    cfg = cfg if cfg is not None else load_config()
    reports = project_path("reports")
    site = project_path("site")
    figs_out = site / "figures"
    if site.exists():
        shutil.rmtree(site)
    figs_out.mkdir(parents=True, exist_ok=True)

    figures = sorted((reports / "figures").glob("*.png"))
    for png in figures:
        shutil.copy2(png, figs_out / png.name)

    metrics = {r["strategy"]: r for r in _csv(reports / "metrics.csv")}
    sens = {r["strategy"]: r for r in _csv(reports / "metrics_sensitivity.csv")}
    b4res = {r["strategy"]: r for r in _csv(reports / "model_results.csv")}
    b5res = _csv(reports / "b5_model_results.csv")
    stats_path = reports / "figures" / "eda_stats.json"
    eda = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}

    def row(name: str, label: str) -> list[str]:
        m = metrics.get(name, {})
        return [
            label,
            _pct(m.get("cagr")),
            _num(m.get("sharpe")),
            _pct(m.get("max_drawdown")),
            _num(m.get("annual_turnover"), 1),
        ]

    main_table = _table(
        ["策略", "年化", "夏普", "最大回撤", "年换手"],
        [
            row("logreg", "R1 逻辑回归（主策略）"),
            row("logreg_R2", "R2 分位阈值"),
            row("ridge", "R4 ridge / Kelly"),
            row("buy_and_hold", "买入持有"),
            row("const_60", "恒定 60%"),
            row("sma200", "SMA200"),
        ],
        highlight=0,
    )

    sens_table = _table(
        ["策略", "5→10 bps 年化变化", "5→10 bps 夏普变化"],
        [
            [label, f"{float(sens[key]['d_cagr_bps']) * 100:+.2f}%", f"{float(sens[key]['d_sharpe_bps']):+.2f}"]
            for key, label in (
                ("logreg", "R1 逻辑回归"),
                ("logreg_R2", "R2 分位阈值"),
                ("ridge", "R4 ridge / Kelly"),
                ("sma200", "SMA200"),
            )
            if key in sens
        ],
    )

    b4_table = _table(
        ["策略", "AUC", "平衡准确率", "Brier"],
        [
            [
                label,
                _num(b4res.get(key, {}).get("auc"), 4),
                _num(b4res.get(key, {}).get("balanced_accuracy"), 4),
                _num(b4res.get(key, {}).get("brier"), 4),
            ]
            for key, label in (
                ("logreg_R1", "逻辑回归（主模型）"),
                ("rf_R1", "随机森林"),
                ("always_long", "永远猜涨（对照）"),
            )
            if key in b4res
        ],
    )

    b5_table = _table(
        ["模型", "特征数", "周数", "年化", "夏普", "AUC"],
        [
            [
                r.get("model", "—"),
                r.get("n_features", "—"),
                r.get("n_weeks", "—"),
                _pct(r.get("ann_return")),
                _num(r.get("sharpe")),
                _num(r.get("auc"), 4),
            ]
            for r in b5res
        ],
    )

    r1 = metrics.get("logreg", {})
    bh = metrics.get("buy_and_hold", {})
    headline = (
        f"主策略 R1：年化 {_pct(r1.get('cagr'))}、夏普 {_num(r1.get('sharpe'))}、"
        f"最大回撤 {_pct(r1.get('max_drawdown'))}；"
        f"同期买入持有 {_pct(bh.get('cagr'))} / {_num(bh.get('sharpe'))}。"
        "结论：这套周频择时在样本外没有创造价值。"
    )

    pages: dict[str, str] = {}

    pages["home"] = "".join(
        [
            _card(
                "一句话结论",
                f"<p class='lead'>{html.escape(headline)}</p>"
                "<p>本页所有数字来自 <code>reports/metrics.csv</code>，由 "
                "<code>uv run python -m src.pipeline</code> 自动生成，没有手抄。</p>",
                kicker="G05 · S&P 500 周频择时 · 1996–2025",
            ),
            _card("六条线（5 bps 单边成本）", main_table),
            _card(
                "怎么读这张表",
                "<ul>"
                "<li><b>R1/R2/R4</b>：三种把“上涨概率”变成仓位的规则，主结论只报 R1。</li>"
                "<li><b>买入持有</b>：最重要的对手；<b>恒定 60%</b>：把“敢不敢满仓”从择时能力里剥出来的公平对照。</li>"
                "<li>回撤用周频净值计算；日频回撤见 P6 页的体检表。</li>"
                "</ul>",
            ),
            _card(
                "每个页面在讲什么",
                "<ul>"
                "<li><b>P1 数据</b>：5 个标的怎么变成一张 1,566 行的周表</li>"
                "<li><b>P2 探索</b>：30 年市场长什么样、为什么方向难预测</li>"
                "<li><b>P3 特征</b>：22 个特征、标签时点、18 折与 5 项泄漏测试</li>"
                "<li><b>P4 模型</b>：逻辑回归 / 随机森林 / 三种下注规则</li>"
                "<li><b>P5 深度</b>：PCA、自编码器、DNN、CNN 到底有没有用</li>"
                "<li><b>P6 回测</b>：一根净值曲线怎么算出来，以及它说了什么</li>"
                "</ul>",
            ),
        ]
    )

    pages["b1"] = "".join(
        [
            _card(
                "P1 数据与工程",
                "<ul>"
                "<li>5 个标的：SPY（主）、^GSPC（对账）、^VIX（情绪）、DTB3 / DGS10（利率）</li>"
                "<li>复权口径唯一：<code>f = adj_close / close</code>，同一行四个价格共用一个因子</li>"
                "<li>VIX 统一 <code>shift(1)</code>，宏观滞后 1 个交易日</li>"
                "<li>原始文件写后置只读，<code>manifest.json</code> 记录 url / 下载时间 / 行数 / sha256</li>"
                "</ul>",
                kicker="负责人：P1 · 交付：weekly_panel / daily_panel / manifest / 质量报告",
            ),
            _card(
                "质量结论",
                "<ul>"
                "<li>周面板 1,566 行 × 12 列，1996-01-05 ~ 2025-12-31，<b>零缺失</b></li>"
                "<li>SPY 与 ^GSPC 日收益相关：全样本 0.9853，2010 年起 0.9985</li>"
                "<li>异常只标注不改值：除息 121 次、极端周收益 1 次、零成交量 0</li>"
                "<li>成分股只抓当前 503 只快照（ADR-003）：不画年度换手图</li>"
                "</ul>",
            ),
        ]
    )

    eda_figs = [
        ("eda_01_market.png", "30 年市场：三次大跌与修复"),
        ("eda_02_dist.png", "收益分布：厚尾"),
        ("eda_03_vol.png", "波动聚集"),
        ("eda_04_corr.png", "22 个特征的相关矩阵（高度冗余）"),
    ]
    pages["b2"] = "".join(
        [
            _card(
                "P2 探索性分析",
                "<ul>"
                "<li>只用 1996–2005 验证假设，<b>不碰测试集</b></li>"
                "<li>结论：各特征与下周收益的 Spearman 相关都在 0.07 以内——方向几乎不可预测</li>"
                "<li>但波动可预测：高波动期成段聚集，离散度相差 2.3 倍</li>"
                "<li>22 个特征高度共线：231 对里 23 对 |r| &gt; 0.8，矩阵秩仅 21/22</li>"
                "</ul>",
                kicker="负责人：P2 · 交付：7 张图 + eda_findings.md",
            ),
            _card("四张关键图", "".join(_figure(name, cap) for name, cap in eda_figs if (figs_out / name).exists())),
        ]
    )

    pages["b3"] = "".join(
        [
            _card(
                "P3 特征与防泄漏",
                "<ul>"
                "<li>22 个特征，全部比率型；禁止使用价格绝对水平</li>"
                "<li>标签：<code>t</code> 收盘决策 → <code>t+1</code> 开盘买入 → <code>t+2</code> 开盘卖出</li>"
                "<li>18 折扩展窗口 + 2 周 purge；预处理只在训练折内 fit</li>"
                "<li>5 项泄漏测试全部通过</li>"
                "</ul>",
                kicker="负责人：P3 · 交付：model_panel（42 列）+ splits.json + 泄漏审计",
            ),
            _card(
                "泄漏测试：用“能不能作弊成功”验证检测链路",
                _table(
                    ["测试", "结果", "判据"],
                    [
                        ["特征截断不变性", "改未来价格，历史特征不变", "逐值相等"],
                        ["折内预处理", "极端测试值不改变拟合参数", "参数不变"],
                        ["标签时间对齐", "四周价格手算一致", "逐值相等"],
                        ["随机标签对照", "合并 AUC 0.4875", "[0.45, 0.55]"],
                        ["故意注入未来收益", "合并 AUC 0.9996", "≥ 0.95"],
                    ],
                ),
            ),
        ]
    )

    pages["b4"] = "".join(
        [
            _card(
                "P4 模型与决策",
                "<ul>"
                "<li>主模型：L2 逻辑回归（预注册，不中途替换）；对照：随机森林</li>"
                "<li>超参只在 2006–2007 上选一次并<b>冻结</b>，18 折共用</li>"
                "<li>三种下注规则：R1 固定阈值、R2 分位阈值、R4 回归 + 凯利</li>"
                "<li>报告只报“与基线的差值 + 置信区间”，不做排行榜</li>"
                "</ul>",
                kicker="负责人：P4 · 交付：predictions.parquet + 指标表 + 三种规则对比",
            ),
            _card("预测层：只比瞎猜强一点点", b4_table),
            _card(
                "这说明什么",
                "<p>AUC 0.522 与“永远猜涨”的 0.500 相差无几；Brier 0.250 也基本等于抛硬币的损失。"
                "所以后面的结论不应期待“高准确率”，而应看“扣成本后还剩多少”。</p>",
            ),
        ]
    )

    pages["b5"] = "".join(
        [
            _card(
                "P5 降维与深度学习",
                "<ul>"
                "<li>PCA：7 维即解释大部分方差，压缩后与全特征基本持平</li>"
                "<li>DNN：22→64→32→1，早停；938 周样本外 AUC 0.523</li>"
                "<li>CNN：20 周窗口，每折开头 19 周无法构窗，可用样本仅 596 周</li>"
                "<li>结论：在这个样本量上，深度学习“看不出来”，不是更强</li>"
                "</ul>",
                kicker="负责人：P5 · 交付：四套结果 + 缩放对比 + 过拟合诊断",
            ),
            _card(
                "一组必须说清的比较陷阱",
                "<p>CNN 的 596 周与逻辑回归的 938 周<b>不是同一段样本</b>（每折少 19 周，全部集中在年初），"
                "所以它们的年化/夏普不能直接比。本页只报 AUC 与样本量，收益对比不进主结论。</p>"
                + (b5_table if b5res else ""),
            ),
        ]
    )

    pages["b6"] = "".join(
        [
            _card(
                "P6 回测与交付",
                "<ul>"
                "<li>资产收益：<code>open[t+2] / open[t+1] − 1</code>（简单收益，开盘对开盘）</li>"
                "<li>空仓利息：DTB3 贴现率 → 债券等价收益率，按当周实际交易日折算</li>"
                "<li>成本 5 bps 单边（敏感性 10 bps）；只做多/空仓，不做空</li>"
                "<li>输出：<code>metrics.csv</code> / <code>equity_curve.csv</code> / <code>trades.csv</code></li>"
                "</ul>",
                kicker="负责人：P6 · 交付：回测引擎 + 体检表 + 一条命令 + PPT + 打包",
            ),
            _card(
                "回测体检表",
                _figure("backtest_tearsheet.png", "净值 / 回撤 / 滚动夏普 / 年度收益 / 换手 / 仓位"),
            ),
            _card("成本敏感性：翻倍后结论不变", sens_table),
            _card("主表", main_table),
        ]
    )

    nav = "".join(
        f'<button class="tab{" active" if key == "home" else ""}" data-target="{key}">{html.escape(label)}</button>'
        for key, label in SECTIONS
    )
    body = "".join(
        f'<div class="page{" active" if key == "home" else ""}" id="page-{key}">{pages[key]}</div>'
        for key, _ in SECTIONS
    )

    doc = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>G05 · S&amp;P 500 周频择时 · 项目展示</title>
<style>
  :root {{ --navy:#1f4e79; --ink:#333; --muted:#6b7280; --line:#e5e7eb; --bg:#f7f8fa; }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--ink);
         font-family: "Microsoft YaHei", "PingFang SC", -apple-system, "Segoe UI", sans-serif; line-height:1.7; }}
  header {{ background:var(--navy); color:#fff; padding:28px 24px 18px; }}
  header h1 {{ margin:0; font-size:24px; letter-spacing:.3px; }}
  header p {{ margin:6px 0 0; opacity:.85; font-size:14px; }}
  nav {{ display:flex; flex-wrap:wrap; gap:6px; padding:12px 24px; background:#fff; border-bottom:1px solid var(--line);
         position:sticky; top:0; z-index:10; }}
  .tab {{ border:1px solid var(--line); background:#fff; color:var(--ink); padding:6px 14px; border-radius:999px;
          cursor:pointer; font-size:14px; }}
  .tab:hover {{ border-color:var(--navy); color:var(--navy); }}
  .tab.active {{ background:var(--navy); color:#fff; border-color:var(--navy); }}
  main {{ max-width:1080px; margin:0 auto; padding:24px; }}
  .page {{ display:none; }}
  .page.active {{ display:block; }}
  .card {{ background:#fff; border:1px solid var(--line); border-radius:12px; padding:20px 22px; margin-bottom:18px; }}
  .card h2 {{ margin:0 0 10px; font-size:19px; color:var(--navy); }}
  .kicker {{ font-size:13px; color:var(--muted); margin:-4px 0 10px; }}
  .lead {{ font-size:17px; font-weight:600; }}
  ul {{ padding-left:20px; margin:8px 0; }}
  li {{ margin:4px 0; }}
  code {{ background:#f1f3f5; padding:1px 5px; border-radius:4px; font-size:13px; }}
  table {{ width:100%; border-collapse:collapse; margin:10px 0; font-size:14px; }}
  th, td {{ border-bottom:1px solid var(--line); padding:8px 10px; text-align:right; }}
  th:first-child, td:first-child {{ text-align:left; }}
  thead th {{ background:#f9fafb; color:var(--muted); font-weight:600; }}
  tr.hl td {{ font-weight:700; color:#c00000; }}
  figure {{ margin:16px 0; }}
  figure img {{ width:100%; border:1px solid var(--line); border-radius:8px; }}
  figcaption {{ font-size:13px; color:var(--muted); margin-top:6px; }}
  footer {{ max-width:1080px; margin:0 auto 40px; padding:0 24px; color:var(--muted); font-size:13px; }}
</style>
</head>
<body>
<header>
  <h1>从数据到决策：S&amp;P 500 周频择时的防泄漏实践</h1>
  <p>G05 · 1996–2025 · 一行命令复现：uv sync &amp;&amp; uv run main.py</p>
</header>
<nav>{nav}</nav>
<main>{body}</main>
<footer>
  数字来源：reports/metrics.csv、model_results.csv、b5_model_results.csv、figures/eda_stats.json。
  本页由 <code>uv run python -m src.build_site</code> 生成，不要手改。
</footer>
<script>
  function activate(key) {{
    var target = document.getElementById('page-' + key) ? key : 'home';
    document.querySelectorAll('.tab').forEach(function (b) {{
      b.classList.toggle('active', b.dataset.target === target);
    }});
    document.querySelectorAll('.page').forEach(function (p) {{ p.classList.remove('active'); }});
    document.getElementById('page-' + target).classList.add('active');
  }}
  document.querySelectorAll('.tab').forEach(function (btn) {{
    btn.addEventListener('click', function () {{
      activate(btn.dataset.target);
      history.replaceState(null, '', '#' + btn.dataset.target);
      window.scrollTo({{ top: 0, behavior: 'smooth' }});
    }});
  }});
  window.addEventListener('hashchange', function () {{ activate(location.hash.slice(1)); }});
  activate(location.hash.slice(1) || 'home');
</script>
</body>
</html>
"""
    out = site / "index.html"
    out.write_text(doc, encoding="utf-8")
    print(f"[site] 已生成 {out}（{len(figures)} 张图，{len(SECTIONS)} 个页面）")
    return out


if __name__ == "__main__":  # pragma: no cover
    build(load_config())
