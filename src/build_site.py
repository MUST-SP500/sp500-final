"""展示网页（B6）：生成 site/index.html，双击可开、离线可用。"""

from __future__ import annotations

import csv
import html
import json
import shutil
from pathlib import Path

from src.config import load_config, project_path

TABS = [("home", "首页"), ("b1", "P1 数据"), ("b2", "P2 探索"), ("b3", "P3 特征"),
        ("b4", "P4 模型"), ("b5", "P5 深度"), ("b6", "P6 回测")]

CSS = """
:root{--navy:#1f4e79;--ink:#333;--muted:#6b7280;--line:#e5e7eb;--bg:#f7f8fa}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);line-height:1.7;
     font-family:"Microsoft YaHei","PingFang SC",-apple-system,"Segoe UI",sans-serif}
header{background:var(--navy);color:#fff;padding:26px 24px 18px}
header h1{margin:0;font-size:23px}header p{margin:6px 0 0;opacity:.85;font-size:14px}
nav{display:flex;flex-wrap:wrap;gap:6px;padding:12px 24px;background:#fff;
    border-bottom:1px solid var(--line);position:sticky;top:0;z-index:9}
.tab{border:1px solid var(--line);background:#fff;color:var(--ink);padding:6px 14px;
     border-radius:999px;cursor:pointer;font-size:14px}
.tab.active{background:var(--navy);color:#fff;border-color:var(--navy)}
main{max-width:1080px;margin:0 auto;padding:24px}
.page{display:none}.page.active{display:block}
.card{background:#fff;border:1px solid var(--line);border-radius:12px;padding:20px 22px;margin-bottom:18px}
.card h2{margin:0 0 8px;font-size:19px;color:var(--navy)}
.kicker{font-size:13px;color:var(--muted);margin:0 0 10px}
.lead{font-size:17px;font-weight:600}
ul{padding-left:20px;margin:8px 0}li{margin:4px 0}
code{background:#f1f3f5;padding:1px 5px;border-radius:4px;font-size:13px}
table{width:100%;border-collapse:collapse;margin:10px 0;font-size:14px}
th,td{border-bottom:1px solid var(--line);padding:8px 10px;text-align:right}
th:first-child,td:first-child{text-align:left}
thead th{background:#f9fafb;color:var(--muted);font-weight:600}
tr.hl td{font-weight:700;color:#c00000}
figure{margin:16px 0}figure img{width:100%;border:1px solid var(--line);border-radius:8px}
figcaption{font-size:13px;color:var(--muted);margin-top:6px}
footer{max-width:1080px;margin:0 auto 40px;padding:0 24px;color:var(--muted);font-size:13px}
"""

JS = """
function show(key){var k=document.getElementById('page-'+key)?key:'home';
  document.querySelectorAll('.tab').forEach(function(b){b.classList.toggle('active',b.dataset.k===k)});
  document.querySelectorAll('.page').forEach(function(p){p.classList.remove('active')});
  document.getElementById('page-'+k).classList.add('active')}
document.querySelectorAll('.tab').forEach(function(b){b.onclick=function(){
  show(b.dataset.k);history.replaceState(null,'','#'+b.dataset.k);scrollTo({top:0,behavior:'smooth'})}});
addEventListener('hashchange',function(){show(location.hash.slice(1))});
show(location.hash.slice(1)||'home');
"""


def _csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _pct(x, digits=1):
    try:
        return f"{float(x) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "—"


def _num(x, digits=2):
    try:
        return f"{float(x):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def _table(headers, rows, highlight=None):
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join(
        f'<tr{" class=hl" if i == highlight else ""}>'
        + "".join(f"<td>{html.escape(str(c))}</td>" for c in row) + "</tr>"
        for i, row in enumerate(rows))
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _card(title, body, kicker=""):
    return (f'<section class="card"><h2>{html.escape(title)}</h2>'
            f'{f"<p class=kicker>{html.escape(kicker)}</p>" if kicker else ""}{body}</section>')


def _fig(name, caption):
    return (f'<figure><img src="figures/{name}" loading="lazy" alt="{html.escape(caption)}">'
            f"<figcaption>{html.escape(caption)}</figcaption></figure>")


def _bullets(items):
    return "<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>"


def build(cfg: dict | None = None) -> Path:
    cfg = cfg if cfg is not None else load_config()
    reports = project_path("reports")
    site = project_path("site")
    if site.exists():
        shutil.rmtree(site)
    (site / "figures").mkdir(parents=True)

    figures = sorted((reports / "figures").glob("*.png"))
    for png in figures:
        shutil.copy2(png, site / "figures" / png.name)

    metrics = {r["strategy"]: r for r in _csv(reports / "metrics.csv")}
    sens = {r["strategy"]: r for r in _csv(reports / "metrics_sensitivity.csv")}
    b4 = {r["strategy"]: r for r in _csv(reports / "model_results.csv")}
    b5 = _csv(reports / "b5_model_results.csv")

    def row(key, label):
        m = metrics.get(key, {})
        return [label, _pct(m.get("cagr")), _num(m.get("sharpe")),
                _pct(m.get("max_drawdown")), _num(m.get("annual_turnover"), 1)]

    main_table = _table(["策略", "年化", "夏普", "最大回撤", "年换手"], [
        row("logreg", "R1 逻辑回归（主策略）"), row("logreg_R2", "R2 分位阈值"),
        row("ridge", "R4 ridge/Kelly"), row("buy_and_hold", "买入持有"),
        row("const_60", "恒定 60%"), row("sma200", "SMA200")], highlight=0)

    sens_table = _table(["策略", "5→10 bps 年化", "5→10 bps 夏普"], [
        [label, f"{float(sens[k]['d_cagr_bps']) * 100:+.2f}%", f"{float(sens[k]['d_sharpe_bps']):+.2f}"]
        for k, label in (("logreg", "R1"), ("logreg_R2", "R2"), ("ridge", "R4"), ("sma200", "SMA200"))
        if k in sens])

    b4_table = _table(["预测器", "AUC", "平衡准确率", "Brier"], [
        [label, _num(b4.get(k, {}).get("auc"), 4), _num(b4.get(k, {}).get("balanced_accuracy"), 4),
         _num(b4.get(k, {}).get("brier"), 4)]
        for k, label in (("logreg_R1", "逻辑回归"), ("rf_R1", "随机森林"), ("always_long", "永远猜涨"))
        if k in b4])

    b5_table = _table(["模型", "特征", "周数", "年化", "夏普", "AUC"], [
        [r["model"], r["n_features"], r["n_weeks"], _pct(r["ann_return"]), _num(r["sharpe"]), _num(r["auc"], 4)]
        for r in b5]) if b5 else ""

    r1, bh = metrics.get("logreg", {}), metrics.get("buy_and_hold", {})
    headline = (f"主策略 R1：年化 {_pct(r1.get('cagr'))}、夏普 {_num(r1.get('sharpe'))}、"
                f"最大回撤 {_pct(r1.get('max_drawdown'))}；买入持有 "
                f"{_pct(bh.get('cagr'))} / {_num(bh.get('sharpe'))}。"
                "结论：这套周频择时在样本外没有创造价值。")

    pages = {
        "home": _card("一句话结论", f"<p class=lead>{html.escape(headline)}</p>"
                      "<p>数字来自 <code>reports/metrics.csv</code>，由 "
                      "<code>uv run python -m src.pipeline</code> 生成。</p>", "G05 · 1996–2025")
        + _card("六条线（5 bps 单边成本）", main_table)
        + _card("页面导览", _bullets([
            "<b>P1</b>：5 个标的怎么变成 1,566 行周表",
            "<b>P2</b>：30 年市场与“方向不可预测”的证据",
            "<b>P3</b>：22 特征、标签时点、18 折、5 项泄漏测试",
            "<b>P4</b>：逻辑回归 / 随机森林 / 三种下注规则",
            "<b>P5</b>：PCA、自编码器、DNN、CNN 的结论",
            "<b>P6</b>：净值怎么算出来，以及它说了什么"])),

        "b1": _card("P1 数据与工程", _bullets([
            "5 个标的：SPY（主）、^GSPC（对账）、^VIX（情绪）、DTB3 / DGS10（利率）",
            "复权口径唯一：<code>f = adj_close / close</code>",
            "VIX 统一 <code>shift(1)</code>，宏观滞后 1 个交易日",
            "manifest 记录 url / 下载时间 / 行数 / sha256"]), "负责人：P1")
        + _card("质量结论", _bullets([
            "周面板 1,566 行 × 12 列，1996-01-05 ~ 2025-12-31，零缺失",
            "SPY 与 ^GSPC 日收益相关：全样本 0.9853，2010 年起 0.9985",
            "异常只标注不改值：除息 121 次、极端周收益 1 次、零成交量 0",
            "成分股只有当前 503 只快照（ADR-003），不画年度换手图"])),

        "b2": _card("P2 探索性分析", _bullets([
            "只用 1996–2005 验证假设，不碰测试集",
            "各特征与下周收益的 Spearman 相关都在 0.07 以内",
            "波动可预测：高波动成段聚集，离散度差 2.3 倍",
            "22 个特征高度共线：23 对 |r| &gt; 0.8，矩阵秩 21/22"]), "负责人：P2")
        + _card("关键图", "".join(_fig(n, c) for n, c in [
            ("eda_01_market.png", "30 年市场与三次大跌"),
            ("eda_02_dist.png", "收益分布：厚尾"),
            ("eda_03_vol.png", "波动聚集"),
            ("eda_04_corr.png", "22 个特征的相关性")])),

        "b3": _card("P3 特征与防泄漏", _bullets([
            "22 个特征，全部比率型；禁止价格绝对水平",
            "标签：<code>t</code> 收盘决策 → <code>t+1</code> 开盘买入 → <code>t+2</code> 开盘卖出",
            "18 折扩展窗口 + 2 周 purge；预处理只在训练折 fit"]), "负责人：P3")
        + _card("5 项泄漏测试", _table(["测试", "结果", "判据"], [
            ["特征截断不变性", "改未来价格，历史特征不变", "逐值相等"],
            ["折内预处理", "极端值不改变拟合参数", "参数不变"],
            ["标签时间对齐", "四周价格手算一致", "逐值相等"],
            ["随机标签对照", "合并 AUC 0.4875", "[0.45, 0.55]"],
            ["注入未来收益", "合并 AUC 0.9996", "≥ 0.95"]])),

        "b4": _card("P4 模型与决策", _bullets([
            "主模型：L2 逻辑回归；对照：随机森林",
            "超参只在 2006–2007 选一次并冻结",
            "三种规则：R1 阈值 / R2 分位 / R4 回归 + 凯利"]), "负责人：P4")
        + _card("预测层：只比瞎猜强一点", b4_table)
        + _card("含义", "<p>AUC 0.522 对 0.500，几乎没有增量。"
                        "所以后面不看准确率，只看扣成本后剩多少。</p>"),

        "b5": _card("P5 降维与深度学习", _bullets([
            "PCA：7 维解释大部分方差，压缩后与全特征持平",
            "DNN：938 周样本外 AUC 0.523",
            "CNN：20 周窗口，每折少 19 周，可用样本 596 周",
            "结论：这个样本量上，深度学习“看不出来”"]), "负责人：P5")
        + _card("比较陷阱", "<p>CNN 的 596 周与逻辑回归的 938 周不是同一段样本，"
                            "年化/夏普不可直接比。本页只报 AUC 与样本量。</p>" + b5_table),

        "b6": _card("P6 回测与交付", _bullets([
            "资产收益：<code>open[t+2] / open[t+1] − 1</code>",
            "空仓利息：DTB3 → 债券等价收益率，按实际交易日折算",
            "成本 5 bps 单边（敏感性 10 bps）；只做多/空仓"]), "负责人：P6")
        + _card("回测体检表", _fig("backtest_tearsheet.png", "净值/回撤/滚动夏普/年度收益/换手/仓位"))
        + _card("成本敏感性", sens_table)
        + _card("主表", main_table),
    }

    nav = "".join(f'<button class="tab{" active" if k == "home" else ""}" data-k="{k}">{label}</button>'
                  for k, label in TABS)
    body = "".join(f'<div class="page{" active" if k == "home" else ""}" id="page-{k}">{pages[k]}</div>'
                   for k, _ in TABS)

    doc = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>G05 · S&P 500 周频择时</title><style>{CSS}</style></head><body>
<header><h1>从数据到决策：S&amp;P 500 周频择时的防泄漏实践</h1>
<p>G05 · 1996–2025 · 复现：uv sync &amp;&amp; uv run main.py</p></header>
<nav>{nav}</nav><main>{body}</main>
<footer>数字来源：reports/metrics.csv 等；本页由 <code>uv run python -m src.build_site</code> 生成。</footer>
<script>{JS}</script></body></html>"""

    out = site / "index.html"
    out.write_text(doc, encoding="utf-8")
    print(f"[site] {out}（{len(figures)} 张图）")
    return out


if __name__ == "__main__":
    build(load_config())
