# 复现日志（B1）

> 生成时间：2026-11-23 13:10　|　由 `uv run python -m src.reproduce` 自动生成

## 1. 环境

| 项目 | 值 |
|---|---|
| 系统 | Windows 11（AMD64） |
| Python | 3.12.15（D:\MUSTLABS\Master_Class_YanYi_Shang\Open-source data tools\sp500-final\.venv\Scripts） |
| uv | uv 0.13.9 (6b2d39dcc 2026-11-17 x86_64-pc-windows-msvc) |
| 项目版本 | v0.1.0 |
| git commit | 99334ad |
| pandas / numpy | 2.3.3 / 2.5.5 |
| 随机种子 | 42（config.yaml） |

## 2. 一键复现命令

```bash
uv sync                                              # 按 uv.lock 装依赖
uv run python -m src.pipeline --stage download        # 抓 5 个标的（约 3 秒）
uv run python -m src.pipeline --stage clean           # 复权 + 周频 + 面板（约 0.5 秒）
uv run python -m src.data_quality                     # 质量报告 + 异常清单
uv run python -m src.scrape_wiki                      # 成分股快照
uv run python -m src.export_csv                       # 导出 CSV 供核对
uv run pytest                                         # 全部测试
```

或一条命令跑完前五步：`uv run python -m src.reproduce`

## 3. 本次运行

| 步骤 | 耗时 (s) | 产出 |
|---|---|---|
| download | 2.34 | 5 个标的（雅虎 3 + FRED 2） |
| clean | 0.06 | 复权 + 周频 + 面板 |
| quality | 0.23 | 质量报告 + 异常清单 |
| wiki | 1.56 | 当前成分表快照 |
| export | 0.06 | 面板导出 CSV 供核对 |
| **合计** | **4.25** |  |

- 周面板：1,566 行 × 12 列，1996-01-05 ~ 2025-12-31
- 日面板：7,550 行 × 12 列
- 本次重跑前清除了 12 个生成文件（原始数据、面板、CSV、manifest）

## 4. 删数据重跑的一致性校验

| 指标 | 重跑前 | 重跑后 | 判定 |
|---|---|---|---|
| weekly.rows | 1566 | 1566 | 一致 |
| weekly.cols | 12 | 12 | 一致 |
| weekly.first_date | 1996-01-05 | 1996-01-05 | 一致 |
| weekly.last_date | 2025-12-31 | 2025-12-31 | 一致 |
| weekly.volume_sum | 690091047500.0 | 690091047500.0 | 一致 |
| weekly.na_total | 0 | 0 | 一致 |
| weekly.first_close | 36.3055 | 36.3055 | 一致 |
| weekly.last_close | 676.635 | 676.635 | 一致 |
| daily.rows | 7550 | 7550 | 一致 |
| daily.cols | 12 | 12 | 一致 |
| daily.first_date | 1996-01-02 | 1996-01-02 | 一致 |
| daily.last_date | 2025-12-31 | 2025-12-31 | 一致 |
| daily.volume_sum | 690091047500.0 | 690091047500.0 | 一致 |
| daily.na_total | 3 | 3 | 一致 |
| daily.first_close | 36.6278 | 36.6278 | 一致 |
| daily.last_close | 676.635 | 676.635 | 一致 |
| raw.yahoo/GSPC.csv | 7550 | 7550 | 一致 |
| raw.yahoo/SPY.csv | 7550 | 7550 | 一致 |
| raw.yahoo/VIX.csv | 7550 | 7550 | 一致 |
| raw.fred/DGS10.csv | 7505 | 7505 | 一致 |
| raw.fred/DTB3.csv | 7505 | 7505 | 一致 |
| raw.wiki/constituents.csv | 503 | 503 | 一致 |

**结论**：全部一致，换台电脑可复现。

> `first_close` / `last_close` 允许 5e-4 的相对偏差：雅虎每次请求会重算复权因子，`adj_close` 天然有 1e-4 量级漂移（见数据字典）。行数、日期、成交量和缺失数必须逐位相同。

## 5. 已知限制

1. 雅虎 `adj_close` 每次重下会漂移约 1e-4，面板的 `close` 随之漂移；跨机核对用「行数 + 日期 + close 允许 5e-4」。
2. FRED 的 DGS10/DTB3 在 H.15 假期不出数，成交日与交易所差 45 天，已在前向填充上限 5 天内补齐。
3. 成分股只有当前快照（503 只），没有历史变更，无法画年度换手图（ADR-003）。
4. 末尾一周是不完整周（数据截至 2025-12-31），建模时该行目标值为空。
