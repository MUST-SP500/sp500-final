# 变量说明总表（Variable Descriptions）

> **想查"这个字段是什么意思"，只看这一个文件。**
> 口径怎么定、为什么这么定的 → `data_dictionary.md`；历史决策 → `decisions.md`；复现记录 → `reproduce_log.md`。

数据层由 B1（P1）负责。所有文件都在仓库内，路径相对仓库根目录。

---

## 0. 速查：哪个文件有什么

| 文件 | 粒度 | 行数 | 列数 | 用途 | 进 git |
|---|---|---|---|---|---|
| `data/raw/yahoo/SPY.csv` | 日 | 7,550 | 7 | 主标的原始价（未复权 + 雅虎复权收盘） | ❌ |
| `data/raw/yahoo/GSPC.csv` | 日 | 7,550 | 7 | `^GSPC` 指数，**只做校验**，不建模 | ❌ |
| `data/raw/yahoo/VIX.csv` | 日 | 7,550 | 7 | `^VIX` 恐慌指数 | ❌ |
| `data/raw/fred/DGS10.csv` | 日 | 7,505 | 2 | 10 年期美债收益率 | ❌ |
| `data/raw/fred/DTB3.csv` | 日 | 7,505 | 2 | 3 月期国库券贴现率 | ❌ |
| `data/raw/wiki/constituents.csv` | 快照 | 503 | 8 | 当前 S&P 500 成分股（幸存者偏差论证） | ❌ |
| `data/raw/manifest.json` | — | 6 条 | — | 每个原始文件的行数/区间/sha256/下载时间 | ❌ |
| `data/processed/daily_panel.parquet` | 日 | 7,550 | 12 | **B3 特征工程的输入** | ❌ |
| `data/processed/weekly_panel.parquet` | 周 | 1,566 | 12 | **建模与回测的主表** | ❌ |
| `data/processed/anomalies.csv` | — | 122 | 5 | 异常点清单（只标注，不改值） | ❌ |
| `data/processed/csv/*.csv` | — | — | — | 上面两个面板的 CSV 副本，供 Excel 核对 | ❌ |

区间统一为 `1996-01-01 ~ 2025-12-31`（交易日实际首日 1996-01-02，周面板首周 1996-01-05）。

---

## 1. `data/processed/weekly_panel.parquet`（建模主表）

| 变量 | 类型 | 单位 | 含义 | 计算方式 / 备注 |
|---|---|---|---|---|
| `date` | datetime | — | 该 ISO 周的**最后一个交易日** | 周键不是自然周五；假期周可能提前 |
| `iso_week` | string | — | ISO 周键 `YYYY-Www` | 例：`2020-W53`。跨年周归属上一年 |
| `open` | float | 美元（合成价） | 周内第一个交易日的复权开盘 | `open_raw × f` |
| `high` | float | 美元（合成价） | 周内复权最高价 | `max(high_adj)` |
| `low` | float | 美元（合成价） | 周内复权最低价 | `min(low_adj)` |
| `close` | float | 美元（合成价） | 周内最后一个交易日的复权收盘 | **算收益/特征一律用它** |
| `adj_close` | float | 美元（合成价） | 雅虎原始复权收盘（周内最后一天的值） | 恒等于 `close`，**只做对账** |
| `volume` | int | 股 | 周内成交量加总 | VIX 无成交量，故本列只来自 SPY |
| `n_trading_days` | int | 天 | 该周实际交易日数 | 5 = 完整周；4 = 有假日；1 = 极端情况（2001-W37） |
| `vix` | float | 点 | VIX 收盘，**已 `shift(1)`** | 等于该周最后交易日的**前一交易日** VIX |
| `dgs10` | float | % | 10 年期美债收益率，滞后 1 个交易日 | 前向填充上限 5 天 |
| `dtb3` | float | % | 3 月期国库券贴现率，滞后 1 个交易日 | 前向填充上限 5 天 |

**不含** `label_ret` / `label_up` / `next_ret`——这三列由 `src/labels.py`（B3）在建模阶段追加。

---

## 2. `data/processed/daily_panel.parquet`（B3 特征工程的输入）

| 变量 | 类型 | 单位 | 含义 | 计算方式 / 备注 |
|---|---|---|---|---|
| `date` | datetime | — | 交易日 | 7,550 个 |
| `f` | float | 无量纲 | **当日复权因子** | `adj_close / close_raw`，1996 年 0.589 → 2025 年 0.989 |
| `open` | float | 美元（合成价） | 复权开盘 | `open_raw × f` |
| `high` | float | 美元（合成价） | 复权最高 | `high_raw × f` |
| `low` | float | 美元（合成价） | 复权最低 | `low_raw × f` |
| `close` | float | 美元（合成价） | 复权收盘 | 恒等于 `adj_close`；**与周面板 `close` 同口径** |
| `adj_close` | float | 美元（合成价） | 雅虎原始复权字段 | 只做对账 |
| `close_raw` | float | 美元 | **未复权收盘**（1996-01-02 为 62.14） | 仅日频有，用于核对 `f` |
| `volume` | int | 股 | 当日成交量 | — |
| `vix` | float | 点 | VIX 收盘，`shift(1)` 后 | 首行（1996-01-02）为空 |
| `dgs10` | float | % | 10Y 收益率，滞后 1 日 + 限长填充 | 首行为空 |
| `dtb3` | float | % | 3M 贴现率，滞后 1 日 + 限长填充 | 首行为空 |

> 全套数据只有这 3 个空值（日频首行的宏观列）。周面板零缺失。

---

## 3. `data/raw/yahoo/{SPY,GSPC,VIX}.csv`（雅虎原始快照）

| 变量 | 类型 | 单位 | 含义 | 备注 |
|---|---|---|---|---|
| `date` | date | — | 交易日 | 交易所日历 |
| `open` | float | 美元 | **未复权**开盘 | 拆分前/后的原始报价 |
| `high` | float | 美元 | 未复权最高 | — |
| `low` | float | 美元 | 未复权最低 | — |
| `close` | float | 美元 | **未复权**收盘 | SPY 1996-01-02 为 62.14 |
| `adj_close` | float | 美元（合成价） | 雅虎复权收盘（含分红再投资） | 同一区间每次重下会漂 1e-4 |
| `volume` | int | 股 | 成交量 | 指数（GSPC/VIX）恒为 0 或缺失 |

---

## 4. `data/raw/fred/{DGS10,DTB3}.csv`（FRED 原始快照）

| 变量 | 类型 | 单位 | 含义 | 备注 |
|---|---|---|---|---|
| `date` | date | — | 发布日 | H.15 假期不出数，行数比交易所少 45 天 |
| `dgs10` | float | 百分点 | 10 年期国债**收益率**（constant maturity） | 4.97 表示 4.97% |
| `dtb3` | float | 百分点 | 3 月期国库券**贴现率** | 可能为负（2020 年最低 −0.05） |

文件里列名就叫 `dgs10` / `dtb3`（小写）。原始值里的 `.` 表示当天无数据，读入后为 NaN。

---

## 5. `data/raw/wiki/constituents.csv`（成分股快照）

| 变量 | 类型 | 含义 | 备注 |
|---|---|---|---|
| `symbol` | string | 交易代码 | 保留 `BRK.B` 这类点号；**不参与建模** |
| `security` | string | 公司名 | — |
| `gics_sector` | string | GICS 一级板块（11 类） | 画板块分布图用 |
| `gics_sub_industry` | string | GICS 三级子行业 | — |
| `headquarters_location` | string | 总部所在地 | — |
| `date_added` | date | 加入指数日期 | 最早可到 1957 年 |
| `cik` | int | SEC 中央索引号 | — |
| `founded` | int | 成立年份 | 源头就有空值 |

---

## 6. `data/processed/anomalies.csv`（异常清单，只标注不改值）

| 变量 | 类型 | 含义 |
|---|---|---|
| `date` | date | 异常发生的日期 |
| `kind` | string | 异常类型，见下表 |
| `value` | float | 该异常的数值（收益率 / 因子变化 / 连续天数） |
| `detail` | string | 人读的描述 |
| `action` | string | 处理方式 |

| `kind` | 含义 | 条数 |
|---|---|---|
| `extreme_daily_return` | 单日 \|收益\| > 20% | 0（SPY 历史最大单日 14.52%） |
| `extreme_weekly_return` | 单周 \|收益\| > 15% | 1（2008-10-10，−19.79%） |
| `zero_volume` | 成交量为 0 | 0 |
| `stale_close` | 连续 ≥3 日收盘完全相同（疑似停牌） | 0 |
| `dividend_event` | 复权因子**上跳** = 除息/拆分 | 121（年均 4.03） |
| `factor_drop` | 复权因子**下跳**（不该发生，说明上游回溯改了历史） | 0 |

---

## 7. `data/raw/manifest.json`（数据快照身份）

| 变量 | 含义 |
|---|---|
| `generated_at` | 本次清单生成时间（ISO 8601，带时区） |
| `files[].ticker` | 标的（`SPY` / `^GSPC` / `^VIX` / `DTB3` / `DGS10` / `SP500_CONSTITUENTS`） |
| `files[].source` | 来源（`yahoo` / `fred` / `wikipedia`） |
| `files[].url` | 实际请求的 URL |
| `files[].retrieved_at` | **下载时间**——报告里引用数据必须写这个 |
| `files[].rows` | 行数 |
| `files[].date_range` | 日期区间；快照类为 `snapshot` |
| `files[].sha256` | 文件指纹（快照身份，不是跨机一致性承诺） |
| `files[].file` | 相对 `data/raw/` 的路径 |

---

## 8. 命名约定

| 后缀 / 词 | 含义 |
|---|---|
| `_raw` | 未复权原始报价（仅 `close_raw` 存在） |
| `_adj` | 复权值（`open_adj` / `high_adj` / `low_adj` / `close_adj`，只在代码中间步骤出现） |
| `f` | 复权因子 `adj_close / close_raw` |
| `iso_week` | ISO 8601 周键 `YYYY-Www` |
| `n_` 前缀 | 计数（`n_trading_days`） |
| `dgs` / `dtb` | FRED 序列名前缀（DGS = Daily Goverment Securities，DTB = Daily Treasury Bill） |

## 9. 常用派生量（B3 会用到，B1 不预计算）

| 派生量 | 公式 | 说明 |
|---|---|---|
| 日简单收益 | `close.pct_change()` | 用复权 `close` |
| 日对数收益 | `np.log(close).diff()` | 标签用对数收益（`label_ret`） |
| 周收益 | 周 `close` 的 `pct_change()` | 周末到周末 |
| 已实现波动 | `ret.rolling(20).std() * sqrt(252)` | 年化 |
| 均线偏离 | `close / close.rolling(20).mean() - 1` | 日频算，周频取样 |

> **禁止**用价格绝对水平做特征（1996 年 36 美元 vs 2025 年 676 美元），只用比率型。
